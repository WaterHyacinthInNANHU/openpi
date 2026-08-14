"""π0.7 quality conditioning: read a precomputed tag artifact and put the tag in the prompt.

ONLINE TIER.

There is deliberately NO randomness here. The dropout was decided offline and baked into the
artifact (`axis.dataset.build_quality_labels`), so the realized tagged fraction is a number read
off disk rather than a reconstruction from eight workers' RNG states.

TWO OBJECTS, and the split is load-bearing. `QualityTaggedDataset` runs BEFORE any transform and
injects the tag using `__getitem__`'s own index argument -- the only unambiguously concat-global
index available. `AxisQualityConditioning` runs at the HEAD of the repack group, before
`RepackTransform` rebuilds the dict and drops every key not in its map. Reading `data["index"]`
instead would have been simpler and wrong: under `ConcatDataset` that key is the SUB-DATASET-local
row (measured -- see quality_conditioning_test.py), so every tag past the first sub-dataset would
land on the wrong frame with no error.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
from typing_extensions import override

from openpi.training.slb_cfg import apply_metadata
import openpi.transforms as _transforms

# The key the wrapper injects and the transform consumes. Not in RepackTransform's map: the
# transform runs first and rewrites `prompt`, after which the key has done its job.
QUALITY_KEY = "quality"

# Mirrors axis.dataset.quality_labels and axis.dataset.index_schedule.N_BINS. Duplicated (three
# ints) rather than imported: those modules are offline tier, and nothing under openpi/ imports
# `axis`. The duplication is caught by the transform's range guard, which refuses any tag that is
# neither NO_TAG nor a bin in [1, N_BINS]. `_NAME_PREFIX` likewise mirrors the builder's filename
# convention (build_quality_labels.check_artifact_filename).
NO_TAG = 0
NOT_TRAINABLE = 255
N_BINS = 5
_NAME_PREFIX = "quality_"


class QualityTags:
    """The artifact, plus the checks that bind it to this run."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        # NOT memory-mapped: `np.load(..., mmap_mode=...)` is silently ignored for a .npz (the
        # members are deflated inside a zip, so there is nothing to map), and asking for it would
        # only put a false claim in the code. The array is 2.2 MB and is read once in the parent
        # before forking; CPython's refcounts touch the ndarray OBJECT header, not its data
        # buffer, so the 2.2 MB of pages stay shared copy-on-write across the 8 workers anyway.
        with np.load(self.path, allow_pickle=False) as z:
            self.tag = np.asarray(z["tag"])
            self.prompts = [str(s) for s in z["prompts"]]
            self.meta = json.loads(str(z["meta"]))
        if self.tag.ndim != 1 or self.tag.dtype != np.uint8:
            # Not cast: a float or int64 array reads through `int()` without complaint, and a
            # truncated tag is a WRONG tag that is indistinguishable from a right one. Same
            # reasoning as ScheduleSampler's refusal to cast a float rows array.
            raise ValueError(
                f"quality artifact {self.path} has tag shape {self.tag.shape} dtype "
                f"{self.tag.dtype}; expected 1-D uint8 dense over the corpus row space"
            )
        if not self.prompts:
            raise ValueError(
                f"quality artifact {self.path} carries no prompts; the token-budget guard would "
                f"have nothing to check and would pass vacuously"
            )
        if self.meta.get("reward_id") is None:
            raise ValueError(
                f"quality artifact {self.path} has no 'reward_id' in its meta, so its filename "
                f"cannot be checked against its contents and the two CFG arms -- which share one "
                f"config name -- become indistinguishable. Rebuild it with "
                f"axis.dataset.build_quality_labels."
            )
        declared = self.meta.get("n_rows")
        if declared is not None and int(declared) != len(self.tag):
            # The builder refuses to WRITE this (build_quality_labels.quality_meta), so the file
            # was truncated or hand-edited after the fact: every provenance line logged from meta
            # would then describe a corpus this file does not cover.
            raise ValueError(
                f"quality artifact {self.path} carries {len(self.tag)} tags but its meta reports "
                f"n_rows={int(declared)}; the file disagrees with itself about the index space it "
                f"covers. Rebuild it with axis.dataset.build_quality_labels."
            )

    @property
    def reward_id(self) -> str:
        return str(self.meta["reward_id"])

    def check_reward_id(self, path: str | pathlib.Path) -> None:
        """Bind a `quality_<reward_id>.npz` filename to the artifact's own reward.

        `cfg_v2` and `cfg_phase` run under ONE config name (`pi05_axis_cfg`) and their artifacts
        are structurally identical, so the filename in `data.quality_path` is the only thing that
        says which reward built the tags. Names that make no claim are not checked.
        """
        stem = pathlib.Path(path).stem
        if not stem.startswith(_NAME_PREFIX):
            return
        claimed = stem[len(_NAME_PREFIX):]
        if claimed != self.reward_id:
            raise ValueError(
                f"quality artifact {path} is named for reward_id={claimed!r} but its own meta "
                f"reports {self.reward_id!r}. Both CFG arms run under pi05_axis_cfg, so this run "
                f"would record itself as {claimed!r} while conditioning on {self.reward_id!r}."
            )

    def check_dataset_rows(self, n_dataset_rows: int, roots_index: str | None = None) -> None:
        """The tag array is dense over the corpus; a different length means different frames."""
        if len(self.tag) != int(n_dataset_rows):
            raise ValueError(
                f"corpus mismatch: quality artifact {self.path} covers {len(self.tag)} rows but "
                f"the dataset built from {roots_index!r} has {int(n_dataset_rows)}. Every index "
                f"would still be in bounds and would mean a different frame. Rebuild the artifact "
                f"against this corpus."
            )

    def tag_for_row(self, index: int) -> int:
        i = int(index)
        if i < 0:
            # numpy reads a negative index from the END, so this would return a real tag for the
            # wrong row rather than raise -- the silent-wrap class ScheduleSampler also refuses.
            raise IndexError(
                f"negative row index {i} into quality artifact {self.path}; numpy would read it "
                f"from the tail and return some other row's tag."
            )
        q = int(self.tag[i])
        if q == NOT_TRAINABLE:
            raise KeyError(
                f"row {i} is marked not trainable in {self.path} but the sampler drew it. The "
                f"tag array and the row plan disagree about which rows exist; a default here "
                f"would silently untag part of the arm."
            )
        return q


class QualityTaggedDataset:
    """Inject `data["quality"]` using the index this was CALLED with.

    Wraps the raw concat dataset BEFORE `transform_dataset`, so the tag is present when the
    repack group's head transform runs. Backed by the dense uint8 array rather than a dict:
    1.3M dict entries copied into each of 8 workers is ~150-200 MB per worker, against 2.2 MB
    for the array -- the same reasoning `StrictWeightedRowDataset` records.
    """

    def __init__(self, dataset, tags: QualityTags):
        self._dataset = dataset
        self._tags = tags

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        # Tag first: a bad index must fail here rather than after the inner dataset has already
        # decoded a frame for it (and, for a negative index, returned a plausible wrong row).
        q = self._tags.tag_for_row(index)
        item = dict(self._dataset[index])
        item[QUALITY_KEY] = q
        return item


@dataclasses.dataclass(frozen=True)
class AxisQualityConditioning(_transforms.DataTransformFn):
    """Append `"\\nQuality: {q}"` to the prompt. Heads the pretrain repack group.

    `q == NO_TAG` yields the BARE prompt: that IS the unconditional branch, and it is why a
    dropped row must not emit `Quality: 0`.

    Every lookup RAISES on a miss. A `return data` fallback on either key would train the plain
    control under the arm's name with no symptom but an absent log line -- once for a missing
    wrapper, once for a repack order that put this transform after `RepackTransform` (which
    rebuilds the dict from its structure map and drops everything else).
    """

    @override
    def __call__(self, data: dict) -> dict:
        if QUALITY_KEY not in data:
            raise KeyError(
                f"{QUALITY_KEY!r} is not in the sample. This transform is wired into the repack "
                f"group but QualityTaggedDataset is not wired into the loader, so no tag ever "
                f"reaches it -- the arm would train as the plain BC control under its own name."
            )
        if "prompt" not in data:
            raise KeyError(
                "'prompt' is not in the sample. This transform must run at the HEAD of the "
                "repack group: RepackTransform rebuilds the dict from its structure map, so a "
                "transform placed after it never sees the prompt and the tag is never applied."
            )
        q = int(np.asarray(data[QUALITY_KEY]).reshape(-1)[0])
        if q != NO_TAG and not 1 <= q <= N_BINS:
            # NOT_TRAINABLE (255) leaking past a bypassed wrapper lands here, as does an artifact
            # built with a different bin count. Either would emit a `Quality: <n>` string the
            # tokenizer has never seen -- a silent sixth condition.
            raise ValueError(
                f"quality tag {q} is neither NO_TAG ({NO_TAG}) nor a bin in [1, {N_BINS}]. "
                f"{NOT_TRAINABLE} is the not-trainable sentinel and must never reach the prompt; "
                f"any other value means the artifact was built with a different bin count."
            )
        if q == NO_TAG:
            return data
        return {**data, "prompt": apply_metadata(data["prompt"], q, None)}
