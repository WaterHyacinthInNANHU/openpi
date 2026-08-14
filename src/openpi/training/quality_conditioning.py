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

TWO OBJECTS, ONE CONSTRUCTOR. The wiring does not get to compose them itself: `wrap_and_transform`
is the only supported entry point, because wrapper-without-transform is silent (see its docstring)
and a wiring task naturally reaches for the wrapper alone.
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
# `axis`.
#
# The transform's range guard catches only HALF of a drift in that duplicate. If the offline
# N_BINS GREW (5 -> 7), tags 6 and 7 are outside [1, N_BINS] and the guard raises. If it SHRANK
# (5 -> 3), every tag is still inside [1, 5], the guard never fires, and the arm trains 3-bin
# conditioning while this file, the config and the write-up all say 5 -- silently. So the offline
# bin count is read back from the artifact instead of inferred from the tags: `quality_meta`
# writes `bin_row_counts` keyed "1".."N_BINS", and `QualityTags.__init__` requires its largest key
# to equal N_BINS. That comparison is the ONLY place in the repo where the two tiers' bin counts
# meet, which is why it lives in the reader and not in the wiring.
#
# `_NAME_PREFIX` likewise mirrors the builder's filename convention
# (build_quality_labels.check_artifact_filename).
NO_TAG = 0
NOT_TRAINABLE = 255
N_BINS = 5
_NAME_PREFIX = "quality_"


class QualityTags:
    """The artifact, plus the checks that bind it to this run."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        # NOT memory-mapped: `np.load(..., mmap_mode=...)` is silently ignored for a .npz. It
        # returns an `NpzFile`, whose `__getitem__` reads each member through `zipfile` into a
        # fresh in-memory array and never forwards `mmap_mode` -- so asking for it would only put
        # a false claim in the code. NOT a compression story: measured, `np.savez` writes its
        # members with compress_type=0, i.e. STORED, uncompressed; `savez_compressed` is the one
        # that deflates. The array is 2.2 MB and is read once in the parent before forking;
        # CPython's refcounts touch the ndarray OBJECT header, not its data buffer, so the 2.2 MB
        # of pages stay shared copy-on-write across the 8 workers anyway.
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
        if declared is None:
            # REQUIRED, matching the sibling reader (ScheduleSampler.check_dataset_rows): the
            # builder always writes it, so an artifact without it predates the field and cannot be
            # told apart from one whose tag array was truncated. Accepting it would make the
            # self-consistency check below pass vacuously on exactly the file that needs it.
            raise ValueError(
                f"quality artifact {self.path} has no meta['n_rows'], so the tag array cannot be "
                f"checked against the corpus size the builder recorded. Rebuild it with "
                f"axis.dataset.build_quality_labels."
            )
        if int(declared) != len(self.tag):
            # The builder refuses to WRITE this (build_quality_labels.quality_meta), so the file
            # was truncated or hand-edited after the fact: every provenance line logged from meta
            # would then describe a corpus this file does not cover.
            raise ValueError(
                f"quality artifact {self.path} carries {len(self.tag)} tags but its meta reports "
                f"n_rows={int(declared)}; the file disagrees with itself about the index space it "
                f"covers. Rebuild it with axis.dataset.build_quality_labels."
            )
        self._check_bin_count()
        if not bool((self.tag == NO_TAG).any()):
            # DROP_WHOLE=0 (or a dropout that misfired) leaves every trainable row tagged, so the
            # model never sees the bare prompt, never learns p(a | no tag), and guidance at
            # inference -- which subtracts an unconditional forward pass -- is undefined. Nothing
            # online would notice: training loss, throughput and every prompt look normal.
            raise ValueError(
                f"quality artifact {self.path} has no NO_TAG ({NO_TAG}) rows, so CFG has no "
                f"unconditional branch to guide away from and the arm degenerates to plain "
                f"conditional BC. Rebuild it with a non-zero dropout "
                f"(axis.dataset.quality_labels.DROP_WHOLE)."
            )

    def _check_bin_count(self) -> None:
        """Bind the offline tier's N_BINS to this file's copy of it.

        The transform's range guard sees a GROWN offline bin count (tags outside [1, N_BINS]) and
        misses a SHRUNK one entirely, since every tag of a 3-bin build is a legal 5-bin tag. The
        artifact carries the answer: `bin_row_counts` is keyed "1".."N_BINS" by the builder.
        """
        counts = self.meta.get("bin_row_counts")
        if not counts:
            raise ValueError(
                f"quality artifact {self.path} has no 'bin_row_counts' in its meta, so the bin "
                f"count it was BUILT with cannot be compared against this tier's N_BINS="
                f"{N_BINS}. A build with fewer bins would pass every range check and train a "
                f"coarser arm than the config reports. Rebuild it with "
                f"axis.dataset.build_quality_labels."
            )
        offline_bins = max(int(k) for k in counts)
        if offline_bins != N_BINS:
            raise ValueError(
                f"bin-count mismatch: quality artifact {self.path} was built with "
                f"{offline_bins} bins (its meta['bin_row_counts'] keys) but this tier's N_BINS is "
                f"{N_BINS}. A smaller offline count is invisible to the transform's range guard "
                f"-- every tag would be in [1, {N_BINS}] and the arm would train "
                f"{offline_bins}-bin conditioning while the config and the write-up say {N_BINS}. "
                f"Rebuild the artifact, or bring the two tiers' N_BINS back into agreement."
            )

    @property
    def reward_id(self) -> str:
        return str(self.meta["reward_id"])

    def check_reward_id(self, path: str | pathlib.Path | None = None) -> None:
        """Bind a `quality_<reward_id>.npz` filename to the artifact's own reward.

        `cfg_v2` and `cfg_phase` run under ONE config name (`pi05_axis_cfg`) and their artifacts
        are structurally identical, so the filename in `data.quality_path` is the only thing that
        says which reward built the tags. Names that make no claim are not checked.

        `path` defaults to `self.path` -- the file these tags were actually read from -- because
        that is the only path whose name this object can speak to. A caller handing in some OTHER
        path gets a check of that name against these tags, which is a vacuous pass whenever the
        other name makes no claim; the argument is kept only so the wiring can name the
        configured path in the error when the two are the same file by construction.
        """
        named = pathlib.Path(self.path if path is None else path)
        stem = named.stem
        if not stem.startswith(_NAME_PREFIX):
            return
        claimed = stem[len(_NAME_PREFIX):]
        if claimed != self.reward_id:
            raise ValueError(
                f"quality artifact {named} is named for reward_id={claimed!r} but its own meta "
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
        if i >= len(self.tag):
            # numpy raises here too, but with a message that names neither the artifact nor the
            # remedy -- and a test asserting a bare IndexError cannot tell that error apart from
            # one thrown by an unrelated line.
            raise IndexError(
                f"row index {i} is beyond quality artifact {self.path} ({len(self.tag)} rows). "
                f"The corpus is larger than the artifact; rebuild the artifact against this "
                f"corpus rather than letting the tail of the run go untagged."
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
        # The bind is HERE, not left to the caller. `tag_for_row`'s bounds check cannot see a
        # GROWN artifact: if len(tag) > len(dataset) every index the sampler can draw is in
        # range, nothing raises, and the whole run is tagged off a longer index space -- i.e. the
        # right-looking tag on the wrong frame, for every frame, for the entire run. This is the
        # only place both lengths are in hand, and the composition test used to have to remember
        # to call it by hand.
        tags.check_dataset_rows(len(dataset))
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

    Every lookup RAISES on a miss, and the two misses catch DIFFERENT things. A missing
    `quality` means no tag reached the transform -- an unwired wrapper, or this transform placed
    after `RepackTransform`, which rebuilds the dict from its structure map and drops `quality`
    along with everything else not in it. A missing `prompt` means the sample never carried one
    (`prompt_from_task=False`), which the tag has nowhere to go on. A `return data` fallback on
    either would train the plain control under the arm's name, with no symptom but an absent log
    line.

    Note the ORDER is load-bearing for the second case only in the sense that it cannot fire for
    a mis-ordered repack: after `RepackTransform` the tag is gone too, so the `quality` guard
    reports first and names the real fault.
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
                "'prompt' is not in the sample, but a quality tag is. The sample carries no "
                "instruction to append the tag to -- a dataset configured with "
                "prompt_from_task=False, or a repack map that does not surface 'prompt'. "
                "Appending to a synthesised empty prompt would condition on a bare "
                "'Quality: <n>' with no task text."
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


def wrap_and_transform(dataset, tags: QualityTags, transforms):
    """Build the arm. THE ONLY supported way to assemble these two objects.

    The wrapper and the transform are useless apart and there is no arrangement of one without
    the other that is anything but a bug -- but only ONE of the two wrong arrangements announces
    itself. Transform wired, wrapper missing: the transform's `quality` guard raises on the first
    batch. Wrapper wired, transform missing: `quality` is injected, `RepackTransform` drops it
    while rebuilding the dict, every prompt stays bare, nothing raises, and the arm trains as the
    plain BC control under its own name. That is the failure this project has shipped three
    times, and it is the arrangement a wiring task most easily produces, because the wrapper is
    the conspicuous new object while the transform is one more entry in a list.

    So the two are not offered separately to the wiring: this function takes the RAW dataset (the
    concat, before `transform_dataset`) and the transform list the loader would have built, and
    returns the composed dataset with `AxisQualityConditioning` at the HEAD -- ahead of
    `RepackTransform`, while `prompt` and `quality` are both still present. `transforms` is the
    full ordered list, exactly as `transform_dataset` assembles it.
    """
    # Imported here, not at module scope: `data_loader` is what wires this arm, so a top-level
    # import would be a cycle.
    from openpi.training.data_loader import TransformedDataset

    return TransformedDataset(
        QualityTaggedDataset(dataset, tags), [AxisQualityConditioning(), *transforms]
    )
