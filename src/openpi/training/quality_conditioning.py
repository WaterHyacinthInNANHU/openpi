"""π0.7 quality conditioning: read a precomputed tag artifact and put the tag in the prompt.

ONLINE TIER.

STAGE 1 (pretraining) has deliberately NO randomness here. The dropout was decided offline and
baked into the artifact (`axis.dataset.build_quality_labels`), so the realized tagged fraction is
a number read off disk rather than a reconstruction from eight workers' RNG states.

STAGE 2 (the LIBERO finetune, `LiberoQualityConditioning` at the bottom of this file) has no
artifact and cannot have one without a second offline pipeline over a third-party dataset, which
is out of scope. It therefore draws its dropout from a KEYED HASH of `(seed, episode_index,
frame_index)` -- a pure function of the row, not a stream. That is a stated deviation from design
decision D5, and it preserves the property D5 exists to protect: the realized dropout is
reproducible from the run record (the seed, which is in the config) without knowing any worker's
RNG state. It is also the mechanism `slb_cfg.SlbCfgMetadataConditioning` already uses. The
deviation belongs in the paper's method note, beside the stage-2 parity cost.

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
import hashlib
import json
import pathlib

import numpy as np
from typing_extensions import override

from openpi.training.slb_cfg import INFER_QUALITY
from openpi.training.slb_cfg import QUALITY_KEY as _PROMPT_TAG_KEY
from openpi.training.slb_cfg import apply_metadata
import openpi.transforms as _transforms

# The key the wrapper injects and the transform consumes. Not in RepackTransform's map: the
# transform runs first and rewrites `prompt`, after which the key has done its job.
QUALITY_KEY = "quality"

# The exact prefix `apply_metadata` puts in front of the bin, and the only trace the tag leaves in
# a sample. Built from slb_cfg's own key rather than spelled out, so the two cannot drift; pinned
# against a real tagged prompt in quality_conditioning_test.py.
PROMPT_TAG_MARKER = f"\n{_PROMPT_TAG_KEY}: "


def is_tagged(prompt) -> bool:
    """True if this prompt already carries a quality tag."""
    return PROMPT_TAG_MARKER in str(prompt)

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

# STAGE 2's copy of π0.7's two dropout levels (design D8), held LITERALLY equal to stage 1's so
# both stages train the unconditional branch at the same 1 - 0.85*0.95 = 0.1925 marginal. A
# different rate here would mean the two stages disagree about how often the model sees a bare
# prompt, which nothing in a loss curve or a checkpoint would reveal and no write-up would state.
#
# Named separately from `axis.dataset.quality_labels.DROP_WHOLE/DROP_COMPONENT` rather than
# imported, for the same reason `N_BINS` is duplicated: those live in the OFFLINE tier and nothing
# under openpi/ imports `axis`. The duplicate is pinned by config_cfg_stage2_test.py against the
# same 0.8075 tagged marginal the offline suite asserts.
DROP_WHOLE_STAGE2 = 0.15
DROP_COMPONENT_STAGE2 = 0.05

# The two LeRobot row keys the stage-2 draw is keyed on. `RepackTransform` drops both, which is
# why `LiberoQualityConditioning` must HEAD the repack group -- see its docstring.
EPISODE_INDEX_KEY = "episode_index"
FRAME_INDEX_KEY = "frame_index"


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
        if is_tagged(data["prompt"]):
            # DOUBLE TAG. This transform is in the chain twice -- the arrangement a wiring task
            # falls into, because `wrap_and_transform` PREPENDS it to a list that already begins
            # with `repack_transforms.inputs`, so heading the repack group with it as well reads
            # like belt-and-braces and is not. The result is `"...\nQuality: 5\nQuality: 5"`: in
            # range, tokenizable, logged nowhere, and matched by no eval-time prompt, which
            # conditions the arm on a string it will never be asked about at inference.
            #
            # `wrap_and_transform` refuses that composition up front, but only for the lists IT is
            # handed; this catches every other route to a tagged prompt, including a second
            # wrapper and a corpus whose task strings already carry the marker.
            raise ValueError(
                f"prompt {str(data['prompt'])[:120]!r} already carries a quality tag "
                f"({PROMPT_TAG_MARKER!r}), so tagging it would apply the tag TWICE. Either this "
                f"transform is in the chain twice (it belongs ONLY at the head of the list "
                f"`wrap_and_transform` builds -- never in the repack group as well), or the "
                f"corpus prompts were built with the tag already in them."
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


# The worst-case discrete state: pi0.5 serialises 32 digitized ints into the prompt
# (`Task: {text}, State: {32 ints};\nAction: `), and `np.digitize(state, bins=linspace(-1,1,257)
# [:-1])-1` maps any value >= 1.0 to the top bin, 255 -- 3 characters, the widest spelling
# (measured: state=-1.0 digitizes to 0, 1 character; the difference is 64 prompt tokens on
# "pick up the bowl", 79 vs 143). Building the worst case here rather than sampling one makes
# the guard a bound, not an observation.
_WORST_STATE_VALUE = 1.0

# `PaligemmaTokenizer.tokenize()` pads OR truncates to ITS OWN configured `max_len` and returns
# a full-length (tokens, mask) pair either way -- so `len(tokenize(...)[0])` is a CONSTANT equal
# to that max_len, never the prompt's real token count, and `mask.sum()` saturates at max_len
# too once truncation happens. A tokenizer sized at (or near) the budget under test therefore
# cannot tell "exactly fits" apart from "far over, truncated" -- and calling it on an over-budget
# prompt has ALREADY run upstream's own `logging.warning` before this function gets to raise,
# which is precisely the warning-not-raise failure this guard exists to replace. So measurement
# here never tokenizes with a budget-sized tokenizer: it probes with one sized far beyond
# anything a real corpus prompt plus tag could produce (measured max over all 1401 AXIS task
# names + worst-case state: ~155 tokens), so upstream's truncation branch can never fire and
# `mask.sum()` is the TRUE, unclamped count.
_PROBE_MAX_LEN = 8192


def check_token_budget(
    prompts,
    max_token_len: int,
    *,
    state_dim: int = 32,
    q: int | None = 5,
    tokenizer=None,
) -> int:
    """Tokenize every corpus prompt WITH the tag and RAISE if any exceeds `max_token_len`.

    `PaligemmaTokenizer.tokenize` truncates from the right and only logs a warning, which under
    `PYTHONUNBUFFERED` redirection is one line in a ten-hour log. What it truncates is the tail of
    `"Task: {text}, State: {32 ints};\\nAction: "` -- the `Action:` marker itself -- and it
    happens for the CFG arm ONLY, because only the CFG arm lengthens the prompt (measured: the
    tag itself costs 4-5 tokens depending on the prompt's trailing text, e.g. trailing
    punctuation or whitespace changes how sentencepiece merges across the `\\n` boundary -- NOT
    a constant, so every corpus prompt has to be measured rather than one representative sample).
    So this is not a convenience check: it is the difference between an arm and a silently
    malformed arm.

    Returns the MINIMUM margin in tokens, for the run record. `q=None` measures the untagged
    prompt, which is what makes "the tag is what pushed it over" a statement a test can make.
    `q` defaults to 5 (`slb_cfg.INFER_QUALITY`, the value CFG conditions on at inference), and
    -- measured across all 1401 AXIS task names -- the tag's cost does not vary across q=1..5 for
    a fixed prompt, only across prompts, so checking one q is not a shortcut that hides risk.
    """
    prompts = [str(p) for p in prompts]
    if not prompts:
        raise ValueError(
            "no prompts to check: the token-budget guard would pass vacuously. The artifact must "
            "carry the corpus's task strings (QualityTags.prompts)."
        )
    max_token_len = int(max_token_len)
    # Only the CLASS is taken from a caller-supplied tokenizer, never the instance: reusing the
    # instance directly would mean measuring with (or near) the exact budget under test, which is
    # the ambiguous, warning-triggering case explained above.
    tokenizer_cls = type(tokenizer) if tokenizer is not None else None
    if tokenizer_cls is None:
        from openpi.models import tokenizer as _tok

        tokenizer_cls = _tok.PaligemmaTokenizer
    probe_max_len = max(_PROBE_MAX_LEN, max_token_len * 2)
    probe = tokenizer_cls(probe_max_len)
    state = np.full(int(state_dim), _WORST_STATE_VALUE, dtype=np.float32)

    margin = max_token_len
    worst = prompts[0]
    for prompt in prompts:
        text = prompt if q is None else apply_metadata(prompt, int(q), None)
        _, mask = probe.tokenize(text, state=state)
        n = int(np.asarray(mask).sum())
        if n >= probe_max_len:
            # The probe itself saturated, so its own truncation branch (and warning) may have
            # fired and `n` is a lower bound, not the true count. Silently continuing would risk
            # UNDER-reporting an overflow this severe. Raise and name the fix rather than guess.
            raise ValueError(
                f"prompt {prompt[:120]!r} tokenizes to >= this guard's own probe length "
                f"({probe_max_len}); its true length is unknown. Raise _PROBE_MAX_LEN before "
                f"trusting a result for a prompt this long."
            )
        candidate_margin = max_token_len - n
        if candidate_margin < margin:
            margin, worst = candidate_margin, prompt
    if margin < 0:
        raise ValueError(
            f"token budget exceeded by {-margin} tokens: the longest corpus prompt "
            f"{worst[:120]!r} plus the {('Quality: %d' % q) if q is not None else 'no'} tag "
            f"tokenizes to {max_token_len - margin} tokens against max_token_len={max_token_len}. "
            f"PaligemmaTokenizer would truncate from the right, deleting the 'Action:' marker "
            f"for THIS ARM ONLY -- and it would only log a warning. Raise max_token_len, or "
            f"shorten the tag."
        )
    return margin


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

    already = [t for t in transforms if isinstance(t, AxisQualityConditioning)]
    if already:
        # THE DOUBLE TAG, refused where it is constructed rather than where it shows up. The list
        # handed in is `transform_dataset`'s, which begins with `repack_transforms.inputs` -- so a
        # conditioning entry added to the repack group in config.py arrives here, and prepending a
        # second one would produce `"...\nQuality: 5\nQuality: 5"` on every tagged row. Nothing
        # downstream raises on that: the tag is in range, the tokenizer is happy, the loss curve
        # is normal, and the arm is conditioned on a string no eval-time prompt can match.
        #
        # The runtime guard in `AxisQualityConditioning.__call__` would also catch this, on the
        # first tagged sample. Refusing here as well makes the arrangement unconstructable rather
        # than merely fatal, and names the file to fix.
        raise ValueError(
            f"the transform list handed to wrap_and_transform already contains "
            f"{len(already)} AxisQualityConditioning transform(s), so the tag would be applied "
            f"twice ('...\\nQuality: 5\\nQuality: 5') -- in range, tokenizable, and matched by no "
            f"eval-time prompt. This function is the ONLY place the conditioning belongs; remove "
            f"it from the repack group (openpi/src/openpi/training/config.py), which is what this "
            f"list starts with."
        )
    return TransformedDataset(
        QualityTaggedDataset(dataset, tags), [AxisQualityConditioning(), *transforms]
    )


# =================================================================================================
# STAGE 2 (the LIBERO finetune). One class, no wrapper, no artifact -- see the module docstring.
# =================================================================================================


@dataclasses.dataclass(frozen=True)
class LiberoQualityConditioning(_transforms.DataTransformFn):
    """Append a CONSTANT `"\\nQuality: {q_ep}"` to the prompt. Heads the LIBERO repack group.

    WHY A CONSTANT (D9). Stage 2 finetunes on LIBERO, which is uniformly expert data -- there is
    nothing to bin, so every sample carries the same tag. Two things follow, and both are the
    point: the conditional branch is anchored at exactly the value inference asks for
    (`slb_cfg.INFER_QUALITY`), and the dropout keeps the unconditional branch trained so guidance
    at eval still has something to subtract. The guidance scale β is therefore swept at EVAL, not
    at training time; there is no β here.

    WHY A KEYED HASH RATHER THAN AN ARTIFACT, unlike stage 1. Building one would mean a second
    offline pipeline over a third-party dataset (LIBERO), which is out of scope. What D5 exists to
    protect is that the realized dropout be reconstructable WITHOUT knowing any worker's RNG
    state -- so the draw is a pure function of `(seed, episode_index, frame_index)`, which
    `dropped()` exposes so a row's fate can be recomputed offline from the run record alone. Two
    passes over one row agree; eight workers agree; a rerun agrees; and unlike a per-batch RNG the
    answer does not depend on batch order, worker count, or how many rows were drawn first. This
    deviation from D5 is disclosed in the plan and belongs in the paper's method note.

    WHERE IT RUNS. At the HEAD of `LeRobotLiberoDataConfig`'s repack group, i.e. before
    `RepackTransform`, which rebuilds the dict from its structure map and drops `episode_index`
    and `frame_index` along with everything else not in it. Placed after it, every lookup would
    miss -- so both keys RAISE rather than defaulting. `slb_cfg.SlbCfgMetadataConditioning`, the
    sibling that uses this same keying, returns `data` unchanged on a miss; that is the silent
    version of this failure and is not repeated here.

    Note there is no `QualityTaggedDataset` twin: stage 1 needs a wrapper because its tag comes
    from a per-row artifact indexed by the concat-global row number, which no transform can see.
    A constant tag needs no lookup at all.
    """

    q_ep: int = INFER_QUALITY
    drop_whole: float = DROP_WHOLE_STAGE2
    drop_component: float = DROP_COMPONENT_STAGE2
    seed: int = 0

    def __post_init__(self) -> None:
        q = int(self.q_ep)
        if not 1 <= q <= N_BINS:
            # NO_TAG (0) is the most likely wrong value -- `quality_tag=0` reads like "off" and
            # would instead emit `Quality: 0`, a sixth condition the tokenizer has never seen and
            # that no eval prompt can match. Refused at CONSTRUCTION so it fails when the config
            # is built, not on the first batch of a 15-hour run.
            raise ValueError(
                f"stage-2 quality tag {q} is not a bin in [1, {N_BINS}]. Stage 2 conditions on a "
                f"CONSTANT tag and inference asks for {INFER_QUALITY} "
                f"(slb_cfg.INFER_QUALITY); {NO_TAG} is the untagged sentinel and must never "
                f"reach a prompt. To disable conditioning, leave `quality_tag` unset."
            )
        for name, p in (("drop_whole", self.drop_whole), ("drop_component", self.drop_component)):
            if not 0.0 <= float(p) < 1.0:
                raise ValueError(
                    f"{name}={p} is outside [0, 1). A rate of 1 would drop every row, leaving the "
                    f"CONDITIONAL branch untrained while the config and the write-up say the arm "
                    f"is conditioned."
                )
        if float(self.drop_whole) == 0.0 and float(self.drop_component) == 0.0:
            # Stage 1's counterpart is `QualityTags.__init__`'s refusal of an artifact with no
            # NO_TAG rows. With no dropout the model never sees the bare prompt, never learns
            # p(a | no tag), and guidance -- which subtracts an unconditional forward pass -- is
            # undefined. Nothing online would notice: loss, throughput and every prompt look fine.
            raise ValueError(
                "both stage-2 dropout levels are zero, so every row is tagged and CFG has no "
                "unconditional branch to guide away from -- the arm degenerates to plain "
                "conditional BC while still being called CFG."
            )

    def dropped(self, episode_index: int, frame_index: int) -> bool:
        """Whether this row's tag is dropped. PURE in `(seed, episode_index, frame_index)`.

        Public because that purity is the whole of the D5 deviation's defence: the realized
        dropout of any row of any run is recomputable from the run record (the config's `seed`)
        plus the row's own identity, with no worker RNG state involved.

        Mixed through SHA-256 rather than an ad-hoc integer product because adjacent `(ep, fr)`
        -- which is what consecutive rows of one episode are -- must not yield correlated draws.
        Both levels are drawn UNCONDITIONALLY (`random(2)`, not a short-circuited second call) so
        a future second metadata component slots in without moving the first component's stream,
        exactly as `axis.dataset.quality_labels.two_level_dropout` does offline.
        """
        digest = hashlib.sha256(
            f"{int(self.seed)}:{int(episode_index)}:{int(frame_index)}".encode()
        ).digest()
        u = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(2)
        return bool(u[0] < float(self.drop_whole) or u[1] < float(self.drop_component))

    @override
    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            raise KeyError(
                "'prompt' is not in the sample, so a quality tag has no instruction to attach "
                "to. Either the dataset is configured with prompt_from_task=False, or this "
                "transform runs after a repack that does not surface 'prompt'. Tagging a "
                "synthesised empty prompt would condition the arm on a bare 'Quality: <n>'."
            )
        for key in (EPISODE_INDEX_KEY, FRAME_INDEX_KEY):
            if key not in data:
                raise KeyError(
                    f"{key!r} is not in the sample. The stage-2 dropout draw is keyed on the row, "
                    f"and RepackTransform drops this key while rebuilding the dict -- so this "
                    f"transform must HEAD the repack group. Placed after it, every row would be "
                    f"treated identically and the arm would silently train with no dropout at "
                    f"all (or none at all tagged), with nothing in the loss curve to show it."
                )
        if is_tagged(data["prompt"]):
            # DOUBLE TAG, the same failure `AxisQualityConditioning` refuses: two conditioning
            # entries in one chain yield `"...\nQuality: 5\nQuality: 5"` -- in range, tokenizable,
            # logged nowhere, and matched by no eval-time prompt.
            raise ValueError(
                f"prompt {str(data['prompt'])[:120]!r} already carries a quality tag "
                f"({PROMPT_TAG_MARKER!r}), so tagging it would apply the tag TWICE. Either this "
                f"transform is in the chain twice (it belongs ONLY at the head of the LIBERO "
                f"repack group -- see LeRobotLiberoDataConfig.quality_tag), or the corpus's task "
                f"strings were built with the tag already in them."
            )
        ep = int(np.asarray(data[EPISODE_INDEX_KEY]).reshape(-1)[0])
        fr = int(np.asarray(data[FRAME_INDEX_KEY]).reshape(-1)[0])
        if self.dropped(ep, fr):
            # The BARE prompt is the unconditional branch, which is why a dropped row must not
            # emit `Quality: 0`.
            return data
        return {**data, "prompt": apply_metadata(data["prompt"], int(self.q_ep), None)}
