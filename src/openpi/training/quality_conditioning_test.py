"""Quality-tag reader, dataset wrapper, and prompt transform.

The FIRST test is a measurement, and it is why this module has a wrapper at all. `slb_cfg`'s
conditioning keys on (episode_index, frame_index) because it could not obtain a flat index; the
spec asks whether `data["index"]` under ConcatDataset is the concat-global row. It is not, and a
tag applied on that assumption lands on the wrong frame for every sub-dataset after the first --
silently, with the arm reverting to the control. Pinned by execution, not by reading torch.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import openpi.transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.training.quality_conditioning import N_BINS
from openpi.training.quality_conditioning import NO_TAG
from openpi.training.quality_conditioning import NOT_TRAINABLE
from openpi.training.quality_conditioning import QUALITY_KEY
from openpi.training.quality_conditioning import AxisQualityConditioning
from openpi.training.quality_conditioning import QualityTaggedDataset
from openpi.training.quality_conditioning import QualityTags
from openpi.training.quality_conditioning import check_token_budget
from openpi.training.quality_conditioning import wrap_and_transform


class _Sub(torch.utils.data.Dataset):
    """A sub-dataset that reports its OWN row number under "index", as LeRobot v3.0 does."""

    def __init__(self, n: int, prompt: str):
        self._n, self._prompt = n, prompt

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        return {"index": int(i), "prompt": self._prompt}


def test_concat_dataset_index_is_not_the_concat_row():
    """THE MEASUREMENT. Catches: keying the tag on `data["index"]`.

    ConcatDataset maps a global index to (sub, local) and returns the sub-dataset's own item, so
    "index" is local. Reading it as a concat row tags the wrong frame everywhere past the first
    sub-dataset -- the disjoint-grid failure `slb_cfg.build_cfg_labels` documents.
    """
    concat = torch.utils.data.ConcatDataset([_Sub(5, "a"), _Sub(7, "b")])
    assert len(concat) == 12
    assert concat[0]["index"] == 0
    # Global row 5 is the FIRST row of the second sub-dataset...
    assert concat[5]["prompt"] == "b"
    # ...and it reports index 0, not 5.
    assert concat[5]["index"] == 0, "if this ever becomes 5, re-read the wrapper decision"


def _write_artifact(tmp_path, tag, *, reward_id="v2", name=None, prompts=("pick up the bowl",),
                    n_bins=N_BINS, meta_overrides=None, drop_meta_keys=()):
    """Write an artifact shaped exactly as `build_quality_labels.quality_meta` writes one.

    `bin_row_counts` is keyed "1".."n_bins" by the builder, and the reader reads the OFFLINE bin
    count back out of those keys, so a test that omitted it would be testing a file the builder
    never emits.
    """
    tag = np.asarray(tag, dtype=np.uint8)
    p = tmp_path / (name or f"quality_{reward_id}.npz")
    meta = {
        "reward_id": reward_id,
        "n_rows": int(len(tag)),
        "bin_row_counts": {str(b): int((tag == b).sum()) for b in range(1, int(n_bins) + 1)},
    }
    meta.update(meta_overrides or {})
    for k in drop_meta_keys:
        meta.pop(k, None)
    np.savez(
        p,
        tag=tag,
        prompts=np.array([str(s) for s in prompts]),
        meta=np.array(json.dumps(meta)),
    )
    return p


def _tags(tmp_path, tag: np.ndarray, **kwargs) -> QualityTags:
    """Build a QualityTags THROUGH ITS REAL CONSTRUCTOR.

    This used to bypass `__init__` with `__new__` and hand-set the attributes, which meant the
    headline wrapper tests -- the ones that matter most -- exercised none of the constructor's
    guards. Every guard added there (the bin-count bind, the n_rows bind, the unconditional-branch
    check) would have been invisible to exactly those tests. Cheap: one small .npz per call.
    """
    return QualityTags(_write_artifact(tmp_path, tag, **kwargs))


def test_the_wrapper_uses_its_own_index_argument(tmp_path):
    """Catches: a wrapper that forwards `item["index"]` instead of the index it was called with.
    Row 5 must receive tag[5], not tag[0].

    The tag pattern must NOT be periodic in the first sub-dataset's length. `arange(1,13) % 5 + 1`
    -- the obvious choice -- has period 5 == len(_Sub(5)), so `tag[i] == tag[i - 5]` for every row
    past the boundary and the local-index bug passes this test unnoticed (verified by mutation).
    The assertion below pins the property so a future edit cannot quietly re-degenerate it.
    """
    concat = torch.utils.data.ConcatDataset([_Sub(5, "a"), _Sub(7, "b")])
    tag = np.array([1, 2, 3, 4, 5, 5, NO_TAG, 4, 2, 3, 1, 5], dtype=np.uint8)
    assert all(tag[i] != tag[i - 5] for i in range(5, 12)), "pattern is blind to the local-index bug"
    tags = _tags(tmp_path, tag)
    wrapped = QualityTaggedDataset(concat, tags)
    assert len(wrapped) == 12
    for i in range(12):
        assert wrapped[i][QUALITY_KEY] == int(tags.tag[i])


def test_reading_a_not_trainable_row_raises(tmp_path):
    """Catches: the 0.0-default pattern. A sentinel that silently reads as NO_TAG turns a
    misaligned map into an untagged arm -- i.e. the control, under the arm's name."""
    tags = QualityTags(_write_artifact(tmp_path, [1, NOT_TRAINABLE, NO_TAG]))
    assert tags.tag_for_row(0) == 1
    with pytest.raises(KeyError, match="not trainable"):
        tags.tag_for_row(1)


def test_a_tag_array_that_disagrees_with_the_dataset_raises(tmp_path):
    """Catches: a grown or shrunk corpus. Every index would still be in bounds and mean a
    different frame -- the case `ScheduleSampler.check_dataset_rows` exists to catch, repeated
    here because the tag artifact has the same exposure."""
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 3]))
    tags.check_dataset_rows(3)
    with pytest.raises(ValueError, match="corpus mismatch") as excinfo:
        tags.check_dataset_rows(4, "roots_5000.json")
    assert "roots_5000.json" in str(excinfo.value)


def test_a_filename_that_disagrees_with_the_reward_id_raises(tmp_path):
    """Catches: quality_phase.npz containing the v2 tags. Both arms share ONE config name."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], reward_id="v2", name="quality_phase.npz")
    tags = QualityTags(p)
    with pytest.raises(ValueError, match="reward_id"):
        tags.check_reward_id(p)


def test_the_filename_check_defaults_to_the_file_it_read(tmp_path):
    """Catches: `check_reward_id(some_other_path)`. The argument used to be REQUIRED, so a caller
    with two paths in scope could hand in the wrong one -- and a name that makes no claim is a
    documented no-op, so the wrong path is a vacuous PASS on a mismatched artifact. The file this
    object was read from is the only name it can speak to, so that is the default."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], reward_id="v2", name="quality_phase.npz")
    with pytest.raises(ValueError, match="reward_id"):
        QualityTags(p).check_reward_id()
    # ...and the vacuous pass the default exists to prevent, shown explicitly:
    QualityTags(p).check_reward_id(tmp_path / "staging.npz")


def test_filename_check_is_a_noop_on_a_name_that_makes_no_claim(tmp_path):
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], reward_id="v2", name="staging.npz")
    QualityTags(p).check_reward_id(p)
    QualityTags(p).check_reward_id()


def test_an_artifact_without_a_reward_id_raises(tmp_path):
    p = _write_artifact(tmp_path, [1, NO_TAG], drop_meta_keys=("reward_id",))
    with pytest.raises(ValueError, match="reward_id"):
        QualityTags(p)


def test_the_transform_appends_the_pi07_tag():
    """Catches: a different tag spelling. `slb_cfg.apply_metadata` is reused, not reinvented --
    a second spelling would make the two code paths incomparable for no gain."""
    out = AxisQualityConditioning()({"prompt": "pick up the bowl", QUALITY_KEY: 5})
    assert out["prompt"] == "pick up the bowl\nQuality: 5"


def test_a_dropped_out_row_yields_the_bare_control_prompt():
    """q == 0 IS the unconditional branch. Catches: emitting `Quality: 0`, which would train a
    sixth bin the tokenizer has never seen and destroy the unconditional branch."""
    data = {"prompt": "pick up the bowl", QUALITY_KEY: NO_TAG}
    out = AxisQualityConditioning()(data)
    assert out["prompt"] == "pick up the bowl"


def test_the_transform_does_not_mutate_its_input():
    data = {"prompt": "p", QUALITY_KEY: 4}
    AxisQualityConditioning()(data)
    assert data["prompt"] == "p"


def test_a_missing_quality_key_raises():
    """THE ANTI-INERT TRIPWIRE. Catches: the transform wired into the repack group while the
    dataset wrapper is not wired into the loader. Silently returning `data` there would train the
    plain control under the arm's name, with no symptom but an absent log line -- the exact
    failure round 1 shipped and Plan A reproduced."""
    with pytest.raises(KeyError, match="quality"):
        AxisQualityConditioning()({"prompt": "pick up the bowl"})


def test_a_missing_prompt_raises():
    """NOT the mis-ordered-repack catch it used to claim to be: after `RepackTransform` the
    `quality` key is gone too (it is not in the repack map), so the guard above fires first and
    this branch is unreachable for that fault. See
    `test_a_mis_ordered_repack_is_caught_by_the_quality_guard`.

    It is load-bearing for a different reason. A dataset configured with `prompt_from_task=False`
    hands back samples that carry a tag and NO prompt; a `return data` fallback there would leave
    every prompt untagged in exactly the config that has no prompt to tag, and the arm would be
    the control."""
    with pytest.raises(KeyError, match="prompt"):
        AxisQualityConditioning()({QUALITY_KEY: 3})


def test_a_mis_ordered_repack_is_caught_by_the_quality_guard():
    """Pins WHICH guard catches the mis-ordered repack, since the docstring above used to credit
    the wrong one. `RepackTransform` rebuilds the dict from its map, dropping `quality` along with
    everything else, so a transform placed after it sees a prompt and no tag."""
    repacked = _transforms.RepackTransform({"prompt": "prompt"})(
        {"prompt": "p", QUALITY_KEY: 3, "index": 0}
    )
    assert QUALITY_KEY not in repacked
    with pytest.raises(KeyError, match="quality"):
        AxisQualityConditioning()(repacked)


def test_the_transform_is_frozen():
    """Repo convention for DataTransformFn, and it is what makes the transform safe to share
    across 8 dataloader workers."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        AxisQualityConditioning().__setattr__("x", 1)


# ---------------------------------------------------------------------------------------------
# Coverage beyond the brief's list, measured against the sibling reader (schedule_sampler_test):
# the artifact's own dtype/shape contract, the negative-index wrap, and the composition the
# loader will actually build. Every one names the broken implementation it catches.
# ---------------------------------------------------------------------------------------------


def test_a_non_uint8_tag_array_is_rejected(tmp_path):
    """Catches: an artifact whose tags were written as int64/float. `int(...)` would happily read
    a float 2.9 as 2 and a negative int as a bin, so the dtype IS the contract -- the same reason
    `ScheduleSampler` refuses a float rows array instead of casting it."""
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.array([1, NO_TAG, 3], np.int64), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 3,
                                       "bin_row_counts": {str(b): 1 for b in range(1, N_BINS + 1)}})))
    with pytest.raises(ValueError, match="uint8"):
        QualityTags(p)


def test_a_two_dimensional_tag_array_is_rejected(tmp_path):
    """Catches: a (steps, batch) schedule artifact handed to the quality reader by mistake --
    both are .npz files of ints and the two arms sit next to each other on disk. Indexing a 2-D
    array by row would yield a whole ROW of tags, and `int()` of it raises far downstream."""
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.zeros((3, 4), np.uint8), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 12,
                                       "bin_row_counts": {str(b): 1 for b in range(1, N_BINS + 1)}})))
    with pytest.raises(ValueError, match="1-D"):
        QualityTags(p)


def test_an_artifact_without_prompts_is_rejected(tmp_path):
    """Catches: an artifact built before `corpus_prompts` existed. The token-budget guard
    tokenizes this list, so an empty one makes the guard pass vacuously -- the builder raises on
    it at write time and the reader must not accept what the builder refuses to emit."""
    p = _write_artifact(tmp_path, [1, NO_TAG], prompts=())
    with pytest.raises(ValueError, match="prompt"):
        QualityTags(p)


def test_meta_row_count_disagreeing_with_the_tag_array_is_rejected(tmp_path):
    """Catches: a truncated or hand-edited artifact. `check_dataset_rows` compares the DATASET
    against `len(tag)`, so a meta that claims a different corpus size would go unnoticed while
    every provenance line printed from meta describes a corpus this file does not cover."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], meta_overrides={"n_rows": 99})
    with pytest.raises(ValueError, match="n_rows"):
        QualityTags(p)


def test_an_artifact_without_n_rows_is_rejected(tmp_path):
    """`n_rows` used to be OPTIONAL here (`if declared is not None`) while the sibling reader
    (`ScheduleSampler.check_dataset_rows`) REQUIRES it, with no reason given for the divergence.
    The builder always writes it, so an artifact missing it predates the field or was hand-built
    -- and accepting it makes the self-consistency check above pass vacuously on exactly the file
    that needs it. Same contract as the sibling now."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], drop_meta_keys=("n_rows",))
    with pytest.raises(ValueError, match="n_rows"):
        QualityTags(p)


def test_a_filename_that_agrees_with_the_reward_id_passes(tmp_path):
    """The happy path of the binding check: the convention must not be a tripwire on itself."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], reward_id="v2")
    tags = QualityTags(p)
    assert tags.reward_id == "v2"
    tags.check_reward_id(p)


# --- the tier-boundary N_BINS bind (the range guard sees only half of a drift) ----------------


def test_an_artifact_built_with_fewer_bins_is_rejected(tmp_path):
    """THE SILENT HALF. If the offline N_BINS shrank 5 -> 3, every tag is still inside [1, 5], the
    transform's range guard never fires, and the arm trains 3-bin conditioning while this file,
    the config and the write-up all say 5. `bin_row_counts`' keys are the only record of the bin
    count the artifact was BUILT with, and this is the only place the two tiers meet."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], n_bins=3)
    with pytest.raises(ValueError, match="bin-count mismatch"):
        QualityTags(p)


def test_an_artifact_built_with_more_bins_is_rejected(tmp_path):
    """The loud half, caught EARLIER. A grown offline N_BINS would eventually trip the transform's
    range guard -- but only on the first row that actually drew a bin above 5, which may be
    thousands of steps in. Reading it off the meta fails at load."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], n_bins=7)
    with pytest.raises(ValueError, match="bin-count mismatch"):
        QualityTags(p)


def test_an_artifact_without_bin_row_counts_is_rejected(tmp_path):
    """Catches: an artifact predating the field, which would make the bind above pass vacuously
    -- the same class as an empty `prompts` making the token-budget guard vacuous."""
    p = _write_artifact(tmp_path, [1, NO_TAG, 3], drop_meta_keys=("bin_row_counts",))
    with pytest.raises(ValueError, match="bin_row_counts"):
        QualityTags(p)


def test_an_artifact_matching_this_tiers_bin_count_passes(tmp_path):
    """The happy path: the bind must not be a tripwire on a correctly built file."""
    QualityTags(_write_artifact(tmp_path, [1, NO_TAG, N_BINS]))


# --- the unconditional branch CFG guides away from --------------------------------------------


def test_an_artifact_with_no_unconditional_rows_is_rejected(tmp_path):
    """Catches: an artifact built with DROP_WHOLE=0 (or a dropout that never fired). Every
    trainable row carries a tag, the model never sees the bare prompt, p(a) is never learned, and
    guidance at inference -- which subtracts an unconditional pass -- is undefined. Nothing online
    notices: loss, throughput and every prompt look exactly right."""
    p = _write_artifact(tmp_path, [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="unconditional"):
        QualityTags(p)


def test_a_single_unconditional_row_is_enough(tmp_path):
    """The guard is on EXISTENCE, not on a rate: the realized fraction is a number read off the
    meta, and re-deriving a threshold here would be a second, disagreeing opinion about it."""
    QualityTags(_write_artifact(tmp_path, [1, 2, 3, 4, NO_TAG]))


def test_a_negative_row_raises_rather_than_reading_from_the_tail(tmp_path):
    """Catches: `self.tag[index]` with no lower-bound check. Numpy indexes -1 as the LAST row, so
    a negative index silently returns a real, wrong tag instead of raising -- the same silent
    wrap `ScheduleSampler` refuses for negative schedule rows."""
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 3]))
    with pytest.raises(IndexError, match="negative"):
        tags.tag_for_row(-1)


def test_a_row_beyond_the_artifact_raises(tmp_path):
    """Catches: a corpus larger than the artifact reaching the reader. The `match=` is not
    decoration -- a bare `pytest.raises(IndexError)` here passed on an IndexError from ANY cause,
    including one raised while building the fixture, so it did not test what it named."""
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 3]))
    with pytest.raises(IndexError, match="beyond quality artifact"):
        tags.tag_for_row(3)


def test_the_wrapper_does_not_mutate_the_underlying_item(tmp_path):
    """Catches: `item = self._dataset[i]; item[QUALITY_KEY] = ...`. LeRobot hands back a fresh
    dict, but a wrapped dataset that caches (or a test double that does) would accumulate the
    tag of whichever row was read last."""

    class _Cached(torch.utils.data.Dataset):
        def __init__(self):
            self.item = {"prompt": "p"}

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return self.item

    inner = _Cached()
    wrapped = QualityTaggedDataset(inner, _tags(tmp_path, [1, NO_TAG]))
    assert wrapped[0][QUALITY_KEY] == 1
    assert QUALITY_KEY not in inner.item


def test_the_wrapper_refuses_a_negative_index(tmp_path):
    """Catches: a negative index reaching the pair. The inner dataset would return its LAST row
    while the tag lookup must not silently agree -- both would be 'valid', and wrong."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    wrapped = QualityTaggedDataset(concat, _tags(tmp_path, [1, NO_TAG]))
    with pytest.raises(IndexError, match="negative"):
        wrapped[-1]


def test_the_wrapper_refuses_a_not_trainable_row(tmp_path):
    """Catches: a sampler whose row plan and the tag artifact disagree about which rows exist.
    The wrapper is where that surfaces, because it is the only place both are in hand."""
    concat = torch.utils.data.ConcatDataset([_Sub(3, "a")])
    wrapped = QualityTaggedDataset(concat, _tags(tmp_path, [1, NOT_TRAINABLE, NO_TAG]))
    assert wrapped[0][QUALITY_KEY] == 1
    with pytest.raises(KeyError, match="not trainable"):
        wrapped[1]


# --- the wrapper binds itself to the dataset (I2) ---------------------------------------------


def test_a_tag_array_longer_than_the_dataset_raises_at_construction(tmp_path):
    """THE GROWN-ARTIFACT CASE, and the reason the bind cannot be left to the caller. When
    len(tag) > len(dataset), EVERY index the sampler can draw is inside the tag array:
    `tag_for_row`'s bounds check never fires, nothing raises, and the whole run trains on tags
    read off a longer index space -- a right-looking tag on the wrong frame, for every frame.
    `check_dataset_rows` used to be the caller's job to remember."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    tags = _tags(tmp_path, [1, NO_TAG, 3, 4])
    with pytest.raises(ValueError, match="corpus mismatch"):
        QualityTaggedDataset(concat, tags)


def test_a_tag_array_shorter_than_the_dataset_raises_at_construction(tmp_path):
    """The other direction. This one would eventually raise from `tag_for_row` -- but only once
    the sampler happened to draw a row past the end, which under a shuffled sampler is thousands
    of steps of already-mistagged training later. It must fail at construction."""
    concat = torch.utils.data.ConcatDataset([_Sub(4, "a")])
    tags = _tags(tmp_path, [1, NO_TAG])
    with pytest.raises(ValueError, match="corpus mismatch"):
        QualityTaggedDataset(concat, tags)


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_every_bin_produces_its_own_prompt(q):
    """Catches: a transform that collapses bins (e.g. `q >= 3` -> good/bad). Five distinct
    conditions is the whole point of the π0.7 tag; a collapse would train a coarser arm than the
    one the paper describes, with no test failing."""
    out = AxisQualityConditioning()({"prompt": "p", QUALITY_KEY: q})
    assert out["prompt"] == f"p\nQuality: {q}"


def test_a_not_trainable_tag_reaching_the_transform_raises():
    """Catches: the sentinel leaking past a wrapper that was bypassed or replaced. `Quality: 255`
    is a token sequence the tokenizer has never seen and would silently become a sixth
    condition."""
    with pytest.raises(ValueError, match="255"):
        AxisQualityConditioning()({"prompt": "p", QUALITY_KEY: NOT_TRAINABLE})


def test_a_tag_outside_the_bin_range_raises():
    """Catches: an artifact built with a different N_BINS, or an off-by-one in the binning. The
    online tier duplicates the bin count across the tier boundary, so this is the only place the
    duplication can be caught."""
    with pytest.raises(ValueError, match="quality"):
        AxisQualityConditioning()({"prompt": "p", QUALITY_KEY: 6})


def test_a_numpy_scalar_tag_is_accepted():
    """The wrapper injects a python int, but nothing forbids a uint8 arriving from an array-typed
    path; `f"{q}"` on a numpy scalar formats differently across numpy 2.x (np.uint8(3) reprs as
    'np.uint8(3)'), which would silently emit a garbage tag."""
    out = AxisQualityConditioning()({"prompt": "p", QUALITY_KEY: np.uint8(3)})
    assert out["prompt"] == "p\nQuality: 3"


def test_the_composition_the_loader_will_build_tags_the_right_rows(tmp_path):
    """THE ARM, end to end at this tier, through the constructor the wiring will call.

    Catches: wrapping where the AWR wrapper sits (i.e. AFTER `transform_dataset`), which injects
    the tag once the prompt has already been rewritten and leaves every prompt bare. Also catches
    a tag/row misalignment across the ConcatDataset boundary, which no single-sub-dataset test
    can see.
    """
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a"), _Sub(3, "b")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 5, 4, 3]))
    dataset = wrap_and_transform(concat, tags, [])
    assert [dataset[i]["prompt"] for i in range(5)] == [
        "a\nQuality: 1",
        "a",  # dropped out -> the bare control prompt
        "b\nQuality: 5",
        "b\nQuality: 4",
        "b\nQuality: 3",
    ]


def test_wrapping_where_the_awr_wrapper_sits_raises_instead_of_training_the_control(tmp_path):
    """The plan's sharpest catch, encoded: the AWR wrapper goes AFTER `transform_dataset`, and a
    quality wrapper copied into that position injects the tag once the transforms have already
    run. The transform then never sees a tag, every prompt stays bare, and the arm trains as the
    plain control with a green suite. It must be impossible to build that order quietly.
    """
    from openpi.training.data_loader import TransformedDataset

    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG]))
    wrong_order = QualityTaggedDataset(
        TransformedDataset(concat, [AxisQualityConditioning()]), tags
    )
    with pytest.raises(KeyError, match="quality"):
        wrong_order[0]


# --- the complementary inert composition, and the constructor that forbids it (I1) ------------


def _repack():
    """The real `RepackTransform` the pretrain group runs: it rebuilds the dict from its map and
    drops `quality` with everything else, which is what makes the omission below SILENT."""
    return _transforms.RepackTransform({"prompt": "prompt"})


def test_the_wrapper_without_the_transform_is_silently_the_baseline(tmp_path):
    """THE REFUTATION, encoded. The previous implementation claimed the inert arm 'cannot be
    assembled quietly at this tier'. It can: wire the wrapper, forget the transform, and the tag
    is injected, dropped by `RepackTransform`, and every prompt comes out bare. No error, no log
    line, an arm byte-identical to the plain BC control.

    This test asserts the BROKEN behaviour on purpose, so the claim stays refuted and nobody
    re-derives it. `wrap_and_transform` is the answer -- see the next test.
    """
    from openpi.training.data_loader import TransformedDataset

    concat = torch.utils.data.ConcatDataset([_Sub(2, "a"), _Sub(3, "b")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 5, 4, 3]))
    hand_wired = TransformedDataset(QualityTaggedDataset(concat, tags), [_repack()])
    assert [hand_wired[i]["prompt"] for i in range(5)] == ["a", "a", "b", "b", "b"]
    assert all(QUALITY_KEY not in hand_wired[i] for i in range(5))


def test_wrap_and_transform_is_the_only_way_to_get_a_wrapped_dataset(tmp_path):
    """Catches: the arrangement above reaching the wiring. The two objects are not offered
    separately -- `wrap_and_transform` takes the raw dataset and the loader's transform list and
    returns the composed dataset with `AxisQualityConditioning` already at the HEAD, so there is
    no call site at which the transform can be omitted while the wrapper is kept.

    Same inputs as the previous test, same repack, opposite outcome."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a"), _Sub(3, "b")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 5, 4, 3]))
    dataset = wrap_and_transform(concat, tags, [_repack()])
    assert [dataset[i]["prompt"] for i in range(5)] == [
        "a\nQuality: 1",
        "a",
        "b\nQuality: 5",
        "b\nQuality: 4",
        "b\nQuality: 3",
    ]


def test_wrap_and_transform_puts_the_conditioning_ahead_of_the_given_transforms(tmp_path):
    """Catches: appending instead of prepending. Order is the whole point -- a repack anywhere in
    the caller's list drops `quality` before a trailing conditioning transform could read it, and
    the result is the silent baseline again. Pinned positionally, not just behaviourally."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG]))
    repack = _repack()
    given = [repack]
    dataset = wrap_and_transform(concat, tags, given)
    assert dataset[0]["prompt"] == "a\nQuality: 1"
    assert given == [repack], "the caller's list must not be mutated in place"


def test_wrap_and_transform_still_binds_the_artifact_to_the_corpus(tmp_path):
    """The constructor must not become a way around the length bind it now owns."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 3]))
    with pytest.raises(ValueError, match="corpus mismatch"):
        wrap_and_transform(concat, tags, [])


# ---------------------------------------------------------------------------------------------
# The token-budget guard: pi0.5 uses max_token_len=200 (Pi0Config.__post_init__, non-KI), and
# PaligemmaTokenizer.tokenize truncates from the right with only a `logging.warning` -- deleting
# the tail of "Task: {text}, State: {32 ints};\nAction: " (the `Action:` marker) for the CFG arm
# ONLY, since only the CFG arm lengthens the prompt. `check_token_budget` must raise instead.
#
# `_tok` below is a REAL PaligemmaTokenizer, never skipped for a missing sentencepiece model: a
# skipped guard test is indistinguishable from a passing one, the same lesson as the anti-inert
# tripwire above. Round-1 training has already populated the download cache on every box this
# runs on.
# ---------------------------------------------------------------------------------------------


def _tok(max_len: int) -> _tokenizer.PaligemmaTokenizer:
    return _tokenizer.PaligemmaTokenizer(max_len)


def test_returns_the_margin_when_every_tagged_prompt_fits():
    """The happy path. Also pins that the margin is measured against the TAGGED prompt: if it
    were measured against the untagged one, `margin < untagged` below could never hold, because
    both sides would be the same number. Caught by mutation (see report)."""
    margin = check_token_budget(["pick up the bowl"], 200, tokenizer=_tok(200))
    assert margin > 0
    untagged = check_token_budget(["pick up the bowl"], 200, q=None, tokenizer=_tok(200))
    assert margin < untagged


def test_a_prompt_that_overflows_raises_and_names_the_margin():
    """Catches: relying on the upstream truncation warning. Upstream logs and continues; here the
    only acceptable behaviour is a raise, because the truncated token is the `Action:` marker.
    The prompt genuinely overflows a 64-token budget (measured raw length 235 with worst-case
    state) -- this is not a test-only artifact of a broken measurement."""
    long_prompt = " ".join(["place the red mug on the second shelf of the wooden cabinet"] * 8)
    with pytest.raises(ValueError, match="token budget") as excinfo:
        check_token_budget([long_prompt], 64, tokenizer=_tok(64))
    assert "64" in str(excinfo.value)


def test_it_raises_rather_than_warning(caplog):
    """Catches: a copy of upstream's `logging.warning` path sneaking in here, OR a same-budget
    tokenizer being used for measurement (which would run upstream's OWN warning before this
    function ever gets to raise). The spec's rule is absolute -- every guard raises, and no
    warning of any kind reaches the log, upstream's included.

    The brief's own version of this test only checked `pytest.raises`, never `caplog.records` --
    so it could not have caught either bug. Fixed here to actually assert silence; mutation-
    tested (see report) by measuring with `tokenizer` directly instead of an oversized probe,
    which reproduces upstream's warning and fails this assertion.
    """
    long_prompt = " ".join(["place the red mug on the shelf"] * 20)
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError):
            check_token_budget([long_prompt], 48, tokenizer=_tok(48))
    assert caplog.records == [], [r.message for r in caplog.records]


def test_the_tag_is_what_pushes_a_borderline_prompt_over():
    """The attribution test: the guard must fire on prompts the CONTROL tokenizes fine. Catches a
    check that tokenizes the bare prompt and therefore can never distinguish the arm's cost.

    Finding the borderline point needs the TRUE, untruncated token count. The brief's own version
    of this loop read `len(tok.tokenize(prompt, state=...)[0])` as if it were that count -- but
    `PaligemmaTokenizer.tokenize` always returns an array padded OR truncated to `tok`'s own
    `max_len`, so that length is the CONSTANT 200 for every prompt, `fits_untagged` is always
    True, `tagged` is never `> 200`, and the loop runs to exhaustion and fails on every input
    (verified: this is not hypothetical, the literal brief code was run and always hit
    `pytest.fail`, regardless of `check_token_budget`'s correctness). A separate, deliberately
    oversized probe recovers the real count via `mask.sum()` instead.
    """
    probe = _tok(4096)
    state = np.ones(32, np.float32)

    def real_len(text: str) -> int:
        _, mask = probe.tokenize(text, state=state)
        return int(mask.sum())

    for n in range(1, 60):
        prompt = " ".join(["open the drawer"] * n)
        fits_untagged = real_len(prompt) <= 200
        tagged = real_len(prompt + "\nQuality: 5")
        if fits_untagged and tagged > 200:
            break
    else:
        pytest.fail("no borderline prompt found; the tag must cost at least one token")
    check_token_budget([prompt], 200, q=None, tokenizer=_tok(200))  # control passes
    with pytest.raises(ValueError, match="token budget"):  # arm does not
        check_token_budget([prompt], 200, tokenizer=_tok(200))


def test_an_empty_prompt_list_raises():
    """A guard with nothing to check must say so, not pass. Same class as the EACCES swallows: a
    probe that answers confidently when it should fail."""
    with pytest.raises(ValueError, match="no prompts"):
        check_token_budget([], 200, tokenizer=_tok(200))


# --- coverage beyond the brief's list -----------------------------------------------------------
# The brief's five tests all exercise a SINGLE prompt. None of them pin the "MINIMUM margin over
# the corpus" contract the docstring promises, none pin that the guard reads the same tag spelling
# the transform actually emits (a drifted reimplementation would check a DIFFERENT string than the
# one that ships), and none probe the guard's own defensive ceiling. Added below, matching the
# pattern of the sibling coverage block above this one.


def test_the_margin_is_the_worst_prompt_in_the_corpus_not_the_first():
    """Catches: returning the FIRST prompt's margin, or the last, instead of the minimum over the
    whole list -- silently passing a corpus whose actual worst prompt overflows because a
    comfortable early or late entry masked it."""
    short = "pick up the bowl"
    long_ = " ".join(["place the red mug on the second shelf of the wooden cabinet"] * 3)
    margin_short_alone = check_token_budget([short], 200, tokenizer=_tok(200))
    margin_long_alone = check_token_budget([long_], 200, tokenizer=_tok(200))
    assert margin_long_alone < margin_short_alone
    combined = check_token_budget([short, long_, short], 200, tokenizer=_tok(200))
    assert combined == margin_long_alone


def test_the_guard_measures_the_exact_string_the_transform_emits():
    """Catches: a reimplementation of the tag format drifting from `slb_cfg.apply_metadata`, the
    one the online transform actually calls. If this check built its own `f"...Quality: {q}"`
    instead of calling `apply_metadata`, a spelling change to the real tag (e.g. a future added
    field) would silently stop being covered by the budget guard."""
    from openpi.training.quality_conditioning import AxisQualityConditioning

    prompt = "pick up the bowl"
    transform_output = AxisQualityConditioning()({"prompt": prompt, QUALITY_KEY: 5})["prompt"]
    probe = _tok(4096)
    state = np.full(32, 1.0, np.float32)
    expected_n = int(probe.tokenize(transform_output, state=state)[1].sum())
    margin = check_token_budget([prompt], 200, tokenizer=_tok(200))
    assert margin == 200 - expected_n


def test_a_prompt_beyond_even_the_probes_reach_raises_rather_than_under_report():
    """Catches: silently trusting a saturated probe's `mask.sum()`, which is a LOWER bound once
    the probe itself truncates -- reporting a margin computed from that number could read as a
    (very negative but bounded) ordinary overflow, when the true excess is unknown and could be
    far larger.

    `match=` deliberately targets the dedicated-branch phrase "own probe length", not just the
    number 8192: the saturated `n` also gets reported as "tokenizes to 8192 tokens" by the
    GENERIC overflow message once the branch that names the real cause is removed, since a
    saturated probe reports exactly `_PROBE_MAX_LEN` either way. Verified by mutation (see
    report) that matching on the bare number does NOT discriminate the two branches -- both
    contain "8192" -- while "own probe length" only appears on the dedicated path.
    """
    absurd_prompt = " ".join(["open the drawer"] * 4000)
    with pytest.raises(ValueError, match="own probe length"):
        check_token_budget([absurd_prompt], 200, tokenizer=_tok(200))


def test_q_defaults_to_five(monkeypatch):
    """Pins the documented default (`slb_cfg.INFER_QUALITY`, what CFG conditions on at
    inference) against silent drift, e.g. someone "fixing" the signature to `q: int | None = 1`.

    Token cost does not vary across q=1..5 for any of the 1401 real AXIS task names (measured),
    so comparing MARGINS between the default and an explicit q=5 cannot tell q=1 from q=5 apart
    -- both would produce the identical number. This spies on the actual value passed to
    `apply_metadata` instead of inferring it from a margin that cannot carry the signal.
    """
    import openpi.training.quality_conditioning as qc

    seen = []
    real_apply_metadata = qc.apply_metadata

    def spy(prompt, q_ep, mistake, **kwargs):
        seen.append(q_ep)
        return real_apply_metadata(prompt, q_ep, mistake, **kwargs)

    monkeypatch.setattr(qc, "apply_metadata", spy)
    check_token_budget(["pick up the bowl"], 200, tokenizer=_tok(200))
    assert seen == [5]
