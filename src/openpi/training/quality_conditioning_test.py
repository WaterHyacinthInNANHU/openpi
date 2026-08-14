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

from openpi.training.quality_conditioning import NO_TAG
from openpi.training.quality_conditioning import NOT_TRAINABLE
from openpi.training.quality_conditioning import QUALITY_KEY
from openpi.training.quality_conditioning import AxisQualityConditioning
from openpi.training.quality_conditioning import QualityTaggedDataset
from openpi.training.quality_conditioning import QualityTags


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


def test_the_wrapper_uses_its_own_index_argument():
    """Catches: a wrapper that forwards `item["index"]` instead of the index it was called with.
    Row 5 must receive tag[5], not tag[0].

    The tag pattern must NOT be periodic in the first sub-dataset's length. `arange(1,13) % 5 + 1`
    -- the obvious choice -- has period 5 == len(_Sub(5)), so `tag[i] == tag[i - 5]` for every row
    past the boundary and the local-index bug passes this test unnoticed (verified by mutation).
    The assertion below pins the property so a future edit cannot quietly re-degenerate it.
    """
    concat = torch.utils.data.ConcatDataset([_Sub(5, "a"), _Sub(7, "b")])
    tag = np.array([1, 2, 3, 4, 5, 5, 1, 4, 2, 3, 1, 5], dtype=np.uint8)
    assert all(tag[i] != tag[i - 5] for i in range(5, 12)), "pattern is blind to the local-index bug"
    tags = _tags(tag)
    wrapped = QualityTaggedDataset(concat, tags)
    assert len(wrapped) == 12
    for i in range(12):
        assert wrapped[i][QUALITY_KEY] == int(tags.tag[i])


def _tags(tag: np.ndarray, tmp_path=None, prompts=("pick up the bowl",), reward_id="v2"):
    """Build a QualityTags without touching disk when the test does not need a file."""
    obj = QualityTags.__new__(QualityTags)
    obj.tag = np.asarray(tag, dtype=np.uint8)
    obj.prompts = [str(p) for p in prompts]
    obj.meta = {"reward_id": reward_id, "n_rows": int(len(tag))}
    obj.path = tmp_path
    return obj


def _write_artifact(tmp_path, tag, *, reward_id="v2", name=None, prompts=("pick up the bowl",)):
    p = tmp_path / (name or f"quality_{reward_id}.npz")
    np.savez(
        p,
        tag=np.asarray(tag, dtype=np.uint8),
        prompts=np.array([str(s) for s in prompts]),
        meta=np.array(json.dumps({"reward_id": reward_id, "n_rows": int(len(tag))})),
    )
    return p


def test_reading_a_not_trainable_row_raises(tmp_path):
    """Catches: the 0.0-default pattern. A sentinel that silently reads as NO_TAG turns a
    misaligned map into an untagged arm -- i.e. the control, under the arm's name."""
    tags = QualityTags(_write_artifact(tmp_path, [1, NOT_TRAINABLE, 3]))
    assert tags.tag_for_row(0) == 1
    with pytest.raises(KeyError, match="not trainable"):
        tags.tag_for_row(1)


def test_a_tag_array_that_disagrees_with_the_dataset_raises(tmp_path):
    """Catches: a grown or shrunk corpus. Every index would still be in bounds and mean a
    different frame -- the case `ScheduleSampler.check_dataset_rows` exists to catch, repeated
    here because the tag artifact has the same exposure."""
    tags = QualityTags(_write_artifact(tmp_path, [1, 2, 3]))
    tags.check_dataset_rows(3)
    with pytest.raises(ValueError, match="corpus mismatch") as excinfo:
        tags.check_dataset_rows(4, "roots_5000.json")
    assert "roots_5000.json" in str(excinfo.value)


def test_a_filename_that_disagrees_with_the_reward_id_raises(tmp_path):
    """Catches: quality_phase.npz containing the v2 tags. Both arms share ONE config name."""
    p = _write_artifact(tmp_path, [1, 2, 3], reward_id="v2", name="quality_phase.npz")
    tags = QualityTags(p)
    with pytest.raises(ValueError, match="reward_id"):
        tags.check_reward_id(p)


def test_filename_check_is_a_noop_on_a_name_that_makes_no_claim(tmp_path):
    p = _write_artifact(tmp_path, [1, 2, 3], reward_id="v2", name="staging.npz")
    QualityTags(p).check_reward_id(p)


def test_an_artifact_without_a_reward_id_raises(tmp_path):
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.array([1], np.uint8), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"n_rows": 1})))
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
    """Catches: a repack order that puts this transform AFTER RepackTransform, which rebuilds the
    dict and drops anything not in its map. Returning `data` unchanged there is the same silent
    revert-to-control."""
    with pytest.raises(KeyError, match="prompt"):
        AxisQualityConditioning()({QUALITY_KEY: 3})


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
    np.savez(p, tag=np.array([1, 2, 3], np.int64), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 3})))
    with pytest.raises(ValueError, match="uint8"):
        QualityTags(p)


def test_a_two_dimensional_tag_array_is_rejected(tmp_path):
    """Catches: a (steps, batch) schedule artifact handed to the quality reader by mistake --
    both are .npz files of ints and the two arms sit next to each other on disk. Indexing a 2-D
    array by row would yield a whole ROW of tags, and `int()` of it raises far downstream."""
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.zeros((3, 4), np.uint8), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 12})))
    with pytest.raises(ValueError, match="1-D"):
        QualityTags(p)


def test_an_artifact_without_prompts_is_rejected(tmp_path):
    """Catches: an artifact built before `corpus_prompts` existed. The token-budget guard
    tokenizes this list, so an empty one makes the guard pass vacuously -- the builder raises on
    it at write time and the reader must not accept what the builder refuses to emit."""
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.array([1], np.uint8), prompts=np.array([], dtype="<U1"),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 1})))
    with pytest.raises(ValueError, match="prompt"):
        QualityTags(p)


def test_meta_row_count_disagreeing_with_the_tag_array_is_rejected(tmp_path):
    """Catches: a truncated or hand-edited artifact. `check_dataset_rows` compares the DATASET
    against `len(tag)`, so a meta that claims a different corpus size would go unnoticed while
    every provenance line printed from meta describes a corpus this file does not cover."""
    p = tmp_path / "quality_v2.npz"
    np.savez(p, tag=np.array([1, 2, 3], np.uint8), prompts=np.array(["x"]),
             meta=np.array(json.dumps({"reward_id": "v2", "n_rows": 99})))
    with pytest.raises(ValueError, match="n_rows"):
        QualityTags(p)


def test_a_filename_that_agrees_with_the_reward_id_passes(tmp_path):
    """The happy path of the binding check: the convention must not be a tripwire on itself."""
    p = _write_artifact(tmp_path, [1, 2, 3], reward_id="v2")
    tags = QualityTags(p)
    assert tags.reward_id == "v2"
    tags.check_reward_id(p)


def test_a_negative_row_raises_rather_than_reading_from_the_tail(tmp_path):
    """Catches: `self.tag[index]` with no lower-bound check. Numpy indexes -1 as the LAST row, so
    a negative index silently returns a real, wrong tag instead of raising -- the same silent
    wrap `ScheduleSampler` refuses for negative schedule rows."""
    tags = QualityTags(_write_artifact(tmp_path, [1, 2, 3]))
    with pytest.raises(IndexError, match="negative"):
        tags.tag_for_row(-1)


def test_a_row_beyond_the_artifact_raises(tmp_path):
    """Catches: a corpus larger than the artifact reaching the wrapper without
    `check_dataset_rows` having been called (i.e. the wiring forgot the bind)."""
    tags = QualityTags(_write_artifact(tmp_path, [1, 2, 3]))
    with pytest.raises(IndexError):
        tags.tag_for_row(3)


def test_the_wrapper_does_not_mutate_the_underlying_item():
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
    wrapped = QualityTaggedDataset(inner, _tags(np.array([1, 2], np.uint8)))
    assert wrapped[0][QUALITY_KEY] == 1
    assert QUALITY_KEY not in inner.item


def test_the_wrapper_refuses_a_negative_index():
    """Catches: a negative index reaching the pair. The inner dataset would return its LAST row
    while the tag lookup must not silently agree -- both would be 'valid', and wrong."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    wrapped = QualityTaggedDataset(concat, _tags(np.array([1, 2], np.uint8)))
    with pytest.raises(IndexError, match="negative"):
        wrapped[-1]


def test_the_wrapper_refuses_a_not_trainable_row():
    """Catches: a sampler whose row plan and the tag artifact disagree about which rows exist.
    The wrapper is where that surfaces, because it is the only place both are in hand."""
    concat = torch.utils.data.ConcatDataset([_Sub(2, "a")])
    wrapped = QualityTaggedDataset(concat, _tags(np.array([1, NOT_TRAINABLE], np.uint8)))
    assert wrapped[0][QUALITY_KEY] == 1
    with pytest.raises(KeyError, match="not trainable"):
        wrapped[1]


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
    """THE ARM, end to end at this tier: wrapper on the RAW concat dataset, transform at the head
    of the transform list -- the exact order `create_torch_data_loader` must use.

    Catches: wrapping where the AWR wrapper sits (i.e. AFTER `transform_dataset`), which injects
    the tag once the prompt has already been rewritten and leaves every prompt bare. Also catches
    a tag/row misalignment across the ConcatDataset boundary, which no single-sub-dataset test
    can see.
    """
    from openpi.training.data_loader import TransformedDataset

    concat = torch.utils.data.ConcatDataset([_Sub(2, "a"), _Sub(3, "b")])
    tags = QualityTags(_write_artifact(tmp_path, [1, NO_TAG, 5, 4, 3]))
    tags.check_dataset_rows(len(concat))
    dataset = TransformedDataset(QualityTaggedDataset(concat, tags), [AxisQualityConditioning()])
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
    tags = QualityTags(_write_artifact(tmp_path, [1, 2]))
    wrong_order = QualityTaggedDataset(
        TransformedDataset(concat, [AxisQualityConditioning()]), tags
    )
    with pytest.raises(KeyError, match="quality"):
        wrong_order[0]
