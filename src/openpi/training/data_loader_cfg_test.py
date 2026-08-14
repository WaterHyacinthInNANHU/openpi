"""The CFG arm, observed through the real construction path.

Three properties, and none of them is visible in a loss curve.

(1) THE ARM IS LIVE. Everything else about quality conditioning is covered by tests that build a
transform, a `QualityTags` or a `TransformedDataset` themselves -- so deleting the wiring inside
`create_torch_data_loader` leaves every one of them green while the arm trains as plain BC under
its own name. This file drives the real function and DECODES the token ids of the real batches
back to text.

(2) THE TAG LANDS EXACTLY ONCE. `wrap_and_transform` prepends `AxisQualityConditioning` to the
transform list it is handed, and that list already begins with `repack_transforms.inputs`; adding
the conditioning to the repack group as well yields `"...\\nQuality: 5\\nQuality: 5"`, which is in
range, tokenizes cleanly, and matches no eval-time prompt. Counting occurrences in the decoded
prompt is the only place that composition is observable.

(3) COVERAGE NEUTRALITY. CFG changes what the model is conditioned on, not which rows are drawn,
so its row sequence must be byte-identical to the control's at the same seed. Plan A's siblings
shipped a 37% coverage difference that no loss curve revealed; here the difference must be zero
and is asserted as `array_equal`, not as a summary statistic.

THE ARTIFACT IS BUILT HERE, with numpy, in the layout `axis.dataset.build_quality_labels` writes
(the same choice `quality_conditioning_test._write_artifact` makes). Importing the offline
builder would bind this file to a checkout path -- `openpi/` is reached through a symlink from
every worktree, so `__file__` resolves to the shared checkout whose `axis/` is not the one under
test. What the offline dropout maths produces is tested offline
(`tests/axis/dataset/test_quality_labels.py`); what is tested HERE is that whatever the artifact
says, per row, is what the model's prompt carries.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import pretrain_dataset
from openpi.training import quality_conditioning
from openpi.training.quality_conditioning import N_BINS
from openpi.training.quality_conditioning import NO_TAG
from openpi.training.quality_conditioning import NOT_TRAINABLE

ARM = "pi05_axis_cfg"

N_ROWS = 512          # the toy corpus's concatenated row space
EPISODE = 8           # rows per episode...
TRAINABLE = 6         # ...of which the idle filter keeps the first 6
BATCH = 16
PROMPTS = ["pick up the bowl", "open the drawer"]


class _Recording(torch.utils.data.Dataset):
    """A toy AXIS corpus that RECORDS every index it is asked for.

    The recorded sequence is what makes coverage neutrality observable: with `num_workers=0` the
    fetch order IS the sampler's order.
    """

    def __init__(self, n: int, seen: list):
        self._n, self.seen = n, seen

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        self.seen.append(int(i))
        rng = np.random.default_rng(int(i))
        return {
            "observation.images.third_person": np.zeros((224, 224, 3), np.uint8),
            "observation.images.wrist": np.zeros((224, 224, 3), np.uint8),
            "state_eef": rng.random(8).astype(np.float32),
            "action_eef": rng.random((10, 7)).astype(np.float32),
            "prompt": PROMPTS[int(i) % len(PROMPTS)],
            # LeRobot v3.0 surfaces this, and it is the SUB-DATASET-local row under ConcatDataset.
            # Present so a wiring that keys the tag off it fails here the way it would on the box.
            "index": int(i),
        }


def _build_tags() -> tuple[np.ndarray, np.ndarray]:
    """`(dense tag array, planned rows)` in the layout `quality_labels.dense_tags` produces.

    Rows outside the idle-filtered plan are NOT_TRAINABLE; planned rows carry their episode's
    quintile, except for a dropped-out ~19% which carry NO_TAG (the unconditional branch --
    `QualityTags` refuses an artifact without one).

    The quintile CHANGES from episode to episode (1..5 over consecutive episodes) so a tag map
    shifted by one episode disagrees with this one on most rows; a constant-per-corpus bin would
    make the per-row assertion pass under exactly that failure.
    """
    blocks = [
        np.arange(e * EPISODE, e * EPISODE + TRAINABLE, dtype=np.int64)
        for e in range(N_ROWS // EPISODE)
    ]
    flat = np.concatenate(blocks)
    q_per_row = np.repeat(
        np.array([(e % N_BINS) + 1 for e in range(len(blocks))], dtype=np.uint8), TRAINABLE
    )
    dropped = np.random.default_rng(0).random(flat.size) < 0.1925
    tag = np.full(N_ROWS, NOT_TRAINABLE, dtype=np.uint8)
    tag[flat] = np.where(dropped, np.uint8(NO_TAG), q_per_row)
    return tag, flat


def _write_artifact(tmp_path, tag, *, reward_id="v2", name=None, prompts=PROMPTS, n_rows=None):
    """Write an artifact shaped exactly as `build_quality_labels.quality_meta` writes one.

    `bin_row_counts` is keyed "1".."N_BINS" by the builder and is where `QualityTags` reads the
    OFFLINE bin count back out, so an artifact without it is one the builder never emits.
    """
    tag = np.asarray(tag, dtype=np.uint8)
    trainable = tag[tag != NOT_TRAINABLE]
    meta = {
        "reward_id": reward_id,
        "n_rows": len(tag) if n_rows is None else int(n_rows),
        "bin_row_counts": {str(b): int((tag == b).sum()) for b in range(1, N_BINS + 1)},
        "realized_tagged_fraction": float((trainable != NO_TAG).sum()) / max(len(trainable), 1),
        "realized_drop_whole": 0.15,
        "realized_drop_component": 0.05,
        "n_trainable": len(trainable),
        "seed": 0,
        "config_hash": "toy",
    }
    path = tmp_path / (name or f"quality_{reward_id}.npz")
    np.savez(
        path,
        tag=tag,
        prompts=np.array([str(p) for p in prompts]),
        meta=np.array(json.dumps(meta)),
    )
    return path


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """A toy corpus, a tag artifact over its row space, and a fixed row plan."""
    tag, rows = _build_tags()
    # Non-vacuity: the headline test asserts BOTH branches, so both must occur.
    assert (tag[rows] == NO_TAG).any()
    assert (tag[rows] != NO_TAG).any()

    seen: list[int] = []
    monkeypatch.setattr(
        pretrain_dataset,
        "build_pretrain_concat_dataset",
        lambda data_config, action_horizon: _Recording(N_ROWS, seen),
    )
    monkeypatch.setattr(pretrain_dataset, "plan_rows_from_roots", lambda *_a, **_k: rows)
    return {"tag": tag, "rows": rows, "seen": seen, "path": _write_artifact(tmp_path, tag)}


def _model() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=10, max_token_len=200)


def _data_config(quality_path=None) -> _config.DataConfig:
    """The REGISTERED arm's own factory, with only what a launch supplies filled in."""
    base = _config.get_config(ARM)
    factory = dataclasses.replace(
        base.data,
        roots_index="roots.json",
        ranges_path="ranges.json",
        quality_required=False,
        quality_path=None if quality_path is None else str(quality_path),
    )
    return factory.create(base.assets_dirs, _model())


def _drive(data_config, *, seed=0, num_batches=8, batch_size=BATCH, framework="jax"):
    return list(
        _data_loader.create_torch_data_loader(
            data_config,
            _model(),
            action_horizon=10,
            batch_size=batch_size,
            num_batches=num_batches,
            num_workers=0,
            seed=seed,
            skip_norm_stats=True,
            framework=framework,
        )
    )


@pytest.fixture(scope="module")
def tok():
    return _tokenizer.PaligemmaTokenizer(200)


def _decode(observation, i, tok) -> str:
    ids = np.asarray(observation.tokenized_prompt)[i]
    mask = np.asarray(observation.tokenized_prompt_mask)[i].astype(bool)
    return tok._tokenizer.decode([int(t) for t in ids[mask]])  # noqa: SLF001


# --- the arm is live -----------------------------------------------------------------------------


def test_the_arm_is_live_and_every_tag_lands_on_its_own_row(toy, tok):
    """THE ANTI-INERT TEST. Catches: the wrapper never wired, the transform never prepended, the
    wrapper wired AFTER `transform_dataset`, or the tag looked up by `data["index"]` instead of
    the index the wrapper was called with. Asserting construction would pass in all four cases;
    decoding the real batch cannot.

    Per-row exactness, not a fraction: a uniform ~80% tagged would also be produced by a tag map
    shifted by one episode, which is the failure that trains happily on the wrong frames.
    """
    toy["seen"].clear()
    batches = _drive(_data_config(toy["path"]))
    n_tagged = n_bare = 0
    for b, (observation, _actions) in enumerate(batches):
        rows = toy["seen"][b * BATCH : (b + 1) * BATCH]
        assert len(rows) == BATCH
        for i, row in enumerate(rows):
            text = _decode(observation, i, tok)
            q = int(toy["tag"][row])
            assert q != NOT_TRAINABLE
            if q == NO_TAG:
                assert "Quality:" not in text, f"row {row} is dropped out but was tagged: {text!r}"
                n_bare += 1
            else:
                assert f"Quality: {q}" in text, f"row {row} expected Quality: {q}, got {text!r}"
                n_tagged += 1
    # Neither branch may be empty, or half of the assertion above never ran.
    assert n_tagged > 0, f"nothing was tagged in {len(batches)} batches"
    assert n_bare > 0, "no dropped-out row was drawn: the unconditional branch went unchecked"


def test_the_tag_appears_exactly_once_in_every_prompt(toy, tok):
    """THE DOUBLE-TAG TEST. `wrap_and_transform` prepends the conditioning to a transform list
    that already starts with the repack group's inputs, so a conditioning entry added to the
    repack group as well would tag twice -- in range, tokenizable, and matched by no eval-time
    prompt. `Quality:` occurring once is the only observable difference."""
    toy["seen"].clear()
    for observation, _actions in _drive(_data_config(toy["path"])):
        for i in range(BATCH):
            assert _decode(observation, i, tok).count("Quality:") <= 1


def test_the_control_config_tags_nothing(toy, tok):
    """The other half of the anti-inert test: the same corpus through the untagged config must
    yield ZERO tagged prompts. Catches a transform that tags unconditionally."""
    toy["seen"].clear()
    for observation, _actions in _drive(_data_config(None)):
        for i in range(BATCH):
            assert "Quality:" not in _decode(observation, i, tok)


def test_re_adding_the_conditioning_to_the_repack_group_is_refused(toy):
    """The double tag must be UNCONSTRUCTABLE, not merely observable.

    A later edit that "helpfully" heads the repack group with `AxisQualityConditioning` -- the
    arrangement the first draft of this task's brief prescribed -- must fail at loader
    construction rather than train an arm on a doubly-tagged prompt.
    """
    import openpi.transforms as _transforms

    data_config = _data_config(toy["path"])
    doubled = dataclasses.replace(
        data_config,
        repack_transforms=_transforms.Group(
            inputs=[
                quality_conditioning.AxisQualityConditioning(),
                *data_config.repack_transforms.inputs,
            ]
        ),
    )
    with pytest.raises(ValueError, match="twice"):
        _drive(doubled)


# --- coverage neutrality -------------------------------------------------------------------------


def test_the_row_sequence_is_byte_identical_to_the_control(toy):
    """COVERAGE NEUTRALITY, the arm's distinguishing property. Catches: a schedule accidentally
    wired in, rows restricted to the tagged subset, or any reseeding of the sampler. Plan A's
    siblings shipped a 37% coverage difference no loss curve revealed; here it must be zero."""
    toy["seen"].clear()
    _drive(_data_config(toy["path"]), seed=3)
    arm = list(toy["seen"])

    toy["seen"].clear()
    _drive(_data_config(None), seed=3)
    control = list(toy["seen"])

    assert len(arm) == len(control) > 0
    assert np.array_equal(np.asarray(arm), np.asarray(control))


def test_the_arm_draws_from_the_whole_planned_row_set(toy):
    """A full pass must reach every planned row exactly once -- i.e. the wrapper did not restrict
    the draw to the tagged subset, which would be a ~19% coverage cut disguised as conditioning."""
    toy["seen"].clear()
    n = len(toy["rows"]) // BATCH
    _drive(_data_config(toy["path"]), num_batches=n)
    assert sorted(toy["seen"]) == sorted(int(r) for r in toy["rows"][: n * BATCH])
    assert len(set(toy["seen"])) == n * BATCH


# --- the bindings are reached through the real path ----------------------------------------------


def test_a_not_trainable_row_raises_through_the_real_path(toy, tmp_path):
    """Catches: a tag array and a row plan that disagree about which rows exist. A default here
    would silently untag part of the arm.

    A FULL PASS (`RowSampler` permutes the planned rows, so 24 batches of 16 is every one of the
    384 exactly once), because a short run would only sometimes draw the row in question -- a
    flaky test here reads as "the guard sometimes does not fire", which is worse than no test.
    """
    tag = toy["tag"].copy()
    tag[int(toy["rows"][0])] = NOT_TRAINABLE
    path = _write_artifact(tmp_path, tag, name="quality_v2.npz")
    with pytest.raises(KeyError, match="not trainable"):
        _drive(_data_config(path), num_batches=len(toy["rows"]) // BATCH)


def test_a_corpus_mismatch_raises_at_loader_construction(toy, tmp_path):
    """The corpus binding must be reached through the real path, not only by calling the checker.
    Every index of a shorter artifact is still in bounds and means a different frame."""
    path = _write_artifact(tmp_path, toy["tag"][:-EPISODE], name="quality_v2.npz")
    with pytest.raises(ValueError, match="corpus mismatch") as excinfo:
        _drive(_data_config(path))
    # ...and the error names the corpus it disagrees with, not just the two lengths.
    assert "roots.json" in str(excinfo.value)


def test_a_filename_that_disagrees_with_the_reward_id_raises_at_loader_construction(toy, tmp_path):
    """The ONLY thing separating cfg_v2 from cfg_phase, checked where the run actually starts."""
    path = _write_artifact(tmp_path, toy["tag"], reward_id="v2", name="quality_phase.npz")
    with pytest.raises(ValueError, match="reward_id"):
        _drive(_data_config(path))


def test_an_overlong_prompt_raises_at_loader_construction(toy, tmp_path):
    """The token-budget guard must be REACHED, not merely exist. Catches a check that was written
    and never called -- which is how the tokenizer's own truncation warning (one line, mid-run)
    stays the only signal that this arm's `Action:` marker is being cut off."""
    path = _write_artifact(
        tmp_path,
        toy["tag"],
        prompts=[" ".join(["place the red mug on the shelf"] * 40)],
        name="quality_v2.npz",
    )
    with pytest.raises(ValueError, match="token budget"):
        _drive(_data_config(path))


def test_the_pytorch_path_refuses_the_arm_through_the_real_loader(toy):
    """The refusal must be REACHED, not merely defined.

    `_check_quality_unsupported_on_pytorch` is unit-tested by calling it directly, which stays
    green if its call site inside `create_torch_data_loader` is ever removed -- and then a CFG
    config run through `scripts/train_pytorch.py` trains as the control, since that branch never
    builds the pretrain row plan the arm's coverage claim is about.
    """
    with pytest.raises(ValueError, match="pytorch"):
        _drive(_data_config(toy["path"]), framework="pytorch")


def test_an_artifact_without_an_unconditional_branch_raises_at_loader_construction(toy, tmp_path):
    """No NO_TAG rows means no unconditional branch to guide away from: CFG degenerates to
    conditional BC, and nothing online would notice."""
    tag = toy["tag"].copy()
    tag[tag == NO_TAG] = 1
    path = _write_artifact(tmp_path, tag, name="quality_v2.npz")
    with pytest.raises(ValueError, match="NO_TAG"):
        _drive(_data_config(path))
