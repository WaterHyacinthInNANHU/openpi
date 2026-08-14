import dataclasses
import json

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.schedule_sampler import ScheduleSampler


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_torch_data_loader_replays_the_schedule_rows_in_order(tmp_path):
    """`TorchDataLoader` must hand a ScheduleSampler's rows through unaltered.

    Scope: this covers only the sampler -> torch DataLoader contract (no reshuffle, no reorder,
    no padding), because it constructs the loader itself. It does NOT exercise the wiring inside
    `create_torch_data_loader` -- see
    `test_create_torch_data_loader_replays_the_schedule_rows_in_order` below, which is the test
    that fails if `sampler=sampler` there ever regresses to `sampler=None`.

    Modeled on `test_torch_data_loader_parallel`: a small `FakeDataset` (deterministic per index
    via `jax.random.key(index)`) driven directly through `TorchDataLoader`.
    """
    model_config = pi0_config.Pi0Config(action_dim=8, action_horizon=4, max_token_len=16)
    dataset = _data_loader.FakeDataset(model_config, num_samples=12)

    total_steps, batch = 3, 4
    rng = np.random.default_rng(0)
    # A non-trivial (not row-major, repeats allowed) permutation of dataset indices, so a
    # sequential or shuffled sampler could not coincidentally reproduce it.
    rows = rng.integers(0, 12, size=(total_steps, batch)).astype(np.int64)
    schedule_path = tmp_path / "schedule.npz"
    np.savez(schedule_path, rows=rows, meta=np.array(json.dumps({"mode": "drop"})))

    sampler = ScheduleSampler(schedule_path)
    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=batch, sampler=sampler, num_workers=0
    )
    # Bypass the jax-sharding wrapper in TorchDataLoader.__iter__ (irrelevant to this property)
    # and read the raw torch DataLoader directly: drop_last=True over len(sampler) rows gives
    # exactly `total_steps` batches, one full pass.
    batches = list(loader.torch_loader)
    assert len(batches) == total_steps

    for t, actual in enumerate(batches):
        expected = _data_loader._collate_fn([dataset[int(i)] for i in rows[t]])
        # Compare leaf-wise rather than key-wise: some values (e.g. "images") are themselves
        # nested dicts, which np.asarray would otherwise wrap into an opaque 0-d object array.
        actual_leaves = jax.tree.leaves(actual)
        expected_leaves = jax.tree.leaves(expected)
        assert len(actual_leaves) == len(expected_leaves) > 0
        for a, e in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e))


@pytest.mark.parametrize(
    "transform", [_data_loader.transform_dataset, _data_loader.transform_iterable_dataset]
)
def test_missing_norm_stats_raise_rather_than_training_unnormalised(transform):
    """`_load_norm_stats` SWALLOWS a missing assets directory and returns None, logging one INFO
    line. That swallow is upstream behaviour and is not removed here, so what matters is whether
    anything downstream can then reach training with `norm_stats=None` -- an arm trained
    unnormalised and compared against a normalised control would look comparable and be invalid,
    which is exactly the shape of the two conclusions this project has already retracted.

    It cannot. Both transform entry points gate on `repo_id != "fake" and not skip_norm_stats`
    and raise, and neither training entry point (scripts/train.py, scripts/train_pytorch.py)
    passes `skip_norm_stats`, so a config with a real repo_id and missing assets fails at loader
    construction. Pinned here for both transforms because this is the only thing standing
    between a swallowed miss and a silently unnormalised run -- the assets binding in
    `_axis_pretrain_config` fixes only the two round-2 arms, while the swallow is generic.
    """
    cfg = _config.get_config("pi05_axis_drop")
    data_config = dataclasses.replace(
        cfg.data,
        # A fresh box, or any config whose assets have not been computed.
        assets=_config.AssetsConfig(
            assets_dir="/nonexistent/assets", asset_id="Devon018/Franka-Datasets-v2"
        ),
        roots_index="roots.json",
        schedule_path="s.npz",
    ).create(cfg.assets_dirs, cfg.model)
    assert data_config.norm_stats is None
    assert data_config.repo_id != "fake"

    with pytest.raises(ValueError, match="Normalization stats not found"):
        transform([{}], data_config)


def test_create_torch_data_loader_replays_the_schedule_rows_in_order(tmp_path):
    """The end-to-end wiring test: drive `create_torch_data_loader` and check what comes out.

    Everything else about the index schedule is covered by tests that construct a
    `ScheduleSampler`, or call a `_check_*` helper, themselves -- so deleting the corresponding
    call site inside `create_torch_data_loader` leaves them all green while the arm silently
    trains as plain BC on sequential rows. That failure (an arm built but never used) has
    already happened twice in this work, so it needs a test that only the real construction path
    can satisfy.

    Verified to FAIL when `sampler=sampler` in `create_torch_data_loader` is changed to
    `sampler=None`: the loader then yields rows 0..N sequentially and the very first batch
    mismatches.

    `repo_id="fake"` makes `create_torch_dataset` return a 1024-sample `FakeDataset` (returned
    before the pretrain-concat branch, so no corpus is touched) and makes `transform_dataset`
    skip norm stats -- but `pretrain_roots_index` is still what routes the loader into the
    pretrain sampler branch, and `pretrain_schedule_path` into the schedule sub-branch, so the
    exact code under test is the code that runs on the box.
    """
    model_config = pi0_config.Pi0Config(action_dim=8, action_horizon=4, max_token_len=16)
    n_dataset_rows = 1024  # create_torch_dataset's FakeDataset size for repo_id="fake"

    total_steps, batch = 3, 4
    rng = np.random.default_rng(0)
    # Repeats allowed and no row-major structure, so neither a sequential nor a shuffled sampler
    # could reproduce this by accident.
    rows = rng.integers(0, n_dataset_rows, size=(total_steps, batch)).astype(np.int64)
    schedule_path = tmp_path / "schedule.npz"
    np.savez(
        schedule_path,
        rows=rows,
        meta=np.array(json.dumps({"mode": "drop", "n_rows": n_dataset_rows})),
    )

    data_config = dataclasses.replace(
        _config.DataConfig(),
        repo_id="fake",
        # Present only to select the pretrain branch; the schedule sub-branch never reads it.
        pretrain_roots_index="roots_for_the_fake_dataset.json",
        pretrain_schedule_path=str(schedule_path),
        pretrain_expected_mode="drop",
    )

    loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config,
        action_horizon=model_config.action_horizon,
        batch_size=batch,
        num_batches=total_steps,
        num_workers=0,
        num_train_steps=total_steps,
    )

    # The same rows, read straight off the dataset the loader built, to compare against.
    reference = _data_loader.transform_dataset(
        _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config),
        data_config,
    )

    batches = list(loader)
    assert len(batches) == total_steps
    for t, (observation, actions) in enumerate(batches):
        expected = _data_loader._collate_fn([reference[int(i)] for i in rows[t]])  # noqa: SLF001
        np.testing.assert_array_equal(np.asarray(actions), np.asarray(expected["actions"]))
        np.testing.assert_array_equal(np.asarray(observation.state), np.asarray(expected["state"]))


def test_create_torch_data_loader_rejects_a_schedule_built_on_another_corpus(tmp_path):
    """The corpus binding must be reached through the real construction path too: a schedule
    whose `meta["n_rows"]` disagrees with the dataset it is handed indexes different frames."""
    model_config = pi0_config.Pi0Config(action_dim=8, action_horizon=4, max_token_len=16)
    rows = np.zeros((3, 4), dtype=np.int64)
    schedule_path = tmp_path / "schedule.npz"
    np.savez(
        schedule_path,
        rows=rows,
        # 1023, not the FakeDataset's 1024: one episode short, every index still in bounds.
        meta=np.array(json.dumps({"mode": "drop", "n_rows": 1023})),
    )
    data_config = dataclasses.replace(
        _config.DataConfig(),
        repo_id="fake",
        pretrain_roots_index="roots_for_the_fake_dataset.json",
        pretrain_schedule_path=str(schedule_path),
        pretrain_expected_mode="drop",
    )
    with pytest.raises(ValueError, match="corpus mismatch") as excinfo:
        _data_loader.create_torch_data_loader(
            data_config,
            model_config,
            action_horizon=model_config.action_horizon,
            batch_size=4,
            num_batches=3,
            num_workers=0,
            num_train_steps=3,
        )
    assert "roots_for_the_fake_dataset.json" in str(excinfo.value)


def test_schedule_mode_mismatch_raises():
    data_config = dataclasses.replace(_config.DataConfig(), pretrain_expected_mode="drop")
    with pytest.raises(ValueError, match="mode mismatch"):
        _data_loader._check_schedule_mode(data_config, {"mode": "anneal"})


def test_schedule_mode_match_does_not_raise():
    data_config = dataclasses.replace(_config.DataConfig(), pretrain_expected_mode="drop")
    _data_loader._check_schedule_mode(data_config, {"mode": "drop"})  # must not raise


def test_schedule_mode_check_is_a_noop_without_an_expected_mode():
    """Every non-schedule config (and a schedule config launched without expected_mode set)
    must not be affected by this guard."""
    data_config = _config.DataConfig()
    _data_loader._check_schedule_mode(data_config, {"mode": "anneal"})  # must not raise


def test_resume_with_a_schedule_raises():
    with pytest.raises(ValueError, match="resum"):
        _data_loader._check_schedule_resume("s.npz", resuming=True)


def test_no_resume_with_a_schedule_does_not_raise():
    _data_loader._check_schedule_resume("s.npz", resuming=False)  # must not raise


def test_pytorch_framework_with_a_schedule_raises():
    data_config = dataclasses.replace(_config.DataConfig(), pretrain_schedule_path="s.npz")
    with pytest.raises(ValueError, match="pytorch"):
        _data_loader._check_schedule_unsupported_on_pytorch(data_config)


def test_pytorch_framework_without_a_schedule_does_not_raise():
    data_config = _config.DataConfig()
    _data_loader._check_schedule_unsupported_on_pytorch(data_config)  # must not raise


def test_pytorch_framework_with_quality_conditioning_raises():
    """The CFG arm's coverage-neutrality claim is defined against the jax path only.

    The pretrain row plan (`plan_rows_from_roots` -> `RowSampler`) is built in the jax branch of
    `create_torch_data_loader` and nowhere else, so under `framework="pytorch"` neither the arm
    nor its control draws the row sequence the arm claims to share -- the comparison the whole
    experiment rests on does not hold there, and nothing in a loss curve says so.
    """
    data_config = dataclasses.replace(_config.DataConfig(), pretrain_quality_path="quality_v2.npz")
    with pytest.raises(ValueError, match="pytorch"):
        _data_loader._check_quality_unsupported_on_pytorch(data_config)


def test_pytorch_framework_without_quality_conditioning_does_not_raise():
    data_config = _config.DataConfig()
    _data_loader._check_quality_unsupported_on_pytorch(data_config)  # must not raise


# --- the filename<->reward_id binding -----------------------------------------------------------
#
# `_check_schedule_mode` cannot separate `drop_v2` from `drop_phase`: same config name
# (`pi05_axis_drop`), same meta["mode"], different reward. Only the artifact's FILENAME says
# which, and nothing checked it against the artifact's contents.


@pytest.mark.parametrize(
    ("stem", "mode", "reward_id"),
    [("drop_v2", "drop", "v2"), ("drop_phase", "drop", "phase"),
     ("anneal_v2", "anneal", "v2"), ("anneal_phase", "anneal", "phase")],
)
def test_the_four_shipped_artifact_names_match_their_own_meta(stem, mode, reward_id):
    """The names and pairings the round-2 arms actually launch with (see
    reports/round2_schedule_artifacts.md); the check must accept every one of them."""
    _data_loader._check_schedule_reward_id(
        f"/some/scratch/schedules/{stem}.npz", {"mode": mode, "reward_id": reward_id}
    )  # must not raise


def test_a_drop_artifact_rebuilt_from_the_other_reward_is_refused():
    """`drop_v2.npz` carrying the phase reward would train under `exp_name=v3_5000_drop_v2` and
    pass every other guard -- the mode matches, the corpus matches, the budget matches."""
    with pytest.raises(ValueError, match="drop_phase"):
        _data_loader._check_schedule_reward_id(
            "schedules/drop_v2.npz", {"mode": "drop", "reward_id": "phase"}
        )


def test_a_mode_shaped_name_without_a_reward_id_in_the_meta_is_refused():
    with pytest.raises(ValueError, match="reward_id"):
        _data_loader._check_schedule_reward_id("schedules/drop_v2.npz", {"mode": "drop"})


@pytest.mark.parametrize("path", ["s.npz", "/tmp/staging/schedule.npz", "run_2026-08-14.npz"])
def test_a_name_that_makes_no_claim_is_left_alone(path):
    """A name outside the `<mode>_<reward_id>` convention claims nothing, so there is nothing to
    contradict: the guard must not turn a convention into a requirement."""
    _data_loader._check_schedule_reward_id(path, {"mode": "drop", "reward_id": "v2"})
