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
    """If `sampler=sampler` regressed to `sampler=None` (or to any sampler other than the
    ScheduleSampler), the loader would draw its own rows -- shuffled or sequential -- instead of
    replaying the artifact, and the arm would silently train as plain BC. This is the exact
    failure the whole index-schedule feature exists to prevent, so it must be caught by a test,
    not only by a review of the wiring.

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
