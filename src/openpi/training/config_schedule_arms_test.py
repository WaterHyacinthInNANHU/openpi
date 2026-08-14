"""The round-2 index-schedule arms: two names, one recipe, one differing field.

Round 1 picked its supervision out of $AXIS_PRETRAIN_AWR_WEIGHTS, so a checkpoint could not say
which arm produced it without also having the launch environment. These tests pin the
correction: the schedule is a config FIELD, it actually reaches the DataConfig the loader reads,
and an arm cannot silently inherit round 1's env var on top of its schedule.
"""

from __future__ import annotations

import dataclasses

import pytest

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config

ARMS = ("pi05_axis_drop", "pi05_axis_anneal")

# Round 1's `bc` budget, which these arms must match exactly to join the same table.
_BUDGET = {"num_train_steps": 20_605, "batch_size": 64, "action_horizon": 10, "ema_decay": 0.999}

# NOT round 1's warmup: round 1 ran 2,060 (10% of the budget) from its arm TOML, overriding this
# default. Pinned here so nobody reads the registered config as the full launch recipe -- the
# `lr_schedule.*` overrides live in conf/experiments and must be copied across for parity.
_PAPER_WARMUP_DEFAULT = 10_000


def _factory(**kwargs) -> _config.AxisFrankaPretrainDataConfig:
    return _config.AxisFrankaPretrainDataConfig(repo_id="Devon018/Franka-Datasets-v2", **kwargs)


@pytest.mark.parametrize("name", ARMS)
def test_arm_matches_the_round_one_recipe(name: str) -> None:
    cfg = _config.get_config(name)
    assert cfg.num_train_steps == _BUDGET["num_train_steps"]
    assert cfg.batch_size == _BUDGET["batch_size"]  # GLOBAL; /fsdp_devices is per-GPU
    assert cfg.model.action_horizon == _BUDGET["action_horizon"]
    assert cfg.ema_decay == _BUDGET["ema_decay"]
    assert cfg.fsdp_devices == 8
    assert cfg.lr_schedule.warmup_steps == _PAPER_WARMUP_DEFAULT
    assert cfg.data.eef_action is True


def test_the_two_arms_differ_only_in_their_name() -> None:
    """Same recipe, same data config; the schedule artifact supplied at launch is the arm."""
    drop, anneal = (_config.get_config(n) for n in ARMS)
    assert dataclasses.replace(drop, name=anneal.name) == anneal


@pytest.mark.parametrize("name", ARMS)
def test_no_schedule_is_baked_into_the_registered_config(name: str) -> None:
    """The artifact path is per-run (conf/experiments), so the registered config carries none."""
    assert _config.get_config(name).data.schedule_path is None


def test_schedule_path_reaches_the_data_config(tmp_path) -> None:
    """`create()` is the only bridge to the loader: a field that stops here is a dead arm."""
    data = _factory(roots_index="roots.json", ranges_path="ranges.json",
                    schedule_path="schedules/drop_v2.npz").create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data.pretrain_schedule_path == "schedules/drop_v2.npz"


def test_no_schedule_leaves_the_field_unset(tmp_path) -> None:
    data = _factory(roots_index="roots.json").create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data.pretrain_schedule_path is None


def test_schedule_without_a_roots_index_raises(tmp_path) -> None:
    """Without roots_index the pretrain branch never runs and the schedule is ignored -- i.e. the
    arm would train as the plain baseline under the arm's name."""
    with pytest.raises(ValueError, match="roots_index"):
        _factory(schedule_path="s.npz").create(tmp_path, pi0_config.Pi0Config(pi05=True))


def test_schedule_together_with_awr_weights_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="two arms at once"):
        _factory(roots_index="roots.json", schedule_path="s.npz",
                 awr_weights="w.json").create(tmp_path, pi0_config.Pi0Config(pi05=True))


def test_schedule_arm_ignores_the_round_one_env_var(monkeypatch) -> None:
    """A box with $AXIS_PRETRAIN_AWR_WEIGHTS exported must not turn a schedule arm into AWR."""
    monkeypatch.setenv("AXIS_PRETRAIN_AWR_WEIGHTS", "/some/awr_weights.json")
    # The factory, not get_config: _CONFIGS is built at import time, so the env var is long read
    # by the time a test could patch it.
    arm = _config._axis_pretrain_config(eef=True, paper=True, batch_size=64,  # noqa: SLF001
                                        num_train_steps=20_605, name="pi05_axis_drop")
    assert arm.data.awr_weights is None
    baseline = _config._axis_pretrain_config(eef=True, paper=True)  # noqa: SLF001
    assert baseline.data.awr_weights == "/some/awr_weights.json"  # unchanged for round 1
