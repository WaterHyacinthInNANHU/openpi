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


def test_the_two_arms_differ_only_in_their_name_and_expected_mode() -> None:
    """Same recipe, same data config, aside from the mode binding each name carries.

    The schedule artifact supplied at launch is still the arm's actual content, but its
    `meta["mode"]` must now match what the config's name promises (`expected_mode`) -- see
    `DataConfig.pretrain_expected_mode`. Before that binding existed the two configs were equal
    modulo `name` alone; now they are equal modulo `name` and this one field, which is exactly
    the fix: `pi05_axis_drop` and `pi05_axis_anneal` are no longer interchangeable data configs
    that merely differ in which artifact happens to be handed to them at launch.
    """
    drop, anneal = (_config.get_config(n) for n in ARMS)
    assert drop.data.expected_mode == "drop"
    assert anneal.data.expected_mode == "anneal"
    twin = dataclasses.replace(
        drop,
        name=anneal.name,
        data=dataclasses.replace(drop.data, expected_mode=anneal.data.expected_mode),
    )
    assert twin == anneal


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


@pytest.mark.parametrize(("name", "mode"), [("pi05_axis_drop", "drop"), ("pi05_axis_anneal", "anneal")])
def test_registered_arms_require_their_schedule(name: str, mode: str) -> None:
    """The registered configs must carry the guard, even though `schedule_path` itself is unset
    (per-run, from conf/experiments) -- see `test_no_schedule_is_baked_into_the_registered_config`."""
    data = _config.get_config(name).data
    assert data.schedule_required is True
    assert data.expected_mode == mode


@pytest.mark.parametrize("name", ARMS)
def test_a_named_schedule_arm_launched_without_a_schedule_path_raises(name: str, tmp_path) -> None:
    """A launch that omits --data.schedule_path must not silently train the plain BC control
    under the arm's name; the only prior symptom was the absence of the "index schedule" log
    line, which is easy to miss."""
    cfg = _config.get_config(name)
    with pytest.raises(ValueError, match="schedule_path"):
        cfg.data.create(tmp_path, cfg.model)


def test_schedule_required_raises_even_off_the_registered_arms(tmp_path) -> None:
    with pytest.raises(ValueError, match="schedule_path"):
        _factory(roots_index="roots.json", schedule_required=True, expected_mode="drop").create(
            tmp_path, pi0_config.Pi0Config(pi05=True)
        )


@pytest.mark.parametrize("mode", ["drop", "anneal"])
def test_expected_mode_reaches_the_data_config(mode: str, tmp_path) -> None:
    """`create()` is the only bridge to the loader (see `test_schedule_path_reaches_the_data_config`
    above): expected_mode must cross it too, or the mismatch check in data_loader.py has nothing
    to compare against."""
    data = _factory(
        roots_index="roots.json", schedule_path="s.npz", schedule_required=True, expected_mode=mode
    ).create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data.pretrain_expected_mode == mode


def test_no_expected_mode_leaves_the_field_unset(tmp_path) -> None:
    data = _factory(roots_index="roots.json").create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data.pretrain_expected_mode is None


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
