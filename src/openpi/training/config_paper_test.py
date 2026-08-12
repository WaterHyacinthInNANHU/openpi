"""Axis-V1 paper-recipe twins (arXiv 2607.21588v1 Appendix F Table 10 / Appendix G Table 11).

The `_paper` configs are exposed the same way the `_ki` twins are: as SEPARATELY NAMED
configs rather than edits to `pi05_axis_pretrain_eef` / `pi05_libero_axisinit`, so any
in-flight run against the original names keeps the config it started with. The
`*_is_untouched` control tests are what enforce that.

Why pin the values in a test at all: every field here is a number a well-meaning reader might
"correct". The 10,000-step warmup is the standout -- a third of stage 2's 30,000-step budget,
which reads as a table typo and is not (confirmed with the author; and upstream openpi's own
`pi05_libero` ships exactly that warmup). A silent drift off-recipe would not fail anything
else, it would just quietly stop reproducing the paper.
"""

from __future__ import annotations

import numpy as np
import pytest

import openpi.training.config as _config
import openpi.training.optimizer as _optimizer

_STAGE1 = "pi05_axis_pretrain_eef_paper"
_STAGE2 = "pi05_libero_axisinit_paper"

# Appendix F Table 10 / Appendix G Table 11, as (config, field) -> value.
_TABLE_10 = {"action_horizon": 10, "num_train_steps": 100_000, "batch_size": 8, "ema_decay": 0.999}
_TABLE_11 = {"action_horizon": 10, "num_train_steps": 30_000, "batch_size": 64, "ema_decay": 0.999}

# Both tables state the same optimizer block, which is openpi's AdamW default field-for-field.
_PAPER_ADAMW = {"b1": 0.9, "b2": 0.95, "eps": 1e-8, "weight_decay": 1e-10, "clip_gradient_norm": 1.0}

# Both stages: 10,000 warmup then 5e-5 held constant. DO NOT scale the warmup to the budget.
_PAPER_WARMUP = 10_000
_PAPER_LR = 5e-5


@pytest.mark.parametrize(("name", "table"), [(_STAGE1, _TABLE_10), (_STAGE2, _TABLE_11)])
def test_paper_twin_matches_its_table(name: str, table: dict) -> None:
    cfg = _config.get_config(name)
    assert cfg.model.action_horizon == table["action_horizon"]
    assert cfg.num_train_steps == table["num_train_steps"]
    assert cfg.batch_size == table["batch_size"]  # GLOBAL; /fsdp_devices is per-GPU
    assert cfg.ema_decay == table["ema_decay"]
    assert cfg.fsdp_devices == 8


@pytest.mark.parametrize("name", [_STAGE1, _STAGE2])
def test_warmup_is_ten_thousand_absolute(name: str) -> None:
    """Both tables say 10,000 -- an absolute count, NOT a fraction of num_train_steps.

    Stage 2 spends a third of its budget in warmup. That is deliberate, is what upstream
    `pi05_libero` does, and was confirmed with the author. If this test fails because someone
    made warmup proportional to the budget, the fix is to revert that, not to update the test.
    """
    assert _config.get_config(name).lr_schedule.warmup_steps == _PAPER_WARMUP


@pytest.mark.parametrize("name", [_STAGE1, _STAGE2])
def test_lr_is_constant_after_warmup(name: str) -> None:
    """peak_lr == decay_lr is openpi's own idiom for a constant LR (`pi05_full_droid_finetune`).

    Asserted behaviourally, on the materialised optax schedule, rather than by reading the
    dataclass fields -- the claim that matters is "the LR does not move after warmup".
    """
    cfg = _config.get_config(name)
    schedule = cfg.lr_schedule.create()
    after = np.array([float(schedule(s)) for s in range(_PAPER_WARMUP, cfg.num_train_steps + 1, 97)])
    assert after.min() == after.max(), f"LR varies after warmup: [{after.min()}, {after.max()}]"
    assert after[0] == pytest.approx(_PAPER_LR, rel=1e-6)
    # And the warmup really is a ramp, not a no-op.
    assert float(schedule(0)) < float(schedule(_PAPER_WARMUP // 2)) < float(schedule(_PAPER_WARMUP))


@pytest.mark.parametrize("name", [_STAGE1, _STAGE2])
def test_optimizer_block_matches_the_paper(name: str) -> None:
    opt = _config.get_config(name).optimizer
    assert isinstance(opt, _optimizer.AdamW)
    for field, want in _PAPER_ADAMW.items():
        assert getattr(opt, field) == want, f"{name}.optimizer.{field}"


def test_stage2_schedule_is_upstream_pi05_libero_verbatim() -> None:
    """Table 11's schedule IS openpi's shipped LIBERO recipe; only batch size differs (64/256).

    This is the durable justification for the odd-looking warmup, so it is worth asserting:
    if upstream ever changes, we want to know that our "paper" claim moved with it.
    """
    assert _config.get_config(_STAGE2).lr_schedule == _config.get_config("pi05_libero").lr_schedule
    assert _config.get_config(_STAGE2).batch_size == 64
    assert _config.get_config("pi05_libero").batch_size == 256


def test_stage1_is_full_model_from_pi05_base_with_own_norm_stats() -> None:
    """The parts of Table 10 that were already true, and must stay true.

    Own norm stats are expressed as the ABSENCE of an `assets=` override (so assets resolve to
    the config's own dir). Reusing pi05_droid's stats is what the released Axis-V1-Training
    repo does and is a measured failure here, so the negative is asserted explicitly.
    """
    cfg = _config.get_config(_STAGE1)
    assert "pi05_base" in cfg.weight_loader.params_path
    assert cfg.freeze_filter == _config.nnx.Nothing()  # no LoRA, nothing frozen
    assert cfg.model.action_dim == 32
    assert cfg.data.eef_action is True  # OSC_POSE 7-D delta, the paper's real action space

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_config.asset_id != "droid"
    assert data_config.asset_id == data_config.repo_id == "Devon018/Franka-Datasets-v2"
    assert cfg.data.assets.assets_dir is None, "an assets override would reuse someone else's stats"
    # Appendix F's normalization formula is openpi's `_normalize_quantile`, on by default for pi0.5.
    assert data_config.use_quantile_norm is True


def test_stage2_transfers_params_only_and_needs_an_explicit_checkpoint() -> None:
    """Appendix G transfers MODEL PARAMS ONLY, from the EMA-smoothed set.

    A `CheckpointWeightLoader` reads the `params` item and nothing else, so optimizer/step/EMA
    state start fresh. Stage 1 sets ema_decay, and `checkpoints._split_params` writes
    `ema_params` into that `params` item -- so <step>/params IS the EMA-smoothed set.

    The default path must stay unusable: silently initialising from the older eef_ckpt_50000
    pilot (horizon 16, no EMA) would look like a successful run and would not be the paper's.
    """
    cfg = _config.get_config(_STAGE2)
    assert isinstance(cfg.weight_loader, _config.weight_loaders.CheckpointWeightLoader)
    path = cfg.weight_loader.params_path
    # Env-driven; unset in the test environment, so this is the sentinel.
    assert "eef_ckpt_50000" not in path
    assert path.startswith("/unset/") or path.endswith("/params")
    assert _config.get_config(_STAGE2).data.extra_delta_transform is False


@pytest.mark.parametrize(
    ("original", "twin"),
    [("pi05_axis_pretrain_eef", _STAGE1), ("pi05_libero_axisinit", _STAGE2)],
)
def test_original_configs_are_untouched(original: str, twin: str) -> None:
    """The whole point of a separately-named twin: in-flight runs keep their config."""
    base = _config.get_config(original)
    assert base.name != twin
    if original == "pi05_axis_pretrain_eef":
        assert base.model.action_horizon == 16
        assert base.ema_decay is None
        assert base.batch_size == 32
        assert base.lr_schedule.peak_lr == 2.5e-5
        assert base.lr_schedule.decay_lr == 2.5e-6
    else:
        assert base.lr_schedule.warmup_steps == 1_000
        assert base.lr_schedule.decay_lr == 5e-6


@pytest.mark.parametrize("name", [_STAGE1, _STAGE2])
def test_supervision_arms_are_off_so_the_loss_is_plain_bc(name: str) -> None:
    """Task B's invariant: no CFG tag, no AWR weight, on either paper config.

    `slb_sidecar_root` is the single gate for both. It gates `build_sampler`, which is the only
    producer of `weights_by_row`, which is the only thing that wraps the dataset in
    `WeightedRowDataset`, which is the only writer of `slb_awr_loss.LOSS_WEIGHT_KEY`. With no
    such key in the batch the data loader yields a 2-tuple, `train_step` gets loss_weights=None,
    and `slb_awr_loss.combine` returns literally `jnp.mean(chunked_loss)`.
    """
    cfg = _config.get_config(name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_config.slb_sidecar_root is None
    assert data_config.slb_task_id is None
    assert data_config.slb_variant == "vanilla"
    # And no CFG conditioning transform was prepended ahead of the repack.
    transform_names = [type(t).__name__ for t in data_config.repack_transforms.inputs]
    assert transform_names == ["RepackTransform"], transform_names


def test_vanilla_loss_is_bitwise_the_unweighted_mean() -> None:
    import jax.numpy as jnp

    import openpi.training.slb_awr_loss as slb_awr_loss

    rng = np.random.default_rng(0)
    chunked = jnp.asarray(rng.normal(size=(8, 10)), dtype=jnp.float32)
    assert slb_awr_loss.combine(chunked, None) == jnp.mean(chunked)
