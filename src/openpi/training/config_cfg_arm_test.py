"""`pi05_axis_cfg` must be round 1's `bc` in every field but what the prompt says.

CFG's whole claim is that it changes what the model is CONDITIONED ON and nothing else -- not
which rows are drawn, not the budget, not the norm stats. Each of those is asserted here against
round 1's (and the sibling schedule arms') OWN registered configs rather than against hardcoded
numbers, so the two cannot drift.

WHERE THE CONDITIONING IS *NOT*. This factory deliberately leaves `repack_transforms` exactly as
the control builds it. `quality_conditioning.wrap_and_transform` -- the only supported way to
assemble the wrapper and the transform, and what the loader wiring must call -- PREPENDS
`AxisQualityConditioning` to the transform list it is handed, and that list is
`transform_dataset`'s, which begins with `repack_transforms.inputs`. Putting the conditioning in
the repack group as well would apply the tag TWICE (`...\\nQuality: 5\\nQuality: 5`), which
nothing raises on and no loss curve shows. So the repack group is asserted here to be the
control's, byte-for-byte, with and without a quality artifact.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config

ARM = "pi05_axis_cfg"

# The round-1 control this arm must be a single-variable change from.
ROUND1 = "pi05_axis_pretrain_eef_paper"

# A round-2 sibling: same recipe, same budget, a DIFFERENT mechanism. Used for the parity twin
# below, which pins model/optimiser/lr/ema/fsdp/weight-loader in one comparison.
SIBLING = "pi05_axis_drop"


def _factory(**kwargs) -> _config.AxisFrankaPretrainDataConfig:
    return _config.AxisFrankaPretrainDataConfig(
        repo_id="fake", roots_index="roots.json", **kwargs
    )


def _model() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(pi05=True)


# --- the field crosses the factory -> DataConfig boundary ---------------------------------------


def test_the_arm_is_registered():
    assert _config.get_config(ARM).name == ARM


def test_quality_path_reaches_the_data_config(tmp_path):
    """THE INERT-ARM TEST at the config seam. Catches: the field added to the factory but never
    forwarded to DataConfig, which is exactly how round 1's AWR arm and Plan A's schedule arms
    each shipped inert. `getattr(data_config, "pretrain_quality_path", None)` would then return
    None forever and the loader branch would never run."""
    dc = _factory(quality_path="quality_v2.npz").create(tmp_path, _model())
    assert dc.pretrain_quality_path == "quality_v2.npz"


def test_no_quality_path_leaves_the_field_unset(tmp_path):
    dc = _factory().create(tmp_path, _model())
    assert dc.pretrain_quality_path is None


def test_no_quality_path_is_baked_into_the_registered_config():
    """The artifact is per-run (conf/experiments), so the registered config carries none -- and
    both CFG rewards (`cfg_v2`, `cfg_phase`) run under this ONE name, distinguished only by the
    filename handed to `--data.quality_path`."""
    assert _config.get_config(ARM).data.quality_path is None


# --- the conditioning is NOT in the repack group -------------------------------------------------


def _repack_inputs(dc) -> list:
    return list(dc.repack_transforms.inputs)


def test_the_repack_group_is_the_controls_even_with_a_quality_artifact(tmp_path):
    """Catches the DOUBLE-TAG mutation: adding `AxisQualityConditioning()` to the head of the
    repack group here, on top of the one `wrap_and_transform` prepends in the loader.

    The result is a prompt carrying `Quality: 5` twice. Nothing raises -- the transform is
    idempotent-looking, the tag is in range, the tokenizer is happy -- and the arm trains on a
    string no eval-time prompt will ever match. The repack group must therefore stay the
    control's, and the conditioning must reach the chain only through `wrap_and_transform`.
    """
    from openpi.training import quality_conditioning

    arm = _factory(quality_path="quality_v2.npz").create(tmp_path, _model())
    control = _factory().create(tmp_path, _model())
    assert len(_repack_inputs(arm)) == len(_repack_inputs(control)) == 1
    assert not any(
        isinstance(t, quality_conditioning.AxisQualityConditioning) for t in _repack_inputs(arm)
    )
    assert _repack_inputs(arm) == _repack_inputs(control)


def test_the_registered_arm_repacks_exactly_what_round_one_repacks(tmp_path):
    """Byte-identical to round 1's, not merely similar: the repack map decides which parquet
    columns become `state`/`actions`, so a difference here is a different action space."""
    arm_cfg, r1_cfg = _config.get_config(ARM), _config.get_config(ROUND1)
    arm = dataclasses.replace(arm_cfg.data, quality_required=False).create(tmp_path, arm_cfg.model)
    control = r1_cfg.data.create(tmp_path, r1_cfg.model)
    assert _repack_inputs(arm) == _repack_inputs(control)


# --- the guards ----------------------------------------------------------------------------------


def test_the_registered_arm_requires_its_artifact(tmp_path):
    """Catches: launching pi05_axis_cfg without --data.quality_path, which trains the plain
    control under the arm's name. The ONLY other symptom is a missing log line."""
    cfg = _config.get_config(ARM)
    assert cfg.data.quality_required is True
    with pytest.raises(ValueError, match="quality_path is not set"):
        cfg.data.create(tmp_path, cfg.model)


def test_quality_required_raises_even_off_the_registered_arm(tmp_path):
    with pytest.raises(ValueError, match="quality_path is not set"):
        _factory(quality_required=True).create(tmp_path, _model())


def test_quality_without_a_roots_index_raises(tmp_path):
    """The tag array indexes the CONCATENATED pretrain dataset, which only exists when
    roots_index names its parts. Without one the pretrain branch never runs and the artifact is
    silently ignored -- i.e. the arm trains as the baseline under its own name."""
    with pytest.raises(ValueError, match="roots_index"):
        _config.AxisFrankaPretrainDataConfig(
            repo_id="fake", quality_path="quality_v2.npz"
        ).create(tmp_path, _model())


def test_quality_together_with_a_schedule_raises(tmp_path):
    """CFG must fall through to RowSampler. A schedule would change WHICH rows are drawn and
    destroy the coverage-neutrality claim -- two arms at once, and the one property that makes
    CFG comparable to the control silently gone."""
    with pytest.raises(ValueError, match="coverage"):
        _factory(quality_path="q.npz", schedule_path="s.npz").create(tmp_path, _model())


def test_quality_together_with_awr_weights_raises(tmp_path):
    with pytest.raises(ValueError, match="two arms at once"):
        _factory(quality_path="q.npz", awr_weights="w.json").create(tmp_path, _model())


def test_the_control_configs_are_untouched_by_the_new_guards(tmp_path):
    """Every other pretrain config must construct exactly as before: the guards fire only on the
    fields this arm introduces."""
    for name in (ROUND1, "pi05_axis_pretrain_eef"):
        cfg = _config.get_config(name)
        assert cfg.data.quality_required is False
        assert cfg.data.quality_path is None
        dc = cfg.data.create(tmp_path, cfg.model)
        assert dc.pretrain_quality_path is None


# --- coverage neutrality, the arm's distinguishing property --------------------------------------


def test_the_arm_draws_the_controls_rows(tmp_path):
    """CFG is the only round-2 arm that is coverage-neutral: `schedule_path` and `awr_weights`
    are both unset, so `create_torch_data_loader` falls through to `RowSampler(rows, seed)` --
    the control's own sampler, at the same seed, giving a byte-identical row sequence and 100%
    coverage. Catches a schedule or a weights artifact quietly attached to this name, which would
    make it two arms at once and would leave the coverage claim in the write-up false.

    Asserted on the CREATED DataConfig, not just the factory: the loader reads these two fields
    off the DataConfig, and `getattr(..., None)` cannot tell "absent" from "never forwarded".
    """
    cfg = _config.get_config(ARM)
    assert cfg.data.schedule_path is None
    assert cfg.data.schedule_required is False
    assert cfg.data.expected_mode is None
    assert cfg.data.awr_weights is None
    # roots_index comes from $AXIS_PRETRAIN_ROOTS_INDEX at launch; supply one the way a launch
    # would, alongside the per-run artifact.
    dc = dataclasses.replace(
        cfg.data, roots_index="roots.json", quality_path="quality_v2.npz"
    ).create(tmp_path, cfg.model)
    assert dc.pretrain_schedule_path is None
    assert dc.pretrain_expected_mode is None
    assert dc.pretrain_awr_weights is None


def test_the_arm_never_picks_up_awr_weights_from_the_environment(monkeypatch):
    """Round 1 passed its weights through $AXIS_PRETRAIN_AWR_WEIGHTS. On a box where that is
    still exported, a named arm inheriting it would silently be CFG + AWR -- and coverage-neutral
    no longer.

    Built through the FACTORY, not `get_config`: `_CONFIGS` is constructed at import time, so the
    environment is long read by the time a test could patch it and a `get_config` assertion here
    would pass no matter what the factory does.
    """
    monkeypatch.setenv("AXIS_PRETRAIN_AWR_WEIGHTS", "/somewhere/awr.json")
    arm = _config._axis_pretrain_config(  # noqa: SLF001
        eef=True, paper=True, batch_size=64, num_train_steps=20_605,
        name=ARM, quality_required=True,
    )
    assert arm.data.awr_weights is None
    baseline = _config._axis_pretrain_config(eef=True, paper=True)  # noqa: SLF001
    assert baseline.data.awr_weights == "/somewhere/awr.json"  # unchanged for round 1


# --- parity with round 1 --------------------------------------------------------------------------


def test_the_arm_matches_round_ones_recipe():
    """Parity, read off round 1's own config rather than hardcoded."""
    arm, r1 = _config.get_config(ARM), _config.get_config(ROUND1)
    assert arm.model == r1.model
    assert arm.ema_decay == r1.ema_decay
    assert arm.optimizer == r1.optimizer
    assert arm.lr_schedule == r1.lr_schedule
    assert arm.fsdp_devices == r1.fsdp_devices
    assert arm.num_workers == r1.num_workers
    assert arm.weight_loader.params_path == r1.weight_loader.params_path
    assert arm.data.eef_action is True
    # Round 2's budget: one epoch of the rung-5000 corpus at GLOBAL batch 64. NOT round 1's, which
    # is why these two are literals rather than read off ROUND1.
    assert arm.num_train_steps == 20_605
    assert arm.batch_size == 64


def test_the_arm_is_the_sibling_schedule_arm_with_a_different_mechanism():
    """One comparison that pins EVERY remaining field against a round-2 sibling.

    `pi05_axis_drop` already carries round 1's model, optimiser, lr schedule, EMA, sharding,
    weight loader and round 2's budget. If `pi05_axis_cfg` equals it modulo the four fields that
    ARE the mechanism (name, schedule_required, expected_mode, quality_required), then no
    hyper-parameter drifted in while the arm was added -- which a field-by-field test would only
    catch for the fields somebody remembered to list.
    """
    arm, sibling = _config.get_config(ARM), _config.get_config(SIBLING)
    twin = dataclasses.replace(
        sibling,
        name=ARM,
        data=dataclasses.replace(
            sibling.data,
            schedule_required=False,
            expected_mode=None,
            quality_required=True,
        ),
    )
    assert twin == arm


# --- norm stats ------------------------------------------------------------------------------------


def _norm_stats_dir(name: str) -> pathlib.Path:
    cfg = _config.get_config(name)
    resolved = cfg.data.norm_stats_dir(cfg.assets_dirs)
    assert resolved is not None
    return pathlib.Path(str(resolved)).resolve()


def test_the_arm_reads_round_ones_norm_stats():
    """Catches: `assets_base_dir / "pi05_axis_cfg"`, which does not exist -- `_load_norm_stats`
    swallows the miss and `transform_dataset` then dies telling you to run compute_norm_stats,
    which CANNOT run because it has no way to pass --data.quality_path and trips
    `quality_required` first. The same trap the schedule arms hit.

    Recomputing per-arm stats is not the alternative: it would make this arm differ from the
    round-1 control in TWO things rather than one.
    """
    assert _config.get_config(ARM).data.norm_stats_from == ROUND1
    assert _norm_stats_dir(ARM) == _norm_stats_dir(ROUND1)


def test_the_arm_does_not_resolve_to_its_own_assets_dir():
    """Guards the assertion above against the degenerate way it could pass: if `assets_dirs` ever
    stopped being keyed on `name`, both configs would agree while each pointed somewhere
    unintended."""
    cfg = _config.get_config(ARM)
    assert ARM not in str(_norm_stats_dir(ARM))
    assert pathlib.Path(cfg.assets_dirs).name == ARM  # ...and the pitfall is still real


def test_the_arm_follows_round_one_under_a_non_default_assets_base_dir(tmp_path):
    """A NAME, not a path: an --assets-base-dir override must move the arm and round 1 TOGETHER.

    Expressed as the literal `./assets/pi05_axis_pretrain_eef_paper` this agrees with round 1
    only while the default holds -- pass `--assets-base-dir` and round 1's stats move while the
    arm stays pinned to the old directory, silently training against stats belonging to a
    different dataset. The default-path assertion above cannot see that, since both sides move
    together there by construction.
    """
    base = str(tmp_path / "elsewhere")
    arm = dataclasses.replace(_config.get_config(ARM), assets_base_dir=base)
    round1 = dataclasses.replace(_config.get_config(ROUND1), assets_base_dir=base)
    resolved = arm.data.norm_stats_dir(arm.assets_dirs)
    assert resolved is not None
    assert pathlib.Path(str(resolved)) == pathlib.Path(
        str(round1.data.norm_stats_dir(round1.assets_dirs))
    )
    # ...and it actually followed the override rather than staying on ./assets
    assert str(resolved).startswith(base)


def test_round_one_keeps_its_own_norm_stats():
    """The override is for the round-2 arms only; round 1 computed its own and must keep them."""
    cfg = _config.get_config(ROUND1)
    assert cfg.data.assets.assets_dir is None
    assert cfg.data.norm_stats_from is None
    assert pathlib.Path(cfg.assets_dirs).name == ROUND1


def test_the_arm_actually_loads_round_ones_norm_stats(monkeypatch):
    """Not just the same string: the arm must come out of `create()` with stats in hand.

    `assets_dir` is repo-relative (as is upstream's `assets_base_dir` default), so this runs from
    the repo root the way a launch does.
    """
    repo_root = pathlib.Path(_config.__file__).parents[3]
    stats = repo_root / "assets" / ROUND1 / "Devon018" / "Franka-Datasets-v2" / "norm_stats.json"
    if not stats.exists():
        pytest.skip(f"round 1 norm stats not present at {stats}")
    monkeypatch.chdir(repo_root)
    cfg = _config.get_config(ARM)
    # quality_path is per-run (conf/experiments), so supply one the way a launch would.
    data = dataclasses.replace(
        cfg.data, roots_index="roots.json", quality_path="quality_v2.npz"
    ).create(cfg.assets_dirs, cfg.model)
    assert data.norm_stats is not None
