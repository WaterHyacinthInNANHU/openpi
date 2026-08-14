"""Serve-time quality conditioning: two branches that differ ONLY in the tag, and no env var.

Guidance compares two velocities that are supposed to share everything but the prompt. If the
branches were built separately, a later edit to one would silently turn guidance into a comparison
of two unrelated inputs -- so both come from one `chain_kwargs` and one `return` inside
`_input_chain`, and this file asserts the two chains differ in exactly one transform's `drop_all`
and, on a real observation through the real tokenizer, in exactly the prompt tokens.

The other thing under test is the HORIZON. The checkpoint being served is a stage-2 one
(`pi05_libero_axisinit_paper_cfg`, action_horizon 10). The repo's other serve config,
`pi05_axis_eef_libero_serve`, is action_horizon 16 and belongs to a different stage-1 arm; cloning
it would emit chunks of the wrong length with nothing raising anywhere, which is the failure mode
that produces plausible-looking but meaningless rollout numbers.
"""

from __future__ import annotations

import dataclasses
import pathlib

import jax
import numpy as np
import pytest

from openpi.policies import libero_policy as _libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import slb_cfg as _slb_cfg
from openpi.training.quality_conditioning import FixedQualityConditioning
import openpi.transforms as transforms

SERVE = "pi05_axis_cfg_libero_serve"
STAGE2 = "pi05_libero_axisinit_paper_cfg"
PARENT = "pi05_libero"
# The OTHER serve config. Named so the horizon test states what it is guarding against.
WRONG_SERVE = "pi05_axis_eef_libero_serve"

# A path no checkpoint lives at, used to intercept exactly one `maybe_download` call.
_FAKE_CKPT = "/nonexistent-checkpoint"

# The owner-fixed sweep, as (beta, guidance_scale). The mapping is checked arithmetically in
# tests/axis/test_cfg_beta_sweep.py against the spec; here it pins what the SERVER publishes.
BETA_CELLS = ((1.0, 0.0), (1.3, 0.3), (1.7, 0.7), (2.2, 1.2))


def _data_config(name: str = SERVE):
    cfg = _config.get_config(name)
    return cfg.data.create(cfg.assets_dirs, cfg.model)


def _chain(**kwargs):
    return _policy_config._input_chain(  # noqa: SLF001
        repack_inputs=[],
        default_prompt=None,
        data_config=_data_config(),
        norm_stats=None,
        cfg_label=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------------------
# The registered serve config
# ---------------------------------------------------------------------------------------


def test_the_serve_config_matches_the_stage_two_checkpoint_it_serves():
    """Catches: cloning the horizon-16 `pi05_axis_eef_libero_serve` for a horizon-10 checkpoint.
    The action chunk would be the wrong length and every rollout silently wrong."""
    serve, stage2 = _config.get_config(SERVE), _config.get_config(STAGE2)
    assert serve.model.action_horizon == stage2.model.action_horizon == 10
    assert serve.model == stage2.model
    # -> asset_id, i.e. the key the checkpoint's own norm stats were saved under.
    assert serve.data.repo_id == stage2.data.repo_id


def test_the_serve_config_is_not_the_other_serve_config():
    """The control for the test above: the horizon it must NOT have is a real, registered 16."""
    assert _config.get_config(WRONG_SERVE).model.action_horizon == 16
    assert _config.get_config(SERVE).model.action_horizon != 16


def test_the_serve_config_mirrors_pi05_libero():
    """`pi05_libero` is what the existing eval spec serves these checkpoints through, so the
    serve config must be that config's model and data, not a near-miss of it."""
    serve, parent = _config.get_config(SERVE), _config.get_config(PARENT)
    assert serve.model == parent.model
    assert serve.data.repo_id == parent.data.repo_id
    assert serve.data.extra_delta_transform == parent.data.extra_delta_transform
    assert serve.data.assets == parent.data.assets
    assert serve.data.base_config == parent.data.base_config


def test_the_serve_config_carries_no_train_time_quality_tag():
    """`LeRobotLiberoDataConfig.quality_tag` wires the TRAIN-time transform into
    `repack_transforms`, which inference never runs -- so setting it here would be inert while
    reading, in the registry, exactly like the thing that makes this config a CFG one. The tag
    comes from `_input_chain`, which also builds the paired unconditional branch."""
    assert _config.get_config(SERVE).data.quality_tag is None
    # The stage-2 twin is where that field DOES belong; without this the assertion above could
    # pass because the field was removed from the class.
    assert _config.get_config(STAGE2).data.quality_tag == 5


# ---------------------------------------------------------------------------------------
# The cond/uncond pair
# ---------------------------------------------------------------------------------------


def test_the_conditional_branch_carries_the_tag_and_the_unconditional_one_drops_it():
    cond = _chain(quality_tag=5)
    uncond = _chain(quality_tag=5, quality_drop_all=True)
    for chain, drop in ((cond, False), (uncond, True)):
        tags = [t for t in chain if isinstance(t, FixedQualityConditioning)]
        assert len(tags) == 1
        assert (tags[0].q_ep, tags[0].drop_all) == (5, drop)


def test_the_two_branches_are_the_same_chain_one_field_apart():
    """Catches any drift between the branches beyond the tag -- which would make the guidance
    difference a comparison of two unrelated inputs rather than of one label."""
    cond = _chain(quality_tag=5)
    uncond = _chain(quality_tag=5, quality_drop_all=True)
    plain = _chain()
    assert [type(t) for t in cond] == [type(t) for t in uncond]
    assert len(cond) == len(plain) + 1
    for chain in (cond, uncond):
        stripped = [type(t) for t in chain if not isinstance(t, FixedQualityConditioning)]
        assert stripped == [type(t) for t in plain]


def test_the_branches_differ_only_in_the_tokenized_prompt():
    """The whole point of guidance, asserted on a REAL observation through the REAL tokenizer.

    A difference that leaked into `state` or the images would make (v_cond - v_uncond) a
    comparison of two unrelated observations rather than of one label.
    """
    obs = _libero_policy.make_libero_example()
    cond = transforms.compose(_chain(quality_tag=5))(dict(obs))
    uncond = transforms.compose(_chain(quality_tag=5, quality_drop_all=True))(dict(obs))

    assert set(cond) == set(uncond)
    differing = set()
    for key in cond:
        a, b = cond[key], uncond[key]
        if isinstance(a, dict):
            assert set(a) == set(b)
            if any(not np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a):
                differing.add(key)
        elif not np.array_equal(np.asarray(a), np.asarray(b)):
            differing.add(key)

    # CONTROL: the tokens must actually differ, or "differs only in X" is vacuous.
    assert differing == {"tokenized_prompt", "tokenized_prompt_mask"}, differing
    assert int(np.sum(cond["tokenized_prompt_mask"])) > int(
        np.sum(uncond["tokenized_prompt_mask"])
    )
    # ...and the unconditional branch is the BARE prompt, i.e. what an untagged serve produces.
    plain = transforms.compose(_chain())(dict(obs))
    assert np.array_equal(uncond["tokenized_prompt"], plain["tokenized_prompt"])


def test_the_placebo_cell_changes_the_tag_value_and_nothing_else():
    """The `Quality: 3` cell is the cheapest way to tell 'the tag changes behaviour' from 'the
    tag carries the right information'. It must differ from the q=5 cell in the VALUE only."""
    five = [t for t in _chain(quality_tag=5) if isinstance(t, FixedQualityConditioning)][0]
    three = [t for t in _chain(quality_tag=3) if isinstance(t, FixedQualityConditioning)][0]
    assert (five.q_ep, three.q_ep) == (5, 3)
    assert five.drop_all == three.drop_all
    assert dataclasses.replace(five, q_ep=3) == three
    # And the tag actually reaches the prompt with the value asked for.
    assert three({"prompt": "p"})["prompt"] == "p\nQuality: 3"


# ---------------------------------------------------------------------------------------
# No env-var mode switch
# ---------------------------------------------------------------------------------------


def test_the_tag_route_ignores_the_slb_cfg_metadata_env_var(monkeypatch):
    """Round 2 forbids env-var mode switches, and this one ALSO appends a `Mistake` token neither
    stage of this arm trained. Catches the quality route falling through to
    `_slb_cfg.metadata_enabled()`."""
    monkeypatch.setenv("SLB_CFG_METADATA", "1")
    assert _slb_cfg.metadata_enabled()  # control: the variable is really on
    chain = _chain(quality_tag=5)
    tags = [t for t in chain if isinstance(t, FixedQualityConditioning)]
    assert len(tags) == 1
    assert not [
        t
        for t in chain
        if isinstance(
            t, (_slb_cfg.FixedCfgMetadataConditioning, _slb_cfg.FixedCfgConditioning)
        )
    ]
    tagged = tags[0]({"prompt": "p"})["prompt"]
    assert tagged == "p\nQuality: 5"
    assert _slb_cfg.MISTAKE_KEY not in tagged


# ---------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------


def test_quality_tag_and_cfg_label_together_raise():
    """Two conditioning schemes at once would make the served arm unrecoverable from the config."""
    with pytest.raises(ValueError, match="two conditioning schemes"):
        _policy_config._input_chain(  # noqa: SLF001
            repack_inputs=[],
            default_prompt=None,
            data_config=_data_config(),
            norm_stats=None,
            cfg_label=0,
            quality_tag=5,
        )


def test_an_unconditional_branch_without_a_tag_raises():
    """`drop_all` with no tag is the plain chain, so both branches would be identical and
    (v_cond - v_uncond) == 0 -- a sweep that reproduces the beta = 1 cell at every scale."""
    with pytest.raises(ValueError, match="quality_drop_all"):
        _chain(quality_drop_all=True)


@pytest.mark.parametrize("bad", [0, 6, -1, 255])
def test_a_tag_outside_the_bins_is_refused_at_construction(bad):
    """`--quality-tag 0` reads like 'off' and would serve `Quality: 0`, a sixth condition nothing
    trained. Refused when the server starts, not after the client has connected."""
    with pytest.raises(ValueError, match="not a bin"):
        FixedQualityConditioning(q_ep=bad)


def test_guidance_without_a_tag_raises():
    """w != 0 with identical branches makes (v_cond - v_uncond) == 0 and guidance vanish at every
    scale -- a beta sweep that silently reproduces the beta = 1 cell four times."""
    with pytest.raises(ValueError, match="requires"):
        _policy_config.create_trained_policy(
            _config.get_config(SERVE), "/nonexistent", guidance_scale=0.3, quality_tag=None
        )


def test_a_tag_and_the_slb_flag_together_raise():
    with pytest.raises(ValueError, match="two conditioning schemes"):
        _policy_config.create_trained_policy(
            _config.get_config(SERVE),
            "/nonexistent",
            cfg_conditioning=True,
            quality_tag=5,
        )


# ---------------------------------------------------------------------------------------
# What the SERVER publishes: beta = 1 + guidance_scale
# ---------------------------------------------------------------------------------------


def _served(monkeypatch, **kwargs):
    """`create_trained_policy` on the real serve config, with only the WEIGHTS stubbed.

    Everything under test -- the branch that builds the uncond chain, the sample kwargs and the
    published metadata -- is the real function body. Only `restore_params`/`model.load` are
    replaced, because a checkpoint is the one thing a test cannot have.
    """
    cfg = _config.get_config(SERVE)
    dummy_model = dataclasses.replace(
        cfg.model, paligemma_variant="dummy", action_expert_variant="dummy"
    ).create(jax.random.key(0))
    # Narrow, not blanket: `download.maybe_download` is the SAME module object the PaliGemma
    # tokenizer fetches its sentencepiece model through, and the model transforms in this very
    # chain build one. Only the checkpoint path is intercepted.
    real_download = _policy_config.download.maybe_download
    monkeypatch.setattr(
        _policy_config.download,
        "maybe_download",
        lambda p, **kw: pathlib.Path(p) if str(p).startswith(_FAKE_CKPT) else real_download(p, **kw),
    )
    monkeypatch.setattr(_policy_config._model, "restore_params", lambda *a, **k: {})
    # The stage-2 norm stats live in the checkpoint, which this test does not have. None
    # makes `Normalize` a no-op; nothing here inspects normalised values.
    monkeypatch.setattr(_policy_config._checkpoints, "load_norm_stats", lambda *a, **k: None)
    monkeypatch.setattr(type(cfg.model), "load", lambda self, params: dummy_model)
    return _policy_config.create_trained_policy(
        cfg, _FAKE_CKPT, norm_stats=None, **kwargs
    )


@pytest.mark.parametrize(("beta", "w"), BETA_CELLS)
def test_the_server_publishes_beta_as_one_plus_the_guidance_scale(monkeypatch, beta, w):
    """THE off-by-one, at the serve path. openpi computes v = v_uncond + (1 + w)(v_cond -
    v_uncond), so pi0.7's beta -- the TOTAL conditional weight -- is w + 1. Publishing both is
    what lets the rollout driver refuse to label a beta = 1.3 run as beta = 1.0."""
    policy = _served(monkeypatch, quality_tag=5, guidance_scale=w)
    meta = policy.metadata
    assert meta["guidance_scale"] == pytest.approx(w)
    assert meta["beta"] == pytest.approx(beta)
    assert meta["beta"] == pytest.approx(1.0 + meta["guidance_scale"])
    assert meta["quality_tag"] == 5
    assert meta["cfg_conditioning"] is True


def test_the_placebo_cell_is_published_as_the_placebo(monkeypatch):
    meta = _served(monkeypatch, quality_tag=3, guidance_scale=0.0).metadata
    assert (meta["quality_tag"], meta["beta"]) == (3, 1.0)


def test_the_served_policy_carries_the_uncond_branch(monkeypatch):
    """Without it `Policy.infer` never builds `uncond_observation` and guidance is a no-op at
    every scale, with the metadata still claiming a beta."""
    policy = _served(monkeypatch, quality_tag=5, guidance_scale=0.7)
    assert policy._uncond_input_transform is not None  # noqa: SLF001
    assert policy._sample_kwargs["guidance_scale"] == pytest.approx(0.7)  # noqa: SLF001

    # ...and it is the UNCONDITIONAL branch, not a second copy of the conditional one. Wiring it
    # without `drop_all` would tag both branches, making (v_cond - v_uncond) == 0 and guidance a
    # no-op at every β -- with the metadata still reporting one.
    obs = _libero_policy.make_libero_example()
    cond = policy._input_transform(dict(obs))  # noqa: SLF001
    uncond = policy._uncond_input_transform(dict(obs))  # noqa: SLF001
    assert not np.array_equal(cond["tokenized_prompt"], uncond["tokenized_prompt"])
    assert int(np.sum(cond["tokenized_prompt_mask"])) > int(
        np.sum(uncond["tokenized_prompt_mask"])
    )
    assert np.array_equal(cond["state"], uncond["state"])


def test_the_beta_one_cell_does_not_pay_for_guidance(monkeypatch):
    """beta = 1.0 short-circuits to plain conditional sampling with the tag still applied. The
    kwarg must be ABSENT, not 0.0-but-present: `Policy.infer` gates the second input chain (and
    `Pi0.sample_actions` the batch doubling) on it, so a stray 1e-9 would pay ~2x policy compute
    to compute the same answer."""
    policy = _served(monkeypatch, quality_tag=5, guidance_scale=0.0)
    assert "guidance_scale" not in policy._sample_kwargs  # noqa: SLF001
    assert policy.metadata["beta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------
# The flag, as the sweep spec actually spells it
# ---------------------------------------------------------------------------------------


def _serve_args(argv: list[str]):
    """`tyro.cli` over the real `scripts/serve_policy.py` Args.

    Loaded by path because `scripts/` is not an importable package, and registered in
    `sys.modules` because tyro reads the dataclass's source for its helptext.
    """
    import importlib.util
    import sys

    import tyro

    path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("_serve_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return tyro.cli(module.Args, args=argv)
    finally:
        sys.modules.pop(spec.name, None)


def test_the_sweep_spelling_parses():
    """The exact argv `conf/experiments/onelayer_v3_round2_cfg_eval.toml` renders. A flag that
    does not parse dies minutes into an allocated job, reported as 'the server never became
    ready' -- this repo has already lost a submission to argument ordering."""
    args = _serve_args(
        [
            "--quality-tag=5",
            "--guidance-scale=0.3",
            "--env",
            "LIBERO",
            "policy:checkpoint",
            f"--policy.config={SERVE}",
            "--policy.dir=/ckpt",
        ]
    )
    assert (args.quality_tag, args.guidance_scale) == (5, 0.3)
    assert args.policy.config == SERVE


def test_the_tag_is_absent_by_default():
    """Every other serve invocation in the repo omits it and must keep its current behaviour."""
    args = _serve_args(["--env", "LIBERO", "policy:checkpoint", "--policy.config=pi05_libero",
                        "--policy.dir=/ckpt"])
    assert args.quality_tag is None


def test_a_top_level_flag_after_the_selector_is_rejected():
    """The control for the ordering test in tests/axis/test_cfg_beta_sweep.py: putting these in
    `server_overrides` really does produce an unrunnable command, it does not merely look wrong."""
    with pytest.raises(SystemExit):
        _serve_args(
            [
                "--env",
                "LIBERO",
                "policy:checkpoint",
                f"--policy.config={SERVE}",
                "--policy.dir=/ckpt",
                "--quality-tag=5",
            ]
        )
