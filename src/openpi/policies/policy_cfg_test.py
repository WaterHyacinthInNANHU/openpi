"""Classifier-free guidance plumbing between `create_trained_policy` and `Policy.infer`.

pi0_cfg_test.py already pins the guidance ARITHMETIC inside `Pi0.sample_actions`. What was
untested -- and what made a trained `cfg` checkpoint evaluate identically to `vanilla` --
is everything upstream of it: whether an unconditional branch is built at all, whether it
differs from the conditional branch by exactly the advantage tag and nothing else, and
whether `guidance_scale` survives the trip through `Policy`'s jit.

WHY THERE IS NO NUMERIC TEST ON AN UNSTUBBED MODEL
    A freshly created `dummy` Pi0 is an identity map: `embed_prefix` returns bit-identical
    tokens for different images and prompts, so cond and uncond branches coincide and EVERY
    guidance scale looks like a no-op. A numeric assertion there would pass vacuously. We
    therefore reuse pi0_cfg_test's stub -- `action_out_proj` made to return 1 for the cond
    half and 0 for the uncond half -- so the two branches differ by construction, and every
    "guidance changed something" claim is checked against a control that is provably
    non-trivial.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.policies.axis_franka_policy as axis_franka_policy
import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import slb_cfg
import openpi.transforms as transforms

# The REAL pi0.5 default (Pi0Config.__post_init__: 200 if pi05). Not shrunk for speed: the
# advantage tag is appended to the prompt, and pi0.5 serialises the discretised state into
# the same string ("Task: <prompt>, State: ...;\nAction: "), so a short max_token_len
# truncates the prompt and can drop the tag -- silently turning cfg back into vanilla.
_AH, _AD, _MAXLEN, _STEPS = 4, 8, 200, 4

_PROMPT = "pick up the red block"


# ---------------------------------------------------------------------------------------
# Transform-chain construction: the cond/uncond pair
# ---------------------------------------------------------------------------------------


def _pi05_data_config() -> _config.DataConfig:
    """The AXIS-Franka pi0.5 inference config, minus anything that needs a checkpoint."""
    model_config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=_AH,
        action_dim=32,
        max_token_len=_MAXLEN,
        pi05=True,
    )
    return dataclasses.replace(
        _config.DataConfig(),
        data_transforms=transforms.Group(
            inputs=[axis_franka_policy.AxisFrankaInputs()],
            outputs=[axis_franka_policy.AxisFrankaOutputs()],
        ),
        model_transforms=_config.ModelTransformFactory()(model_config),
    )


def _chain(cfg_label: int | None) -> list:
    return _policy_config._input_chain(  # noqa: SLF001
        repack_inputs=[],
        default_prompt=None,
        data_config=_pi05_data_config(),
        norm_stats=None,
        cfg_label=cfg_label,
    )


def _raw_obs() -> dict:
    """Exactly the payload rollout_pi05._build_obs sends over the websocket."""
    rng = np.random.default_rng(0)
    return {
        "state": rng.normal(size=(8,)).astype(np.float32),
        "base_0_rgb": rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb": rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8),
        "prompt": _PROMPT,
    }


def test_uncond_chain_is_the_cond_chain_with_a_different_tag():
    """Structural: the two branches are the same list of transforms, one label apart."""
    cond, uncond, plain = _chain(0), _chain(slb_cfg.NULL_LABEL), _chain(None)

    assert [type(t) for t in cond] == [type(t) for t in uncond]
    # Exactly one extra transform relative to the un-guided chain, and it is the tag.
    assert len(cond) == len(plain) + 1
    tags = [t for t in cond if isinstance(t, slb_cfg.FixedCfgConditioning)]
    assert len(tags) == 1 and tags[0].label == _policy_config.CFG_POSITIVE_LABEL == 0
    uncond_tags = [t for t in uncond if isinstance(t, slb_cfg.FixedCfgConditioning)]
    assert len(uncond_tags) == 1 and uncond_tags[0].label == slb_cfg.NULL_LABEL
    # Dropping the tag from either branch leaves the un-guided chain, i.e. nothing else moved.
    for chain in (cond, uncond):
        stripped = [type(t) for t in chain if not isinstance(t, slb_cfg.FixedCfgConditioning)]
        assert stripped == [type(t) for t in plain]


def test_cond_and_uncond_observations_differ_only_in_the_tokenized_prompt():
    """The whole point of guidance: two inputs identical except for the prompt tokens.

    Run through the REAL pi0.5 model transforms (real PaliGemma tokenizer), because a
    difference that leaked into `state` or the images would make (v_cond - v_uncond) a
    comparison of two unrelated observations rather than of two conditionings.
    """
    obs = _raw_obs()
    cond = transforms.compose(_chain(0))(dict(obs))
    uncond = transforms.compose(_chain(slb_cfg.NULL_LABEL))(dict(obs))

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

    # CONTROL: the tokens must actually differ, otherwise "differs only in X" is vacuous.
    assert differing == {"tokenized_prompt", "tokenized_prompt_mask"}, differing
    assert int(np.sum(cond["tokenized_prompt_mask"])) > int(np.sum(uncond["tokenized_prompt_mask"]))


def test_cfg_conditioning_appends_the_advantage_tag_before_tokenization():
    """The tag reaches the tokenizer, not just the dict: more prompt tokens are unmasked."""
    obs = _raw_obs()
    tagged = transforms.compose(_chain(0))(dict(obs))
    plain = transforms.compose(_chain(None))(dict(obs))
    assert not np.array_equal(tagged["tokenized_prompt"], plain["tokenized_prompt"])
    # `NULL_LABEL` is the bare prompt, so the un-guided chain and the uncond branch agree.
    uncond = transforms.compose(_chain(slb_cfg.NULL_LABEL))(dict(obs))
    assert np.array_equal(uncond["tokenized_prompt"], plain["tokenized_prompt"])


# ---------------------------------------------------------------------------------------
# Policy.infer: does the guidance actually reach sample_actions?
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ReplacePromptTokens(transforms.DataTransformFn):
    """Stand-in for the uncond chain in the model-level tests: changes ONLY the tokens."""

    value: int = 7

    @override
    def __call__(self, data: dict) -> dict:
        return {**data, "tokenized_prompt": np.full_like(np.asarray(data["tokenized_prompt"]), self.value)}


def _stub_model_and_obs():
    """A `dummy` Pi0 whose two guidance branches differ by construction.

    Without the stub the transformer is an identity map and guidance is unmeasurable; see
    the module docstring. `action_out_proj` is a plain callable attribute, which `nnx.split`
    keeps in the graphdef (static), so it survives `module_jit`.
    """
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=_AH,
        action_dim=_AD,
        max_token_len=16,
        pi05=True,
    )
    model = cfg.create(jax.random.key(0))

    def fake_proj(x):
        b = x.shape[0]
        if b == 2:  # batch-doubled: cond half -> v=1, uncond half -> v=0
            return jnp.concatenate([jnp.ones((1, _AH, _AD)), jnp.zeros((1, _AH, _AD))], axis=0)
        return jnp.ones((b, _AH, _AD))

    model.action_out_proj = fake_proj

    obs = cfg.fake_obs(batch_size=1)
    flat = {k: v for k, v in obs.to_dict().items() if v is not None}
    unbatched = jax.tree.map(lambda x: np.asarray(x)[0], flat)
    return model, unbatched


def _make_policy(model, *, uncond: bool, guidance_scale: float | None):
    sample_kwargs = {"num_steps": _STEPS, "noise": jnp.zeros((1, _AH, _AD))}
    if guidance_scale is not None:
        sample_kwargs["guidance_scale"] = guidance_scale
    return _policy.Policy(
        model,
        transforms=[],
        output_transforms=[],
        uncond_transforms=[_ReplacePromptTokens()] if uncond else None,
        sample_kwargs=sample_kwargs,
    )


@pytest.fixture(scope="module")
def _stubbed():
    return _stub_model_and_obs()


def test_guidance_scale_zero_is_byte_identical_to_no_uncond_transforms(_stubbed):
    """w=0 must reproduce non-guided sampling EXACTLY, not merely closely."""
    model, obs = _stubbed
    baseline = _make_policy(model, uncond=False, guidance_scale=None).infer(dict(obs))["actions"]
    for kwargs in ({"uncond": True, "guidance_scale": 0.0}, {"uncond": True, "guidance_scale": None}):
        out = _make_policy(model, **kwargs).infer(dict(obs))["actions"]
        assert np.array_equal(np.asarray(baseline), np.asarray(out)), kwargs


@pytest.mark.parametrize("guidance_scale", [0.5, 1.0, 2.0])
def test_nonzero_guidance_changes_the_actions(_stubbed, guidance_scale):
    """Closed form: v_uncond=0, v_cond=1 => v=1+w and x_0 integrates to -(1+w) from x=0.

    Checking the VALUE, not just inequality, pins that the scale reaching the model is the
    one asked for -- a plumbing bug that silently rescaled w would still "change the output".
    """
    model, obs = _stubbed
    out = _make_policy(model, uncond=True, guidance_scale=guidance_scale).infer(dict(obs))["actions"]
    baseline = _make_policy(model, uncond=False, guidance_scale=None).infer(dict(obs))["actions"]
    assert not np.allclose(np.asarray(out), np.asarray(baseline))
    assert float(np.asarray(out)[0, 0]) == pytest.approx(-(1.0 + guidance_scale), abs=1e-5)
    assert float(np.asarray(baseline)[0, 0]) == pytest.approx(-1.0, abs=1e-5)


def test_uncond_branch_is_skipped_when_guidance_is_off(_stubbed):
    """A configured-but-unused uncond chain must not be run (it costs a full transform pass)."""
    model, obs = _stubbed

    calls = []

    @dataclasses.dataclass(frozen=True)
    class _Counting(transforms.DataTransformFn):
        @override
        def __call__(self, data: dict) -> dict:
            calls.append(1)
            return data

    policy = _policy.Policy(
        model,
        transforms=[],
        output_transforms=[],
        uncond_transforms=[_Counting(), _ReplacePromptTokens()],
        sample_kwargs={"num_steps": _STEPS, "noise": jnp.zeros((1, _AH, _AD)), "guidance_scale": 0.0},
    )
    policy.infer(dict(obs))
    assert calls == []


def test_guidance_scale_survives_the_policy_jit(_stubbed):
    """Regression: a Python float kwarg is TRACED by jax.jit, and Pi0.sample_actions does
    `bool(guidance_scale != 0.0)`. Without static_argnames this raises
    TracerBoolConversionError -- i.e. CFG would have crashed the first time it was served."""
    model, obs = _stubbed
    policy = _make_policy(model, uncond=True, guidance_scale=2.0)
    policy.infer(dict(obs))  # must not raise


# ---------------------------------------------------------------------------------------
# create_trained_policy guard rails
# ---------------------------------------------------------------------------------------


def test_guidance_without_conditioning_is_rejected():
    """Guiding an untagged policy is a silent no-op; refuse it instead of shipping vanilla."""
    with pytest.raises(ValueError, match="cfg_conditioning"):
        _policy_config.create_trained_policy(
            _config.TrainConfig(name="unused", model=_pi0_config.Pi0Config()),
            "/nonexistent",
            guidance_scale=2.0,
        )


def test_observation_roundtrip_keys_are_unchanged_by_the_tag():
    """The tagged chain still produces a valid model Observation (no key added or dropped)."""
    obs = _raw_obs()
    for label in (None, 0, slb_cfg.NULL_LABEL):
        out = transforms.compose(_chain(label))(dict(obs))
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], out)
        observation = _model.Observation.from_dict(batched)
        assert observation.tokenized_prompt is not None
        assert observation.state.shape == (1, 32)
