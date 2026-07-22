"""Knowledge-insulation gradient test: the action loss must not reach the VLM.

Ported from TensorAuto/OpenTau's test_pi05_knowledge_insulation_controls_vlm_gradient
(the only KI implementation found with a behavioural gradient test).

TWO THINGS THIS TEST HAS TO WORK AROUND, both of which silently make it pass vacuously:

1. A freshly created Pi0 has degenerate attention/MLP projections -- measured, 43 of 53
   param leaves get exactly zero gradient, including every q/kv/attn_vec einsum and both
   experts' mlp. Each layer collapses to its residual, the transformer is an identity map,
   and the prefix cannot influence the action output at all. `create()` builds an
   architecture meant to have pretrained weights loaded into it. So we randomize the
   parameters; without this the "insulated" case reads 0.0 for the trivial reason that the
   un-insulated case is 0.0 too.
2. The `dummy` variant's embed_prefix is not a usable input path either, so the prefix is
   stubbed with fixed random tokens. Consequence: this test covers the LLM side of the cut.
   The vision tower sits upstream of the same severed path but is not exercised here.

The control assertion (`joint` MUST leak) is what keeps the above from passing silently.
"""

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0_config as _pi0_config
from openpi.models.pi0 import make_attn_mask

_B, _AH, _AD = 2, 4, 8


def _randomized_model():
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy",
        action_horizon=_AH, action_dim=_AD, max_token_len=16, pi05=True,
        knowledge_insulation=True,
    )
    model = cfg.create(jax.random.key(0))
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    new = [
        jax.random.normal(keys[i], x.shape, dtype=jnp.float32).astype(x.dtype) * 0.05
        if hasattr(x, "shape") and jnp.issubdtype(x.dtype, jnp.floating) else x
        for i, x in enumerate(leaves)
    ]
    nnx.update(model, jax.tree_util.tree_unflatten(treedef, new))
    return model, cfg


def _bucket(path: str) -> str:
    # openpi packs BOTH experts into PaliGemma.llm; the action expert's params carry the
    # "_1" suffix (pi0_test.py: the frozen VLM params are the llm paths WITHOUT "_1").
    if "_1" in path:
        return "expert"
    if "img" in path:
        return "vlm_img"
    if "llm" in path:
        return "vlm_llm"
    return "heads"


def _grads(model, obs, x_t, t, prefix, mode):
    def loss(mod):
        pt, pm, pam = prefix
        st, sm, sam, ac = mod.embed_suffix(obs, x_t, t)
        if mode == "joint":
            im = jnp.concatenate([pm, sm], axis=1)
            am = jnp.concatenate([pam, sam], axis=0)
            (_, so), _ = mod.PaliGemma.llm(
                [pt, st], mask=make_attn_mask(im, am),
                positions=jnp.cumsum(im, axis=1) - 1, adarms_cond=[None, ac],
            )
        else:
            (_, _), kv = mod.PaliGemma.llm(
                [pt, None], mask=make_attn_mask(pm, pam), positions=jnp.cumsum(pm, axis=1) - 1
            )
            if mode == "stop":
                kv = jax.tree.map(jax.lax.stop_gradient, kv)
            pp = einops.repeat(pm, "b p -> b s p", s=st.shape[1])
            (_, so), _ = mod.PaliGemma.llm(
                [None, st],
                mask=jnp.concatenate([pp, make_attn_mask(sm, sam)], axis=-1),
                positions=jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1,
                kv_cache=kv, adarms_cond=[None, ac],
            )
        return jnp.mean(mod.action_out_proj(so[:, -_AH:]) ** 2)

    out: dict[str, float] = {}
    for path, v in jax.tree_util.tree_flatten_with_path(nnx.grad(loss)(model))[0]:
        if v is not None and hasattr(v, "shape") and v.size:
            k = _bucket(jax.tree_util.keystr(path))
            out[k] = max(out.get(k, 0.0), float(jnp.max(jnp.abs(v))))
    return out


@pytest.fixture(scope="module")
def _setup():
    model, cfg = _randomized_model()
    obs = cfg.fake_obs(batch_size=_B)
    x_t = jax.random.normal(jax.random.key(3), (_B, _AH, _AD))
    t = jnp.full((_B,), 0.5)
    pt, pm, pam = model.embed_prefix(obs)
    prefix = (jax.random.normal(jax.random.key(9), pt.shape), pm, pam)
    return {m: _grads(model, obs, x_t, t, prefix, m) for m in ("joint", "2pass", "stop")}


def test_control_joint_forward_leaks_into_vlm(_setup):
    """Guards every other assertion here: if the un-insulated path does not leak, there is
    no gradient to cut and a passing insulation test would mean nothing."""
    assert _setup["joint"].get("vlm_llm", 0.0) > 1e-9


def test_two_pass_alone_does_not_insulate(_setup):
    """Splitting the forward is NOT sufficient -- the KV cache stays differentiable, so the
    stop_gradient is the actual mechanism rather than a redundant safety net."""
    assert _setup["2pass"].get("vlm_llm", 0.0) > 1e-9


def test_stop_gradient_cuts_the_vlm_path(_setup):
    assert _setup["stop"].get("vlm_llm", 0.0) == 0.0


def test_action_expert_still_trains_under_insulation(_setup):
    """Insulation must remove the VLM gradient WITHOUT weakening the action expert."""
    assert _setup["stop"].get("expert", 0.0) == pytest.approx(
        _setup["joint"].get("expert", 0.0), rel=1e-3
    )
    assert _setup["stop"].get("heads", 0.0) > 1e-9
