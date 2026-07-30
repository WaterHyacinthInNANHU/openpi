"""The KI discrete loss must actually read the observation.

`_fast_token_loss` supervises `prefix_out[:, -ki_num_action_tokens:]`, i.e. the LAST n
positions of the prefix. The prefix is [image tokens ..., max_token_len prompt tokens], and
the prompt is right-padded by `PaligemmaTokenizer`. For pi05 (max_token_len=200, n=64) that
window is prompt positions 136-199, while a real prompt is ~60-90 tokens -- so the readout
lands entirely in padding.

`make_attn_mask` zeroes a padded query's whole attention row
(`valid_mask = input_mask[:,None,:] * input_mask[:,:,None]`). `gemma.py` then fills those
logits with `big_neg` and softmaxes, which for an all-masked row is UNIFORM over every key.
So each padded readout position returns the same unweighted mean-pool of all value vectors:
measured pairwise cosine 1.0000 across the window, versus 0.70-0.86 for real positions.

The consequence is not that the loss is constant -- the mean-pool still moves with the image
(`test_ki_loss_depends_on_the_observation` passes). It is that all n positions are IDENTICAL,
so the discrete head is one global classifier applied n times and its optimum is the
position-independent marginal FAST-token distribution. That is nothing like the paper's
autoregressive `sum_j log p(l_{j+1} | x_{1:j})`, and it is close to the failure
`_fast_token_loss`'s own docstring warns about (KI degrading to backbone freezing).

`pi0_ki_test.py` cannot catch this: it stubs the prefix with a fully-valid mask, so every
readout position is real there.

See reports/pi05_knowledge_insulation_design.md sec. 4 for the fix options.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
import openpi.models.pi0_config as _pi0_config

_B, _AH, _AD = 1, 4, 8
# Wide enough that a short prompt leaves a padded tail, like the real pi05 max_token_len=200.
_MAX_TOKEN_LEN = 48
_N_FAST = 16
_PROMPT_LEN = 10  # real tokens; the remaining positions are padding

# Strict xfail: these encode the CORRECT behaviour, which the current readout does not have.
# Strict so that whoever fixes `_fast_token_loss` is told to drop the marker instead of
# quietly leaving a passing xfail behind. See reports/pi05_knowledge_insulation_design.md.
_KNOWN_DEFECT = pytest.mark.xfail(
    strict=True,
    reason=(
        "KI reads prefix_out[:, -n:], which for a padded prompt is entirely padding. Padded "
        "query rows are all-masked, so softmax over big_neg logits is UNIFORM over every key: "
        "all n readout vectors collapse to the same mean-pooled vector (measured pairwise "
        "cosine 1.0000 vs 0.70-0.86 for real positions). The discrete head can therefore only "
        "predict a position-independent marginal, not the ordered FAST sequence."
    ),
)


def _randomized_model() -> object:
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=_AH,
        action_dim=_AD,
        max_token_len=_MAX_TOKEN_LEN,
        pi05=True,
        knowledge_insulation=True,
        ki_num_action_tokens=_N_FAST,
        ki_fast_vocab_size=64,
    )
    model = cfg.create(jax.random.key(0))
    # A freshly created Pi0 is a near-identity map (see pi0_ki_test.py); randomize so the
    # prefix can actually influence the readout.
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    keys = jax.random.split(jax.random.key(11), len(leaves))
    nnx.update(
        model,
        jax.tree_util.tree_unflatten(
            treedef,
            [
                jax.random.normal(keys[i], x.shape, dtype=jnp.float32).astype(x.dtype) * 0.05
                if hasattr(x, "shape") and jnp.issubdtype(x.dtype, jnp.floating)
                else x
                for i, x in enumerate(leaves)
            ],
        ),
    )
    return model


def _observation(image_seed: int) -> _model.Observation:
    rng = np.random.default_rng(image_seed)
    img = jnp.asarray(rng.uniform(-1, 1, (_B, *_model.IMAGE_RESOLUTION, 3)), dtype=jnp.float32)
    mask = jnp.ones((_B, _MAX_TOKEN_LEN), dtype=bool).at[:, _PROMPT_LEN:].set(False)
    return _model.Observation(
        images={"base_0_rgb": img},
        image_masks={"base_0_rgb": jnp.ones((_B,), dtype=bool)},
        state=jnp.zeros((_B, _AD), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((_B, _MAX_TOKEN_LEN), dtype=jnp.int32),
        tokenized_prompt_mask=mask,
        fast_action_tokens=jnp.asarray(
            np.random.default_rng(7).integers(0, 64, (_B, _N_FAST)), dtype=jnp.int32
        ),
    )


def _fast_loss(model, obs):
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
    return model._fast_token_loss(prefix_out, obs), prefix_out


@_KNOWN_DEFECT
def test_ki_readout_positions_are_not_all_padding():
    """The supervised window must overlap real (unmasked) prompt positions."""
    model = _randomized_model()
    obs = _observation(0)
    _, prefix_out = _fast_loss(model, obs)
    n_prefix = prefix_out.shape[1]
    n_image = n_prefix - _MAX_TOKEN_LEN

    readout_start = n_prefix - _N_FAST
    last_real = n_image + _PROMPT_LEN  # exclusive
    assert readout_start < last_real, (
        f"KI reads prefix positions [{readout_start}, {n_prefix}) but the last real token is at "
        f"{last_real - 1} (image tokens: {n_image}, prompt tokens: {_PROMPT_LEN} of {_MAX_TOKEN_LEN}). "
        "The whole readout window is padding."
    )


def test_ki_loss_depends_on_the_observation():
    """A different image must change the discrete loss.

    This one PASSES even with the defect: an all-masked query attends uniformly over every
    key, so the collapsed readout is a mean-pool of the real activations and does still move
    with the image. Kept as the floor -- if it ever fails, KI is training on nothing at all.
    """
    model = _randomized_model()
    obs_a = _observation(0)
    obs_b = dataclasses.replace(_observation(1), fast_action_tokens=obs_a.fast_action_tokens)

    loss_a, _ = _fast_loss(model, obs_a)
    loss_b, _ = _fast_loss(model, obs_b)

    assert not jnp.allclose(loss_a, loss_b, atol=1e-6), (
        f"KI cross-entropy is identical ({float(loss_a[0])}) for two different images: the "
        "readout carries no information about the observation."
    )


@_KNOWN_DEFECT
def test_ki_readout_positions_are_distinguishable():
    """The n supervised positions must differ from each other.

    THE decisive property. The head has to emit an ORDERED sequence of n FAST ids, so the n
    positions it reads must carry different information. If they are identical the head is a
    single global classifier applied n times, and the best it can do is the marginal token
    distribution -- no ordering, no chunk structure, essentially no signal for the VLM.
    """
    model = _randomized_model()
    _, prefix_out = _fast_loss(model, _observation(0))

    window = np.asarray(prefix_out[0, -_N_FAST:].astype(jnp.float32))
    unit = window / np.linalg.norm(window, axis=-1, keepdims=True)
    cos = unit @ unit.T
    off_diagonal = cos[~np.eye(len(cos), dtype=bool)]

    assert off_diagonal.mean() < 0.99, (
        f"the {_N_FAST} KI readout positions are near-identical (mean pairwise cosine "
        f"{off_diagonal.mean():.4f}); the discrete head cannot represent an ordered sequence."
    )
