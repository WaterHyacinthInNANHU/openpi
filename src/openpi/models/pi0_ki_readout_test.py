"""Knowledge insulation reads real, teacher-forced FAST tokens -- not prompt padding.

The original KI implementation supervised `prefix_out[:, -n:]` while never putting the FAST ids
into the token stream at all, so the readout window was prompt padding. Padded query rows are
all-masked, `gemma` fills them with `big_neg`, and softmax over an all-`big_neg` row is UNIFORM
over every key -- so all n positions collapsed to one mean-pooled vector (measured pairwise
cosine 1.0000 vs 0.70-0.86 for real positions). The head was a single global classifier applied
n times, whose optimum is the position-independent marginal.

Option C fixes this the way every surveyed implementation does: the FAST ids are spliced into
`tokenized_prompt` as real PaliGemma vocab ids (`FASTTokenizer`), appended last with a causal
`token_ar_mask`, and the loss is a shift-by-one cross-entropy through the EXISTING tied LM head,
masked by `token_loss_mask`. See reports/pi05_knowledge_insulation_design.md sec. 1a and 4.

Properties pinned down here:
  1. the supervised positions are real tokens (was xfail)
  2. those positions are distinguishable from each other (was xfail)
  3. the action expert cannot see the FAST tokens -- splicing the ground-truth answer into the
     prefix otherwise leaks it into the flow-matching loss. The paper: "we set the attention
     mask A such that no discrete FAST action token can attend to continuous action tokens and
     vice-versa."
  4. KI adds no parameters (the whole point of routing the CE through the tied LM head)
"""

from __future__ import annotations

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.models.tokenizer as _tokenizer

_B, _AH, _AD = 1, 4, 8
_MAX_TOKEN_LEN = 250


def _config(*, ki: bool = True) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=_AH,
        action_dim=_AD,
        max_token_len=_MAX_TOKEN_LEN,
        pi05=True,
        knowledge_insulation=ki,
    )


def _randomized_model():
    model = _config(ki=True).create(jax.random.key(0))
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


def _smooth_actions(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, _AH)[:, None]
    return np.sin(
        2 * np.pi * t * rng.uniform(0.2, 0.8, (1, _AD)) + rng.uniform(0, 2 * np.pi, (1, _AD))
    ).astype(np.float32)


def _observation(action_seed: int = 0, image_seed: int = 0) -> _model.Observation:
    """An observation carrying a real FAST-tokenized prompt, as the KI transform group builds it."""
    tok = _tokenizer.FASTTokenizer(max_len=_MAX_TOKEN_LEN)
    tokens, token_mask, ar_mask, loss_mask = tok.tokenize(
        "pick up the red block", np.zeros(_AD, dtype=np.float32), _smooth_actions(action_seed)
    )
    rng = np.random.default_rng(image_seed)
    img = jnp.asarray(rng.uniform(-1, 1, (_B, *_model.IMAGE_RESOLUTION, 3)), dtype=jnp.float32)
    return _model.Observation(
        images={"base_0_rgb": img},
        image_masks={"base_0_rgb": jnp.ones((_B,), dtype=bool)},
        state=jnp.zeros((_B, _AD), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(tokens, dtype=jnp.int32)[None],
        tokenized_prompt_mask=jnp.asarray(token_mask, dtype=bool)[None],
        token_ar_mask=jnp.asarray(ar_mask, dtype=jnp.int32)[None],
        token_loss_mask=jnp.asarray(loss_mask, dtype=bool)[None],
    )


def _prefix_out(model, obs):
    from openpi.models.pi0 import make_attn_mask

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
    return out


def _suffix_out(model, obs):
    """Run the insulated forward and return the action expert's output."""
    x_t = jnp.asarray(_smooth_actions(99))[None]
    time = jnp.full((_B,), 0.5, dtype=jnp.float32)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)
    _, suffix_out = model._forward_insulated(
        prefix_tokens, prefix_mask, prefix_ar_mask,
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond,
        model.fast_span_mask(obs, prefix_mask.shape[1]),
    )
    return suffix_out


def test_supervised_positions_are_real_tokens():
    """The loss-masked positions must all be valid (unmasked) prompt tokens, and non-empty.

    This is the property the padding defect violated: the supervised window previously sat
    entirely outside `tokenized_prompt_mask`.
    """
    obs = _observation()
    loss_mask = np.asarray(obs.token_loss_mask[0])
    valid = np.asarray(obs.tokenized_prompt_mask[0])

    assert loss_mask.sum() > 0, "no supervised positions at all"
    assert np.all(valid[loss_mask]), "some supervised positions are padding"


def test_supervised_positions_are_distinguishable():
    """THE decisive property. The supervised positions must carry different information.

    If they are identical the head is one global classifier applied n times and can only learn
    the marginal FAST-token distribution. Measured 1.0000 pairwise cosine before the fix.
    """
    model = _randomized_model()
    obs = _observation()
    out = _prefix_out(model, obs)

    n_image = int(out.shape[1] - _MAX_TOKEN_LEN)
    loss_mask = np.asarray(obs.token_loss_mask[0])
    window = np.asarray(out[0, n_image:][loss_mask].astype(jnp.float32))

    unit = window / np.linalg.norm(window, axis=-1, keepdims=True)
    cos = unit @ unit.T
    off_diagonal = cos[~np.eye(len(cos), dtype=bool)]

    assert off_diagonal.mean() < 0.99, (
        f"the {len(window)} supervised positions are near-identical (mean pairwise cosine "
        f"{off_diagonal.mean():.4f}); the head cannot represent an ordered sequence."
    )


def test_action_expert_cannot_see_the_fast_tokens():
    """No label leakage: the ground-truth FAST ids are in the prefix, so the flow-matching
    expert must be masked off them. Changing ONLY the spliced action tokens must leave the
    expert's output unchanged.

    Without the mask the expert reads the answer at training time and sees nothing at
    inference -- the flow loss would collapse to copying the tokens.
    """
    model = _randomized_model()
    obs_a = _observation(action_seed=0)
    obs_b = _observation(action_seed=1)

    # Sanity: the two observations really do differ in the spliced action tokens.
    assert not np.array_equal(np.asarray(obs_a.tokenized_prompt), np.asarray(obs_b.tokenized_prompt))

    assert jnp.allclose(_suffix_out(model, obs_a), _suffix_out(model, obs_b), atol=1e-5), (
        "the action expert's output changed when only the ground-truth FAST tokens changed: "
        "the label is leaking into the flow-matching branch."
    )


def test_knowledge_insulation_adds_no_parameters():
    """Option C's whole point: the CE goes through the tied LM head, so a KI model and a stock
    pi0.5 have identical parameter trees and checkpoints stay interchangeable."""

    def keys(*, ki: bool) -> set[str]:
        params = nnx.state(_config(ki=ki).create(jax.random.key(0)), nnx.Param).to_pure_dict()
        return set(traverse_util.flatten_dict(params, sep="/"))

    assert keys(ki=True) == keys(ki=False)
