"""WVM Eq E.5 weighted-BC loss combination for the SLB ``awr`` arm (openpi venv, jax only).

Why this module exists: the ``awr`` arm used to be implemented as weighted
RESAMPLING -- ``torch.multinomial(weights, ..., replacement=True)`` inside
``slb_variant_sampler.RowSampler``. That is not what WVM (arXiv 2606.24742)
Appendix E specifies, and the difference is not cosmetic: resampling has a
different estimator variance than reweighting, and drawing WITH REPLACEMENT also
changes the epoch composition (duplicates, omissions) independently of the
weights. The paper puts the weights in the OBJECTIVE:

    Eq E.4  advantage:   Delta_i = V(t_tail) - V(t_head),
                         t_head = t, t_tail = min(t + H - 1, T - 1)
    Eq E.5  objective:   L = sum_i w_i * l_i / (sum_i w_i + eps),   eps = 1e-6
    Eq E.7  AWR weight:  w_i = min(exp(tau * Delta_i), delta),
                         delta = 2.0, tau = 10 (H=10) / 2 (H=50)

and, verbatim:

    "We further renormalise the per-batch weights so that (1/B) sum_i w_i = 1,
     keeping the gradient scale aligned with vanilla BC and allowing us to reuse
     the same learning rate and weight decay as the BC baseline."

So rows are drawn UNIFORMLY (same path as vanilla) and the weight travels with
each sampled row into the batch; this module turns (per-example losses, weights)
into the scalar Eq E.5 objective.

``combine(chunked_loss, None)`` is the vanilla path and returns exactly
``jnp.mean(chunked_loss)`` -- the same expression the training loop used before
this module existed, so vanilla/filt_bin/top70/cfg are bit-for-bit unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp

import openpi.shared.array_typing as at

# Key under which the per-row Eq E.7 weight rides through the torch collate_fn into
# the training batch dict. Emitted ONLY for the `awr` variant, so every other
# config's batch pytree keeps exactly the keys it had before.
LOSS_WEIGHT_KEY = "slb_loss_weight"

# WVM Eq E.5's epsilon. Also the zero-weight guard: an all-zero (or degenerately
# tiny) weight vector yields 0.0 instead of NaN.
EPS = 1e-6


def renormalize(weights: at.Array, *, eps: float = EPS) -> at.Array:
    """Scale weights so that (1/B) sum_i w_i == 1 (the WVM per-batch renormalisation).

    Keeping the batch mean weight at 1 is what lets the awr arm reuse the BC arm's
    learning rate and weight decay. ``+ eps`` in the denominator is the all-zero guard.
    """
    b = weights.shape[0]
    return b * weights / (jnp.sum(weights) + eps)


def weighted_mean(
    per_example_loss: at.Float[at.Array, " b"],
    weights: at.Float[at.Array, " b"],
    *,
    eps: float = EPS,
) -> at.Float[at.Array, ""]:
    """WVM Eq E.5: ``L = sum_i w_i l_i / (sum_i w_i + eps)``, on renormalised weights.

    With equal weights this collapses to ``mean(per_example_loss)`` (up to eps), which
    is the scale-alignment property the paper relies on.
    """
    w = renormalize(jnp.asarray(weights, dtype=per_example_loss.dtype), eps=eps)
    return jnp.sum(w * per_example_loss) / (jnp.sum(w) + eps)


def combine(chunked_loss: at.Array, weights: at.Array | None = None, *, eps: float = EPS) -> at.Array:
    """Reduce a ``[*b ah]`` per-chunk loss to the scalar training objective.

    ``weights is None`` is the vanilla-BC path and returns literally
    ``jnp.mean(chunked_loss)``. Otherwise the chunk axes are averaged into one
    per-example loss l_i and Eq E.5 is applied over the batch axis.
    """
    if weights is None:
        return jnp.mean(chunked_loss)
    per_example = jnp.mean(chunked_loss, axis=tuple(range(1, chunked_loss.ndim)))
    return weighted_mean(per_example, weights, eps=eps)
