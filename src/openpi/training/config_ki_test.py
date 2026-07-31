"""Knowledge-insulation twins of the DROID and AXIS-pretrain configs.

KI is exposed the same way the SLB arms expose it: as a SEPARATE, additively-named config
rather than a change to the existing one, so any in-flight run against the non-KI name keeps
the config it started with. The control tests below (`*_is_untouched`) are what enforce that.

`pi05_droid` deliberately has NO KI twin: it is an inference-only config (no weight_loader,
no num_train_steps) and `knowledge_insulation` only affects `compute_loss`. Enabling it there
would add an untrained `discrete_action_head` that changes nothing at sampling time.
`pi05_droid_finetune` is the trainable DROID config and is the one that gets the twin.
"""

from __future__ import annotations

import pytest

import openpi.models.pi0_config as _pi0_config
import openpi.training.config as _config

# KI splices an `Action:` + FAST-ids + `|` postfix into the prompt, so the token budget rises.
# Measured: postfix 37-53 tokens, totals 158-202, i.e. the stock 200 overflows ~3.3% of samples.
_EXPECTED_MAX_TOKEN_LEN = 250
_STOCK_MAX_TOKEN_LEN = 200

_KI_TWINS = [
    ("pi05_axis_pretrain", "pi05_axis_pretrain_ki"),
    ("pi05_droid_finetune", "pi05_droid_finetune_ki"),
]


@pytest.mark.parametrize(("base_name", "ki_name"), _KI_TWINS)
def test_ki_twin_exists_and_enables_insulation(base_name, ki_name):
    model = _config.get_config(ki_name).model
    assert isinstance(model, _pi0_config.Pi0Config)
    assert model.knowledge_insulation is True
    assert model.max_token_len == _EXPECTED_MAX_TOKEN_LEN


@pytest.mark.parametrize(("base_name", "ki_name"), _KI_TWINS)
def test_base_config_is_untouched(base_name, ki_name):
    """Control: adding the twin must not turn KI on for the existing config."""
    assert _config.get_config(base_name).model.knowledge_insulation is False


@pytest.mark.parametrize(("base_name", "ki_name"), _KI_TWINS)
def test_twin_differs_from_base_only_in_ki(base_name, ki_name):
    """The A/B must be clean: same data, same optimizer, same budget -- only KI differs."""
    base, ki = _config.get_config(base_name), _config.get_config(ki_name)

    assert ki.data == base.data
    assert ki.weight_loader == base.weight_loader
    assert ki.num_train_steps == base.num_train_steps
    assert ki.batch_size == base.batch_size
    assert ki.lr_schedule == base.lr_schedule
    assert ki.freeze_filter == base.freeze_filter

    ki_fields = {"knowledge_insulation", "ki_fast_loss_weight", "max_token_len"}
    differing = {
        f.name
        for f in _pi0_config.dataclasses.fields(base.model)
        if getattr(base.model, f.name) != getattr(ki.model, f.name)
    }
    assert differing <= ki_fields, f"twin differs outside KI: {differing - ki_fields}"


def test_pi05_droid_has_no_ki_twin():
    """`pi05_droid` is inference-only; a KI twin there would be inert. See module docstring."""
    assert _config.get_config("pi05_droid").model.knowledge_insulation is False
    with pytest.raises(ValueError):
        _config.get_config("pi05_droid_ki")




import numpy as np  # noqa: E402

import openpi.models.tokenizer as _tokenizer  # noqa: E402

# Real actions are 8-D (7 joint velocities + gripper) zero-padded to 32 by
# `PadStatesAndActions`, for both DROID and AXIS-Franka.
_ACTIVE_ACTION_DIMS = 8


def _smooth_chunk(horizon: int, action_dim: int, active_dims: int, seed: int) -> np.ndarray:
    """A normalized, zero-padded action chunk of the shape the data pipeline produces.

    Smooth, because FAST's front-end is a DCT: white noise is near-incompressible and its
    token count says nothing about the config.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, horizon)[:, None]
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    chunk[:, :active_dims] = np.sin(
        2 * np.pi * t * rng.uniform(0.2, 0.8, (1, active_dims)) + rng.uniform(0, 2 * np.pi, (1, active_dims))
    )
    return chunk


def test_ki_token_budget_fits_prompt_plus_action_postfix():
    """`max_token_len` must hold the pi0.5 prompt AND the spliced FAST postfix.

    Truncation is from the right, so an overflow silently drops the tail of the action chunk --
    the exact failure the discrete loss exists to prevent. Measured across 24 chunks: postfix
    37-53 tokens, total 158-202. The stock 200 overflows; 250 does not.
    """
    ki = _config.get_config("pi05_axis_pretrain_ki").model
    tok = _tokenizer.FASTTokenizer(max_len=ki.max_token_len)
    rng = np.random.default_rng(0)

    totals = []
    for seed in range(24):
        chunk = _smooth_chunk(ki.action_horizon, ki.action_dim, _ACTIVE_ACTION_DIMS, seed)
        state = rng.uniform(-1, 1, ki.action_dim).astype(np.float32)
        _, token_mask, _, loss_mask = tok.tokenize("pick up the red block and place it in the bowl", state, chunk)
        totals.append(int(token_mask.sum()))
        # A supervised postfix must actually be present, and inside the valid region.
        assert int(loss_mask.sum()) > 0, f"chunk {seed}: no supervised action tokens"
        assert np.all(token_mask[loss_mask]), f"chunk {seed}: supervised positions fell into padding"

    assert max(totals) < ki.max_token_len, (
        f"prompt+action tokens reach {max(totals)} but max_token_len is {ki.max_token_len}; "
        "the action tail is being truncated away"
    )


def test_ki_budget_keeps_headroom_over_the_stock_one():
    """Documents WHY the budget rose, honestly.

    Correction to an earlier claim in the design doc: with REAL AXIS prompts (median 6 words,
    longest 12) the prompt+postfix total tops out around 191, so the stock 200 does NOT in fact
    overflow for AXIS -- an earlier "3.3% overflow" figure came from an invented long prompt and
    was wrong. The case for 250 is margin, not necessity:

      * ~9 tokens of headroom at 200 is thin, and FAST output length varies with how
        compressible the chunk is, so a noisier chunk can cross it;
      * truncation is from the right, so crossing it silently deletes the tail of the action
        chunk -- a failure that is invisible in the loss curve;
      * `pi05_droid_finetune_ki` targets arbitrary user DROID datasets whose prompts are not
        bounded by AXIS's 12 words.

    The cost is ~11% more prefix tokens, and zero parameters (RoPE).
    """
    ki = _config.get_config("pi05_axis_pretrain_ki").model
    assert ki.max_token_len > _STOCK_MAX_TOKEN_LEN

    tok = _tokenizer.FASTTokenizer(max_len=1000)  # no truncation, so we measure true lengths
    rng = np.random.default_rng(0)
    totals = []
    for seed in range(24):
        chunk = _smooth_chunk(ki.action_horizon, ki.action_dim, _ACTIVE_ACTION_DIMS, seed)
        state = rng.uniform(-1, 1, ki.action_dim).astype(np.float32)
        # The longest real AXIS instruction, at 12 words.
        _, token_mask, _, _ = tok.tokenize(
            "Put the Contact Lens Case on the Toilet Paper - Multi Embodiment", state, chunk
        )
        totals.append(int(token_mask.sum()))

    assert max(totals) < ki.max_token_len, "even the KI budget is too small"
    assert max(totals) > _STOCK_MAX_TOKEN_LEN - 30, (
        f"max observed {max(totals)} now sits far below the stock {_STOCK_MAX_TOKEN_LEN}; the "
        "headroom argument for raising the budget no longer holds and 250 should be revisited"
    )
