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

# A normalized 16x32 action chunk FAST-compresses to ~40-50 ids, so the 32 default would
# truncate almost every chunk; 2049 = FAST's own 2048 vocabulary plus the pad/end class.
_EXPECTED_NUM_TOKENS = 64
_EXPECTED_VOCAB = 2049

_KI_TWINS = [
    ("pi05_axis_pretrain", "pi05_axis_pretrain_ki"),
    ("pi05_droid_finetune", "pi05_droid_finetune_ki"),
]


@pytest.mark.parametrize(("base_name", "ki_name"), _KI_TWINS)
def test_ki_twin_exists_and_enables_insulation(base_name, ki_name):
    model = _config.get_config(ki_name).model
    assert isinstance(model, _pi0_config.Pi0Config)
    assert model.knowledge_insulation is True
    assert model.ki_num_action_tokens == _EXPECTED_NUM_TOKENS
    assert model.ki_fast_vocab_size == _EXPECTED_VOCAB


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

    ki_fields = {"knowledge_insulation", "ki_num_action_tokens", "ki_fast_vocab_size", "ki_fast_loss_weight"}
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


def _tokenizer_for(model) -> _tokenizer.FASTActionTokenizer:
    return _tokenizer.FASTActionTokenizer(
        num_tokens=model.ki_num_action_tokens, pad_token_id=model.ki_fast_vocab_size - 1
    )


def test_ki_action_token_budget_covers_a_real_chunk():
    """The token budget must not silently truncate.

    If a chunk needs more than `ki_num_action_tokens` ids the tail is dropped, and KI then
    trains the VLM on a partial action while still reporting itself as KI. Measured: a 16x32
    chunk with 8 active dims takes ~33-43 ids, so 64 has real headroom.
    """
    ki = _config.get_config("pi05_axis_pretrain_ki").model
    tok = _tokenizer_for(ki)

    for seed in range(8):
        chunk = _smooth_chunk(ki.action_horizon, ki.action_dim, _ACTIVE_ACTION_DIMS, seed)
        tokens = tok.tokenize_actions(chunk)
        assert tokens.shape == (ki.ki_num_action_tokens,)
        assert int(tokens.max()) < ki.ki_fast_vocab_size
        # Padding present => the chunk fit without truncation.
        assert int((tokens == ki.ki_fast_vocab_size - 1).sum()) > 0, f"chunk {seed} filled the budget"


def test_token_budget_headroom_depends_on_zero_padding():
    """Documents WHY 64 is enough, so the number is not silently invalidated later.

    The headroom comes from 24 of the 32 action dims being constant zero padding. A chunk
    whose 32 dims are all genuinely active needs ~116-148 ids and WOULD truncate. So
    `ki_num_action_tokens=64` is tied to the 8-D-padded-to-32 action space; a future config
    with more real dims must raise it.
    """
    ki = _config.get_config("pi05_axis_pretrain_ki").model
    tok = _tokenizer_for(ki)
    pad = ki.ki_fast_vocab_size - 1

    padded = tok.tokenize_actions(_smooth_chunk(ki.action_horizon, ki.action_dim, _ACTIVE_ACTION_DIMS, 0))
    dense = tok.tokenize_actions(_smooth_chunk(ki.action_horizon, ki.action_dim, ki.action_dim, 0))

    assert int((padded == pad).sum()) > 0, "8 active dims should fit in the budget"
    assert int((dense == pad).sum()) == 0, "32 active dims should saturate the budget"
