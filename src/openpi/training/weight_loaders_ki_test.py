"""A KI model must load a pre-KI checkpoint with no special-casing at all.

History: the first KI design added a `discrete_action_head`, which no released checkpoint
carries. `CheckpointWeightLoader` keeps only the intersection of checkpoint and reference params
plus reference params matching `missing_regex` (`.*lora.*`), so the head was dropped and
`scripts/train.py` `_load_weights_and_validate` raised on the structural mismatch before step 0
-- every `_ki` config was unlaunchable. That was first fixed by widening the regex.

Option C then removed the head entirely: the discrete cross-entropy goes through the TIED LM
head (`gemma.Module.decode`), so KI allocates no parameters and the regex went back to
`.*lora.*`. These tests keep that true -- if anyone reintroduces a KI-only parameter, the first
test fails and tells them they have also reintroduced the loading bug.
"""

from __future__ import annotations

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import pytest

import openpi.models.pi0_config as _pi0_config
import openpi.training.weight_loaders as _weight_loaders


def _params(*, ki: bool) -> dict:
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        pi05=True,
        knowledge_insulation=ki,
    )
    return nnx.state(cfg.create(jax.random.key(0)), nnx.Param).to_pure_dict()


@pytest.fixture(scope="module")
def ki_params() -> dict:
    return _params(ki=True)


@pytest.fixture(scope="module")
def stock_params() -> dict:
    return _params(ki=False)


def test_ki_needs_no_params_the_checkpoint_lacks(ki_params, stock_params):
    """The root property: a KI model asks for exactly the params a stock pi0.5 checkpoint has."""
    assert set(traverse_util.flatten_dict(ki_params, sep="/")) == set(
        traverse_util.flatten_dict(stock_params, sep="/")
    )


def test_stock_checkpoint_loads_into_a_ki_model(ki_params, stock_params):
    """End to end: merging a pre-KI checkpoint into a KI model leaves nothing missing.

    This is exactly the structural equality `scripts/train.py` asserts before step 0.
    """
    checkpoint = {k: np.asarray(v) for k, v in traverse_util.flatten_dict(stock_params, sep="/").items()}
    merged = _weight_loaders._merge_params(
        traverse_util.unflatten_dict(checkpoint, sep="/"),
        ki_params,
        missing_regex=_weight_loaders.CheckpointWeightLoader.missing_regex,
    )
    assert set(traverse_util.flatten_dict(merged, sep="/")) == set(
        traverse_util.flatten_dict(ki_params, sep="/")
    )


def test_unknown_reference_params_are_still_dropped(ki_params, stock_params):
    """Control: the regex must not be so wide that a real mismatch slips through.

    A param the checkpoint lacks and LoRA did not introduce is a typo or arch drift, and must
    stay dropped so train.py's equality check reports it.
    """
    reference = {**ki_params, "typo_head": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    merged = _weight_loaders._merge_params(
        stock_params,
        reference,
        missing_regex=_weight_loaders.CheckpointWeightLoader.missing_regex,
    )
    assert "typo_head/kernel" not in traverse_util.flatten_dict(merged, sep="/")
