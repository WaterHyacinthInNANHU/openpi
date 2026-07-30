"""The knowledge-insulation head must survive loading a pre-KI checkpoint.

`discrete_action_head` is introduced by `knowledge_insulation=True` and exists in NO
released checkpoint -- every KI config initialises from `pi05_droid/params`, which predates
it. `CheckpointWeightLoader` keeps only the intersection of checkpoint and reference params,
plus reference params matching `missing_regex` (was `.*lora.*` only). The head matched
neither, so it was dropped from the merged tree, and `scripts/train.py`
`_load_weights_and_validate` then compares the merged tree against the full params shape
with `check_pytree_equality` -- a hard ValueError before step 0.

The control test is what keeps the fix honest: widening the regex to `.*` would make the
first test pass while silently accepting genuinely misnamed params.
"""

from __future__ import annotations

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import numpy as np
import pytest

import openpi.models.pi0_config as _pi0_config
import openpi.training.weight_loaders as _weight_loaders


def _ki_params_shape() -> dict:
    """Reference params of a tiny KI-enabled pi0.5, as train.py passes to the loader."""
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
        max_token_len=16,
        pi05=True,
        knowledge_insulation=True,
    )
    return nnx.state(cfg.create(jax.random.key(0)), nnx.Param).to_pure_dict()


@pytest.fixture(scope="module")
def ki_params() -> dict:
    return _ki_params_shape()


def _pre_ki_checkpoint(params: dict) -> dict:
    """The same params as a released (pre-KI) checkpoint: everything except the KI head."""
    flat = traverse_util.flatten_dict(params, sep="/")
    return traverse_util.unflatten_dict(
        {k: np.asarray(v) for k, v in flat.items() if "discrete_action_head" not in k}, sep="/"
    )


def test_ki_head_survives_a_pre_ki_checkpoint(ki_params):
    merged = _weight_loaders._merge_params(
        _pre_ki_checkpoint(ki_params),
        ki_params,
        missing_regex=_weight_loaders.CheckpointWeightLoader.missing_regex,
    )
    flat = traverse_util.flatten_dict(merged, sep="/")
    assert "discrete_action_head/kernel" in flat
    assert "discrete_action_head/bias" in flat


def test_merged_tree_matches_the_reference_structure(ki_params):
    """What train.py actually asserts: the merged tree must be structurally complete."""
    merged = _weight_loaders._merge_params(
        _pre_ki_checkpoint(ki_params),
        ki_params,
        missing_regex=_weight_loaders.CheckpointWeightLoader.missing_regex,
    )
    assert set(traverse_util.flatten_dict(merged, sep="/")) == set(
        traverse_util.flatten_dict(ki_params, sep="/")
    )


def test_unknown_reference_params_are_still_dropped(ki_params):
    """Control: the regex must not have been widened to `.*`.

    A param the checkpoint does not carry and KI did not introduce is a real mismatch
    (typo, arch drift). It must still be dropped so train.py's equality check reports it.
    """
    reference = {**ki_params, "typo_head": {"kernel": np.zeros((2, 2), dtype=np.float32)}}
    merged = _weight_loaders._merge_params(
        _pre_ki_checkpoint(ki_params),
        reference,
        missing_regex=_weight_loaders.CheckpointWeightLoader.missing_regex,
    )
    assert "typo_head/kernel" not in traverse_util.flatten_dict(merged, sep="/")
