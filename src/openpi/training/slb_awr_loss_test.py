"""Tests for the WVM Eq E.5 weighted-BC loss and the path the weight takes to reach it.

Run under the openpi venv:

    cd openpi && JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src ./.venv/bin/pytest \
        src/openpi/training/slb_awr_loss_test.py -q

Deliberately model-free where possible. A freshly `create()`d Pi0 has degenerate
attention/MLP projections (see models/pi0_ki_test.py: 43 of 53 param leaves get exactly
zero gradient and the transformer is an identity map), so numeric assertions made through
a stock model pass vacuously. The Eq E.5 combination is tested directly on synthetic
losses and weights, and the one test that does run the real `train_step` uses a tiny
model whose loss provably depends on its parameter -- with a CONTROL assertion that the
unweighted and weighted losses differ, so an inert weight path cannot pass silently.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.slb_awr_loss as _awr
import openpi.training.slb_variant_sampler as _svs
import openpi.training.utils as training_utils

_EPS = _awr.EPS


# ---------------------------------------------------------------------------
# Eq E.5 itself
# ---------------------------------------------------------------------------


def _chunked(seed: int = 0, b: int = 6, ah: int = 4):
    return jax.random.normal(jax.random.key(seed), (b, ah)) ** 2


def test_no_weights_is_bitwise_the_old_plain_mean():
    """Non-AWR configs must be numerically unchanged: the None path IS jnp.mean."""
    x = _chunked()
    assert _awr.combine(x, None) == jnp.mean(x)  # exact, not approximate
    assert _awr.combine(x) == jnp.mean(x)


@pytest.mark.parametrize("c", [1.0, 0.25, 7.5, 2.0])
def test_equal_weights_equal_plain_mean(c):
    """The paper's scale-alignment property: uniform weights reproduce vanilla BC."""
    x = _chunked(seed=1)
    w = jnp.full((x.shape[0],), c)
    np.testing.assert_allclose(_awr.combine(x, w), jnp.mean(x), rtol=1e-6, atol=0)


def test_unequal_weights_match_hand_computed_eq_e5():
    x = _chunked(seed=2)
    w = jnp.asarray([0.3, 2.0, 1.0, 0.05, 2.0, 0.7])
    per_example = np.asarray(jnp.mean(x, axis=1), dtype=np.float64)
    wn = np.asarray(w, dtype=np.float64)
    # per-batch renormalisation so (1/B) sum w = 1, then L = sum w*l / (sum w + eps)
    b = len(wn)
    wn = b * wn / (wn.sum() + _EPS)
    expected = float((wn * per_example).sum() / (wn.sum() + _EPS))
    np.testing.assert_allclose(float(_awr.combine(x, w)), expected, rtol=1e-6, atol=0)


def test_unequal_weights_actually_move_the_loss():
    """CONTROL: without this, the two tests above could both hold on an inert weight path."""
    x = _chunked(seed=2)
    w = jnp.asarray([0.05, 2.0, 0.05, 2.0, 0.05, 2.0])
    weighted = float(_awr.combine(x, w))
    plain = float(jnp.mean(x))
    assert abs(weighted - plain) / plain > 0.05  # >5% apart, far outside float noise


def test_renormalize_gives_unit_mean_weight():
    w = jnp.asarray([0.3, 2.0, 1.0, 0.05])
    np.testing.assert_allclose(float(jnp.mean(_awr.renormalize(w))), 1.0, rtol=1e-6, atol=0)


def test_all_zero_weights_do_not_divide_by_zero():
    """eps guard: a degenerate batch must give 0.0, never NaN/inf."""
    x = _chunked(seed=3)
    for w in (jnp.zeros((x.shape[0],)), jnp.full((x.shape[0],), 1e-30)):
        out = _awr.combine(x, w)
        assert jnp.isfinite(out), out
        np.testing.assert_allclose(float(out), 0.0, atol=1e-6)


def test_combine_handles_1d_per_example_loss():
    """compute_loss returns [*b ah]; a [b] loss must still reduce over the batch axis only."""
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    w = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(float(_awr.combine(x, w)), float(jnp.mean(x)), rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# The sampler no longer resamples
# ---------------------------------------------------------------------------


def test_rowsampler_rejects_a_weight_vector():
    """The multinomial branch is gone; passing weights must fail loudly, not silently."""
    rows = np.array([10, 20, 30], dtype=np.int64)
    with pytest.raises(TypeError):
        _svs.RowSampler(rows, np.array([0.1, 2.0, 1.0]))  # type: ignore[call-arg]


def test_rowsampler_draws_uniformly():
    """Rows that were ~20x likelier under the old multinomial branch are now equiprobable.

    Same setup as the deleted test_rowsampler_weighting_biases_high_weight_rows, which
    asserted counts[20] > counts[10] * 5. Here 5x is far outside the tolerated band, so
    this test fails against the old weighted sampler.
    """
    rows = np.array([10, 20], dtype=np.int64)
    drawn = np.asarray(list(_svs.RowSampler(rows, num_samples=4000, seed=0)))
    n10, n20 = int((drawn == 10).sum()), int((drawn == 20).sum())
    assert n10 + n20 == 4000
    assert abs(n10 - n20) < 250, (n10, n20)  # ~8 binomial sigma; 5x bias would be ~2700


def test_rowsampler_source_has_no_multinomial():
    """The draw itself must contain no weighted branch (docstring mentions it by name)."""
    import inspect

    assert "multinomial" not in inspect.getsource(_svs.RowSampler.__iter__)
    assert "multinomial" not in inspect.getsource(_svs.RowSampler.__init__)


def test_row_weight_map_averages_collisions():
    rows = np.array([100, 101, 100], dtype=np.int64)
    weights = np.array([2.0, 0.5, 1.0], dtype=np.float64)
    m = _svs.row_weight_map(rows, weights)
    assert m == {100: 1.5, 101: 0.5}


# ---------------------------------------------------------------------------
# The weight reaches the batch
# ---------------------------------------------------------------------------


class _DictDataset:
    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {"state": np.full((2,), float(i), dtype=np.float32)}


def test_weighted_row_dataset_attaches_the_row_weight():
    ds = _svs.WeightedRowDataset(_DictDataset(5), {1: 2.0, 3: 0.25})
    assert float(ds[1][_awr.LOSS_WEIGHT_KEY]) == 2.0
    assert float(ds[3][_awr.LOSS_WEIGHT_KEY]) == 0.25
    # a row outside the keep-set contributes nothing rather than entering at weight 1
    assert float(ds[0][_awr.LOSS_WEIGHT_KEY]) == 0.0
    assert ds[1]["state"].tolist() == [1.0, 1.0]
    assert len(ds) == 5


def test_collate_stacks_the_weight_into_a_batch_vector():
    ds = _svs.WeightedRowDataset(_DictDataset(4), {0: 1.0, 1: 2.0, 2: 3.0})
    batch = _data_loader._collate_fn([ds[i] for i in range(3)])  # noqa: SLF001
    np.testing.assert_allclose(batch[_awr.LOSS_WEIGHT_KEY], [1.0, 2.0, 3.0])


def _fake_batch(b: int = 3, *, with_weight: bool):
    batch = {
        "image": {"base_0_rgb": np.zeros((b, 4, 4, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.ones((b,), dtype=bool)},
        "state": np.zeros((b, 5), dtype=np.float32),
        "actions": np.zeros((b, 2, 5), dtype=np.float32),
    }
    if with_weight:
        batch[_awr.LOSS_WEIGHT_KEY] = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
    return batch


class _FakeTorchLoader:
    def __init__(self, batch):
        self._batch = batch

    def __iter__(self):
        yield self._batch


def test_data_loader_yields_two_tuple_without_weights():
    loader = _data_loader.DataLoaderImpl(_config.DataConfig(), _FakeTorchLoader(_fake_batch(with_weight=False)))
    (item,) = list(loader)
    assert len(item) == 2  # unchanged for vanilla/filt_bin/top70/cfg


def test_data_loader_yields_the_weight_as_a_third_element():
    loader = _data_loader.DataLoaderImpl(_config.DataConfig(), _FakeTorchLoader(_fake_batch(with_weight=True)))
    (item,) = list(loader)
    assert len(item) == 3
    np.testing.assert_allclose(item[2], [0.5, 1.0, 2.0])


# ---------------------------------------------------------------------------
# End-to-end through the real train_step
# ---------------------------------------------------------------------------

_B, _AH, _AD, _SD = 4, 2, 3, 3


def _load_train_module():
    path = pathlib.Path(__file__).parents[3] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_openpi_train_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TinyModel(_model.BaseModel):
    """Smallest model whose loss genuinely depends on a trainable parameter.

    Not a Pi0: a freshly created Pi0 is an identity map with mostly-zero gradients, so a
    numeric train_step assertion made through it would pass for the wrong reason.
    """

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(_AD, _AH, 4)
        self.proj = nnx.Param(jax.random.normal(rngs.params(), (_SD, _AD)) * 0.5)

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        pred = observation.state @ self.proj.value  # (B, AD)
        return jnp.mean(jnp.square(pred[:, None, :] - actions), axis=-1)  # (B, AH)

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError


def _tiny_state(config):
    model = _TinyModel(nnx.Rngs(params=jax.random.key(7)))
    params = nnx.state(model)
    tx = optax.sgd(0.1)
    return model, training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def _tiny_batch():
    obs = _model.Observation(
        images={"base_0_rgb": jnp.zeros((_B, 4, 4, 3))},
        image_masks={"base_0_rgb": jnp.ones((_B,), dtype=bool)},
        state=jax.random.normal(jax.random.key(3), (_B, _SD)),
    )
    actions = jax.random.normal(jax.random.key(4), (_B, _AH, _AD))
    return obs, actions


def test_train_step_applies_eq_e5_and_is_unchanged_without_weights():
    train = _load_train_module()
    config = _config.TrainConfig(name="slb_awr_loss_test", batch_size=_B)
    model, state = _tiny_state(config)
    obs, actions = _tiny_batch()
    rng = jax.random.key(0)

    chunked = model.compute_loss(rng, obs, actions)
    assert chunked.shape == (_B, _AH)

    _, info_plain = train.train_step(config, rng, state, (obs, actions))
    # the 2-tuple path still reports exactly the old jnp.mean
    np.testing.assert_allclose(float(info_plain["loss"]), float(jnp.mean(chunked)), rtol=1e-6, atol=0)

    # One-hot on the worst example: Eq E.5 must then report that example's own loss,
    # which is a value the plain mean cannot produce.
    per_example = jnp.mean(chunked, axis=1)
    weights = jnp.zeros((_B,)).at[int(jnp.argmax(per_example))].set(1.0)
    _, info_weighted = train.train_step(config, rng, state, (obs, actions, weights))
    np.testing.assert_allclose(
        float(info_weighted["loss"]), float(_awr.combine(chunked, weights)), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(float(info_weighted["loss"]), float(jnp.max(per_example)), rtol=1e-6, atol=0)

    # CONTROL: the weights must actually change loss AND gradient, otherwise the
    # agreement above would be the vacuous "both equal the plain mean" case.
    assert abs(float(info_weighted["loss"]) - float(info_plain["loss"])) / float(info_plain["loss"]) > 0.05
    assert abs(float(info_weighted["grad_norm"]) - float(info_plain["grad_norm"])) > 1e-4

    # equal weights reproduce the vanilla step
    _, info_equal = train.train_step(config, rng, state, (obs, actions, jnp.ones((_B,))))
    np.testing.assert_allclose(float(info_equal["loss"]), float(info_plain["loss"]), rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        float(info_equal["grad_norm"]), float(info_plain["grad_norm"]), rtol=1e-5, atol=0
    )
