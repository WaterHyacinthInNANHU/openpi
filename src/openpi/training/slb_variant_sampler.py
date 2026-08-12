"""Online SLB variant sampler: restrict / weight openpi rows per WVM variant.

This is the piece that makes the SLB variant bake-off non-trivial. The offline
build materialises numpy-only sidecars keyed by (task, attempt, window); here we
map each kept window back to a flat openpi row and hand the loader a torch
`Sampler` that restricts which rows are drawn.

EVERY variant, including AWR, draws its rows UNIFORMLY. AWR additionally returns a
row -> weight map: WVM (arXiv 2606.24742) Eq E.5 puts the weights in the OBJECTIVE
(a weighted LOSS), not in the sampler. See openpi.training.slb_awr_loss for the
objective and for why weighted resampling is not the same estimator.

Window semantics (see benchmarks/slb_pilot/adv_proxy.chunk_deltas): window index
``s`` is the sliding-window START frame ``s`` of the attempt's episode. So the
window->row map is the identity offset: ``row = episode_from[episode_index] + s``.

The camera_fixed HF dataset holds ALL attempts as episodes (success + skipped),
but only success attempts carry a sidecar entry. Therefore EVERY variant --
including vanilla -- must restrict to success-episode windows; the non-success
episodes and each episode's last H-1 frames are never emitted.

Imports only numpy + torch here plus the numpy-only sidecar readers from the
outer ``benchmarks`` package (bridged via PYTHONPATH). Never imports
scipy/mujoco/overlay, so it is safe under the openpi venv.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Mapping

import numpy as np
import torch


def awr_weight(delta: float, tau: float, cap: float) -> float:
    """WVM AWR weight w = min(exp(tau*delta), cap) (Eq E.7, pre-renorm)."""
    return float(min(np.exp(tau * float(delta)), cap))


# Frames the renderer emits per sim step before the one whose state matches the trace.
# The renderer writes 3 frames per 0.2 s sim step and the trace's joint_qpos matches the
# THIRD, so a window whose sim_time is t belongs at index round((t - t0)*fps) + 2 within
# its episode, where t0 is the attempt's FIRST sim_time (most traces start at 0.2 s, not 0).
#
# Measured directly, not inferred: LeRobot observation.state[:7] holds the trace joint_qpos
# bit-exactly, so the true window->row map is readable by nearest-neighbour search.
# Over 90 (attempt, window) samples on task 1644, true_row minus this rule is 0 for 90/90,
# while true_row minus the legacy `round(t*fps)` is -1 for 66 and +2 for 24 -- i.e. the
# legacy rule is never correct. Median ||qpos - state|| is 0.00000 at the aligned row vs
# 0.02491 at the legacy row.
RENDER_FRAME_OFFSET = 2


def plan_indices(
    episode_from: Mapping[int, int],
    sidecar,
    join_index,
    variant: str,
    *,
    fps: float,
    ep_len: Mapping[int, int],
    awr_tau: float = 3.0,
    awr_delta: float = 2.0,
    render_aligned_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map kept (attempt, window) pairs to flat rows (+ AWR weights).

    Args:
        episode_from: episode_index -> flat start row offset in the LeRobot dataset.
        sidecar: a ``VariantSidecar`` for (task, variant).
        join_index: a ``JoinIndex`` mapping attempt_id -> EpisodeRef.
        variant: one of vanilla/filt_bin/top70/awr/cfg.
        fps: video frame rate of the LeRobot dataset (e.g. 15.0).
        ep_len: episode_index -> number of frames in that episode.
        awr_tau, awr_delta: AWR temperature and cap (only used when variant=="awr").

    Returns:
        (rows, weights). ``rows`` is an int64 array of flat dataset indices computed
        as ``row = base + round(t_start[w] * fps)``, clamped to [base, base+ep_len[ep]-1].
        For AWR ``weights`` is a float64 array aligned to ``rows``; otherwise ``None``
        (uniform sampling over the restricted rows).
    """
    rows: list[int] = []
    weights: list[float] = []
    is_awr = variant == "awr"
    # vanilla and cfg both keep every success window (keep_mask is all-True); cfg's
    # conditioning is injected in the action expert at train time, not by row
    # restriction here. Only filt_bin/top70 drop rows; only awr attaches weights.
    n_missing_ep = 0
    for aid in sorted(sidecar._per_attempt):  # noqa: SLF001 - reader-owned map
        ref = join_index.episode_for(aid)
        if ref is None:
            n_missing_ep += 1
            continue
        ep = int(ref.episode_index)
        if ep not in episode_from:
            n_missing_ep += 1
            continue
        base = int(episode_from[ep])
        hi = base + int(ep_len[ep]) - 1
        mask = sidecar.keep_mask(aid)
        t_start = sidecar.t_start(aid)
        t0 = float(t_start[0]) if len(t_start) else 0.0
        for w in np.nonzero(mask)[0]:
            if render_aligned_rows:
                row = base + int(round((float(t_start[int(w)]) - t0) * fps)) + RENDER_FRAME_OFFSET
            else:
                # LEGACY, measurably wrong by one frame (see RENDER_FRAME_OFFSET). Retained
                # only to reproduce runs made before 2026-07-22; never use it for new work.
                row = base + int(round(float(t_start[int(w)]) * fps))
            row = min(max(row, base), hi)  # clamp into this episode
            rows.append(row)
            if is_awr:
                weights.append(awr_weight(sidecar.delta(aid, int(w)), awr_tau, awr_delta))
    if n_missing_ep:
        logging.warning("slb_variant_sampler: %d attempts had no episode in dataset", n_missing_ep)
    rows_arr = np.asarray(rows, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float64) if is_awr else None
    return rows_arr, weights_arr


class RowSampler(torch.utils.data.Sampler[int]):
    """Sample flat rows uniformly -- for EVERY variant, AWR included.

    Unlike ``WeightedRandomSampler`` -- which yields positions into its weight
    vector -- this yields the actual flat dataset rows, so the restricted index
    space maps straight onto the openpi dataset.

    There is deliberately no weighted branch here any more. AWR used to draw with
    ``torch.multinomial(weights, ..., replacement=True)``, which is a different
    estimator from WVM Eq E.5 and additionally perturbs the epoch composition
    (duplicates/omissions) independently of the weights. Eq E.5 is applied in the
    training loss instead; see ``row_weight_map`` / ``WeightedRowDataset`` for how the
    weight travels with each uniformly-drawn row.
    """

    def __init__(
        self,
        rows: np.ndarray,
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        if len(rows) == 0:
            raise ValueError("RowSampler got an empty row set; check sidecar/join_index/episode_from")
        self._rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        self._num_samples = int(num_samples) if num_samples is not None else len(self._rows)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        self._epoch += 1
        perm = torch.randperm(len(self._rows), generator=gen)
        if self._num_samples > len(perm):
            extra = torch.randint(len(self._rows), (self._num_samples - len(perm),), generator=gen)
            pos = torch.cat([perm, extra])
        else:
            pos = perm[: self._num_samples]
        yield from self._rows[pos].tolist()


def row_weight_map(rows: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    """Collapse the row-aligned AWR weight array into a row -> weight lookup.

    Two windows can land on the same flat row (rounding, or the episode-end clamp).
    Their multiplicity is already expressed by the row appearing twice in ``rows``,
    which uniform sampling honours, so the weight of a shared row is the MEAN of the
    colliding windows' weights rather than an arbitrary winner.
    """
    acc: dict[int, list[float]] = {}
    for r, w in zip(rows.tolist(), weights.tolist(), strict=True):
        acc.setdefault(int(r), []).append(float(w))
    n_collisions = sum(1 for v in acc.values() if len(v) > 1)
    if n_collisions:
        logging.warning("slb_variant_sampler: %d rows carry >1 window; averaging their AWR weights", n_collisions)
    return {r: float(np.mean(v)) for r, v in acc.items()}


class WeightedRowDataset:
    """Attach each row's Eq E.7 weight to the sample dict, under ``LOSS_WEIGHT_KEY``.

    Wraps the FULLY TRANSFORMED dataset, so the weight cannot be dropped by a repack
    transform, and is added only when a weight map exists (i.e. only for ``awr``);
    every other variant's sample dict is untouched.
    """

    def __init__(self, dataset, weights_by_row: Mapping[int, float]):
        # Imported here, not at module scope: slb_awr_loss pulls in jax, and this module
        # must stay importable from the numpy/torch-only offline test venv.
        from openpi.training.slb_awr_loss import LOSS_WEIGHT_KEY

        self._dataset = dataset
        self._weights_by_row = dict(weights_by_row)
        self._key = LOSS_WEIGHT_KEY

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        # A row outside the AWR keep-set can only be reached if something bypassed the
        # sampler; weight 0 makes such a row contribute nothing rather than silently
        # entering at weight 1.
        item[self._key] = np.float32(self._weights_by_row.get(int(index), 0.0))
        return item


def _unwrap_to_base(dataset):
    """Peel TransformedDataset wrappers to reach the underlying LeRobotDataset."""
    base = dataset
    seen = 0
    # Cap the unwrap depth: openpi stacks at most a couple of TransformedDataset
    # wrappers; 8 is a safety bound so a cyclic/self-referential wrapper can't spin.
    while hasattr(base, "_dataset") and seen < 8:
        base = base._dataset  # noqa: SLF001 - openpi TransformedDataset internal
        seen += 1
    return base


def episode_from_offsets(dataset) -> dict[int, int]:
    """episode_index -> flat start row offset for the underlying LeRobot dataset."""
    base = _unwrap_to_base(dataset)
    # LeRobot exposes episode_data_index["from"] (v2) or per-episode offsets in meta.
    edi = getattr(base, "episode_data_index", None)
    if edi is not None and "from" in edi:
        # LeRobot's episode_data_index["from"] is dense and positionally 0-indexed by
        # episode: element i is the flat start row of episode i, for i in 0..N-1. So
        # enumerate() yields the correct episode_index->offset map only because every
        # episode is present; a sparse episode space would need the meta path below.
        froms = np.asarray(edi["from"]).astype(np.int64)
        return {int(ep): int(off) for ep, off in enumerate(froms)}
    meta = getattr(base, "meta", None)
    episodes = getattr(meta, "episodes", None) if meta is not None else None
    if episodes is not None:
        # LeRobot v3.0 stores per-episode offsets in meta.episodes, which may be a
        # HuggingFace datasets.Dataset (column_names), a pandas DataFrame (columns),
        # or a dict. Column access episodes[col] returns a list/Series either way.
        cols = getattr(episodes, "column_names", None)
        if cols is None:
            cols = list(getattr(episodes, "columns", []) or (episodes.keys() if hasattr(episodes, "keys") else []))
        if "dataset_from_index" in cols and "episode_index" in cols:
            idx = np.asarray(episodes["episode_index"]).astype(np.int64)
            froms = np.asarray(episodes["dataset_from_index"]).astype(np.int64)
            return {int(e): int(f) for e, f in zip(idx, froms, strict=True)}
    raise RuntimeError(
        "Could not derive episode_from offsets from dataset "
        f"({type(base).__name__}); expected episode_data_index or meta.episodes"
    )


def _fps_and_ep_len(dataset, data_config) -> tuple[float, dict[int, int]]:
    """Read fps + per-episode length, asserting the LeRobot dataset is uniform-fps."""
    import glob
    import pandas as pd
    root = data_config.slb_dataset_root
    info = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    ep = pd.concat(
        [pd.read_parquet(p) for p in sorted(glob.glob(str(pathlib.Path(root) / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))],
        ignore_index=True,
    )
    ep_len = {int(e): int(l) for e, l in zip(ep["episode_index"], ep["length"])}
    # uniformity guard: check one episode's timestamps step by ~1/fps.
    # Read the `timestamp` column across ALL data chunk files in sorted (flat) order,
    # then index by the flat dataset_from_index/dataset_to_index. This is robust to
    # multi-chunk datasets where iloc on the first chunk alone would slice the wrong rows.
    import numpy as np
    data_files = sorted(
        glob.glob(str(pathlib.Path(root) / "data" / "**" / "*.parquet"), recursive=True)
    )
    ts_all = np.concatenate([
        pd.read_parquet(f, columns=["timestamp"])["timestamp"].to_numpy(dtype=float)
        for f in data_files
    ])
    e0 = int(ep["episode_index"].iloc[0])
    a, b = int(ep["dataset_from_index"].iloc[0]), int(ep["dataset_to_index"].iloc[0])
    ts = ts_all[a:b]
    if len(ts) > 2:
        dt = np.diff(ts)
        if np.max(np.abs(dt - 1.0 / fps)) > 0.5 / fps:
            raise RuntimeError(
                f"LeRobot episode {e0} is not uniform-fps (fps={fps}); "
                "sim_time->row arithmetic requires uniform fps"
            )
    return fps, ep_len


def build_sampler(dataset, data_config, *, seed: int = 0) -> tuple[RowSampler, dict[int, float] | None]:
    """Build the variant RowSampler (+ AWR row weights) for a transformed SLB dataset.

    Returns ``(sampler, weights_by_row)``. ``weights_by_row`` is None for every variant
    except ``awr``; when it is not None the caller must wrap the dataset in
    ``WeightedRowDataset`` so Eq E.5 can be applied in the loss.
    """
    from axis.episode.join_index import JoinIndex
    from axis.dataset.sidecar_reader import VariantSidecar

    variant = data_config.slb_variant
    sidecar = VariantSidecar.load(
        int(data_config.slb_task_id), variant, sidecar_root=data_config.slb_sidecar_root
    )
    join_index = JoinIndex.from_manifest(data_config.slb_manifest_path)
    episode_from = episode_from_offsets(dataset)
    fps, ep_len = _fps_and_ep_len(dataset, data_config)
    rows, weights = plan_indices(
        episode_from, sidecar, join_index, variant,
        fps=fps, ep_len=ep_len,
        awr_tau=data_config.awr_tau, awr_delta=data_config.awr_delta,
        render_aligned_rows=getattr(data_config, "slb_render_aligned_rows", False),
    )
    weights_by_row = None if weights is None else row_weight_map(rows, weights)
    logging.info(
        "slb_variant_sampler[%s]: %d rows, uniformly sampled (loss-weighted=%s) over %d episodes",
        variant,
        len(rows),
        weights_by_row is not None,
        len(episode_from),
    )
    return RowSampler(rows, seed=seed), weights_by_row
