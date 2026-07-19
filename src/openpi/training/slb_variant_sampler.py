"""Online SLB variant sampler: restrict / weight openpi rows per WVM variant.

This is the piece that makes the SLB variant bake-off non-trivial. The offline
build materialises numpy-only sidecars keyed by (task, attempt, window); here we
map each kept window back to a flat openpi row and hand the loader a torch
`Sampler` that either restricts (hard filters + vanilla) or weights (AWR) which
rows are drawn.

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

import logging
from typing import Mapping

import numpy as np
import torch


def awr_weight(delta: float, tau: float, cap: float) -> float:
    """WVM AWR weight w = min(exp(tau*delta), cap) (Eq E.7, pre-renorm)."""
    return float(min(np.exp(tau * float(delta)), cap))


def plan_indices(
    episode_from: Mapping[int, int],
    sidecar,
    join_index,
    variant: str,
    *,
    awr_tau: float = 3.0,
    awr_delta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map kept (attempt, window) pairs to flat rows (+ AWR weights).

    Args:
        episode_from: episode_index -> flat start row offset in the LeRobot dataset.
        sidecar: a ``VariantSidecar`` for (task, variant).
        join_index: a ``JoinIndex`` mapping attempt_id -> EpisodeRef.
        variant: one of vanilla/filt_bin/top70/awr/cfg.
        awr_tau, awr_delta: AWR temperature and cap (only used when variant=="awr").

    Returns:
        (rows, weights). ``rows`` is an int64 array of flat dataset indices. For
        AWR ``weights`` is a float64 array aligned to ``rows``; otherwise ``None``
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
        mask = sidecar.keep_mask(aid)
        for w in np.nonzero(mask)[0]:
            rows.append(base + int(w))
            if is_awr:
                weights.append(awr_weight(sidecar.delta(aid, int(w)), awr_tau, awr_delta))
    if n_missing_ep:
        logging.warning("slb_variant_sampler: %d attempts had no episode in dataset", n_missing_ep)
    rows_arr = np.asarray(rows, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float64) if is_awr else None
    return rows_arr, weights_arr


class RowSampler(torch.utils.data.Sampler[int]):
    """Sample flat rows uniformly (hard filters/vanilla) or by weight (AWR).

    Unlike ``WeightedRandomSampler`` -- which yields positions into its weight
    vector -- this yields the actual flat dataset rows, so the restricted index
    space maps straight onto the openpi dataset.
    """

    def __init__(
        self,
        rows: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        if len(rows) == 0:
            raise ValueError("RowSampler got an empty row set; check sidecar/join_index/episode_from")
        self._rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        self._weights = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self._num_samples = int(num_samples) if num_samples is not None else len(self._rows)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        self._epoch += 1
        if self._weights is None:
            perm = torch.randperm(len(self._rows), generator=gen)
            if self._num_samples > len(perm):
                extra = torch.randint(len(self._rows), (self._num_samples - len(perm),), generator=gen)
                pos = torch.cat([perm, extra])
            else:
                pos = perm[: self._num_samples]
        else:
            pos = torch.multinomial(self._weights, self._num_samples, replacement=True, generator=gen)
        yield from self._rows[pos].tolist()


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


def build_sampler(dataset, data_config, *, seed: int = 0) -> RowSampler:
    """Build the variant RowSampler for a transformed SLB dataset."""
    from benchmarks.dataloader.join_index import JoinIndex
    from benchmarks.dataloader.sidecar_reader import VariantSidecar

    variant = data_config.slb_variant
    sidecar = VariantSidecar.load(
        int(data_config.slb_task_id), variant, sidecar_root=data_config.slb_sidecar_root
    )
    join_index = JoinIndex.from_manifest(data_config.slb_manifest_path)
    episode_from = episode_from_offsets(dataset)
    rows, weights = plan_indices(
        episode_from,
        sidecar,
        join_index,
        variant,
        awr_tau=data_config.awr_tau,
        awr_delta=data_config.awr_delta,
    )
    logging.info(
        "slb_variant_sampler[%s]: %d rows (weighted=%s) over %d episodes",
        variant,
        len(rows),
        weights is not None,
        len(episode_from),
    )
    return RowSampler(rows, weights, seed=seed)
