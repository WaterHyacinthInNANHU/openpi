"""Multi-task pretraining dataset assembly (online, openpi venv).

Two concerns live here, split by dependency weight so the risky part is testable off-box:

  * ROW SELECTION -- read each per-task __droid8d sub-dataset's episode offsets from its
    ``meta`` parquet, join the non-idle sample-ranges into a single flat row list. Touches only
    json + pandas + numpy, so it is unit-tested under the offline venv without lerobot or a GPU.
  * DATASET CONSTRUCTION -- build one ``LeRobotDataset`` per root and concatenate them. Needs
    lerobot (lazy-imported); exercised only on the box.

Both read the SAME ``roots_index`` in the SAME (task-id-sorted) order, so the flat index space
the sampler plans over matches the ``ConcatDataset`` the loader iterates. ``build_pretrain_concat_dataset``
asserts each sub-dataset's ``len()`` equals the total its ``meta`` reported, failing loudly if
that invariant -- the one seam the offline tests cannot cover -- ever breaks.

Offline-safe: NEVER import torch / lerobot at module scope (the offline venv lacks them).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from openpi.training import pretrain_sampler


def read_episode_offsets(root: str | pathlib.Path) -> tuple[dict[int, tuple[int, int]], int]:
    """``{episode_index: (local_start_row, length)}`` and total row count for one sub-dataset.

    Read from ``meta/episodes/**/*.parquet`` rather than by loading the dataset, so the sampler
    can place every sub-dataset in the concat index space without paying a full LeRobot load.
    """
    import glob

    import pandas as pd

    root = pathlib.Path(root)
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no meta/episodes parquet under {root}")
    ep = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    episodes = {
        int(e): (int(f), int(n))
        for e, f, n in zip(ep["episode_index"], ep["dataset_from_index"], ep["length"], strict=True)
    }
    total = int(ep["length"].sum())
    return episodes, total


CORPUS_ROOT_ENV = "AXIS_PRETRAIN_CORPUS_ROOT"


def _ordered_roots(roots_index: str | pathlib.Path) -> list[tuple[int, str]]:
    """``(task_id, root_path)`` pairs from the roots-index JSON, sorted by task id.

    Sorting (not JSON insertion order) makes the concat index space deterministic, so the
    dataset build and the row planner agree regardless of how the index file was written.

    TWO FORMATS, because the corpus outlived the first one.

    * **v2 (write this)** -- ``{"version": 2, "base": "../derived224", "roots": {"1000":
      "task_1000"}}``. Roots are RELATIVE and ``base`` is itself relative to the index file, so
      the whole corpus tree can be copied to another machine, or moved on this one, and still
      resolve. ``$AXIS_PRETRAIN_CORPUS_ROOT`` overrides ``base`` when a copy is split apart.
    * **v1 (still read)** -- a bare ``{"1000": "/abs/path/task_1000"}`` mapping.

    v1 embeds absolute paths, which pinned an index to the machine that built it: shipping the
    identical shards to another box produced a file naming directories that did not exist there,
    and the corpus had to be re-indexed by hand. Reading it is kept because a training run in
    flight is using one, and changing what an already-launched run reads is not a refactor.
    """
    import os

    path = pathlib.Path(roots_index)
    data = json.loads(path.read_text())

    if isinstance(data, dict) and "roots" in data:
        mapping = data["roots"]
        override = os.environ.get(CORPUS_ROOT_ENV)
        base = pathlib.Path(override) if override else path.parent / str(data.get("base", "."))
    else:
        mapping = data          # v1: values are absolute and self-sufficient
        base = None

    out: list[tuple[int, str]] = []
    for key, value in mapping.items():
        root = pathlib.Path(str(value))
        if root.is_absolute():
            resolved = root
        elif base is None:
            raise ValueError(
                f"{path} has a relative root {value!r} but no 'base' key. A bare mapping is "
                f"read as the legacy absolute-path format; add \"version\": 2 and \"base\", or "
                f"set ${CORPUS_ROOT_ENV}."
            )
        else:
            resolved = base / root
        out.append((int(key), str(resolved)))
    return sorted(out, key=lambda kv: kv[0])


def plan_rows_and_weights_from_roots(
    roots_index: str | pathlib.Path,
    ranges_path: str | pathlib.Path,
    weights_path: str | pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat training rows AND their AWR weights, resolved in ONE pass.

    Returned together, from the same loop over the same sub-datasets, because the flat index
    space depends on the task order and on every episode length across ~476 sub-datasets. A
    weights artifact keyed by flat index would have to agree with this loop exactly, and would
    fail SILENTLY if one episode were added or dropped, or if the offline builder sorted task
    ids as strings ("1000" < "800") -- applying the right weights to the wrong frames, with no
    error and no log line.

    The artifact is therefore keyed by `episode_key(task_id, episode_index)`, the same key this
    function computes, and the join happens here.

    A row whose episode is absent from the artifact, or whose weight is non-finite, RAISES. The
    alternative -- defaulting to 0.0, as the SLB wrapper does -- would leave those rows in the
    batch contributing nothing, which is a training-set reduction disguised as a weighting.
    """


    rows = plan_rows_from_roots(roots_index, ranges_path)

    payload = json.loads(pathlib.Path(weights_path).read_text())
    by_key = payload["weights"] if isinstance(payload, dict) and "weights" in payload else payload

    # Rebuild the same flat space the rows were planned in, then place each episode's weights.
    entries: list[tuple[int, dict[int, tuple[int, int]], int]] = []
    for task_id, root in _ordered_roots(roots_index):
        episodes, total = read_episode_offsets(root)
        entries.append((task_id, episodes, total))
    subdatasets, total_rows = pretrain_sampler.build_subdatasets(entries)

    flat = np.full(int(total_rows), np.nan, dtype=np.float32)
    missing: list[str] = []
    for sub in subdatasets:
        for ep_index, (local_start, length) in sub.episodes.items():
            key = pretrain_sampler.episode_key(sub.task_id, ep_index)
            w = by_key.get(key)
            if w is None:
                missing.append(key)
                continue
            if len(w) != length:
                raise ValueError(
                    f"{key}: weights have {len(w)} entries but the episode has {length} rows. "
                    f"The weights artifact was built against a different corpus."
                )
            base = sub.global_base + local_start
            flat[base : base + length] = np.asarray(w, dtype=np.float32)

    if missing:
        raise ValueError(
            f"{len(missing)} episodes have no weights (e.g. {missing[:5]}). Every episode the "
            f"loader can sample must carry one, even if that weight is the neutral 1.0 -- a "
            f"missing key would otherwise be filled by a reader default and silently reweight "
            f"training."
        )

    weights = flat[rows]
    if not np.isfinite(weights).all():
        bad = int(np.sum(~np.isfinite(weights)))
        raise ValueError(
            f"{bad} planned rows have no finite weight. Rows and weights disagree about the "
            f"corpus; refusing to train on a partially-weighted objective."
        )
    return rows, weights


def plan_rows_from_roots(roots_index: str | pathlib.Path, ranges_path: str | pathlib.Path) -> np.ndarray:
    """Flat row indices to train on, across every per-task sub-dataset in ``roots_index``.

    Concatenates the sub-datasets in task-id order (matching ``build_pretrain_concat_dataset``)
    and keeps only rows whose within-episode frame falls in a non-idle range from ``ranges_path``.
    """
    if roots_index is None or ranges_path is None:
        raise ValueError(
            "pretraining needs both AXIS_PRETRAIN_ROOTS_INDEX and AXIS_PRETRAIN_RANGES set "
            "(build them with axis_data.build_pretrain_datasets / pretrain_ranges); "
            f"got roots_index={roots_index!r}, ranges_path={ranges_path!r}"
        )
    entries: list[tuple[int, dict[int, tuple[int, int]], int]] = []
    for task_id, root in _ordered_roots(roots_index):
        episodes, total = read_episode_offsets(root)
        entries.append((task_id, episodes, total))
    subdatasets, _ = pretrain_sampler.build_subdatasets(entries)
    ranges = json.loads(pathlib.Path(ranges_path).read_text())
    rows = pretrain_sampler.plan_pretrain_rows(subdatasets, ranges)
    return np.asarray(rows, dtype=np.int64)


def build_pretrain_concat_dataset(data_config, action_horizon: int):
    """Concatenate the per-task ``__droid8d`` LeRobot sub-datasets into one flat dataset.

    Sub-datasets are built and concatenated in task-id order -- the SAME order
    ``plan_rows_from_roots`` plans rows over -- so the sampler's flat indices land on the right
    frames. Each sub-dataset's ``len()`` is asserted against the total its ``meta`` reported;
    that invariant is what lets the offline-planned index space match this concat, and it is
    the one seam the offline tests cannot cover.

    Lazy-imports lerobot/torch: this must stay importable under the offline venv.
    """
    from lerobot.datasets import lerobot_dataset
    import torch

    from openpi import transforms as _transforms

    datasets = []
    for _task_id, root in _ordered_roots(data_config.pretrain_roots_index):
        _episodes, total = read_episode_offsets(root)
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=root)
        # Same sub-millisecond frame-timestamp jitter tolerance the SLB path uses (v3.0 default
        # tolerance_s=1e-4 is too tight for our re-rendered videos).
        tolerance_s = 0.25 / dataset_meta.fps
        ds = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            root=root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
            tolerance_s=tolerance_s,
        )
        if len(ds) != total:
            raise RuntimeError(
                f"pretrain sub-dataset {root}: len(LeRobotDataset)={len(ds)} != meta total {total}; "
                "the sampler's flat index space would be misaligned. Rebuild the roots index / dataset."
            )
        if data_config.prompt_from_task:
            tasks = {int(ti): str(task) for task, ti in dataset_meta.tasks["task_index"].items()}
            from openpi.training.data_loader import TransformedDataset

            ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(tasks)])
        datasets.append(ds)
    if not datasets:
        raise ValueError(f"pretrain_roots_index named no datasets: {data_config.pretrain_roots_index}")
    return torch.utils.data.ConcatDataset(datasets)
