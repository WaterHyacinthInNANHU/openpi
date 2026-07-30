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


def _ordered_roots(roots_index: str | pathlib.Path) -> list[tuple[int, str]]:
    """``(task_id, root_path)`` pairs from the roots-index JSON, sorted by task id.

    Sorting (not JSON insertion order) makes the concat index space deterministic, so the
    dataset build and the row planner agree regardless of how the index file was written.
    """
    mapping = json.loads(pathlib.Path(roots_index).read_text())
    return sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda kv: kv[0])


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
