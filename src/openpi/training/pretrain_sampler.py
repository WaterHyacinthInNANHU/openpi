"""Row planning for the multi-task AXIS pretraining loader (online, openpi venv).

WHY THIS FILE EXISTS
    The SLB path (``slb_variant_sampler``) is single-task: one ``.npz`` sidecar and one
    ``task_manifest.json`` join select the rows of ONE task's LeRobot dataset. Pretraining
    trains over ~182 tasks at once. Those live as separate per-task ``__droid8d`` LeRobot
    sub-datasets, concatenated into one flat index space; a row is kept iff its within-episode
    frame index falls inside a precomputed non-idle range (``pretrain_ranges``).

    The heart of that filter is a pure index computation with no jax/lerobot dependency, kept
    here so it is unit-testable without a real dataset or GPU. The concat construction and the
    LeRobot episode-offset extraction that FEED this function live in ``data_loader``.

The ranges dict is openpi's own sample-ranges artifact, keyed ``task_<task_id>--<episode_index>``
with half-open ``[start, end)`` frame ranges -- byte-identical to what
``benchmarks.dataloader.pretrain_ranges`` emits. Ranges are clamped to each episode's length.
"""

from __future__ import annotations

import dataclasses


def episode_key(task_id: int, episode_index: int) -> str:
    """Stable episode identity used by the ranges file -- must match pretrain_ranges."""
    return f"task_{task_id}--{episode_index}"


@dataclasses.dataclass(frozen=True)
class SubDataset:
    """One per-task ``__droid8d`` LeRobot sub-dataset's placement in the concat.

    ``global_base`` is this sub-dataset's flat row offset in the concatenated index space
    (cumulative length of all earlier sub-datasets). ``episodes`` maps each episode index to
    its ``(local_start_row, length)`` within THIS sub-dataset.
    """

    task_id: int
    global_base: int
    episodes: dict[int, tuple[int, int]]


def build_subdatasets(
    entries: list[tuple[int, dict[int, tuple[int, int]], int]],
) -> tuple[list[SubDataset], int]:
    """Assign cumulative ``global_base`` offsets to per-task sub-datasets.

    ``entries`` is ordered ``(task_id, episodes, length)`` per sub-dataset, where ``length``
    is that sub-dataset's total row count (``len(LeRobotDataset)``) and ``episodes`` maps
    episode index to ``(local_start_row, episode_length)`` WITHIN that sub-dataset. Sub-dataset
    order must match the ``torch.utils.data.ConcatDataset`` order, since that is what defines
    the flat index space. Returns the ``SubDataset`` list plus the total concatenated length.
    """
    subs: list[SubDataset] = []
    acc = 0
    for task_id, episodes, length in entries:
        subs.append(SubDataset(task_id=int(task_id), global_base=acc, episodes=dict(episodes)))
        acc += int(length)
    return subs, acc


def plan_pretrain_rows(subdatasets: list[SubDataset], ranges: dict[str, list[list[int]]]) -> list[int]:
    """Flat global row indices kept by the sample-ranges filter.

    A row is kept iff its within-episode frame index ``f`` falls in some half-open
    ``[s, e)`` range for that ``(task_id, episode_index)``. Ranges are clamped to
    ``[0, length)``; an episode with no key in ``ranges`` contributes nothing.
    """
    rows: list[int] = []
    for sub in subdatasets:
        for ep_idx, (local_start, length) in sub.episodes.items():
            key = episode_key(sub.task_id, ep_idx)
            base = sub.global_base + local_start
            for s, e in ranges.get(key, []):
                lo = max(0, int(s))
                hi = min(int(e), length)
                rows.extend(base + f for f in range(lo, hi))
    return rows
