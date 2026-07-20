"""Patch LeRobot's _query_hf_dataset column-first fast path, which is a pessimisation.

WHY THIS EXISTS
    lerobot 0.4.4 `LeRobotDataset._query_hf_dataset` tries a "fast" column-first
    lookup before falling back to a row-first one:

        try:    torch.stack(self.hf_dataset[key][relative_indices])
        except: torch.stack(self.hf_dataset[relative_indices][key])

    On HF `datasets` 3.6.0 the first form ALWAYS raises TypeError -- `ds[key]`
    returns a fully materialised Python list, and a list cannot be indexed by a
    list. But materialising it first pushes the WHOLE column through
    `hf_transform_to_torch`, so every training sample pays an O(total_frames)
    cost whose result is then discarded.

    Measured on the task-1424 __droid8d dataset (268 131 rows, 17-row query):
        dead try branch : 3.164 s   (raises TypeError, result thrown away)
        fallback path   : 0.0029 s  (the one that actually produces the batch)
        -> 1078x waste, per sample.

    At batch 32 / 8 workers this accounted for most of the ~15.4 s/step observed
    in SLURM job 26490164 -- i.e. the SLB training bottleneck was this line, not
    MP4 decode as originally diagnosed.

THE FIX
    Swap the order so the row-first form (the one that works) is tried first.
    The column-first form is KEPT as the fallback, so other lerobot/datasets
    version combinations that genuinely benefit from it still work. This is a
    pure reordering -- both expressions return identical values.

    Idempotent and self-verifying: safe to re-run, and it fails loudly rather
    than silently no-op'ing if upstream changes the code shape.

USAGE
    ./.venv/bin/python patch_lerobot_query.py           # apply (from openpi/)
    ./.venv/bin/python patch_lerobot_query.py --check   # report status only

    Re-run after any `uv sync` or `setup_lerobot_v3.sh`, which reinstall lerobot
    and revert this. setup_lerobot_v3.sh calls it automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

BEFORE = """            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
"""

AFTER = """            # AXIS PATCH (patch_lerobot_query.py): row-first, NOT column-first.
            # `self.hf_dataset[key]` materialises the entire column through
            # hf_transform_to_torch before raising TypeError on datasets>=3.x,
            # costing O(total_frames) per sample for a discarded result
            # (measured 3.164s vs 0.0029s = 1078x on a 268k-row dataset).
            try:
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
"""


def target_path() -> Path:
    import lerobot.datasets.lerobot_dataset as mod

    return Path(mod.__file__)


def main() -> int:
    check_only = "--check" in sys.argv
    path = target_path()
    src = path.read_text()

    if AFTER in src:
        print(f"OK: already patched -> {path}")
        return 0

    if BEFORE not in src:
        print(
            f"ERROR: could not find the expected _query_hf_dataset block in {path}.\n"
            "Upstream lerobot has changed shape. Re-check whether the column-first\n"
            "pessimisation still exists before assuming this patch is still needed.",
            file=sys.stderr,
        )
        return 1

    if check_only:
        print(f"UNPATCHED (patch would apply cleanly) -> {path}")
        return 1

    path.write_text(src.replace(BEFORE, AFTER, 1))
    assert AFTER in path.read_text(), "patch write-back verification failed"
    print(f"PATCHED: row-first query order -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
