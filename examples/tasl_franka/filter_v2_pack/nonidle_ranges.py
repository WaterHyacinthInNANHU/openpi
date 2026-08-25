"""
Iterates through the DROID dataset and creates a json mapping from episode unique IDs to ranges of time steps
that should be sampled during training (all others are filtered out).

Filtering logic:
We look for ranges of consecutive steps that contain at most min_idle_len consecutive idle frames
(default to 7 -- as most DROID action-chunking policies run the first 8 actions generated in each chunk, filtering
this way means the policy will not get stuck outputting stationary actions). Additionally, we also only keep non-idle
ranges of length at least min_non_idle_len (default to 16 frames = ~1 second), while also removing the last
filter_last_n_in_ranges frames from the end of each range (as those all correspond to action chunks with many idle actions).

This leaves us with trajectory segments consisting of contiguous, significant movement. Training on this filtered set
yields policies that output fewer stationary actions (i.e., get "stuck" in states less).

Two dataset sources are supported; the filtering logic and constants are identical for both:

  --source rlds     the original DROID RLDS/TFDS dataset. Episode key is
                    "{recording_folderpath}--{file_path}", joint velocities come from
                    action_dict/joint_velocity (rad/s).

  --source lerobot  a LeRobot v2.1 dataset collected on the TASL FR3 bench. Episode key is
                    the episode_index as a string, joint velocities come from the parquet
                    "actions" column. Those actions are joint-velocity commands normalized
                    to [-1, 1], so they are multiplied by --vel-scale to get rad/s before
                    the 1e-3 threshold is applied. The default 0.509 was measured on this
                    bench by regressing the realized joint motion (diff of state[1:8] * fps)
                    against the commanded action (per-joint correlation 0.64-0.78).

Examples:
    python compute_droid_nonidle_ranges.py --source rlds \
        --builder-dir /path/to/droid --out /path/to/droid_sample_ranges.json

    python compute_droid_nonidle_ranges.py --source lerobot \
        --repo-id franka/tasl_fr3_10task_250ep \
        --out $HF_LEROBOT_HOME/franka/tasl_fr3_10task_250ep/meta/nonidle_ranges.json

The resulting json is consumed via `DataConfig.filter_dict_path` (both the RLDS and the
LeRobot data paths honour it).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

min_idle_len = 7  # If more than this number of consecutive idle frames, filter all of them out
min_non_idle_len = 16  # If fewer than this number of consecutive non-idle frames, filter all of them out
filter_last_n_in_ranges = 10  # When using a filter dict, remove this many frames from the end of each range
idle_threshold = 1e-3  # rad/s -- a step is idle if no joint velocity changed by more than this


def nonidle_ranges(joint_velocities: np.ndarray, protect: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Return the [start, end) ranges of `joint_velocities` that should be kept.

    `joint_velocities` is (T, n_joints) in rad/s.
    `protect` (optional, (T,) bool): frames that must never be treated as idle,
    e.g. a window around gripper open/close events (arm still, gripper moving).
    """
    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < idle_threshold, axis=1)]
    )
    if protect is not None:
        is_idle_array &= ~np.asarray(protect, dtype=bool)

    # Find what steps go from idle to non-idle and vice-versa
    is_idle_padded = np.concatenate(
        [[False], is_idle_array, [False]]
    )  # Start and end with False, so idle at first step is a start of motion

    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]  # +1 transitions --> going from idle to non-idle
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]  # -1 transitions --> going from non-idle to idle

    # Find which steps correspond to idle segments of length at least min_idle_len
    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    # Get all non-idle ranges of at least min_non_idle_len
    # Same logic as above, but for keep_mask, allowing us to filter out contiguous ranges of length < min_non_idle_len
    keep_padded = np.concatenate([[False], keep_mask, [False]])

    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]  # +1 transitions --> going from filter out to keep
    keep_true_ends = np.where(keep_diff == -1)[0]  # -1 transitions --> going from keep to filter out

    # Find which steps correspond to non-idle segments of length at least min_non_idle_len
    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    ranges = [(int(s), int(e) - filter_last_n_in_ranges) for s, e in zip(keep_true_starts, keep_true_ends, strict=True)]
    return [(s, e) for s, e in ranges if e > s]


def run_rlds(args) -> dict[str, list[tuple[int, int]]]:
    import tensorflow as tf  # noqa: PLC0415
    import tensorflow_datasets as tfds  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to the GPU you want to use, or leave empty for CPU

    builder = tfds.builder_from_directory(builder_dir=args.builder_dir)  # path to `droid` dir (not its parent)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    tf.data.experimental.ignore_errors(ds)

    keep_ranges_map: dict[str, list[tuple[int, int]]] = {}
    if Path(args.out).exists():
        with Path(args.out).open("r") as f:
            keep_ranges_map = json.load(f)
        print(f"Resuming from {len(keep_ranges_map)} episodes already processed")

    for ep_idx, ep in enumerate(tqdm(ds)):
        recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
        file_path = ep["episode_metadata"]["file_path"].numpy().decode()

        key = f"{recording_folderpath}--{file_path}"
        if key in keep_ranges_map:
            continue

        joint_velocities = np.array([step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]])
        keep_ranges_map[key] = nonidle_ranges(joint_velocities)

        if ep_idx % 1000 == 0:
            with Path(args.out).open("w") as f:
                json.dump(keep_ranges_map, f)

    return keep_ranges_map


def run_lerobot(args) -> dict[str, list[tuple[int, int]]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    root = args.root or os.environ.get("HF_LEROBOT_HOME")
    if root is None:
        raise ValueError("Pass --root or set HF_LEROBOT_HOME")
    data_dir = Path(root) / args.repo_id
    files = sorted(glob.glob(str(data_dir / "data" / "chunk-*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}/data/chunk-*/")

    fps = None
    if args.vel_source == "state":
        with (data_dir / "meta" / "info.json").open() as fi:
            fps = float(json.load(fi)["fps"])

    keep_ranges_map: dict[str, list[tuple[int, int]]] = {}
    total = kept = 0
    for f in files:
        episode_index = int(Path(f).stem.split("_")[1])
        table = pq.read_table(f, columns=["actions", "state"])
        actions = np.array(table.column("actions").to_pylist(), dtype=float)
        if args.vel_source == "action":
            # actions[:, :7] are joint-velocity commands normalized to [-1, 1]; scale back to rad/s.
            joint_velocities = actions[:, :7] * args.vel_scale
        else:
            # Measured joint velocity from the recorded state ([grip, q0..q6]): diff(q) * fps.
            # v[t] is the motion between t and t+1, i.e. the response to cmd[t] (lag ~1 frame);
            # the last row is repeated so the array stays (T, 7).
            q = np.array(table.column("state").to_pylist(), dtype=float)[:, 1:8]
            v = np.diff(q, axis=0) * fps
            joint_velocities = np.vstack([v, v[-1:]]) if len(v) else np.zeros_like(q)
        protect = None
        if args.grip_guard > 0:
            # Never treat frames near a gripper open/close event as idle (arm still, gripper moving).
            grip_event = np.hstack([[False], np.abs(np.diff(actions[:, 7])) > args.grip_delta])
            protect = grip_event.copy()
            for k in range(1, args.grip_guard + 1):
                protect[k:] |= grip_event[:-k]
                protect[:-k] |= grip_event[k:]
        ranges = nonidle_ranges(joint_velocities, protect=protect)
        keep_ranges_map[str(episode_index)] = ranges
        total += len(actions)
        kept += sum(e - s for s, e in ranges)

    empty = sum(1 for v in keep_ranges_map.values() if not v)
    n_ranges = [len(v) for v in keep_ranges_map.values()]
    print(f"  episodes         : {len(keep_ranges_map)} (fully filtered out: {empty})")
    print(f"  frames           : {total} -> keep {kept} ({kept / total * 100:.1f}%), drop {total - kept}")
    print(f"  ranges per episode: median {int(np.median(n_ranges))}, max {max(n_ranges)}")
    return keep_ranges_map


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["rlds", "lerobot"], default="rlds")
    p.add_argument("--out", required=True, help="where to write the json")
    # rlds
    p.add_argument("--builder-dir", help="[rlds] path to the `droid` tfds directory (not its parent)")
    # lerobot
    p.add_argument("--repo-id", help="[lerobot] e.g. franka/tasl_fr3_10task_250ep")
    p.add_argument("--root", help="[lerobot] dataset root; defaults to $HF_LEROBOT_HOME")
    p.add_argument(
        "--vel-scale",
        type=float,
        default=0.509,
        help="[lerobot] rad/s per unit of the normalized action (measured on the TASL FR3 bench)",
    )
    p.add_argument(
        "--vel-source",
        choices=["action", "state"],
        default="action",
        help="[lerobot] 'action': commanded velocity (actions[:, :7] * vel_scale, the original behaviour); "
        "'state': measured velocity diff(state[:, 1:8]) * fps (use when commands are noisy / never zero)",
    )
    p.add_argument(
        "--grip-guard",
        type=int,
        default=0,
        help="[lerobot] protect +/- this many frames around gripper events from being marked idle (0 = off)",
    )
    p.add_argument(
        "--grip-delta",
        type=float,
        default=0.02,
        help="[lerobot] |delta gripper action| above this counts as a gripper event (for --grip-guard)",
    )
    args = p.parse_args()

    if args.source == "rlds":
        if not args.builder_dir:
            p.error("--source rlds requires --builder-dir")
        keep_ranges_map = run_rlds(args)
    else:
        if not args.repo_id:
            p.error("--source lerobot requires --repo-id")
        keep_ranges_map = run_lerobot(args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        json.dump(keep_ranges_map, f)
    print(f"wrote {args.out}")
    print(
        f"  constants: min_idle_len={min_idle_len} min_non_idle_len={min_non_idle_len} "
        f"filter_last_n_in_ranges={filter_last_n_in_ranges} idle_threshold={idle_threshold}"
    )
    if args.source == "lerobot":
        print(f"  lerobot: vel_source={args.vel_source} grip_guard={args.grip_guard} grip_delta={args.grip_delta}")


if __name__ == "__main__":
    main()
