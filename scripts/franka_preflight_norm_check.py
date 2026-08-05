"""Preflight: do the RLinf FR3 actions/state fit the DROID norm stats?

The pi05_droid_finetune recipe reuses the original DROID norm stats
(quantile normalization). DROID recorded raw joint velocities in rad/s,
while the RLinf bench records velocities normalized to [-1, 1] — so before
training we check where our data lands after DROID quantile normalization:
    norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1        (openpi Normalize)

Verdict: if the bulk of our normalized actions stays within roughly [-1.5, 1.5],
reusing DROID stats is fine; if it blows far outside, use
runme_cal_stats_franka.sh and switch the config to local stats instead.

Run via: bash runme_preflight_franka.sh (sets HF_LEROBOT_HOME, uses uv).
"""

import glob
import os

import numpy as np
import pandas as pd

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize

DATASET_DIR = os.path.join(
    os.environ.get("HF_LEROBOT_HOME", "/data1/Franka_RealRobot/lerobot_home"),
    "franka/test_finetune",
)
DROID_ASSETS = "gs://openpi-assets/checkpoints/pi05_droid/assets/droid"


def quantile_norm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    q01, q99 = q01[: x.shape[-1]], q99[: x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def report(name: str, normed: np.ndarray) -> bool:
    p01, p50, p99 = np.percentile(normed, [1, 50, 99], axis=0)
    frac_out = (np.abs(normed) > 1.5).mean(axis=0)
    print(f"\n=== {name}: normalized with DROID quantile stats ===")
    print("dim  p01     p50     p99     frac|>1.5|")
    for i in range(normed.shape[1]):
        print(f"{i:>3}  {p01[i]:>6.2f}  {p50[i]:>6.2f}  {p99[i]:>6.2f}  {frac_out[i]:>8.3f}")
    worst = frac_out.max()
    print(f"worst-dim fraction outside [-1.5, 1.5]: {worst:.3f}")
    return worst < 0.05


def main():
    stats = _normalize.load(_download.maybe_download(DROID_ASSETS))
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "data/chunk-*/episode_*.parquet")))
    assert files, f"no parquet under {DATASET_DIR}"
    actions, states = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["actions", "state"])
        actions.append(np.stack(df["actions"].to_numpy()))
        states.append(np.stack(df["state"].to_numpy()))
    actions = np.concatenate(actions)  # [N, 8] = [dq0..dq6 in [-1,1], grip in [0,1]]
    states = np.concatenate(states)  # [N, 8] = [grip, q0..q6]
    # DroidInputs feeds state as [q0..q6, grip]; reorder before applying stats.
    states = np.concatenate([states[:, 1:8], states[:, 0:1]], axis=1)
    print(f"{len(files)} episodes, {len(actions)} frames from {DATASET_DIR}")

    ok_a = report("actions", quantile_norm(actions, stats["actions"].q01, stats["actions"].q99))
    ok_s = report("state", quantile_norm(states, stats["state"].q01, stats["state"].q99))

    print()
    if ok_a and ok_s:
        print("VERDICT: REUSE DROID STATS — distributions fit the DROID quantile range.")
    else:
        print(
            "VERDICT: RECOMMEND OWN STATS — run runme_cal_stats_franka.sh and drop the\n"
            "gs:// assets override in the pi05_droid_franka_lora config."
        )


if __name__ == "__main__":
    main()
