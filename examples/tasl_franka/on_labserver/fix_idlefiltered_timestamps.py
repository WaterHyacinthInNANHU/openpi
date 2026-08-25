"""Fix LeRobot timestamp sync for the idle-filtered dataset.

The idle filter dropped frames but kept absolute collection timestamps,
leaving gaps (e.g. 1.2s) that fail lerobot check_timestamps_sync
(tolerance 1e-4s). After filtering, frame position IS the time axis:
re-stamp timestamp = arange(n)/fps per episode, and re-stamp the global
index column contiguously (loader builds ep_data_index from it).
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = [
    Path("/data1/Franka_RealRobot/lerobot_home/franka/test_finetune_idlefiltered"),
    Path("/data1/Franka_RealRobot/test-finetune-idlefiltered"),
]
FPS = 15

for ds in TARGETS:
    files = sorted(glob.glob(str(ds / "data/chunk-000" / "episode_*.parquet")))
    if not files:
        print(f"{ds}: no parquet files, skip"); continue
    global_idx = 0
    for f in files:
        df = pd.read_parquet(f)
        n = len(df)
        df["timestamp"] = np.arange(n, dtype=float) / FPS
        df["frame_index"] = np.arange(n)
        df["index"] = np.arange(global_idx, global_idx + n)
        df.to_parquet(f, index=False)
        global_idx += n
    print(f"{ds}: {len(files)} episodes, {global_idx} rows re-stamped")
