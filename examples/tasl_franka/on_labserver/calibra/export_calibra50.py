"""Export Calibra keep0.5 coreset (8 episodes) as a new LeRobot dataset.

Keeps episodes [1,3,6,7,10,12,13,14] from franka/test_finetune_idlefiltered
(4067 -> 2121 frames), re-numbered 0..7 to match LeRobot split semantics
("0:8" = episode indices 0..7). Parquet files keep the embedded huggingface
metadata; episode_index and global `index` columns are re-stamped; meta
files (info/episodes/episodes_stats) rewritten accordingly.
"""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/data1/Franka_RealRobot/lerobot_home/franka/test_finetune_idlefiltered")
DST = Path("/data1/Franka_RealRobot/lerobot_home/franka/test_finetune_idlefiltered_calibra50")
KEEP = [1, 3, 6, 7, 10, 12, 13, 14]
NEW_ID = {old: new for new, old in enumerate(sorted(KEEP))}  # {1:0, 3:1, 6:2, ...}

if DST.exists():
    shutil.rmtree(DST)
(DST / "data/chunk-000").mkdir(parents=True)
(DST / "meta").mkdir(parents=True)

files = sorted(SRC.glob("data/chunk-000/episode_*.parquet"))
kept_files = [f for f in files if int(f.stem.split("_")[1]) in KEEP]
assert len(kept_files) == 8

global_idx = 0
total = 0
for f in kept_files:
    old_ep = int(f.stem.split("_")[1])
    new_ep = NEW_ID[old_ep]
    t = pq.read_table(f)
    n = t.num_rows
    t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                     pa.array([new_ep] * n, type=pa.int64()))
    t = t.set_column(t.schema.get_field_index("index"), "index",
                     pa.array(range(global_idx, global_idx + n), type=pa.int64()))
    out = DST / "data/chunk-000" / f"episode_{new_ep:06d}.parquet"
    pq.write_table(t, out)
    global_idx += n
    total += n
print(f"copied {len(kept_files)} episodes -> {NEW_ID}, {total} frames (index 0..{global_idx - 1})")

info = json.load(open(SRC / "meta/info.json"))
info["total_episodes"] = 8
info["total_frames"] = total
info["splits"] = {"train": "0:8"}
json.dump(info, open(DST / "meta/info.json", "w"), indent=4)

for name in ("episodes.jsonl", "episodes_stats.jsonl"):
    rows = [json.loads(l) for l in (SRC / "meta" / name).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if int(r["episode_index"]) in KEEP]
    for r in rows:
        r["episode_index"] = NEW_ID[int(r["episode_index"])]
    rows.sort(key=lambda r: r["episode_index"])
    (DST / "meta" / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"meta/{name}: {len(rows)} rows")

shutil.copy(SRC / "meta/tasks.jsonl", DST / "meta/tasks.jsonl")
if (SRC / "meta/layout.json").exists():
    shutil.copy(SRC / "meta/layout.json", DST / "meta/layout.json")
print("meta done")
