import pyarrow.parquet as pq
from pathlib import Path

DS = Path("/data1/Franka_RealRobot/lerobot_home/franka/test_finetune_idlefiltered")
ep2task = {}
ep2frames = {}
for f in sorted(DS.glob("data/chunk-000/episode_*.parquet")):
    t = pq.read_table(f, columns=["task_index", "episode_index"])
    ep = int(f.stem.split("_")[1])
    ep2task[ep] = t.column("task_index")[0].as_py()
    ep2frames[ep] = t.num_rows

for label, kept in [
    ("ALL16", list(ep2task)),
    ("KEEP50", [1, 3, 6, 7, 10, 12, 13, 14]),
    ("KEEP75", [0, 1, 3, 4, 6, 7, 10, 11, 12, 13, 14, 15]),
]:
    t0 = [e for e in kept if ep2task[e] == 0]
    t1 = [e for e in kept if ep2task[e] == 1]
    f0 = sum(ep2frames[e] for e in t0)
    f1 = sum(ep2frames[e] for e in t1)
    print(
        f"{label}: task0={len(t0)}eps/{f0}f  task1={len(t1)}eps/{f1}f  "
        f"| total {sum(ep2frames[e] for e in kept)}f"
    )
