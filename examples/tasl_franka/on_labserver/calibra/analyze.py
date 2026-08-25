import json, sys
import pyarrow.parquet as pq
from pathlib import Path

def ep_frames(ds):
    out = {}
    for f in sorted(Path(ds).glob("data/chunk-000/episode_*.parquet")):
        out[int(f.stem.split("_")[1])] = pq.read_table(f).num_rows
    return out

frames = ep_frames("ds_bridge")
total = sum(frames.values())
print("total frames:", total)

for label, kept in [("keep0.5", [1,3,6,7,10,12,13,14])]:
    kf = sum(frames[e] for e in kept)
    print(f"{label}: kept episodes {sorted(kept)} | frames {kf} ({kf/total:.1%})")

d = json.load(open("prune_k075.json"))
print("keep0.75: n_original", d["n_original"], "n_kept", d["n_kept"], "frac", d["keep_fraction_actual"])
print("  quality_fail:", d.get("quality_fail_ids"))
print("  diversity_pruned:", d.get("diversity_pruned_ids"))
kept75 = [str(i) for i in range(d["n_original"]) if str(i) not in set(d.get("diversity_pruned_ids", []))]
kf = sum(frames[int(e)] for e in kept75)
print(f"  kept: {sorted(map(int, kept75))} | frames {kf} ({kf/total:.1%})")
