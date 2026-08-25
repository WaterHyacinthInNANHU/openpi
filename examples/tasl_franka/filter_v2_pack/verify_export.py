"""Check an exported dataset against its source: index continuity, meta consistency, and byte-level
row equality (images / state / actions / task_index / is_success) via meta/source_segments.json.

    python verify_export.py --src <source dataset dir> --out <exported dataset dir>
"""
import argparse, collections, glob, json, numpy as np, pyarrow.parquet as pq
p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True); a = p.parse_args()
SRC, OUT = a.src, a.out
info = json.load(open(OUT + "/meta/info.json")); fps = float(info["fps"])
eps = [json.loads(l) for l in open(OUT + "/meta/episodes.jsonl")]
stats = [json.loads(l) for l in open(OUT + "/meta/episodes_stats.jsonl")]
prov = json.load(open(OUT + "/meta/source_segments.json"))["episodes"]
files = sorted(glob.glob(OUT + "/data/chunk-*/*.parquet"))
print(f"info: episodes {info['total_episodes']} frames {info['total_frames']} | episodes.jsonl {len(eps)} | stats {len(stats)} | parquet {len(files)} | prov {len(prov)}")
cmp_cols = [c for c in ("image", "extra_view_image", "state", "actions", "task_index", "is_success") if c in info["features"]]
ok = True; tot = 0; last_idx = -1; src_cache = {}
for f in files:
    t = pq.read_table(f); n = t.num_rows; ep = int(t.column("episode_index")[0].as_py()); tot += n
    fi = np.array(t.column("frame_index").to_pylist()); ts = np.array(t.column("timestamp").to_pylist())
    ix = np.array(t.column("index").to_pylist()); d = np.array(t.column("done").to_pylist())
    c1 = np.array_equal(fi, np.arange(n)); c2 = np.allclose(ts, np.arange(n) / fps, atol=1e-4)
    c3 = ix[0] == last_idx + 1 and np.array_equal(ix, np.arange(ix[0], ix[0] + n)); c4 = (d.sum() == 1 and d[-1])
    c5 = eps[ep]["length"] == n and stats[ep]["episode_index"] == ep
    last_idx = ix[-1]
    pr = prov[str(ep)]; se = pr["src_episode"]
    if se not in src_cache: src_cache = {se: pq.read_table(f"{SRC}/data/chunk-{se//1000:03d}/episode_{se:06d}.parquet")}
    s = src_cache[se].slice(pr["start"], n)
    c6 = all(s.column(k).to_pylist() == t.column(k).to_pylist() for k in cmp_cols)
    c7 = bool(t.schema.metadata and b"huggingface" in t.schema.metadata)
    if not all([c1, c2, c3, c4, c5, c6, c7]): ok = False; print("FAIL ep", ep, [c1, c2, c3, c4, c5, c6, c7])
print(f"frames sum {tot} == info {info['total_frames']}: {tot == info['total_frames']} | all per-episode checks: {ok}")
src_eps = set(v["src_episode"] for v in prov.values())
n_src = json.load(open(SRC + "/meta/info.json"))["total_episodes"]
print("segments per src ep: max", max(sum(1 for v in prov.values() if v["src_episode"] == e) for e in src_eps),
      "| src eps with 0 segments:", n_src - len(src_eps), "(of", n_src, "; 0 segments is expected when running with --limit)")
L = [e["length"] for e in eps]; print("episode length min/median/max:", min(L), int(np.median(L)), max(L))
c = collections.Counter(e["tasks"][0] for e in eps); print("episodes per task:", sorted(c.values()))
print("VERIFY", "OK" if ok and tot == info["total_frames"] else "FAILED")
