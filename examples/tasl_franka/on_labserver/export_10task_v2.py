"""把 v2 过滤结果物理导出成独立的 LeRobot v2.1 数据集 (原数据集一帧不动)。

每个保留段 -> 一条新 episode; 图像/state/action/task 整行原样拷贝, 只重编
episode_index / frame_index / timestamp / index / done。段的判据与常量直接 import
openpi 官方 compute_droid_nonidle_ranges.py (实测速度 + 夹爪保护 ±5), 尾砍设为 0:
物理导出后段与段之间没有"缺口", chunk 越过段尾时按 LeRobot 常规补末帧, 不需要再砍。
meta/episodes_stats.jsonl 按 lerobot.common.datasets.compute_stats 的算法重算。
meta/source_segments.json 记录 新 episode -> (源 episode, start, end) 便于回溯。
"""
import argparse, glob, io, json, os, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, "/data1/Franka_RealRobot/openpi/examples/droid")
import compute_droid_nonidle_ranges as M
sys.path.insert(0, "/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages")
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices, auto_downsample_height_width

p = argparse.ArgumentParser()
p.add_argument("--src", default="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep")
p.add_argument("--out", default="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_v2")
p.add_argument("--grip-guard", type=int, default=5)
p.add_argument("--grip-delta", type=float, default=0.02)
args = p.parse_args()
assert not os.path.exists(args.out), f"{args.out} 已存在, 不覆盖"
os.makedirs(os.path.join(args.out, "data", "chunk-000")); os.makedirs(os.path.join(args.out, "meta"))
info = json.load(open(os.path.join(args.src, "meta", "info.json"))); fps = float(info["fps"]); feats = info["features"]
M.filter_last_n_in_ranges = 0   # 物理导出: 不砍段尾

def seg_ranges(A, S):
    q = S[:, 1:8]; v = np.diff(q, axis=0) * fps; v = np.vstack([v, v[-1:]]) if len(v) else np.zeros_like(q)
    ge = np.hstack([[False], np.abs(np.diff(A[:, 7])) > args.grip_delta]); pr = ge.copy()
    for k in range(1, args.grip_guard + 1): pr[k:] |= ge[:-k]; pr[:-k] |= ge[k:]
    return M.nonidle_ranges(v, protect=pr)

def ep_stats(tbl, n, ep_out, idx0, tix):
    st = {}
    for name in feats:
        dt = feats[name]["dtype"]
        if dt == "image":
            col = tbl.column(name).to_pylist(); imgs = []
            for i in sample_indices(n):
                im = np.asarray(Image.open(io.BytesIO(col[i]["bytes"])).convert("RGB")).transpose(2, 0, 1)
                imgs.append(auto_downsample_height_width(im))
            s = get_feature_stats(np.stack(imgs), axis=(0, 2, 3), keepdims=True)
            st[name] = {k: (v if k == "count" else np.squeeze(v / 255.0, axis=0)) for k, v in s.items()}
        else:
            if name == "episode_index": arr = np.full(n, ep_out, dtype=np.int64)
            elif name == "index": arr = np.arange(idx0, idx0 + n, dtype=np.int64)
            elif name == "frame_index": arr = np.arange(n, dtype=np.int64)
            elif name == "timestamp": arr = (np.arange(n) / fps).astype(np.float32)
            elif name == "task_index": arr = np.full(n, tix, dtype=np.int64)
            else: arr = np.asarray(tbl.column(name).to_pylist())
            if arr.dtype == bool: arr = arr.astype(np.float32)   # 与原 meta 一致 (bool 当数值)
            st[name] = get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
    return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in st.items()}

ep_out = 0; idx_off = 0; episodes = []; stats_out = []; prov = {}; dropped_frames = 0; src_frames = 0
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(os.path.join(args.src, "meta", "tasks.jsonl"))}
for pqf in sorted(glob.glob(os.path.join(args.src, "data", "chunk-*", "*.parquet"))):
    t = pq.read_table(pqf); schema = t.schema; n_src = t.num_rows; src_frames += n_src
    assert schema.metadata and b"huggingface" in schema.metadata, pqf
    A = np.asarray(t.column("actions").to_pylist(), dtype=float); S = np.asarray(t.column("state").to_pylist(), dtype=float)
    src_ep = int(t.column("episode_index")[0].as_py()); tix = int(t.column("task_index")[0].as_py())
    ranges = seg_ranges(A, S); kept = sum(e - s for s, e in ranges); dropped_frames += n_src - kept
    for s, e in ranges:
        n = e - s; sub = t.slice(s, n); cols = []
        for f in schema:
            if f.name == "episode_index": cols.append(pa.array(np.full(n, ep_out), type=f.type))
            elif f.name == "frame_index": cols.append(pa.array(np.arange(n), type=f.type))
            elif f.name == "timestamp": cols.append(pa.array((np.arange(n) / fps).astype(np.float32), type=f.type))
            elif f.name == "index": cols.append(pa.array(np.arange(idx_off, idx_off + n), type=f.type))
            elif f.name == "done": d = np.zeros(n, bool); d[-1] = True; cols.append(pa.array(d, type=f.type))
            else: cols.append(sub.column(f.name))
        new_t = pa.Table.from_arrays(cols, schema=schema)
        with pq.ParquetWriter(os.path.join(args.out, "data", "chunk-000", "episode_%06d.parquet" % ep_out), schema) as w:
            w.write_table(new_t)
        episodes.append({"episode_index": ep_out, "tasks": [tasks[tix]], "length": n})
        stats_out.append({"episode_index": ep_out, "stats": ep_stats(new_t, n, ep_out, idx_off, tix)})
        prov[ep_out] = {"src_episode": src_ep, "start": int(s), "end": int(e)}
        ep_out += 1; idx_off += n
    if src_ep % 50 == 0: print(f"  src ep{src_ep} done -> {ep_out} episodes / {idx_off} frames", flush=True)

meta = os.path.join(args.out, "meta")
os.system(f"cp {os.path.join(args.src,'meta','tasks.jsonl')} {meta}/")
with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
    for e in episodes: f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
    for s in stats_out: f.write(json.dumps(s) + "\n")
with open(os.path.join(meta, "source_segments.json"), "w") as f:
    json.dump({"source": args.src, "criterion": "compute_droid_nonidle_ranges: vel_source=state, grip_guard=%d, grip_delta=%g, filter_last_n_in_ranges=0" % (args.grip_guard, args.grip_delta),
               "constants": {"min_idle_len": M.min_idle_len, "min_non_idle_len": M.min_non_idle_len, "idle_threshold": M.idle_threshold},
               "episodes": prov}, f, indent=1)
info = dict(info); info["total_episodes"] = ep_out; info["total_frames"] = idx_off; info["splits"] = {"train": "0:%d" % ep_out}
json.dump(info, open(os.path.join(meta, "info.json"), "w"), indent=4)
print(f"\nDONE {args.out}\n  source {src_frames} frames / 250 ep  ->  {idx_off} frames / {ep_out} ep  (dropped {dropped_frames} = {dropped_frames/src_frames*100:.1f}%)")
