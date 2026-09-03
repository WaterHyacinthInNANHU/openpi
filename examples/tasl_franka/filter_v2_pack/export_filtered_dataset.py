"""Physically export a dummy-action-filtered copy of a LeRobot v2.1 dataset (source is never modified).

Criterion = openpi's official compute_droid_nonidle_ranges (same 4 constants), but run on the
*measured* joint velocity diff(state[:,1:8])*fps instead of the commanded one, plus a +/-N frame
guard around gripper events (arm still, gripper moving = valid supervision).  See nonidle_ranges.py.

Every kept [start, end) segment becomes one new episode.  image / wrist image / state / actions /
task / is_success rows are copied byte-for-byte; only episode_index / frame_index / timestamp /
index / done are re-numbered.  meta/episodes_stats.jsonl is recomputed with the lerobot
compute_stats algorithm (vendored below, no lerobot install needed).  meta/source_segments.json
records new_episode -> (source_episode, start, end) for provenance.

Usage:
    python export_filtered_dataset.py --src <lerobot_home>/franka/tasl_fr3_10task_250ep \
                                      --out <lerobot_home>/franka/tasl_fr3_10task_v2
    # smoke test on the first 5 source episodes:
    python export_filtered_dataset.py --src ... --out /tmp/smoke --limit 5
"""
import argparse, glob, io, json, os, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nonidle_ranges as M

# ---- vendored from lerobot/common/datasets/compute_stats.py (lerobot 0.1.0) ----
def estimate_num_samples(dataset_len, min_num_samples=100, max_num_samples=10_000, power=0.75):
    if dataset_len < min_num_samples: min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

def sample_indices(data_len):
    return np.round(np.linspace(0, data_len - 1, estimate_num_samples(data_len))).astype(int).tolist()

def auto_downsample_height_width(img, target_size=150, max_size_threshold=300):
    _, height, width = img.shape
    if max(width, height) < max_size_threshold: return img
    f = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::f, ::f]

def get_feature_stats(array, axis, keepdims):
    return {"min": np.min(array, axis=axis, keepdims=keepdims), "max": np.max(array, axis=axis, keepdims=keepdims),
            "mean": np.mean(array, axis=axis, keepdims=keepdims), "std": np.std(array, axis=axis, keepdims=keepdims),
            "count": np.array([len(array)])}
# ---------------------------------------------------------------------------------

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", required=True, help="source LeRobot v2.1 dataset dir (never modified)")
p.add_argument("--out", required=True, help="output dataset dir (must not exist)")
p.add_argument("--vel-source", choices=["state", "action"], default="state", help="state = measured velocity (v2 default)")
p.add_argument("--vel-scale", type=float, default=0.509, help="[--vel-source action] rad/s per unit action")
p.add_argument("--grip-guard", type=int, default=5, help="protect +/-N frames around gripper events (0 = off)")
p.add_argument("--grip-delta", type=float, default=0.02, help="|delta gripper action| counting as a gripper event")
p.add_argument("--tail", type=int, default=0, help="frames to cut from the end of each kept segment (0 for physical export; "
               "the official 10-frame tail cut is applied later as a ranges json, see run_all.sh)")
p.add_argument("--limit", type=int, default=0, help="only process the first N source episodes (smoke test)")
args = p.parse_args()
assert not os.path.exists(args.out), f"{args.out} already exists, refusing to overwrite"
os.makedirs(os.path.join(args.out, "data", "chunk-000")); os.makedirs(os.path.join(args.out, "meta"))
info = json.load(open(os.path.join(args.src, "meta", "info.json"))); fps = float(info["fps"]); feats = info["features"]
M.filter_last_n_in_ranges = args.tail

def seg_ranges(A, S):
    if args.vel_source == "state":
        q = S[:, 1:8]; v = np.diff(q, axis=0) * fps; v = np.vstack([v, v[-1:]]) if len(v) else np.zeros_like(q)
    else:
        v = A[:, :7] * args.vel_scale
    pr = None
    if args.grip_guard > 0:
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
            if arr.dtype == bool: arr = arr.astype(np.float32)   # same as original meta (bool treated as numeric)
            st[name] = get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
    return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in st.items()}

ep_out = 0; idx_off = 0; episodes = []; stats_out = []; prov = {}; dropped_frames = 0; src_frames = 0; n_src_ep = 0
tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(os.path.join(args.src, "meta", "tasks.jsonl"))}
files = sorted(glob.glob(os.path.join(args.src, "data", "chunk-*", "*.parquet")))
if args.limit: files = files[: args.limit]
for pqf in files:
    t = pq.read_table(pqf); schema = t.schema; n_src = t.num_rows; src_frames += n_src; n_src_ep += 1
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
shutil.copy(os.path.join(args.src, "meta", "tasks.jsonl"), meta)
with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
    for e in episodes: f.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
    for s in stats_out: f.write(json.dumps(s) + "\n")
with open(os.path.join(meta, "source_segments.json"), "w") as f:
    json.dump({"source": os.path.abspath(args.src),
               "criterion": "compute_droid_nonidle_ranges: vel_source=%s, grip_guard=%d, grip_delta=%g, filter_last_n_in_ranges=%d"
                            % (args.vel_source, args.grip_guard, args.grip_delta, args.tail),
               "constants": {"min_idle_len": M.min_idle_len, "min_non_idle_len": M.min_non_idle_len, "idle_threshold": M.idle_threshold},
               "episodes": prov}, f, indent=1)
info = dict(info); info["total_episodes"] = ep_out; info["total_frames"] = idx_off; info["splits"] = {"train": "0:%d" % ep_out}
json.dump(info, open(os.path.join(meta, "info.json"), "w"), indent=4)
print(f"\nDONE {args.out}\n  source {src_frames} frames / {n_src_ep} ep  ->  {idx_off} frames / {ep_out} ep"
      f"  (dropped {dropped_frames} = {dropped_frames/max(src_frames,1)*100:.1f}%)")
