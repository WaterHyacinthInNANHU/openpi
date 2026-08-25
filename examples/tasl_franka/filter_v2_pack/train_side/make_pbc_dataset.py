#!/usr/bin/env python3
"""Build the PBC (centre-crop geometry) copy of a letterboxed RLinf LeRobot dataset.

    /data1/Franka_RealRobot/openpi/.venv/bin/python make_pbc_dataset.py \
        --src /data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep \
        --dst /data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep_pbc

WHY. The 10-task data was collected with RLinf `image_resize_mode: pad`: each 1280x720 ZED frame was
letterboxed into a 224x224 square (content rows 49..174, 126 rows high; black above and below).
That is pi05_droid's `resize_with_pad` convention.  The in-house PBC base
(`axis_pi05_droid_plainbc_v1`) was pretrained with a FULL-HEIGHT CENTRE-SQUARE crop instead
(AXIS-Bench round-3 `center_crop`).  Fine-tuning it on letterboxed frames would be a silent
geometry mismatch, so this script derives a *_pbc dataset whose frames are what the PBC serving
path produces from a live frame:

    live:    1280x720  --centre square-->  720x720  --ResizeImages-->  224x224
    stored:  224x224 letterbox  --un-pad-->  224x126  --centre square-->  126x126  --resize-->  224x224

Same geometry (the central 56 % of the width, full height); the stored path is a 1.78x upsample of
what the live path sees, because the raw frames were not kept.  That loss is unavoidable without
re-collecting; it is recorded in meta/info.json under "pbc_geometry" so nobody mistakes this for a
native-resolution corpus.

WHAT IS PRESERVED, byte-for-byte where possible:
  * every non-image column, row order, episode/frame/task indices (so the source dataset's
    nonidle_ranges*.json filter files apply unchanged);
  * the parquet schema INCLUDING the embedded huggingface metadata (image columns declared
    {"_type": "Image"}) -- dropping it breaks LeRobot iteration ("Could not infer dtype of dict");
  * meta/episodes.jsonl, meta/tasks.jsonl, meta/info.json (+ "pbc_geometry" block).
  * meta/episodes_stats.jsonl is copied with the image entries RECOMPUTED on the new frames
    (openpi never reads them, LeRobot's aggregate stats do).

The crop helper is imported from openpi (`openpi.policies.rlinf_franka_pbc.pbc_center_square`) so the
offline builder and the online transform cannot drift apart.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, "/data1/Franka_RealRobot/openpi/src")
from openpi.policies.rlinf_franka_pbc import PBC_GEOMETRY, pbc_center_square  # noqa: E402

IMAGE_COLS = ("image", "extra_view_image")
OUT_SIZE = 224
# Letterbox geometry of the source frames (1280x720 -> 224x126 centred on a 224 canvas).
SRC_W, SRC_H = 1280, 720
CONTENT_H = int(SRC_H / max(SRC_W / OUT_SIZE, SRC_H / OUT_SIZE))  # 126
PAD_TOP = (OUT_SIZE - CONTENT_H) // 2  # 49


def _unpad_crop_resize(png_bytes: bytes) -> tuple[bytes, np.ndarray]:
    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    if img.shape != (OUT_SIZE, OUT_SIZE, 3):
        raise ValueError(f"expected {OUT_SIZE}x{OUT_SIZE}x3 letterboxed frame, got {img.shape}")
    content = img[PAD_TOP : PAD_TOP + CONTENT_H]  # 126 x 224
    square = pbc_center_square(content)  # 126 x 126 (same function the serving path uses)
    out = np.asarray(Image.fromarray(np.ascontiguousarray(square)).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue(), out


def _verify_letterbox(png_bytes: bytes, name: str) -> None:
    """Refuse frames whose padding rows are not black: they were not produced by the pad path."""
    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    top, bot = img[:PAD_TOP], img[PAD_TOP + CONTENT_H :]
    if top.max() > 8 or bot.max() > 8:
        raise ValueError(
            f"{name}: rows outside {PAD_TOP}..{PAD_TOP + CONTENT_H - 1} are not black "
            f"(max {max(top.max(), bot.max())}); not a 1280x720 letterbox -- refusing to crop blindly"
        )


def _image_stats(frames: np.ndarray) -> dict:
    """LeRobot v2.1 per-episode image stats: (3,1,1) lists over [0,1] pixels, plus a count."""
    x = frames.astype(np.float32) / 255.0  # (N, H, W, 3)
    per_ch = lambda f: f(x, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()  # noqa: E731
    return {"min": per_ch(np.min), "max": per_ch(np.max), "mean": per_ch(np.mean), "std": per_ch(np.std), "count": [int(len(x))]}


def convert_one(args: tuple[str, str, bool]) -> dict:
    src_pq, dst_pq, verify = args
    t = pq.read_table(src_pq)
    schema = t.schema
    assert schema.metadata and b"huggingface" in schema.metadata, f"{src_pq}: missing huggingface schema metadata"
    cols = []
    stats = {}
    for field in schema:
        col = t.column(field.name)
        if field.name not in IMAGE_COLS:
            cols.append(col)
            continue
        new_rows, frames = [], []
        for i, v in enumerate(col.to_pylist()):
            if verify and i == 0:
                _verify_letterbox(v["bytes"], f"{os.path.basename(src_pq)}:{field.name}")
            b, arr = _unpad_crop_resize(v["bytes"])
            new_rows.append({"bytes": b, "path": v.get("path")})
            frames.append(arr)
        cols.append(pa.array(new_rows, type=field.type))
        stats[field.name] = _image_stats(np.stack(frames))
    new_t = pa.Table.from_arrays(cols, schema=schema)
    with pq.ParquetWriter(dst_pq, schema) as w:
        w.write_table(new_t)
    ep = int(t.column("episode_index")[0].as_py())
    return {"episode_index": ep, "n": t.num_rows, "image_stats": stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="delete an existing --dst first")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N episodes (smoke test)")
    a = ap.parse_args()

    src, dst = a.src.rstrip("/"), a.dst.rstrip("/")
    if not dst.endswith("_pbc"):
        sys.exit(f"--dst must end in _pbc so the geometry is visible in the name: {dst}")
    if os.path.exists(dst):
        if not a.force:
            sys.exit(f"{dst} exists; pass --force to rebuild")
        shutil.rmtree(dst)
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    for c in IMAGE_COLS:
        assert info["features"][c]["shape"] == [OUT_SIZE, OUT_SIZE, 3], (c, info["features"][c]["shape"])

    os.makedirs(os.path.join(dst, "data", "chunk-000"))
    os.makedirs(os.path.join(dst, "meta"))
    src_files = sorted(
        os.path.join(src, "data", "chunk-000", f) for f in os.listdir(os.path.join(src, "data", "chunk-000")) if f.endswith(".parquet")
    )
    if a.limit:
        src_files = src_files[: a.limit]
    jobs = [(f, os.path.join(dst, "data", "chunk-000", os.path.basename(f)), True) for f in src_files]

    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(convert_one, jobs), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes  {time.time() - t0:.0f}s", flush=True)
    by_ep = {r["episode_index"]: r for r in results}
    n_frames = sum(r["n"] for r in results)

    # meta/: copy, patch episodes_stats image entries, stamp geometry into info.json.
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(os.path.join(src, "meta", name), os.path.join(dst, "meta", name))
    with open(os.path.join(src, "meta", "episodes_stats.jsonl")) as fin, open(os.path.join(dst, "meta", "episodes_stats.jsonl"), "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["episode_index"] in by_ep:
                rec["stats"].update(by_ep[rec["episode_index"]]["image_stats"])
                fout.write(json.dumps(rec) + "\n")
    if a.limit:
        # Keep meta consistent with the truncated data for the smoke test.
        keep = set(by_ep)
        eps = [json.loads(l) for l in open(os.path.join(src, "meta", "episodes.jsonl")) if json.loads(l)["episode_index"] in keep]
        with open(os.path.join(dst, "meta", "episodes.jsonl"), "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        info["total_episodes"] = len(eps)
        info["total_frames"] = n_frames
        info["splits"] = {"train": f"0:{len(eps)}"}
    info["pbc_geometry"] = {
        "geometry": PBC_GEOMETRY,
        "source_dataset": os.path.basename(src),
        "source_frame": "1280x720 letterboxed to 224x224 (content rows %d..%d)" % (PAD_TOP, PAD_TOP + CONTENT_H - 1),
        "stored_frame": "un-padded 224x126 -> centre %dx%d -> LANCZOS resize to %dx%d" % (CONTENT_H, CONTENT_H, OUT_SIZE, OUT_SIZE),
        "serving_equivalent": "live 1280x720 -> centre 720x720 -> ResizeImages(224,224) (openpi.policies.rlinf_franka_pbc)",
        "upsample_factor_vs_live": round(OUT_SIZE / CONTENT_H, 3),
        "for_base_checkpoint": "axis_pi05_droid_plainbc_v1",
    }
    json.dump(info, open(os.path.join(dst, "meta", "info.json"), "w"), indent=4)
    # nonidle range files are keyed by episode index / frame offsets, which are unchanged.
    src_ranges = [f for f in os.listdir(os.path.join(src, "meta")) if f.startswith("nonidle_ranges")]
    for f in src_ranges:
        shutil.copy2(os.path.join(src, "meta", f), os.path.join(dst, "meta", f))
    print(f"done: {len(results)} episodes / {n_frames} frames -> {dst}  ({time.time() - t0:.0f}s); "
          f"copied filter files: {src_ranges or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
