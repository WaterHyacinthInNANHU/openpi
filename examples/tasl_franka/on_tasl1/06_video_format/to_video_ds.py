"""把图像内嵌型 LeRobot v2.1 数据集转成视频型,以便 HF 的 lerobot visualizer 能打开。

visualizer 的判据(读源码 fetch-data.ts 确认):
  - codebase_version != "v3.0" 走 v2 分支 → 我们的 v2.1 可用
  - 视频 key = info.features 里 dtype=="video" 的项,不要求 observation.images. 前缀
  - video_path 模板用 {video_key}/{episode_chunk}/{episode_index} 变量替换
  - info.video_path 为 null 直接报错拒绝
"""
import io, os, json, glob, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image

SRC="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
DST="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep_video"
# 源列名 -> 视频 key(视频 key 会成为 visualizer 里的面板标题)
VIDMAP={"image":"observation.images.exterior", "extra_view_image":"observation.images.wrist"}
FPS=json.load(open(SRC+"/meta/info.json"))["fps"]

def gb(e): return e["bytes"] if isinstance(e,dict) else e.as_py()["bytes"]

def encode(frames_bytes, out_path, fps):
    """把一串 PNG bytes 用 ffmpeg 编成 h264 mp4(浏览器兼容性最好)。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    first=np.asarray(Image.open(io.BytesIO(frames_bytes[0])).convert("RGB"))
    h,w=first.shape[:2]
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
         "-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0",
         "-c:v","libx264","-pix_fmt","yuv420p","-crf","20","-preset","medium",
         "-movflags","+faststart", out_path]
    p=subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for b in frames_bytes:
        p.stdin.write(np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).tobytes())
    p.stdin.close()
    if p.wait()!=0: raise RuntimeError("ffmpeg failed: "+out_path)

def one(ep):
    f=f"{SRC}/data/chunk-000/episode_{ep:06d}.parquet"
    t=pq.read_table(f); schema=t.schema
    for col,key in VIDMAP.items():
        encode([gb(e) for e in t.column(col).to_pylist()],
               f"{DST}/videos/chunk-000/{key}/episode_{ep:06d}.mp4", FPS)
    # parquet 去掉两列图像,其余原样(schema 元数据要重建,因为列变了)
    keep=[f.name for f in schema if f.name not in VIDMAP]
    sub=t.select(keep)
    hf=json.loads(schema.metadata[b"huggingface"].decode())
    if "info" in hf and "features" in hf["info"]:
        hf["info"]["features"]={k:v for k,v in hf["info"]["features"].items() if k not in VIDMAP}
    ns=sub.schema.with_metadata({b"huggingface": json.dumps(hf).encode()})
    sub=pa.Table.from_arrays(sub.columns, schema=ns)
    out=f"{DST}/data/chunk-000/episode_{ep:06d}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with pq.ParquetWriter(out, ns) as w: w.write_table(sub)
    return ep, t.num_rows

if __name__=="__main__":
    if os.path.exists(DST): shutil.rmtree(DST)
    os.makedirs(DST+"/meta")
    eps=[int(os.path.basename(p)[8:14]) for p in sorted(glob.glob(SRC+"/data/chunk-000/*.parquet"))]
    done=0
    with ProcessPoolExecutor(max_workers=12) as ex:
        for ep,n in ex.map(one, eps):
            done+=1
            if done%25==0: print(f"  {done}/{len(eps)} 条完成", flush=True)
    print("视频+parquet 转换完成", flush=True)
