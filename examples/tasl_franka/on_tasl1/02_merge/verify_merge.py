import json, os, glob, io
import numpy as np, pyarrow.parquet as pq
from PIL import Image
OUT="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(OUT+"/meta/tasks.jsonl")}
eps={json.loads(l)["episode_index"]: json.loads(l) for l in open(OUT+"/meta/episodes.jsonl")}
info=json.load(open(OUT+"/meta/info.json"))
files=sorted(glob.glob(OUT+"/data/chunk-000/*.parquet"))
print("parquet 数:", len(files))
expect_idx=0; bad=[]; per_tix={}
for i,f in enumerate(files):
    t=pq.read_table(f); n=t.num_rows
    ei=set(t.column("episode_index").to_pylist()); ti=set(t.column("task_index").to_pylist())
    ix=t.column("index").to_pylist()
    if ei!={i}: bad.append((f,"episode_index",ei))
    if len(ti)!=1: bad.append((f,"task_index 不唯一",ti))
    tix=list(ti)[0]
    # parquet 的 task_index 解析出的 prompt,必须等于 episodes.jsonl 里记的 task
    if tasks[tix]!=eps[i]["tasks"][0]: bad.append((f,"prompt 不一致",tasks[tix],eps[i]["tasks"][0]))
    if ix[0]!=expect_idx or ix[-1]!=expect_idx+n-1 or len(ix)!=n: bad.append((f,"index 不连续",ix[0],ix[-1]))
    if eps[i]["length"]!=n: bad.append((f,"length 不符",eps[i]["length"],n))
    expect_idx+=n
    per_tix.setdefault(tix,[0,0]); per_tix[tix][0]+=1; per_tix[tix][1]+=n
    if t.schema.metadata is None or b"huggingface" not in t.schema.metadata: bad.append((f,"hf 元数据丢失",))
print("总帧数:", expect_idx, "| info.json 记的:", info["total_frames"], "| 一致" if expect_idx==info["total_frames"] else "| ✗不一致")
print("\n每个 task 的分布:")
for k in sorted(per_tix): print(f"  tix {k}: {per_tix[k][0]:>3} 条 / {per_tix[k][1]:>6} 帧  | {tasks[k]!r}")
# 抽查:每个 task 的第一条,解码一帧图像
print("\n抽查图像解码 + prompt 解析:")
seen=set()
for i,f in enumerate(files):
    t=pq.read_table(f); tix=t.column("task_index").to_pylist()[0]
    if tix in seen: continue
    seen.add(tix)
    e=t.column("image").to_pylist()[0]; b=e["bytes"] if isinstance(e,dict) else e
    im=Image.open(io.BytesIO(b)); w2=t.column("extra_view_image").to_pylist()[0]
    print(f"  ep{i:03d} tix{tix} image={im.size}{im.mode} -> {tasks[tix]!r}")
print("\n问题数:", len(bad))
for b in bad[:10]: print("  ✗", b)
