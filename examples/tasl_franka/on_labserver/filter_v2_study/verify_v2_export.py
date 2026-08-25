import json, glob, numpy as np, pyarrow.parquet as pq
SRC="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"; OUT="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_v2"
info=json.load(open(OUT+"/meta/info.json")); eps=[json.loads(l) for l in open(OUT+"/meta/episodes.jsonl")]
stats=[json.loads(l) for l in open(OUT+"/meta/episodes_stats.jsonl")]; prov=json.load(open(OUT+"/meta/source_segments.json"))["episodes"]
files=sorted(glob.glob(OUT+"/data/chunk-000/*.parquet"))
print(f"info: episodes {info['total_episodes']} frames {info['total_frames']} | episodes.jsonl {len(eps)} | stats {len(stats)} | parquet files {len(files)} | prov {len(prov)}")
ok=True; tot=0; last_idx=-1; src_cache={}
for f in files:
    t=pq.read_table(f); n=t.num_rows; ep=int(t.column("episode_index")[0].as_py()); tot+=n
    fi=np.array(t.column("frame_index").to_pylist()); ts=np.array(t.column("timestamp").to_pylist()); ix=np.array(t.column("index").to_pylist())
    d=np.array(t.column("done").to_pylist())
    c1=np.array_equal(fi,np.arange(n)); c2=np.allclose(ts,np.arange(n)/15,atol=1e-4); c3=ix[0]==last_idx+1 and np.array_equal(ix,np.arange(ix[0],ix[0]+n)); c4=(d.sum()==1 and d[-1])
    c5=eps[ep]["length"]==n and stats[ep]["episode_index"]==ep
    last_idx=ix[-1]
    # 与源逐行比对 (图像 bytes / state / actions / task_index)
    p=prov[str(ep)]; se=p["src_episode"]
    if se not in src_cache: src_cache={se: pq.read_table(f"{SRC}/data/chunk-000/episode_{se:06d}.parquet")}
    s=src_cache[se].slice(p["start"],n)
    c6=all(s.column(k).to_pylist()==t.column(k).to_pylist() for k in ("image","extra_view_image","state","actions","task_index","is_success"))
    c7=t.schema.metadata and b"huggingface" in t.schema.metadata
    if not all([c1,c2,c3,c4,c5,c6,c7]): ok=False; print("FAIL ep",ep,[c1,c2,c3,c4,c5,c6,c7])
print(f"frames sum {tot} == info {info['total_frames']}: {tot==info['total_frames']} | all per-episode checks: {ok}")
print("segments per src ep: max", max(sum(1 for v in prov.values() if v['src_episode']==e) for e in set(v['src_episode'] for v in prov.values())), "| src eps with 0 segments:", 250-len(set(v['src_episode'] for v in prov.values())))
L=[e["length"] for e in eps]; print("episode length min/median/max:",min(L),int(np.median(L)),max(L))
import collections; c=collections.Counter(e["tasks"][0] for e in eps); print("episodes per task:",sorted(c.values()))
# 与 v2 ranges (尾砍10) 的关系: 导出帧数应 = 58760 + 每段尾砍回来的帧
print("image stats shape:",np.array(stats[0]["stats"]["image"]["mean"]).shape, "state stats len:",len(stats[0]["stats"]["state"]["mean"]))
