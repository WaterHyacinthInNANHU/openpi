import json, glob, subprocess, os
import pyarrow.parquet as pq
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep_video"
eps={json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(D+"/meta/episodes.jsonl")}
keys=["observation.images.exterior","observation.images.wrist"]
bad=[]
def nb(p):
    r=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                      "-show_entries","stream=nb_read_frames","-of","csv=p=0",p],
                     capture_output=True,text=True).stdout.strip()
    return int(r) if r.isdigit() else -1
import concurrent.futures as cf
def chk(ei):
    out=[]
    n=eps[ei]
    pqn=pq.read_metadata(f"{D}/data/chunk-000/episode_{ei:06d}.parquet").num_rows
    if pqn!=n: out.append((ei,"parquet 行数",pqn,n))
    for k in keys:
        f=f"{D}/videos/chunk-000/{k}/episode_{ei:06d}.mp4"
        if not os.path.exists(f): out.append((ei,"缺 mp4",k)); continue
        v=nb(f)
        if v!=n: out.append((ei,"mp4 帧数 "+k,v,n))
    return out
with cf.ThreadPoolExecutor(16) as ex:
    for r in ex.map(chk, sorted(eps)): bad+=r
print("检查 %d 条 episode / %d 个 mp4" % (len(eps), len(eps)*2))
print("问题数:", len(bad))
for b in bad[:10]: print("  ✗", b)
