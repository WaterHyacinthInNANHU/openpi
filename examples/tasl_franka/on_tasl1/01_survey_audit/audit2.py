import json, glob
import numpy as np, pyarrow.parquet as pq
M="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(M+"/meta/tasks.jsonl")}
rows=[]
for f in sorted(glob.glob(M+"/data/chunk-000/*.parquet")):
    t=pq.read_table(f, columns=["actions","task_index","episode_index"])
    a=np.array(t.column("actions").to_pylist(), dtype=float)
    tix=t.column("task_index").to_pylist()[0]; ei=t.column("episode_index").to_pylist()[0]
    v=np.abs(a[:,:7]).max(axis=1)          # 每帧 7 个关节速度的最大绝对值
    g=a[:,7]                                # 夹爪
    rows.append((ei,tix,len(a),v,g))
lens=np.array([r[2] for r in rows])
print("=== episode 长度分布(帧) ===")
for p in [0,1,5,25,50,75,95,100]:
    print("  %3d%%: %5.0f" % (p, np.percentile(lens,p)))
short=[r for r in rows if r[2]<50]
print("\n=== 异常短的 episode(<50 帧 = <3.3 秒)===")
for ei,tix,n,_,_ in sorted(short,key=lambda x:x[2]):
    print("  ep%03d  tix%d  %4d 帧 (%.1fs)  | %s" % (ei,tix,n,n/15,tasks[tix][:46]))
print("  共 %d 条" % len(short))
print("\n=== 用不同阈值统计\"低速帧\"占比 ===")
allv=np.concatenate([r[3] for r in rows])
for th in [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
    print("  |关节速度|max < %-6g : %6d / %d  (%.1f%%)" % (th, (allv<th).sum(), len(allv), (allv<th).mean()*100))
print("\n  关节速度分位: 1%%=%.4f 25%%=%.4f 中位=%.4f 75%%=%.4f 99%%=%.4f max=%.3f" % tuple(np.percentile(allv,[1,25,50,75,99,100])))
