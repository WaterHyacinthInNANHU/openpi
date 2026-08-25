import glob, json
import numpy as np, pyarrow.parquet as pq
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"; FPS=15.0; TH=0.01
tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D+"/meta/tasks.jsonl")}
pos=[]; head=[]; tail=[]; per_tix={}
tot=idl=0
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    t=pq.read_table(f, columns=["state","task_index"])
    s=np.array(t.column("state").to_pylist(),float); tix=t.column("task_index").to_pylist()[0]
    dq=np.abs(np.diff(s[:,1:8],axis=0)*FPS).max(axis=1)
    n=len(dq); still=dq<TH
    tot+=n; idl+=still.sum()
    pos.append(np.where(still)[0]/max(n-1,1))
    # 开头连续静止多少帧 / 结尾连续静止多少帧
    k=0
    while k<n and still[k]: k+=1
    head.append(k)
    k2=0
    while k2<n and still[n-1-k2]: k2+=1
    tail.append(k2)
    a=per_tix.setdefault(tix,[0,0]); a[0]+=still.sum(); a[1]+=n
p=np.concatenate(pos)
print("静止判据: 实测 max|Δq| < %g rad/s" % TH)
print("总静止帧: %d / %d  (%.1f%%)" % (idl, tot, idl/tot*100))
print()
print("=== 静止帧在轨迹中的位置分布(0=开头, 1=结尾)===")
h,_=np.histogram(p, bins=10, range=(0,1))
for i,c in enumerate(h):
    print("  %.1f-%.1f : %5d  %s" % (i/10,(i+1)/10,c,"#"*int(c/max(h.max(),1)*50)))
print()
head=np.array(head); tail=np.array(tail)
print("=== 每条 episode 开头/结尾的连续静止段 ===")
print("  开头: 中位 %d 帧 (%.1fs), 均值 %.0f, 最大 %d | 有 %d/%d 条开头静止>15帧(1秒)"
      % (np.median(head), np.median(head)/FPS, head.mean(), head.max(), (head>15).sum(), len(head)))
print("  结尾: 中位 %d 帧 (%.1fs), 均值 %.0f, 最大 %d | 有 %d/%d 条结尾静止>15帧"
      % (np.median(tail), np.median(tail)/FPS, tail.mean(), tail.max(), (tail>15).sum(), len(tail)))
print("  开头+结尾合计占总帧数: %.1f%%" % ((head.sum()+tail.sum())/tot*100))
print()
print("=== 各 task 的静止帧比例 ===")
for k in sorted(per_tix):
    v=per_tix[k]; print("  tix%d %5.1f%%  (%5d/%5d)  %s" % (k, v[0]/v[1]*100, v[0], v[1], tasks[k][:48]))
