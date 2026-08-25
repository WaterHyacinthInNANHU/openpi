import glob
import numpy as np, pyarrow.parquet as pq
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"; FPS=15.0
f=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))[5]
t=pq.read_table(f, columns=["state","actions"])
s=np.array(t.column("state").to_pylist(),float); a=np.array(t.column("actions").to_pylist(),float)
d=np.diff(s,axis=0)*FPS
print("单条 episode:", f.split("/")[-1], "帧数", len(s))
print()
print("=== 单条内、带时延的相关(关节0-2) ===")
for j in range(3):
    row=[]
    for lag in range(0,6):
        aa=a[:len(d)-lag, j]; dd=d[lag:, j]
        n=min(len(aa),len(dd)); r=np.corrcoef(aa[:n],dd[:n])[0,1]
        row.append("lag%d %+.2f"%(lag,r))
    print("  关节%d: %s"%(j," | ".join(row)))
print()
print("=== 头 15 帧对照(关节0) ===")
print("  帧   action[0]   实测Δstate[0]*15 (rad/s)   state[0]")
for i in range(15):
    print("  %3d  %+8.4f      %+8.4f              %+8.4f"%(i,a[i,0], d[i,0] if i<len(d) else float(nan), s[i,0]))
print()
print("=== action 的整体形态 ===")
print("  非零 action 帧占比:", "%.1f%%"%((np.abs(a[:,:7]).max(axis=1)>1e-6).mean()*100))
print("  |action| 恰好=1 的比例(饱和):", "%.1f%%"%((np.abs(a[:,:7])>=0.999).any(axis=1).mean()*100))
print("  action[:,7] (夹爪) 取值:", np.unique(np.round(a[:,7],3))[:10])
