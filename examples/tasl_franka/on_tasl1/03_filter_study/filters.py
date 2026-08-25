import sys, glob, json
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"; FPS=15.0
files=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))
EP=[]
for f in files:
    t=pq.read_table(f,columns=["state","actions"])
    s=np.array(t.column("state").to_pylist(),float); a=np.array(t.column("actions").to_pylist(),float)
    T=fk(s[:,1:8])
    p=T[:,:3,3]; R=T[:,:3,:3]
    dp=np.linalg.norm(np.diff(p,axis=0),axis=1)*1000                 # mm / frame
    dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
    ang=np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))  # deg / frame
    dq=np.abs(np.diff(s[:,1:8],axis=0)*FPS).max(axis=1)
    EP.append(dict(n=len(s),state=s,act=a,dp=dp,ang=ang,dq=dq,p=p))
dp=np.concatenate([e["dp"] for e in EP]); ang=np.concatenate([e["ang"] for e in EP])
dq=np.concatenate([e["dq"] for e in EP]); N=len(dp)
print("总相邻帧对: %d" % N)
print()
print("=== 笛卡尔逐帧位移/转角的分布 ===")
print("  Δ位移 mm/帧 : 10%%=%.3f 20%%=%.3f 50%%=%.3f 80%%=%.3f 99%%=%.2f" % tuple(np.percentile(dp,[10,20,50,80,99])))
print("  Δ转角 度/帧 : 10%%=%.3f 20%%=%.3f 50%%=%.3f 80%%=%.3f 99%%=%.2f" % tuple(np.percentile(ang,[10,20,50,80,99])))
print("  (15Hz,所以 mm/帧 × 15 = mm/s)")
print()
print("=== 方法 A:Δxyz + Δ转角,低于 20%% 分位判静止 ===")
tp,ta=np.percentile(dp,20),np.percentile(ang,20)
print("  阈值: 位移 < %.3f mm/帧 (=%.1f mm/s) 且 转角 < %.3f 度/帧 (=%.2f 度/s)" % (tp,tp*15,ta,ta*15))
for mode,mask in [("两个都低于(AND)", (dp<tp)&(ang<ta)), ("任一低于(OR)", (dp<tp)|(ang<ta))]:
    print("    %s: 删 %d/%d = %.1f%%  | 这些帧的 max|dq| 中位 %.4f rad/s" % (mode,mask.sum(),N,mask.mean()*100,np.median(dq[mask])))
np.save("/tmp/_dp.npy",dp); np.save("/tmp/_ang.npy",ang); np.save("/tmp/_dq.npy",dq)
import pickle; pickle.dump([{k:v for k,v in e.items() if k in("n","dp","ang","dq")} for e in EP], open("/tmp/_ep.pkl","wb"))
np.save("/tmp/_feat.npy", np.concatenate([np.hstack([e["state"][:-1],e["act"][:-1]]) for e in EP]))
