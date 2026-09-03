import sys, glob
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
TH_POS,TH_ROT=0.198,0.044
DP=[];AG=[];DG=[];G=[]
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    s=np.array(pq.read_table(f,columns=["state"]).column("state").to_pylist(),float)
    T=fk(s[:,1:8]);p=T[:,:3,3];R=T[:,:3,:3]
    DP.append(np.concatenate([[0],np.linalg.norm(np.diff(p,axis=0),axis=1)*1000]))
    dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
    AG.append(np.concatenate([[0],np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))]))
    DG.append(np.concatenate([[0],np.abs(np.diff(s[:,0]))]))   # 夹爪逐帧变化
    G.append(s[:,0])
dp=np.concatenate(DP);ag=np.concatenate(AG);dg=np.concatenate(DG);g=np.concatenate(G)
idle=(dp<TH_POS)&(ag<TH_ROT)
print("=== 夹爪信号 ===")
print("  开度取值范围: [%.3f, %.3f]" % (g.min(),g.max()))
print("  逐帧变化 |Δgrip| 分位: 50%%=%.5f 80%%=%.5f 90%%=%.5f 99%%=%.4f max=%.4f" % tuple(np.percentile(dg,[50,80,90,99,100])))
print("  有明显夹爪动作的帧(|Δgrip|>0.005): %d (%.1f%%)" % ((dg>0.005).sum(),(dg>0.005).mean()*100))
print()
print("=== 问题有多大:被方法A(纯笛卡尔)判为 idle 的帧里,有多少其实在动夹爪 ===")
for th in [0.002,0.005,0.01,0.02]:
    moving=dg>th
    print("  |Δgrip|>%-5g : idle 帧中有 %4d 帧在动夹爪 (占 idle 的 %.1f%%,占全部夹爪动作帧的 %.1f%%)"
          % (th,(idle&moving).sum(),(idle&moving).sum()/idle.sum()*100,(idle&moving).sum()/max(moving.sum(),1)*100))
print()
print("=== 加上夹爪维度后的新判据 ===")
for th in [0.002,0.005]:
    new=idle&(dg<=th)
    print("  idle = Δpos<%.3f 且 Δrot<%.3f 且 |Δgrip|<=%g  → 逐帧命中 %d (%.1f%%),比原来少删 %d 帧"
          % (TH_POS,TH_ROT,th,new.sum(),new.mean()*100,idle.sum()-new.sum()))
