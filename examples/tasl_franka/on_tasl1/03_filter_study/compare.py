import sys, glob, pickle
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"; FPS=15.; SCALE=0.509
def ranges(idle,mi=7,mn=16,dl=10):
    n=len(idle); keep=np.ones(n,bool); i=0
    while i<n:
        if idle[i]:
            j=i
            while j<n and idle[j]: j+=1
            if j-i>=mi: keep[i:j]=False
            i=j
        else: i+=1
    out=np.zeros(n,bool); i=0
    while i<n:
        if keep[i]:
            j=i
            while j<n and keep[j]: j+=1
            if j-i>=mn: out[i:max(i,j-dl)]=True
            i=j
        else: i+=1
    return out
dp=np.load("/tmp/_dp.npy"); ang=np.load("/tmp/_ang.npy"); dq=np.load("/tmp/_dq.npy")
OFF=[]
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    a=np.array(pq.read_table(f,columns=["actions"]).column("actions").to_pylist(),float)[:,:7]
    idle=np.all(np.abs(np.diff(a,axis=0))<1e-3/SCALE,axis=1)
    OFF.append(~ranges(np.concatenate([idle,[idle[-1]]]))[:-1])
off=np.concatenate(OFF)                    # True = 官方判据会删
tp,ta=np.percentile(dp,20),np.percentile(ang,20)
A=(dp<tp)&(ang<ta)                          # 方法A
print("三种过滤删掉的帧集合对比(共 %d 帧对)" % len(dq))
print("  官方 openpi idle filter : %5d (%4.1f%%)" % (off.sum(),off.mean()*100))
print("  方法A 笛卡尔 20%% 分位   : %5d (%4.1f%%)" % (A.sum(),A.mean()*100))
print()
print("  交集 : %5d  (占官方的 %.0f%%,占方法A的 %.0f%%)" % ((off&A).sum(),(off&A).sum()/max(off.sum(),1)*100,(off&A).sum()/max(A.sum(),1)*100))
print("  并集 : %5d (%.1f%%)" % ((off|A).sum(),(off|A).mean()*100))
print()
print("=== 各方法删掉/保留帧的实际运动量对照 ===")
print("  %-24s %-22s %-22s" % ("方法","删掉帧 max|dq| 中位","保留帧 max|dq| 中位"))
for nm,m in [("官方 openpi",off),("方法A 笛卡尔20%",A),("A∩官方",off&A),("A 但官方不删",A&~off)]:
    if m.sum()==0: continue
    print("  %-24s %-22.5f %-22.5f" % (nm,np.median(dq[m]),np.median(dq[~m])))
print()
print("=== 方法A 删掉的帧的实际速度分布 ===")
print("  max|dq| 分位: 50%%=%.5f 90%%=%.5f 99%%=%.4f" % tuple(np.percentile(dq[A],[50,90,99])))
print("  Δ位移 mm/帧: 50%%=%.4f 99%%=%.3f" % tuple(np.percentile(dp[A],[50,99])))
