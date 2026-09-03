import sys, glob, json
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
TH_POS,TH_ROT=0.198,0.044
def seg(idle,mi=7,mn=16,dl=10):
    n=len(idle);k=np.ones(n,bool);i=0
    while i<n:
        if idle[i]:
            j=i
            while j<n and idle[j]:j+=1
            if j-i>=mi:k[i:j]=False
            i=j
        else:i+=1
    o=np.zeros(n,bool);i=0
    while i<n:
        if k[i]:
            j=i
            while j<n and k[j]:j+=1
            if j-i>=mn:o[i:max(i,j-dl)]=True
            i=j
        else:i+=1
    return o
tot=kept=0;raws=0;nseg=[];dropep=0;lens=[]
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    s=np.array(pq.read_table(f,columns=["state"]).column("state").to_pylist(),float)
    T=fk(s[:,1:8]);p=T[:,:3,3];R=T[:,:3,:3]
    dp=np.concatenate([[0],np.linalg.norm(np.diff(p,axis=0),axis=1)*1000])
    dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
    ag=np.concatenate([[0],np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))])
    raw=(dp<TH_POS)&(ag<TH_ROT); k=seg(raw)
    tot+=len(s);kept+=k.sum();raws+=raw.sum()
    c=0;i=0;L=[]
    while i<len(k):
        if k[i]:
            j=i
            while j<len(k) and k[j]:j+=1
            c+=1;L.append(j-i);i=j
        else:i+=1
    nseg.append(c); lens+=L
    if c==0: dropep+=1
print("全部 250 条 episode:")
print("  原始              : %d 帧" % tot)
print("  逐帧判据命中      : %d (%.1f%%)" % (raws,raws/tot*100))
print("  加段级规则后实删  : %d (%.1f%%)   ← 段规则多删了 %.1f 个百分点" % (tot-kept,(tot-kept)/tot*100,((tot-kept)-raws)/tot*100))
print("  保留              : %d (%.1f%%)" % (kept,kept/tot*100))
print()
ns=np.array(nseg)
print("  轨迹被切成几段    : 1段 %d条 | 2段 %d条 | 3段 %d条 | >=4段 %d条 | 整条没了 %d条"
      % ((ns==1).sum(),(ns==2).sum(),(ns==3).sum(),(ns>=4).sum(),dropep))
L=np.array(lens)
print("  保留段长度(帧)   : 中位 %d, 最短 %d, 最长 %d | <32帧(2秒)的碎片段 %d 个" % (np.median(L),L.min(),L.max(),(L<32).sum()))
