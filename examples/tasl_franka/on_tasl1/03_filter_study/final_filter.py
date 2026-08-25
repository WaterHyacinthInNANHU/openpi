import sys, glob
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
TH_POS,TH_ROT,TH_GRIP=0.198,0.044,0.002

def feats(s):
    T=fk(s[:,1:8]);p=T[:,:3,3];R=T[:,:3,:3]
    dp=np.concatenate([[0],np.linalg.norm(np.diff(p,axis=0),axis=1)*1000])
    dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
    ag=np.concatenate([[0],np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))])
    dg=np.concatenate([[0],np.abs(np.diff(s[:,0]))])
    return dp,ag,dg

def dilate(m,k):
    if k<=0: return m
    o=m.copy()
    for s_ in range(1,k+1):
        o[s_:]|=m[:-s_]; o[:-s_]|=m[s_:]
    return o

def segrule(idle,mi,mn,dl,mf):
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
            if j-i>=mn:
                e=max(i,j-dl)
                if e-i>=mf:o[i:e]=True
            i=j
        else:i+=1
    return o

VAR=[("① 纯笛卡尔(有 bug)",       0,  7,16,10, 0),
     ("② +夹爪判据",              0, 15,26,10,16),
     ("③ +夹爪判据 +保护窗 ±5",   5, 15,26,10,16),
     ("④ +夹爪判据 +保护窗 ±8",   8, 15,26,10,16)]
res={}
for nm,K,mi,mn,dl,mf in VAR:
    tot=kept=0;gm_del=0;gm_all=0;ns=[];L=[]
    for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
        s=np.array(pq.read_table(f,columns=["state"]).column("state").to_pylist(),float)
        dp,ag,dg=feats(s)
        idle=(dp<TH_POS)&(ag<TH_ROT)
        if nm!=VAR[0][0]:
            idle&=(dg<TH_GRIP)
            idle&=~dilate(dg>TH_GRIP,K)
        o=segrule(idle,mi,mn,dl,mf)
        tot+=len(s);kept+=o.sum()
        gmask=dg>0.005; gm_all+=gmask.sum(); gm_del+=(gmask&~o).sum()
        c=0;i=0
        while i<len(o):
            if o[i]:
                j=i
                while j<len(o) and o[j]:j+=1
                c+=1;L.append(j-i);i=j
            else:i+=1
        ns.append(c)
    L=np.array(L);ns=np.array(ns)
    print("%-26s 删 %4.1f%% | 夹爪动作帧被误删 %4d/%d (%4.1f%%) | 段长中位%3d 最短%2d | 完整1段 %3d/250"
          % (nm,(1-kept/tot)*100,gm_del,gm_all,gm_del/gm_all*100,np.median(L),L.min(),(ns==1).sum()))
