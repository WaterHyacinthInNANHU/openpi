import sys, glob
sys.path.insert(0,"/tmp")
import numpy as np, pyarrow.parquet as pq
from fk import fk
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
TH_POS,TH_ROT=0.198,0.044
def run(mi,mn,dl,min_final):
    tot=kept=0;nseg=[];L=[];dropep=0
    for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
        s=np.array(pq.read_table(f,columns=["state"]).column("state").to_pylist(),float)
        T=fk(s[:,1:8]);p=T[:,:3,3];R=T[:,:3,:3]
        dp=np.concatenate([[0],np.linalg.norm(np.diff(p,axis=0),axis=1)*1000])
        dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
        ag=np.concatenate([[0],np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))])
        idle=(dp<TH_POS)&(ag<TH_ROT)
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
                    if e-i>=min_final: o[i:e]=True
                i=j
            else:i+=1
        tot+=n;kept+=o.sum()
        c=0;i=0
        while i<n:
            if o[i]:
                j=i
                while j<n and o[j]:j+=1
                c+=1;L.append(j-i);i=j
            else:i+=1
        nseg.append(c)
        if c==0:dropep+=1
    L=np.array(L);ns=np.array(nseg)
    return tot,kept,L,ns,dropep
for nm,(mi,mn,dl,mf) in [("官方原版 (mn=16, 不检查段尾后长度)",(7,16,10,0)),
                          ("修正版 A (段尾裁完仍需 >=16 帧)",(7,26,10,16)),
                          ("修正版 B (更保守: idle 段 >=15 才删)",(15,26,10,16))]:
    tot,kept,L,ns,de=run(mi,mn,dl,mf)
    print("%s" % nm)
    print("   保留 %d/%d (%.1f%%) | 删 %.1f%% | 段长 中位%d 最短%d | <16帧碎片 %d 个 | 整条没了 %d 条 | 1段%d 2段%d 3段%d >=4段%d"
          % (kept,tot,kept/tot*100,(1-kept/tot)*100,np.median(L),L.min(),(L<16).sum(),de,(ns==1).sum(),(ns==2).sum(),(ns==3).sum(),(ns>=4).sum()))
