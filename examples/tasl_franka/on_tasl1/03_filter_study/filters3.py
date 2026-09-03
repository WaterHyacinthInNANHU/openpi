import pickle, numpy as np
dq=np.load("/tmp/_dq.npy"); EP=pickle.load(open("/tmp/_ep.pkl","rb")); F=np.load("/tmp/_feat.npy")
N=len(dq)
from sklearn.cluster import MiniBatchKMeans
X=(F-F.mean(0))/(F.std(0)+1e-8); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
km=MiniBatchKMeans(n_clusters=500,random_state=0,n_init=5,batch_size=8192).fit(X)
lab=km.labels_
print("=== 方法 B:SCIZOR 式 state-action 去重(真 KMeans,500 类,不含进度模型)===")
for tau in [0.9999,0.9995,0.999,0.995]:
    keep=np.ones(N,bool)
    for c in np.unique(lab):
        idx=np.where(lab==c)[0]
        if len(idx)<2: continue
        Xi=X[idx]; kept=[0]
        for i in range(1,len(idx)):
            if (Xi[i]@Xi[kept].T).max()>tau: keep[idx[i]]=False
            else: kept.append(i)
    r=~keep
    print("  τ=%.4f : 删 %5d/%d = %4.1f%% | 删掉帧 max|dq| 中位 %.4f  保留帧 %.4f  (比值 %.2f)"
          % (tau,r.sum(),N,r.mean()*100,np.median(dq[r]) if r.any() else np.nan,np.median(dq[keep]),
             (np.median(dq[r])/np.median(dq[keep])) if r.any() else np.nan))
print()
print("=== 方法 C:ISR 等运动学距离重采样(步长扫大)===")
W=5.0
d_all=np.concatenate([e["dp"]+W*e["ang"] for e in EP])
print("  每帧运动学距离(mm 等效): 均值 %.2f 中位 %.2f 80%%=%.2f" % (d_all.mean(),np.median(d_all),np.percentile(d_all,80)))
for step in [3,4,5,6,8,10]:
    tot_out=sum(max(int((e["dp"]+W*e["ang"]).sum()//step)+1,2) for e in EP)
    tot_in=sum(e["n"] for e in EP)
    print("  步长 %2d mm: %d -> %6d 帧 (%.0f%%,压缩 %.0f%%)" % (step,tot_in,tot_out,tot_out/tot_in*100,(1-tot_out/tot_in)*100))
