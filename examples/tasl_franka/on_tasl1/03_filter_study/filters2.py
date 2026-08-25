import pickle, numpy as np
dp=np.load("/tmp/_dp.npy"); ang=np.load("/tmp/_ang.npy"); dq=np.load("/tmp/_dq.npy")
EP=pickle.load(open("/tmp/_ep.pkl","rb")); F=np.load("/tmp/_feat.npy")
N=len(dp)

print("=== 方法 B:SCIZOR 式 state-action 去重(不含进度预测模型)===")
X=(F-F.mean(0))/(F.std(0)+1e-8)
X=X/ (np.linalg.norm(X,axis=1,keepdims=True)+1e-8)          # 单位化 -> 余弦=点积
try:
    from sklearn.cluster import MiniBatchKMeans
    km=MiniBatchKMeans(n_clusters=200, random_state=0, n_init=3, batch_size=4096).fit(X)
    lab=km.labels_
except Exception as e:
    print("  (sklearn 不可用: %s,退化为随机投影分桶)"%e)
    rng=np.random.default_rng(0); P=rng.normal(size=(X.shape[1],8))
    lab=((X@P)>0)@(2**np.arange(8))
print("  特征 = [state(8), action(8)] 标准化后单位化,聚成 %d 类" % len(np.unique(lab)))
for tau in [0.995,0.99,0.98,0.95]:
    keep=np.ones(N,bool)
    for c in np.unique(lab):
        idx=np.where(lab==c)[0]
        if len(idx)<2: continue
        Xi=X[idx]; kept=[]
        for i in range(len(idx)):
            if not kept: kept.append(i); continue
            if (Xi[i]@Xi[kept].T).max()>tau: keep[idx[i]]=False
            else: kept.append(i)
    r=~keep
    print("  余弦相似度 τ=%.3f : 删 %d/%d = %4.1f%%  | 删掉帧的 max|dq| 中位 %.4f rad/s (保留帧 %.4f)"
          % (tau, r.sum(), N, r.mean()*100, np.median(dq[r]) if r.any() else float("nan"), np.median(dq[keep])))

print()
print("=== 方法 C:ISR 式轨迹标准化(等运动学距离重采样)===")
W=5.0   # 1 度 折算成 5 mm
off=0
for step_mm in [1.0, 2.0, 3.0]:
    tot_out=0; tot_in=0
    for e in EP:
        d=e["dp"]+W*e["ang"]
        c=np.concatenate([[0],np.cumsum(d)])
        n_out=max(int(c[-1]//step_mm)+1, 2)
        tot_out+=n_out; tot_in+=e["n"]
    print("  步长 %.1f mm(等效): %d -> %d 帧 (保留 %.1f%%, 压缩 %.1f%%)"
          % (step_mm, tot_in, tot_out, tot_out/tot_in*100, (1-tot_out/tot_in)*100))
print("  注:ISR 是【重采样】不是删帧 —— 停顿被压缩,运动段被均匀化,时序不断裂")
