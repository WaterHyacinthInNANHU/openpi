import json, glob
import numpy as np, pyarrow.parquet as pq
M="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
S="/home/franka_desktop/rlinf_data/datasets"
SPEC=json.load(open("/tmp/prompts.json"))["tasks"]
src_tot=0
for sp in SPEC:
    for d in sp["dirs"]:
        src_tot+=json.load(open(S+"/"+d+"/meta/info.json"))["total_frames"]
mgf=json.load(open(M+"/meta/info.json"))["total_frames"]
print("源 11 个数据集帧数合计 :", src_tot)
print("合并后 info.json 帧数  :", mgf)
print("→", "完全一致,一帧没丢(= 没做任何过滤)" if src_tot==mgf else "不一致!")
print()
tot=n_succ_f=n_interv=still=eps=0; ep_all=ep_any=ep_iv=0; lens=[]
for f in sorted(glob.glob(M+"/data/chunk-000/*.parquet")):
    t=pq.read_table(f, columns=["is_success","intervene_flag","actions"])
    n=t.num_rows; eps+=1; lens.append(n); tot+=n
    su=np.array([x[0] if isinstance(x,(list,tuple)) else x for x in t.column("is_success").to_pylist()])
    iv=np.array([x[0] if isinstance(x,(list,tuple)) else x for x in t.column("intervene_flag").to_pylist()])
    n_succ_f+=int(su.sum()); n_interv+=int(iv.sum())
    ep_all+=int(su.all()); ep_any+=int(su.any()); ep_iv+=int(iv.any())
    a=np.array(t.column("actions").to_pylist(), dtype=float)
    still+=int((np.abs(a[:,:7]).max(axis=1) < 1e-3).sum())
print("episode 数:", eps, " 总帧数:", tot)
print("episode 长度: min %d  中位 %d  max %d  均值 %.0f" % (min(lens), int(np.median(lens)), max(lens), tot/eps))
print()
print("is_success=True 的帧     : %6d / %d  (%.1f%%)" % (n_succ_f, tot, n_succ_f/tot*100))
print("至少一帧 success 的 ep   : %d / %d" % (ep_any, eps))
print("整条都 success 的 ep     : %d / %d" % (ep_all, eps))
print("intervene_flag=True 的帧 : %6d / %d  (%.1f%%)" % (n_interv, tot, n_interv/tot*100))
print("有人工介入的 ep          : %d / %d" % (ep_iv, eps))
print()
print("关节速度 <1e-3 的静止帧  : %6d / %d  (%.1f%%)  ← 未被过滤" % (still, tot, still/tot*100))
