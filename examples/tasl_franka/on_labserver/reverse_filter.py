"""从 8/9 那对「过滤前/过滤后」数据集反推当时用的静止判据。"""
import glob, json
import numpy as np, pyarrow.parquet as pq
RAW="/data1/Franka_RealRobot/lerobot_home/franka/test_finetune"
FLT="/data1/Franka_RealRobot/lerobot_home/franka/test_finetune_idlefiltered"
FPS=15.0
tot_raw=tot_flt=0; removed_dq=[]; kept_dq=[]
for i in range(16):
    r=f"{RAW}/data/chunk-000/episode_{i:06d}.parquet"
    f=f"{FLT}/data/chunk-000/episode_{i:06d}.parquet"
    try:
        tr=pq.read_table(r, columns=["state","actions"]); tf=pq.read_table(f, columns=["state","actions"])
    except Exception as e:
        print("跳过", i, e); continue
    sr=np.array(tr.column("state").to_pylist(),float); sf=np.array(tf.column("state").to_pylist(),float)
    tot_raw+=len(sr); tot_flt+=len(sf)
    # 用 state 全 8 维做指纹匹配,找出哪些原始帧被保留了
    keyf={tuple(np.round(x,7)) for x in sf}
    keep=np.array([tuple(np.round(x,7)) in keyf for x in sr])
    dq=np.abs(np.diff(sr[:,1:8],axis=0)*FPS).max(axis=1)
    k=keep[:-1]
    removed_dq.append(dq[~k]); kept_dq.append(dq[k])
rm=np.concatenate(removed_dq); kp=np.concatenate(kept_dq)
print("原始 %d 帧 -> 过滤后 %d 帧,删掉 %d (%.1f%%)" % (tot_raw,tot_flt,tot_raw-tot_flt,(tot_raw-tot_flt)/tot_raw*100))
print()
print("=== 被删掉的帧的实测关节角速度 max|dq| (rad/s) ===")
print("  n=%d  分位 50%%=%.4f 90%%=%.4f 95%%=%.4f 99%%=%.4f max=%.4f" % (len(rm),*np.percentile(rm,[50,90,95,99,100])))
print("=== 被保留的帧 ===")
print("  n=%d  分位 1%%=%.4f 5%%=%.4f 10%%=%.4f 50%%=%.4f" % (len(kp),*np.percentile(kp,[1,5,10,50])))
print()
# 找最佳分割阈值
best=None
for th in np.logspace(-4,0,80):
    tp=(rm<th).sum(); fp=(kp<th).sum()
    f1=2*tp/(2*tp+fp+(len(rm)-tp)) if tp else 0
    if best is None or f1>best[1]: best=(th,f1,tp,fp)
print("最能区分「被删 vs 被留」的阈值: %.4f rad/s (F1=%.2f, 覆盖被删帧 %d/%d, 误伤保留帧 %d/%d)"
      % (best[0],best[1],best[2],len(rm),best[3],len(kp)))
