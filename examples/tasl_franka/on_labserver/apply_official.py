"""用 openpi 官方 compute_droid_nonidle_ranges.py 的判据,在我们数据上实测。
官方: idle = all(|joint_vel[t+1]-joint_vel[t]| < 1e-3)
      min_idle_len=7, min_non_idle_len=16, filter_last_n_in_ranges=10
我们的 actions 是归一化关节速度(斜率 ~0.509 rad/s per unit),
所以 1e-3 rad/s 对应 action 差分阈值 ~0.002。
"""
import glob
import numpy as np, pyarrow.parquet as pq
D="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"
SCALE=0.509
def ranges(idle, min_idle=7, min_non=16, drop_last=10):
    n=len(idle); keep=np.ones(n,bool)
    i=0
    while i<n:
        if idle[i]:
            j=i
            while j<n and idle[j]: j+=1
            if j-i>=min_idle: keep[i:j]=False
            i=j
        else: i+=1
    out=[];i=0
    while i<n:
        if keep[i]:
            j=i
            while j<n and keep[j]: j+=1
            if j-i>=min_non: out.append((i,max(i,j-drop_last)))
            i=j
        else: i+=1
    return out
for th_rad in [1e-3, 5e-3, 1e-2]:
    th=th_rad/SCALE
    tot=kept=0; n_ep_drop=0
    for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
        a=np.array(pq.read_table(f,columns=["actions"]).column("actions").to_pylist(),float)[:,:7]
        idle=np.all(np.abs(np.diff(a,axis=0))<th,axis=1)
        idle=np.concatenate([idle,[idle[-1]]])
        rs=ranges(idle); k=sum(e-s for s,e in rs)
        tot+=len(a); kept+=k
        if k==0: n_ep_drop+=1
    print("官方判据 @ %.0e rad/s (action 差分 <%.4f):保留 %d/%d 帧 (%.1f%%),删掉 %.1f%%,整条被删的 episode %d"
          % (th_rad, th, kept, tot, kept/tot*100, (1-kept/tot)*100, n_ep_drop))
