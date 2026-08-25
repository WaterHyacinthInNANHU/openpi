"""判断 actions 到底是不是「机械臂实际执行的关节速度」,并用 state 差分重新判静止。"""
import json, glob
import numpy as np, pyarrow.parquet as pq
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
FPS=15.0
S=[];A=[];DS=[]
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    t=pq.read_table(f, columns=["state","actions"])
    s=np.array(t.column("state").to_pylist(),dtype=float)     # (n,8) 关节角+夹爪
    a=np.array(t.column("actions").to_pylist(),dtype=float)   # (n,8)
    d=np.diff(s,axis=0)*FPS                                   # (n-1,8) 实测关节角速度 rad/s
    S.append(s); A.append(a); DS.append(d)
s=np.concatenate(S); a=np.concatenate(A); d=np.concatenate(DS)
print("总帧数:", len(s))
print()
print("=== actions 是不是实测关节速度?对每个关节做相关 ===")
for j in range(7):
    aa=np.concatenate([x[:-1,j] for x in A]); dd=d[:,j]
    r=np.corrcoef(aa,dd)[0,1]
    k=np.polyfit(aa,dd,1)[0] if np.std(aa)>0 else float("nan")
    print("  关节%d: 相关系数 %+.3f | 线性斜率 d(state)/action = %6.3f | action范围[%+.2f,%+.2f] 实测速度范围[%+.2f,%+.2f] rad/s"
          % (j, r, k, aa.min(), aa.max(), dd.min(), dd.max()))
print()
print("=== 用【实测关节角速度】判静止(rad/s,7 关节取最大绝对值) ===")
m=np.abs(d[:,:7]).max(axis=1)
print("  分位: 1%%=%.4f  5%%=%.4f  25%%=%.4f  中位=%.4f  75%%=%.4f  99%%=%.4f  max=%.3f"
      % tuple(np.percentile(m,[1,5,25,50,75,99,100])))
for th in [0.001,0.005,0.01,0.02,0.05,0.1]:
    print("    < %-6g rad/s : %6d / %d  (%.1f%%)" % (th,(m<th).sum(),len(m),(m<th).mean()*100))
