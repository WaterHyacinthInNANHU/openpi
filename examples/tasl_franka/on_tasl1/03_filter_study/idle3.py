import glob
import numpy as np, pyarrow.parquet as pq
D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"; FPS=15.0
A=[];DQ=[];GR=[]
for f in sorted(glob.glob(D+"/data/chunk-000/*.parquet")):
    t=pq.read_table(f, columns=["state","actions"])
    s=np.array(t.column("state").to_pylist(),float)   # [grip, q0..q6]
    a=np.array(t.column("actions").to_pylist(),float) # [dq0..dq6, grip]
    dq=np.diff(s[:,1:8],axis=0)*FPS                   # 实测关节角速度 rad/s
    A.append(a[:-1,:7]); DQ.append(dq); GR.append(s[:,0])
a=np.concatenate(A); dq=np.concatenate(DQ)
print("对齐后的帧数:", len(a))
print()
print("=== action[dq_j] vs 实测 Δq_j —— 现在对齐了 ===")
for j in range(7):
    r=np.corrcoef(a[:,j],dq[:,j])[0,1]; k=np.polyfit(a[:,j],dq[:,j],1)[0]
    print("  q%d: 相关 %+.3f | 斜率 %6.3f rad/s per unit action" % (j,r,k))
ra=np.corrcoef(a.ravel(),dq.ravel())[0,1]
print("  全部关节合起来: 相关 %+.3f | 斜率 %.4f" % (ra, np.polyfit(a.ravel(),dq.ravel(),1)[0]))
print()
print("=== 静止判据一:实测关节角速度 max|Δq| (rad/s) ===")
m=np.abs(dq).max(axis=1)
print("  分位 1%%=%.4f 5%%=%.4f 25%%=%.4f 中位=%.4f 75%%=%.4f 99%%=%.4f" % tuple(np.percentile(m,[1,5,25,50,75,99])))
for th in [0.001,0.005,0.01,0.02,0.05]:
    print("    < %-6g rad/s : %6d / %d  (%.1f%%)" % (th,(m<th).sum(),len(m),(m<th).mean()*100))
print()
print("=== 静止判据二:指令 max|dq_action| ===")
ma=np.abs(a).max(axis=1)
print("  分位 1%%=%.4f 5%%=%.4f 25%%=%.4f 中位=%.4f 75%%=%.4f 99%%=%.4f" % tuple(np.percentile(ma,[1,5,25,50,75,99])))
for th in [0.01,0.02,0.05,0.1]:
    print("    < %-5g : %6d / %d  (%.1f%%)" % (th,(ma<th).sum(),len(ma),(ma<th).mean()*100))
print()
print("=== 两个判据同时判静止(既没发指令、也没动) ===")
for tha,thd in [(0.05,0.01),(0.1,0.02)]:
    both=(ma<tha)&(m<thd)
    print("    |action|<%-4g 且 |Δq|<%-5g : %6d / %d  (%.1f%%)" % (tha,thd,both.sum(),len(both),both.mean()*100))
