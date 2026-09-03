import glob, numpy as np, pyarrow.parquet as pq
D="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"; FPS=15.0
files=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))
CMD=[];VEL={0:[],1:[],2:[]}
for f in files:
    t=pq.read_table(f,columns=["actions","state"])
    A=np.stack(t.column("actions").to_pylist()).astype(float)[:,:7]; Q=np.stack(t.column("state").to_pylist()).astype(float)[:,1:8]
    v=np.diff(Q,axis=0)*FPS  # v[t] = motion between t and t+1
    n=len(v)
    for lag in (0,1,2):
        # cmd[t-lag] vs v[t]
        if n-lag<=0: continue
        VEL[lag].append(v[lag:]); 
    CMD.append(A)
    # store per-lag cmd slices later
A_all=np.concatenate(CMD)
for lag in (0,1,2):
    cs=[];vs=[]
    for f in files:
        t=pq.read_table(f,columns=["actions","state"])
        A=np.stack(t.column("actions").to_pylist()).astype(float)[:,:7]; Q=np.stack(t.column("state").to_pylist()).astype(float)[:,1:8]
        v=np.diff(Q,axis=0)*FPS; n=len(v)
        if n-lag<=0: continue
        cs.append(A[:n][: n-lag] if lag else A[:n]); vs.append(v[lag:])
    c=np.concatenate(cs); v=np.concatenate(vs)
    r=[np.corrcoef(c[:,j],v[:,j])[0,1] for j in range(7)]
    slope=[np.polyfit(c[:,j],v[:,j],1)[0] for j in range(7)]
    print(f"lag {lag}: corr per joint {np.round(r,2).tolist()} | slope(rad/s per unit cmd) {np.round(slope,3).tolist()}")
    if lag==1: C1,V1=c,v
# use lag 1 for deadband analysis
cm=np.abs(C1).max(1); vm=np.abs(V1).max(1)
print("\n== lag1: measured max|vel| vs commanded max|cmd| (bins) ==")
bins=[0,0.05,0.08,0.1,0.13,0.16,0.2,0.25,0.3,0.4,0.6,1.01]
print(" |cmd| bin      n     frac robot idle(<0.01rad/s)   median |vel|   median |vel|/(|cmd|*0.509)")
for a,b in zip(bins[:-1],bins[1:]):
    m=(cm>=a)&(cm<b)
    if m.sum()==0: continue
    print(f" [{a:.2f},{b:.2f})  {m.sum():6d}   {(vm[m]<0.01).mean()*100:5.1f}%                  {np.median(vm[m]):.3f}         {np.median(vm[m]/(cm[m]*0.509+1e-9)):.2f}")
idle=vm<0.01
print(f"\nmax|cmd| on robot-idle frames: median {np.median(cm[idle]):.3f}  p75 {np.quantile(cm[idle],.75):.3f}  p90 {np.quantile(cm[idle],.9):.3f}")
print(f"max|cmd| on moving frames   : median {np.median(cm[~idle]):.3f}  p75 {np.quantile(cm[~idle],.75):.3f}  p90 {np.quantile(cm[~idle],.9):.3f}")
# per-joint: on idle frames, which joint carries the residual command?
print("per-joint mean|cmd| on idle frames:",np.round(np.abs(C1[idle]).mean(0),3).tolist())
print("per-joint mean|cmd| on moving frames:",np.round(np.abs(C1[~idle]).mean(0),3).tolist())
