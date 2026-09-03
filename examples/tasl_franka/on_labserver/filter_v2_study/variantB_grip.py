import sys, glob, json, numpy as np, pyarrow.parquet as pq
sys.path.insert(0,"/data1/Franka_RealRobot/openpi/examples/droid")
import compute_droid_nonidle_ranges as M
D="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"; FPS=15.0
files=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))
EP=[]
for f in files:
    t=pq.read_table(f,columns=["actions","state"])
    EP.append((np.stack(t.column("actions").to_pylist()).astype(float), np.stack(t.column("state").to_pylist()).astype(float)))
def segrule(idle, mi, mn, dl):
    n=len(idle); keep=np.ones(n,bool); i=0
    while i<n:
        if idle[i]:
            j=i
            while j<n and idle[j]: j+=1
            if j-i>=mi: keep[i:j]=False
            i=j
        else: i+=1
    out=np.zeros(n,bool); i=0
    while i<n:
        if keep[i]:
            j=i
            while j<n and keep[j]: j+=1
            if j-i>=mn: out[i:max(i,j-dl)]=True
            i=j
        else: i+=1
    return out
def meas_vel(S_):
    v=np.diff(S_[:,1:8],axis=0)*FPS; return np.vstack([v, v[-1:]])
def dilate(m,k):
    o=m.copy()
    for s in range(1,k+1): o[s:]|=m[:-s]; o[:-s]|=m[s:]
    return o
def run(name, idle_fn, guard):
    tot=kept=gm_all=gm_del=0; leftover=0
    for A,S_ in EP:
        n=len(A); idle=idle_fn(A,S_)
        if guard:
            dg=np.r_[0,np.abs(np.diff(A[:,7]))]>0.02
            idle&=~dilate(dg,guard)
        keep=segrule(idle,M.min_idle_len,M.min_non_idle_len,M.filter_last_n_in_ranges)
        tot+=n; kept+=keep.sum()
        gmask=np.r_[0,np.abs(np.diff(A[:,7]))]>0.05; gm_all+=gmask.sum(); gm_del+=(gmask&~keep).sum()
        m=np.abs(meas_vel(S_)).max(1)<0.01; i=0
        while i<n:
            if m[i]:
                j=i
                while j<n and m[j]: j+=1
                if j-i>15: leftover+=keep[i:j].sum()
                i=j
            else: i+=1
    print(f"{name:34s} 删 {(1-kept/tot)*100:5.1f}%  夹爪误删 {gm_del:4d}/{gm_all} ({gm_del/gm_all*100:4.1f}%)  长静止段残留 {leftover}")
B=lambda A,S_: np.r_[False, np.all(np.abs(np.diff(meas_vel(S_),axis=0))<1e-3,1)]
C=lambda A,S_: np.abs(meas_vel(S_)).max(1)<0.01
run("B 实测Δv<1e-3, 无保护",B,0)
run("B 实测Δv<1e-3 + 夹爪保护±5",B,5)
run("E 实测|v|<0.01 + 夹爪保护±5",C,5)
run("E 实测|v|<0.01 + 夹爪保护±8",C,8)
