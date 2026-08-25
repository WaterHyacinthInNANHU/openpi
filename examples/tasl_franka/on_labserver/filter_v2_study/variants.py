import sys, glob, json, numpy as np, pyarrow.parquet as pq
sys.path.insert(0,"/data1/Franka_RealRobot/openpi/examples/droid")
import compute_droid_nonidle_ranges as M
D="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"; FPS=15.0
tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D+"/meta/tasks.jsonl")}
files=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))
EP=[]
for f in files:
    t=pq.read_table(f,columns=["actions","state","task_index","episode_index"])
    A=np.stack(t.column("actions").to_pylist()).astype(float); S_=np.stack(t.column("state").to_pylist()).astype(float)
    EP.append((t.column("episode_index")[0].as_py(), t.column("task_index")[0].as_py(), A, S_))

def segrule(idle, mi, mn, dl):
    """openpi 段级规则: 连续>=mi 的 idle 段删掉; 剩余段<mn 删; 每段尾砍 dl. 返回 keep mask"""
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
    v=np.diff(S_[:,1:8],axis=0)*FPS       # v[t]: t->t+1
    return np.vstack([v, v[-1:]])         # 对齐到 cmd[t] (lag≈1)

VAR={
 "A 官方原样 (Δcmd*0.509<1e-3)":      lambda A,S_: np.r_[False, np.all(np.abs(np.diff(A[:,:7],axis=0))*0.509<1e-3,1)],
 "B 官方规则+实测Δv<1e-3":            lambda A,S_: np.r_[False, np.all(np.abs(np.diff(meas_vel(S_),axis=0))<1e-3,1)],
 "C 实测|v|max<0.01 rad/s":           lambda A,S_: np.abs(meas_vel(S_)).max(1)<0.01,
 "D 实测|v|max<0.02 rad/s":           lambda A,S_: np.abs(meas_vel(S_)).max(1)<0.02,
 "E C + 夹爪保护(|Δgrip|>0.02 ±5帧)": None,
 "F |cmd|max<0.13 (死区指令)":         lambda A,S_: np.abs(A[:,:7]).max(1)<0.13,
}
def dilate(m,k):
    o=m.copy()
    for s in range(1,k+1): o[s:]|=m[:-s]; o[:-s]|=m[s:]
    return o
print(f"{'variant':38s} {'idle%':>6s} {'删除%':>6s} {'夹爪动作帧误删':>14s} {'整条删':>6s} {'段/条 med/max':>13s} {'tail-run>15 剩余':>16s}")
for name,fn in VAR.items():
    tot=kept=0; gm_all=gm_del=0; whole=0; nseg=[]; idle_tot=0; tail_left=0; per_task={}
    for ep,tix,A,S_ in EP:
        n=len(A)
        if name.startswith("E"):
            idle=np.abs(meas_vel(S_)).max(1)<0.01
            dg=np.r_[0,np.abs(np.diff(A[:,7]))]>0.02
            idle&=~dilate(dg,5)
        else: idle=fn(A,S_)
        idle_tot+=idle.sum()
        keep=segrule(idle, M.min_idle_len, M.min_non_idle_len, M.filter_last_n_in_ranges)
        tot+=n; kept+=keep.sum()
        gmask=np.r_[0,np.abs(np.diff(A[:,7]))]>0.05; gm_all+=gmask.sum(); gm_del+=(gmask&~keep).sum()
        if keep.sum()==0: whole+=1
        c=0;i=0
        while i<n:
            if keep[i]:
                j=i
                while j<n and keep[j]: j+=1
                c+=1;i=j
            else:i+=1
        nseg.append(c)
        # kept frames that sit in a measured-idle run >15 (dummy leftover)
        m=np.abs(meas_vel(S_)).max(1)<0.01
        i=0
        while i<n:
            if m[i]:
                j=i
                while j<n and m[j]: j+=1
                if j-i>15: tail_left+=keep[i:j].sum()
                i=j
            else:i+=1
        p=per_task.setdefault(tix,[0,0]); p[0]+=keep.sum(); p[1]+=n
    nseg=np.array(nseg)
    print(f"{name:38s} {idle_tot/tot*100:5.1f}% {(1-kept/tot)*100:5.1f}% {gm_del:5d}/{gm_all} ({gm_del/gm_all*100:4.1f}%) {whole:6d} {np.median(nseg):6.0f}/{nseg.max():<6d} {tail_left:8d}")
    if name.startswith("C") or name.startswith("E"):
        print("     per-task kept%: "+"  ".join(f"t{k}:{v[0]/v[1]*100:.0f}" for k,v in sorted(per_task.items())))
