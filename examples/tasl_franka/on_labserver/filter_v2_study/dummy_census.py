import glob, json, collections, numpy as np, pyarrow.parquet as pq
D="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"; FPS=15.0; VS=0.509
tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D+"/meta/tasks.jsonl")}
rng=json.load(open(D+"/meta/nonidle_ranges.json"))
files=sorted(glob.glob(D+"/data/chunk-000/*.parquet"))
tot=0; C=collections.Counter(); per_task=collections.defaultdict(lambda: collections.Counter())
head=collections.defaultdict(list); tail=collections.defaultdict(list); pos=collections.defaultdict(list)
grip_only=0; cmd_idle_but_sampleable=0; sampleable=0; last15=0
absA=[]
for f in files:
    t=pq.read_table(f,columns=["actions","state","task_index","episode_index"])
    A=np.stack(t.column("actions").to_pylist()).astype(float); Sx=np.stack(t.column("state").to_pylist()).astype(float)
    tix=t.column("task_index")[0].as_py(); ep=t.column("episode_index")[0].as_py(); n=len(A); tot+=n
    absA.append(np.abs(A[:,:7]).max(1))
    crit={
      "exact0": np.all(A[:,:7]==0,1),
      "cmd<0.01": np.abs(A[:,:7]).max(1)<0.01,
      "cmd<0.02": np.abs(A[:,:7]).max(1)<0.02,
      "cmd<0.05": np.abs(A[:,:7]).max(1)<0.05,
      "meas<0.01rad/s": np.r_[False, np.abs(np.diff(Sx[:,1:8],axis=0)*FPS).max(1)<0.01],
      "official(dA*vs<1e-3)": np.r_[np.all(np.abs(np.diff(A[:,:7],axis=0))*VS<1e-3,1), False],
    }
    dg=np.r_[0,np.abs(np.diff(A[:,7]))]
    for k,m in crit.items():
        C[k]+=m.sum(); per_task[tix][k]+=m.sum()
        # head/tail runs
        i=0
        while i<n and m[i]: i+=1
        head[k].append(i)
        j=0
        while j<n and m[n-1-j]: j+=1
        tail[k].append(j)
        pos[k].append(np.where(m)[0]/max(n-1,1))
    per_task[tix]["n"]+=n
    m=crit["cmd<0.02"]
    grip_only+=(m&(dg>0.05)).sum()
    keep=np.zeros(n,bool)
    for s,e in rng.get(str(ep),[]): keep[s:e]=True
    sampleable+=keep.sum(); cmd_idle_but_sampleable+=(m&keep).sum()
    last15+=min(15,n)
absA=np.concatenate(absA)
print("frames",tot,"| sampleable after official filter",sampleable)
print("max|cmd| quantiles 5/25/50/75/95:",np.round(np.quantile(absA,[.05,.25,.5,.75,.95]),3).tolist())
print("\n== dummy/idle fraction under each criterion ==")
for k in C:
    h=np.array(head[k]); tl=np.array(tail[k]); p=np.concatenate(pos[k])
    hist,_=np.histogram(p,bins=5,range=(0,1))
    print(f"{k:22s} {C[k]/tot*100:5.1f}%  head-run med {np.median(h):3.0f} max {h.max():3d} (>15: {(h>15).sum():3d}/250) | tail-run med {np.median(tl):3.0f} max {tl.max():3d} (>15: {(tl>15).sum():3d}/250) | pos-hist(5 bins) {hist.tolist()}")
print(f"\ncmd<0.02 frames that are STILL sampleable after official filter: {cmd_idle_but_sampleable} ({cmd_idle_but_sampleable/max(C['cmd<0.02'],1)*100:.0f}% of them)")
print(f"cmd<0.02 frames with gripper moving (dg>0.05): {grip_only}")
print(f"frames in last 15 of episode (chunk end-pad zone): {last15} ({last15/tot*100:.1f}%)")
print("\n== per task: cmd<0.02 / meas<0.01 / official ==")
for k in sorted(per_task):
    v=per_task[k]; print(f"  tix{k} n={v['n']:5d}  cmd<0.02 {v['cmd<0.02']/v['n']*100:5.1f}%  meas {v['meas<0.01rad/s']/v['n']*100:5.1f}%  official {v['official(dA*vs<1e-3)']/v['n']*100:5.1f}%  {tasks[k][:45]}")
