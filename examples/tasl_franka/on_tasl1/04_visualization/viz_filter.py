"""方法A 过滤的删前/删后对照视频:被删帧标红 + 底部完整时间轴。"""
import sys, io, os, json, glob, random
sys.path.insert(0,"/tmp")
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image
from fk import fk

D="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
OUT="/tmp/filter_videos"; os.makedirs(OUT,exist_ok=True)
FPS=15.0; S=2
TH_POS, TH_ROT = 0.198, 0.044          # 全局 20% 分位(mm/帧, 度/帧)
MIN_IDLE, MIN_NON, DROP_LAST = 7, 16, 10   # 官方段级规则

def seg_rules(idle):
    n=len(idle); keep=np.ones(n,bool); i=0
    while i<n:
        if idle[i]:
            j=i
            while j<n and idle[j]: j+=1
            if j-i>=MIN_IDLE: keep[i:j]=False      # 连续 >=7 帧 idle 整段删
            i=j
        else: i+=1
    out=np.zeros(n,bool); i=0
    while i<n:
        if keep[i]:
            j=i
            while j<n and keep[j]: j+=1
            if j-i>=MIN_NON: out[i:max(i,j-DROP_LAST)]=True   # 段>=16帧才留,再砍段尾10帧
            i=j
        else: i+=1
    return out

def gb(e): return e["bytes"] if isinstance(e,dict) else e.as_py()["bytes"]
def dec(b): return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D+"/meta/tasks.jsonl")}
by={}
for l in open(D+"/meta/episodes.jsonl"):
    e=json.loads(l); f="%s/data/chunk-000/episode_%06d.parquet"%(D,e["episode_index"])
    tix=pq.read_table(f,columns=["task_index"]).column("task_index").to_pylist()[0]
    by.setdefault(tix,[]).append(e["episode_index"])
random.seed(21)
print("%-4s %-6s %5s %6s %6s %6s %5s %6s" % ("tix","ep","帧数","逐帧命中","最终删","保留","段数","最长段"))
for tix in sorted(by):
    ep=random.choice(by[tix]); prompt=tasks[tix]
    t=pq.read_table("%s/data/chunk-000/episode_%06d.parquet"%(D,ep))
    s=np.array(t.column("state").to_pylist(),float); n=len(s)
    imgs=t.column("image").to_pylist(); wr=t.column("extra_view_image").to_pylist()
    T=fk(s[:,1:8]); p=T[:,:3,3]; R=T[:,:3,:3]
    dp=np.concatenate([[0],np.linalg.norm(np.diff(p,axis=0),axis=1)*1000])
    dR=np.einsum("nij,nkj->nik",R[1:],R[:-1])
    ang=np.concatenate([[0],np.degrees(np.arccos(np.clip((np.trace(dR,axis1=1,axis2=2)-1)/2,-1,1)))])
    raw=(dp<TH_POS)&(ang<TH_ROT)            # 逐帧判据命中
    keep=seg_rules(raw)                      # 段级规则后最终保留
    # 统计段数
    segs=[];i=0
    while i<n:
        if keep[i]:
            j=i
            while j<n and keep[j]: j+=1
            segs.append(j-i); i=j
        else: i+=1
    print("%-4d %-6d %5d %6d(%3.0f%%) %5d(%3.0f%%) %5d %5d %6d" %
          (tix,ep,n,raw.sum(),raw.mean()*100,(~keep).sum(),(~keep).mean()*100,keep.sum(),len(segs),max(segs) if segs else 0))
    h,w=224,224; W=w*S*2; HDR=76; BAR=46
    vw=cv2.VideoWriter("%s/T%02d.mp4"%(OUT,tix),cv2.VideoWriter_fourcc(*"mp4v"),FPS,(W,HDR+h*S+BAR))
    for i in range(n):
        tile=cv2.resize(np.concatenate([dec(gb(imgs[i])),dec(gb(wr[i]))],axis=1),(W,h*S),interpolation=cv2.INTER_CUBIC)
        tile=cv2.cvtColor(tile,cv2.COLOR_RGB2BGR)
        if not keep[i]:
            ov=tile.copy(); cv2.rectangle(ov,(0,0),(W-1,h*S-1),(0,0,220),-1)
            tile=cv2.addWeighted(ov,0.28,tile,0.72,0)
            cv2.rectangle(tile,(2,2),(W-3,h*S-3),(0,0,255),4)
        hdr=np.zeros((HDR,W,3),np.uint8)
        cv2.putText(hdr,"task %d  %s"%(tix,prompt[:64]),(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(hdr,"ep%06d  frame %d/%d   dpos %.3f mm  drot %.3f deg"%(ep,i+1,n,dp[i],ang[i]),
                    (10,46),cv2.FONT_HERSHEY_SIMPLEX,0.45,(190,190,190),1,cv2.LINE_AA)
        st="DROP (idle)" if not keep[i] else "KEEP"
        cv2.putText(hdr,st,(10,68),cv2.FONT_HERSHEY_SIMPLEX,0.52,((0,0,255) if not keep[i] else (0,220,0)),2,cv2.LINE_AA)
        cv2.putText(hdr,"red = would be removed by filter A",(W-330,68),cv2.FONT_HERSHEY_SIMPLEX,0.42,(150,150,150),1,cv2.LINE_AA)
        bar=np.zeros((BAR,W,3),np.uint8)
        for x in range(W):
            k=int(x/W*n)
            bar[8:30,x]=(0,190,0) if keep[k] else (0,0,230)
        cx=int(i/max(n-1,1)*(W-1)); cv2.line(bar,(cx,4),(cx,34),(255,255,255),2)
        cv2.putText(bar,"timeline: green=keep  red=drop   |  kept %d/%d (%.0f%%), %d segments"%(keep.sum(),n,keep.mean()*100,len(segs)),
                    (10,43),cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),1,cv2.LINE_AA)
        vw.write(np.concatenate([hdr,tile,bar],axis=0))
    vw.release()
print("DONE")
