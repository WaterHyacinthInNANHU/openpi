import io, os, glob, json, sys
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image
ROOT="/home/franka_desktop/rlinf_data/datasets"; OUT="/tmp/contact"; S=2
def gb(e): return e["bytes"] if isinstance(e,dict) else e.as_py()["bytes"]
def dec(b): return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
def strip(name, want_tix=None, col="image", tag=""):
    d=os.path.join(ROOT,name)
    tasks={json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(d+"/meta/tasks.jsonl")}
    for ep in sorted(glob.glob(d+"/data/chunk-000/*.parquet")):
        t=pq.read_table(ep); tix=t.column("task_index").to_pylist()[0]
        if want_tix is None or tix==want_tix: break
    n=t.num_rows; imgs=t.column(col).to_pylist()
    picks=[int(n*f) for f in (0.02,0.2,0.4,0.6,0.8,0.98)]
    row=np.concatenate([dec(gb(imgs[min(i,n-1)])) for i in picks],axis=1)
    h,w=row.shape[:2]; row=cv2.resize(row,(w*S,h*S),interpolation=cv2.INTER_CUBIC)
    row=cv2.cvtColor(row,cv2.COLOR_RGB2BGR)
    hdr=np.zeros((40,w*S,3),np.uint8)
    cv2.putText(hdr,f"{name}{tag} {col} tix={tix} n={n} prompt={tasks.get(tix)!r}",(8,26),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    fn=os.path.join(OUT,f"strip_{name}{tag}_{col}.png")
    cv2.imwrite(fn,np.concatenate([hdr,row],axis=0)); print("wrote",fn)
strip("T4-b"); strip("T4-b",col="extra_view_image")
strip("T5-b"); strip("T3-a-25ep"); strip("T1-b",want_tix=0,tag="_tix0")
strip("T5-b", col="extra_view_image")
