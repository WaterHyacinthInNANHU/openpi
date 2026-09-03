import io, os, glob, json
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image

DS = ["T1-a","T1-b","T2-a-25ep","T2-b","T3-a-25ep","T3-b","T4-a-25ep","T4-b","T5-a-25ep","T5-b"]
ROOT = "/home/franka_desktop/rlinf_data/datasets"
OUT = "/tmp/contact"; os.makedirs(OUT, exist_ok=True)
S = 2   # upscale

def gb(e): return e["bytes"] if isinstance(e, dict) else e.as_py()["bytes"]
def dec(b): return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

for name in DS:
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        # T2-a is split
        d = os.path.join(ROOT, name.replace("-25ep","-15ep"))
        if not os.path.isdir(d): print("skip", name); continue
    eps = sorted(glob.glob(os.path.join(d,"data","chunk-000","*.parquet")))
    ep = eps[len(eps)//2]
    t = pq.read_table(ep); n = t.num_rows
    imgs = t.column("image").to_pylist(); wr = t.column("extra_view_image").to_pylist()
    tix = t.column("task_index").to_pylist()[0]
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(os.path.join(d,"meta","tasks.jsonl"))}
    picks = [int(n*0.08), int(n*0.5), min(n-1,int(n*0.92))]
    top = np.concatenate([dec(gb(imgs[i])) for i in picks], axis=1)
    bot = np.concatenate([dec(gb(wr[i])) for i in picks], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    h, w = grid.shape[:2]
    grid = cv2.resize(grid, (w*S, h*S), interpolation=cv2.INTER_CUBIC)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    hdr = np.zeros((46, w*S, 3), np.uint8)
    cv2.putText(hdr, f"{os.path.basename(d)}  ep{os.path.basename(ep)[8:14]}  n={n}  prompt={tasks.get(tix)!r}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"{os.path.basename(d)}.png"), np.concatenate([hdr, grid], axis=0))
    print("wrote", os.path.basename(d), "ep", os.path.basename(ep), "n", n, "| prompt:", tasks.get(tix))
