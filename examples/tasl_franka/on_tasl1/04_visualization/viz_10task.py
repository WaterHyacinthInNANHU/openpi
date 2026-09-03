"""合并数据集里每个 task 随机抽一条轨迹,渲染成左右拼接视频。
左半 = 桌面外部视角(image),右半 = 腕部相机(extra_view_image)。
"""
import io, os, json, glob, random
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image

D   = "/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
OUT = "/tmp/task_videos_10"; os.makedirs(OUT, exist_ok=True)
SCALE = 2
random.seed(21)

tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(D+"/meta/tasks.jsonl")}
fps   = json.load(open(D+"/meta/info.json"))["fps"]

# 按 task_index 归并 episode
by_tix = {}
for l in open(D+"/meta/episodes.jsonl"):
    e = json.loads(l)
    f = "%s/data/chunk-000/episode_%06d.parquet" % (D, e["episode_index"])
    tix = pq.read_table(f, columns=["task_index"]).column("task_index").to_pylist()[0]
    by_tix.setdefault(tix, []).append(e["episode_index"])

def gb(e): return e["bytes"] if isinstance(e, dict) else e.as_py()["bytes"]
def dec(b): return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

for tix in sorted(by_tix):
    ep = random.choice(by_tix[tix])
    prompt = tasks[tix]
    t = pq.read_table("%s/data/chunk-000/episode_%06d.parquet" % (D, ep))
    n = t.num_rows
    imgs = t.column("image").to_pylist(); wr = t.column("extra_view_image").to_pylist()
    h, w = dec(gb(imgs[0])).shape[:2]
    name = "T%02d" % tix
    vw = cv2.VideoWriter("%s/%s.mp4" % (OUT, name), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w*SCALE*2, h*SCALE))
    for i in range(n):
        tile = cv2.resize(np.concatenate([dec(gb(imgs[i])), dec(gb(wr[i]))], axis=1),
                          (w*SCALE*2, h*SCALE), interpolation=cv2.INTER_CUBIC)
        # PIL 解码是 RGB,cv2.VideoWriter 要 BGR,不转红蓝会对调
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, "task %d  %s" % (tix, prompt), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(tile, "episode %06d   frame %d/%d" % (ep, i+1, n), (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205,205,205), 1, cv2.LINE_AA)
        vw.write(tile)
    vw.release()
    print("tix %d  ep %06d  %4d frames  | %s" % (tix, ep, n, prompt), flush=True)
print("DONE")
