"""把 10 个 task 的 11 个 LeRobot 数据集合并成一个。

三件必须做对、否则会静默训坏的事:
 1. task_index 列必须按数据集重映射 —— 每个源数据集都从 0 编号,直接拼接会让
    250 条全部落到全局 index 0 那一个指令上,加载不报错、训练不报错,但训出来的
    模型完全无视语言。
 2. 重写 parquet 时必须保留内嵌的 huggingface schema 元数据(它把 image 列声明为
    {"_type":"Image"}),丢了会在迭代时报 Could not infer dtype of dict。
    做法:用原 schema 构造新 table。
 3. episode_index 重编号 0..249,全局 index 连续 0..N-1。
"""
import json, os, glob, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

SRC  = "/home/franka_desktop/rlinf_data/datasets"
OUT  = "/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
SPEC = json.load(open("/tmp/prompts.json"))["tasks"]

if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "data", "chunk-000"))
os.makedirs(os.path.join(OUT, "meta"))

REMAP = ("episode_index", "index", "task_index")

ep_out = 0
idx_off = 0
episodes, stats_out = [], []
info_ref = None

for tix, spec in enumerate(SPEC):
    prompt = spec["new"]
    for dname in spec["dirs"]:
        d = os.path.join(SRC, dname)
        info = json.load(open(os.path.join(d, "meta", "info.json")))
        if info_ref is None: info_ref = info
        assert info["fps"] == info_ref["fps"], dname
        assert info["features"].keys() == info_ref["features"].keys(), dname
        stats_by_ep = {json.loads(l)["episode_index"]: json.loads(l)["stats"]
                       for l in open(os.path.join(d, "meta", "episodes_stats.jsonl"))}
        for pqf in sorted(glob.glob(os.path.join(d, "data", "chunk-000", "*.parquet"))):
            src_ep = int(os.path.basename(pqf)[8:14])
            t = pq.read_table(pqf)
            n = t.num_rows
            schema = t.schema                      # 带 b"huggingface" 元数据
            assert schema.metadata and b"huggingface" in schema.metadata, pqf
            cols = []
            for f in schema:
                if f.name == "episode_index":
                    cols.append(pa.array(np.full(n, ep_out), type=f.type))
                elif f.name == "index":
                    cols.append(pa.array(np.arange(idx_off, idx_off + n), type=f.type))
                elif f.name == "task_index":
                    cols.append(pa.array(np.full(n, tix), type=f.type))
                else:
                    cols.append(t.column(f.name))
            new_t = pa.Table.from_arrays(cols, schema=schema)   # schema 原样带过来
            out_pq = os.path.join(OUT, "data", "chunk-000", "episode_%06d.parquet" % ep_out)
            with pq.ParquetWriter(out_pq, schema) as w:
                w.write_table(new_t)

            episodes.append({"episode_index": ep_out, "tasks": [prompt], "length": n})
            st = dict(stats_by_ep[src_ep])
            ix = np.arange(idx_off, idx_off + n, dtype=float)
            st["episode_index"] = {"min":[ep_out],"max":[ep_out],"mean":[float(ep_out)],"std":[0.0],"count":[n]}
            st["task_index"]    = {"min":[tix],"max":[tix],"mean":[float(tix)],"std":[0.0],"count":[n]}
            st["index"]         = {"min":[int(ix[0])],"max":[int(ix[-1])],"mean":[float(ix.mean())],
                                   "std":[float(ix.std())],"count":[n]}
            stats_out.append({"episode_index": ep_out, "stats": st})

            ep_out += 1; idx_off += n
        print("  %-12s -> tix %d  (%s)" % (dname, tix, prompt), flush=True)

with open(os.path.join(OUT,"meta","tasks.jsonl"),"w") as f:
    for tix, spec in enumerate(SPEC):
        f.write(json.dumps({"task_index": tix, "task": spec["new"]}, ensure_ascii=False)+"\n")
with open(os.path.join(OUT,"meta","episodes.jsonl"),"w") as f:
    for e in episodes: f.write(json.dumps(e, ensure_ascii=False)+"\n")
with open(os.path.join(OUT,"meta","episodes_stats.jsonl"),"w") as f:
    for s in stats_out: f.write(json.dumps(s)+"\n")

info = dict(info_ref)
info["total_episodes"] = ep_out
info["total_frames"]   = idx_off
info["total_tasks"]    = len(SPEC)
info["total_chunks"]   = 1
info["total_videos"]   = 0
info["splits"]         = {"train": "0:%d" % ep_out}
json.dump(info, open(os.path.join(OUT,"meta","info.json"),"w"), indent=4)

print("\nDONE  episodes=%d  frames=%d  tasks=%d  ->  %s" % (ep_out, idx_off, len(SPEC), OUT))
