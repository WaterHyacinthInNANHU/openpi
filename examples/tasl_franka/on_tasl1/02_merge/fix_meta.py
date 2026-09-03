import json
OUT="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
SPEC=json.load(open("/tmp/prompts.json"))["tasks"]
prompts={i: s["new"] for i,s in enumerate(SPEC)}
# tasks.jsonl 重写
with open(OUT+"/meta/tasks.jsonl","w") as f:
    for i,s in enumerate(SPEC):
        f.write(json.dumps({"task_index":i,"task":s["new"]},ensure_ascii=False)+"\n")
# episodes.jsonl:按 parquet 里的真实 task_index 重写 tasks 字段
import glob, pyarrow.parquet as pq
rows=[json.loads(l) for l in open(OUT+"/meta/episodes.jsonl")]
changed=0
for r in rows:
    f="%s/data/chunk-000/episode_%06d.parquet"%(OUT,r["episode_index"])
    tix=pq.read_table(f,columns=["task_index"]).column("task_index").to_pylist()[0]
    if r["tasks"]!=[prompts[tix]]:
        r["tasks"]=[prompts[tix]]; changed+=1
with open(OUT+"/meta/episodes.jsonl","w") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("episodes.jsonl 改动条数:", changed)
print("tasks.jsonl 现在:")
for l in open(OUT+"/meta/tasks.jsonl"): print("  ", l.rstrip())
