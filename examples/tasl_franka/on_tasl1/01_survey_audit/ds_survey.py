import json, glob, os
rows = []
for d in sorted(glob.glob(os.path.expanduser("~/rlinf_data/datasets/T[1-5]-*"))):
    if d.endswith("_svo"): continue
    inf = os.path.join(d, "meta", "info.json")
    if not os.path.exists(inf):
        rows.append((os.path.basename(d), "-", "-", "-", "no info.json")); continue
    i = json.load(open(inf))
    try:
        ts = [json.loads(l)["task"] for l in open(os.path.join(d,"meta","tasks.jsonl"))]
    except Exception as e:
        ts = ["<unreadable: %s>" % e]
    feats = sorted(i["features"].keys())
    rows.append((os.path.basename(d), i["total_episodes"], i["total_frames"],
                 i.get("fps"), " | ".join(ts), i["splits"]["train"], i["codebase_version"], feats))
tot_ep = tot_fr = 0
for r in rows:
    if r[1] == "-": print(f"{r[0]:<12} {r[4]}"); continue
    print(f"{r[0]:<12} ep={r[1]:>3}  frames={r[5-1]:>5}" if False else
          f"{r[0]:<12} ep={r[1]:>3}  frames={r[2]:>5}  fps={r[3]}  split={r[5]:<8} v{r[6]}  | {r[4]}")
    tot_ep += r[1]; tot_fr += r[2]
print(f"\n合计: {tot_ep} episodes, {tot_fr} frames")
print("\n特征列一致性检查:")
sigs = {}
for r in rows:
    if r[1] == "-": continue
    sigs.setdefault(tuple(r[7]), []).append(r[0])
for k, v in sigs.items():
    print(f"  {len(v)} 个数据集共享同一组列: {v}")
    print(f"     列 = {list(k)}")
