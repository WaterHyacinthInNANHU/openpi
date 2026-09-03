import json, glob, os
STORE="/home/franka_desktop/RLinf/tasl/tasks_store.json"
have={os.path.basename(p)[7:-5] for p in glob.glob("/home/franka_desktop/RLinf/saved_demo/*/layout_*.json")}
ts=json.load(open(STORE))
for t in ts:
    if t["layout"] and t["layout"] not in have:
        print("  清掉不存在的 layout: %-6s <- %s" % (t["id"], t["layout"]))
        t["layout"]=""; t["layouts"]=[]
    elif t["layout"]:
        t["layouts"]=[t["layout"]]
json.dump(ts, open(STORE,"w"), ensure_ascii=False, indent=2)
os.chmod(STORE,0o644)
print("\n可用 layout:", sorted(have))
print("\n最终:")
for t in ts: print("  %-6s layout=%-8s | %s" % (t["id"], t["layout"] or "(无)", t["prompt"]))
