import json, datetime, shutil, os
STORE="/home/franka_desktop/RLinf/tasl/tasks_store.json"
SPEC=json.load(open("/tmp/prompts.json"))["tasks"]
old=json.load(open(STORE))
by_id={t["id"].lower(): t for t in old}
# 旧 id -> 新 id 的映射(尽量保住 layout / datasets 关联)
LEGACY={"T1-a":"pick-blue-cup-and-put-into-red-cup","T1-b":"T1-b","T2-a":"T2","T2-b":"T2-b",
        "T3-a":"T3","T3-b":"T3-b","T4-a":"T4","T4-b":"t4-b","T5-a":"T5","T5-b":"T5-b"}
now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
new=[]
for sp in SPEC:
    tid=sp["id"]; legacy=by_id.get(LEGACY.get(tid,"").lower(), {})
    new.append({
        "id": tid,
        "prompt": sp["new"],
        "layout": legacy.get("layout",""),
        "layouts": legacy.get("layouts",[]),
        "datasets": sp["dirs"],
        "created": legacy.get("created", now),
        "updated": now,
    })
shutil.copy(STORE, STORE+".bak-20260821")
json.dump(new, open(STORE,"w"), ensure_ascii=False, indent=2)
os.chmod(STORE, 0o644)
print("备份:", STORE+".bak-20260821")
print("写入", len(new), "个 task:")
for t in new: print("  %-6s | %-66s | layout=%-8s | datasets=%s" % (t["id"], t["prompt"], t["layout"] or "-", ",".join(t["datasets"])))
