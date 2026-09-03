import json, glob, os, collections
for name in ["T1-a","T1-b","T4-b","T2-b","T5-b"]:
    d = os.path.expanduser("~/rlinf_data/datasets/"+name)
    print("="*60); print(name)
    print("  tasks.jsonl:")
    for l in open(os.path.join(d,"meta","tasks.jsonl")):
        print("   ", l.rstrip())
    cnt = collections.Counter()
    for l in open(os.path.join(d,"meta","episodes.jsonl")):
        e = json.loads(l)
        for t in e.get("tasks", []): cnt[t] += 1
    print("  episodes.jsonl 里每个 task 出现的 episode 数:", dict(cnt))
