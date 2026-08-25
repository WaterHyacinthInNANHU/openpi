import json, glob, os
print("%-12s %-8s %-8s %-16s %-16s %s" % ("name","state","action","image","extra_view","chunks"))
for d in sorted(glob.glob(os.path.expanduser("~/rlinf_data/datasets/T[1-5]-*"))):
    if d.endswith("_svo"): continue
    i = json.load(open(os.path.join(d,"meta","info.json")))
    f = i["features"]
    nch = len(glob.glob(os.path.join(d,"data","chunk-*")))
    npq = len(glob.glob(os.path.join(d,"data","chunk-*","*.parquet")))
    print("%-12s %-8s %-8s %-16s %-16s %d chunk / %d parquet, chunks_size=%d, total_videos=%s" % (
        os.path.basename(d), f["state"]["shape"], f["actions"]["shape"],
        f["image"]["shape"], f["extra_view_image"]["shape"], nch, npq,
        i["chunks_size"], i.get("total_videos")))
