import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
api=HfApi(); REPO="Litian2002/tasl-fr3-10task-250ep"
L="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep/meta"
for f in ["tasks.jsonl","episodes.jsonl"]:
    api.upload_file(path_or_fileobj=f"{L}/{f}", path_in_repo=f"meta/{f}",
                    repo_id=REPO, repo_type="dataset",
                    commit_message="fix T4-a prompt: green -> orange")
    print("uploaded meta/"+f)
# 回读校验
import json, io
from huggingface_hub import hf_hub_download
p=hf_hub_download(REPO, "meta/tasks.jsonl", repo_type="dataset", force_download=True)
for l in open(p):
    d=json.loads(l)
    if d["task_index"]==6: print("远端 tix6 =", repr(d["task"]))
