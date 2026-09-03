import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
REPO="Litian2002/pi05-droid-franka-lora-10task"
SRC="/data1/Franka_RealRobot/checkpoints/pi05_droid_franka_lora_10task/pi05_droid_franka_lora_10task_v0"
api=HfApi()
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
print("repo:", REPO, flush=True)
steps=sorted(os.listdir(SRC), key=lambda s: int(s) if s.isdigit() else -1)
steps=[s for s in steps if s.isdigit()]
print("steps:", steps, flush=True)
for s in steps:
    print("=== uploading", s, flush=True)
    api.upload_folder(folder_path=os.path.join(SRC,s), repo_id=REPO, repo_type="model",
                      path_in_repo=s, ignore_patterns=["train_state/*","**/train_state/*"],
                      commit_message=f"step {s} (params+assets, no train_state)")
    print("=== done", s, flush=True)
print("ALL CKPTS UPLOADED", flush=True)
