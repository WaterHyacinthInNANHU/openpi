import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
REPO="Litian2002/tasl-fr3-10task-250ep"
LOCAL="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep"
api=HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print("repo ready:", REPO)
api.upload_folder(folder_path=LOCAL, repo_id=REPO, repo_type="dataset",
                  commit_message="TASL FR3 10-task merged LeRobot v2.1 (250 ep / 66463 frames)")
print("UPLOAD DONE")
