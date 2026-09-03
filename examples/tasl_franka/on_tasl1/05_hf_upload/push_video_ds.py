import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi
REPO="Litian2002/tasl-fr3-10task-250ep-video"
L="/home/franka_desktop/work/merged/tasl_fr3_10task_250ep_video"
api=HfApi()
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
print("repo:", REPO, flush=True)
api.upload_folder(folder_path=L, repo_id=REPO, repo_type="dataset",
                  commit_message="video-format copy (h264 mp4) for the LeRobot visualizer")
print("VIDEO DS UPLOADED", flush=True)
