import os
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
DEST="/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep"
os.makedirs(DEST, exist_ok=True)
p=snapshot_download(repo_id="Litian2002/tasl-fr3-10task-250ep", repo_type="dataset",
                    local_dir=DEST, max_workers=8)
print("DOWNLOAD DONE ->", p)
