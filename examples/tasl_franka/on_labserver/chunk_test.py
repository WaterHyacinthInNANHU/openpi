import os
os.environ["HF_LEROBOT_HOME"]="/data1/Franka_RealRobot/lerobot_home"
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
meta=LeRobotDatasetMetadata("franka/tasl_fr3_10task_250ep")
H=16
ds=LeRobotDataset("franka/tasl_fr3_10task_250ep",
                  delta_timestamps={"actions":[t/meta.fps for t in range(H)]})
ep0_len=[e["length"] for e in [__import__("json").loads(l) for l in open("/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_250ep/meta/episodes.jsonl")]][0]
print("episode 0 长度:", ep0_len, "| action_horizon:", H)
for i in [0, ep0_len-20, ep0_len-8, ep0_len-2, ep0_len-1]:
    it=ds[i]
    a=np.asarray(it["actions"])
    keys=[k for k in it if "pad" in k]
    pad=np.asarray(it[keys[0]]) if keys else None
    tail=a[:,0]
    print("  frame %3d (距 ep 末 %2d): actions[:,0] 前后 = %s ... %s | 末尾重复? %s | pad掩码=%s"
          % (i, ep0_len-1-i, np.round(tail[:3],4), np.round(tail[-3:],4),
             "是" if np.allclose(tail[-1],tail[-2]) and i>ep0_len-H else "否",
             (pad.sum() if pad is not None else "无此字段")))
    if keys: print("       掩码字段名:", keys, "值:", np.asarray(it[keys[0]]).astype(int))
