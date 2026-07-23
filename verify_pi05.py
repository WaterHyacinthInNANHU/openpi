import numpy as np

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as C

cfg = C.get_config("pi05_droid")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
print("checkpoint dir:", ckpt)

policy = policy_config.create_trained_policy(cfg, ckpt)
print("policy loaded")

obs = {
    "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(7, dtype=np.float32),
    "observation/gripper_position": np.zeros(1, dtype=np.float32),
    "prompt": "pick up the block",
}
out = policy.infer(obs)
a = np.asarray(out["actions"])
print("infer ok, actions shape:", a.shape, "dtype:", a.dtype)
