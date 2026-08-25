"""Repack transform for LeRobot datasets collected by the RLinf FR3 bench.

The RLinf collect dashboard (RLinf/tasl) writes LeRobot v2.1 datasets with:
  state[8]         = [gripper_position, q0..q6]   (alphabetical concat of the
                     obs dict {gripper_position(1), joint_position(7)})
  actions[8]       = [dq0..dq6 normalized to [-1,1], gripper in [0,1]]
                     (joint-velocity convention, pi05_droid-native, 15 Hz)
  image            = exterior ZED 2i, 224x224
  extra_view_image = wrist ZED Mini (eye-in-hand), 224x224

This transform splits/renames those columns into the keys DroidInputs expects.
It mirrors RLinf's examples/embodiment/validate_droid_lerobot.py::
rlinf_frame_to_droid (unit-tested there); keep the two in sync.
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class RLinfFrankaDroidRepack(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"]).reshape(-1)
        if state.shape[0] != 8:
            raise ValueError(f"expected 8-D state [grip,q0..q6], got {state.shape}")
        out = {
            "observation/joint_position": state[1:8],
            "observation/gripper_position": state[0:1],
            "observation/exterior_image_1_left": data["image"],
            "observation/wrist_image_left": data["extra_view_image"],
            # [action_horizon, 8] during training (windowed by the data loader).
            "actions": np.asarray(data["actions"]),
        }
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out
