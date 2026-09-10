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


POLICY_IMAGE_SIZE = 224


def letterbox_square_224(image: np.ndarray, size: int = POLICY_IMAGE_SIZE) -> np.ndarray:
    """Pad to a centred black square, then ``cv2.resize`` to 224x224 (INTER_LINEAR) -- the exact
    ops the letterboxed training data was built with (RLinf ``FrankaEnv._pad_to_square`` +
    ``cv2.resize``; ``tasl/tools/make_centercrop_dataset.py::pad224``). A ``size``x``size`` input
    is returned untouched (identity at train time), a raw 1280x720 camera frame at serve time
    becomes the same pixels the checkpoint saw. `ResizeImages` afterwards is then a no-op; its
    antialiased jax resize would not reproduce cv2's output.
    """
    image = np.asarray(image)
    if image.ndim < 3:
        raise ValueError(f"expected (..., H, W, C), got shape {image.shape}")
    h, w = image.shape[-3], image.shape[-2]
    if h == size and w == size:
        return image
    import cv2  # openpi depends on opencv-python; imported lazily

    def one(frame: np.ndarray) -> np.ndarray:
        side = max(frame.shape[0], frame.shape[1])
        sq = np.zeros((side, side, frame.shape[2]), dtype=frame.dtype)
        y, x = (side - frame.shape[0]) // 2, (side - frame.shape[1]) // 2
        sq[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
        return cv2.resize(sq, (size, size))

    if image.ndim == 3:
        return one(np.ascontiguousarray(image))
    flat = image.reshape(-1, h, w, image.shape[-1])
    out = np.stack([one(np.ascontiguousarray(f)) for f in flat])
    return out.reshape(*image.shape[:-3], size, size, image.shape[-1])


@dataclasses.dataclass(frozen=True)
class LetterboxImages(_transforms.DataTransformFn):
    """Apply `letterbox_square_224` to every image in `data["image"]` (after DroidInputs)."""

    def __call__(self, data: dict) -> dict:
        data["image"] = {k: letterbox_square_224(v) for k, v in data["image"].items()}
        return data
