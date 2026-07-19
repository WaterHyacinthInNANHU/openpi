"""AXIS Franka (robosuite) input/output transforms for pi0.5.

State[9] = 7 arm + 2 finger qpos; action[9] = absolute joint ctrl targets.
Mirrors libero_policy but for our 9-DoF Franka. Images arrive already
repacked to base_0_rgb / left_wrist_0_rgb by the DataConfig repack step.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms as _transforms


def _pad(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: len(vec)] = vec
    return out


def _to_hwc_uint8(img) -> np.ndarray:
    """LeRobot v3.0 yields images as torch CHW float[0,1]; openpi wants numpy HWC uint8."""
    arr = np.asarray(img)  # torch CPU tensor -> numpy via __array__
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


@dataclasses.dataclass(frozen=True)
class AxisFrankaInputs(_transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = _pad(np.asarray(data["state"], dtype=np.float32), self.action_dim)
        base = _to_hwc_uint8(data["base_0_rgb"])
        raw_wrist = data.get("left_wrist_0_rgb")
        has_wrist = raw_wrist is not None
        wrist = _to_hwc_uint8(raw_wrist) if has_wrist else np.zeros_like(base)
        # openpi wants separate `image` / `image_mask` dicts with the model's 3 camera
        # slots; pad the unused right wrist with zeros (masked off for the pi0.5 flow).
        out = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if has_wrist else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            act = np.asarray(data["actions"], dtype=np.float32)
            out["actions"] = np.stack([_pad(a, self.action_dim) for a in act])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class AxisFrankaOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :9]}
