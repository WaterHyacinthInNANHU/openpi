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


@dataclasses.dataclass(frozen=True)
class AxisFrankaInputs(_transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = _pad(np.asarray(data["state"], dtype=np.float32), self.action_dim)
        base = data["base_0_rgb"]
        wrist = data.get("left_wrist_0_rgb")
        images = {"base_0_rgb": base, "base_0_rgb_mask": np.True_}
        if wrist is not None:
            images["left_wrist_0_rgb"] = wrist
            images["left_wrist_0_rgb_mask"] = np.True_
        out = {"state": state, "image": images}
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
