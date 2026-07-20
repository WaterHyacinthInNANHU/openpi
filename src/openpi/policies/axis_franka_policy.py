"""AXIS Franka (DROID 8D) input/output transforms for pi0.5.

State[8] = 7 arm + 1 gripper (DROID format); action[8] = absolute joint ctrl targets.
Mirrors libero_policy. Images arrive already
repacked to base_0_rgb / left_wrist_0_rgb by the DataConfig repack step.

DO NOT PAD state/actions here. The pipeline order is
    data_transforms.inputs -> Normalize -> model_transforms.inputs
(`training/data_loader.py`), and `PadStatesAndActions(32)` already lives in
`model_transforms`, i.e. AFTER `Normalize`. `LiberoInputs` -- which this mirrors --
passes state/actions through unpadded for exactly this reason.

Padding here instead made `Normalize` see 32 dims. pi0.5 uses quantile normalisation
(`config.py:204`) and the reused `pi05_droid` action stats have q01 == q99 == 0 for
dims 8-31, so each padded zero became
    (0 - 0) / (0 - 0 + 1e-6) * 2 - 1 = -1.0
turning 24 of 32 action dims into a constant -1.0 the pretrained decoder never emits,
where pi05_droid saw 0.0. Benign under z-score, pathological under quantile.
Regression-tested in benchmarks/dataloader/tests/test_axis_franka_policy.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms as _transforms


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
    """Repack AXIS-Franka rows into the pi0.5 input layout.

    Deliberately does NOT pad state/actions to the model action dim -- see the module
    docstring. `PadStatesAndActions` in `model_transforms` does that, after `Normalize`.
    """

    def __call__(self, data: dict) -> dict:
        # Pass 8-D state through unpadded: Normalize runs next, and padding here would
        # feed it 24 dims of zeros that quantile-normalize to -1.0 (see module docstring).
        state = np.asarray(data["state"], dtype=np.float32)
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
            # Unpadded, for the same reason as `state`. AxisFrankaOutputs slices the
            # model's 32-D output back to 8 on the way out.
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class AxisFrankaOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :8]}
