"""AXIS Franka (DROID 8D) input/output transforms for pi0.5.

State[8] = 7 arm joint positions (rad) + 1 gripper closedness [0,1] (DROID 8-D layout).
action[8] = 7 arm joint VELOCITY (rad/s) + 1 gripper closedness [0,1] -- NOT absolute
position targets (see benchmarks/dataloader/convert_droid_actions.py and the sim-side
integrator benchmarks/slb_pilot/action_mapping_pi05.py, which integrates v*dt into a
persistent position command). Mirrors libero_policy. Images arrive already
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
Regression-tested in axis_data/tests/test_axis_franka_policy.py.
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


# Relative-EEF action math lives in the dependency-free `eef_math` module (so the offline
# corpus column-add can import it without flax). Re-exported here for the transforms + tests.
# Convention CONFIRMED against LIBERO robosuite OSC_POSE (world-frame delta, 3-D axis-angle,
# absolute gripper, 7-D). NB: eef_pose is baked as MuJoCo **wxyz** -- reorder to xyzw first.
from openpi.policies.eef_math import (  # noqa: E402
    eef_pose_to_delta_actions,
    quat_xyzw_to_axisangle,
)


def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    """Crop the WIDER axis to the shorter one, centred. A no-op on an already-square frame.

    The 5k `camera_fixed` render stores 640x360 (16:9) for BOTH cameras. The model input is
    224x224, so a straight resize would squash 16:9 into 1:1 and distort every frame. LIBERO
    renders square, so cropping to square matches the benchmark's framing rather than
    letterboxing ours into it.

    CROPPING IS DECLARED, NEVER INFERRED -- the caller passes `center_crop`, this function does
    not sniff the aspect ratio and decide. The corpora disagree (v3 is 126x224, randcam 224x224,
    the 5k 640x360) and a transform that silently activates on shape would change what an old
    corpus means without anything in the config recording it. That is the same failure as the
    180-degree orientation incident, which cost three trained models.
    """
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return arr[top:top + side, left:left + side]


@dataclasses.dataclass(frozen=True)
class AxisFrankaInputs(_transforms.DataTransformFn):
    """Repack AXIS-Franka rows into the pi0.5 input layout.

    Deliberately does NOT pad state/actions to the model action dim -- see the module
    docstring. `PadStatesAndActions` in `model_transforms` does that, after `Normalize`.
    """

    # Crop both cameras to square before the downstream resize to 224. Needed for the 5k
    # `camera_fixed` corpus (640x360); a no-op on the square randcam frames, and DELIBERATELY
    # off by default so no existing config changes meaning.
    center_crop: bool = False

    def __call__(self, data: dict) -> dict:
        # Pass 8-D state through unpadded: Normalize runs next, and padding here would
        # feed it 24 dims of zeros that quantile-normalize to -1.0 (see module docstring).
        state = np.asarray(data["state"], dtype=np.float32)
        # WIDTH IS A CONTRACT. Both legitimate stage-1 layouts are 8-D -- `state_eef`
        # [pos3, axis-angle3, closedness, closedness] and droid8 [7 joint angles, closedness] --
        # so anything else means the config and the corpus disagree. The case this catches: an
        # `eef_action=False` config pointed at an UNCONVERTED corpus, whose observation.state is
        # 9-D (7 joint angles + 2 finger widths in metres). That trains happily on absolute joint
        # positions and slices the output to [:8], keeping one finger width and dropping the
        # other. Nothing errors, the loss falls, and the policy is meaningless -- the same shape
        # of failure as the 180-degree orientation incident, which cost three trained models.
        if state.shape[-1] != 8:
            raise ValueError(
                f"state is {state.shape[-1]}-D; AxisFranka stage-1 requires 8-D. A 9-D state is "
                f"the RAW corpus layout (7 joint angles + 2 finger widths): convert it with "
                f"axis.episode.convert_droid_actions and point AXIS_PRETRAIN_ROOTS_INDEX at the "
                f"converted roots index, or use an eef_action=True config with state_eef."
            )
        base = _to_hwc_uint8(data["base_0_rgb"])
        raw_wrist = data.get("left_wrist_0_rgb")
        has_wrist = raw_wrist is not None
        wrist = _to_hwc_uint8(raw_wrist) if has_wrist else np.zeros_like(base)
        if self.center_crop:
            # BOTH cameras, identically. The wrist is 16:9 too, and cropping only one would
            # give the two views different effective fields of view.
            base = _center_crop_square(base)
            wrist = _center_crop_square(wrist)
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


@dataclasses.dataclass(frozen=True)
class AxisFrankaJoint9Outputs(_transforms.DataTransformFn):
    """Joint-POSITION variant: slice the padded 32-D output back to 9 = 7 joint targets + 2 finger
    widths, the space the stage-1 corpus and the held-out render store.

    The 8-D slice above is for the DROID velocity space; applying it to a 9-D policy drops the last
    finger width and shifts the first one into the closedness slot the eval reads.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :9]}


@dataclasses.dataclass(frozen=True)
class AxisFrankaEEFOutputs(_transforms.DataTransformFn):
    """Relative-EEF action variant: slice the model's padded 32-D output back to the 7-D
    LIBERO/robosuite OSC_POSE action [Δpos(3), Δaxis-angle(3), gripper]. Inputs are shared
    with the joint variant (AxisFrankaInputs passes state/actions through unpadded); only the
    output width differs (7 vs 8), and the DataConfig feeds state_eef/action_eef columns."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :7]}


# ---------------------------------------------------------------------------------------
# LIBERO-Plus serve-time adapter (inference only) for the relative-EEF pretrain checkpoint.
#
# The LIBERO client (benchmarks_eval/run_libero_plus_eval.py) sends the pi05_libero contract:
#   observation/image, observation/wrist_image  (HWC uint8, already 180°-rotated + pad-224)
#   observation/state  = [eef_pos(3), quat->axisangle(3), robot0_gripper_qpos(2)]  (8-D)
#   prompt
# Our pretrain_eef model was trained on `state_eef` = [pos(3), CANONICAL axis-angle(3, w>=0),
# closedness, closedness] and emits `action_eef` = [Δpos(3), Δaxis-angle(3), closedness[0,1]].
# Two channels of the LIBERO state and the action gripper therefore need a convention remap;
# pos + Δpose already share robosuite OSC_POSE world-frame conventions (see eef_math).
#
# Serve chain (SimpleDataConfig.data_transforms.inputs, since serve_policy.py passes NO
# repack_transforms -> the DataConfig repack runs only in training):
#   RepackTransform(observation/*->base_0_rgb/left_wrist_0_rgb/state) -> LiberoStateToAxisEEF
#   -> AxisFrankaInputs ; outputs: AxisFrankaEEFLiberoOutputs.
# ---------------------------------------------------------------------------------------

# LIBERO Panda finger half-open width used to map robot0_gripper_qpos -> closedness. Only the
# monotonic open->0 / closed->1 mapping matters: our own quantile norm-stats (bimodal at 0/1)
# absorb the exact scale, so a coarse linear map on the mean finger width is sufficient.
_LIBERO_GRIPPER_OPEN_W = 0.04   # |finger| ~0.04 when fully open
_LIBERO_GRIPPER_CLOSED_W = 0.0  # ~0 when fully closed


def _libero_gripper_qpos_to_closedness(qpos2) -> float:
    """robot0_gripper_qpos (2 finger positions, meters) -> closedness in [0,1] (0 open / 1 close).

    Mirrors convert_droid_actions.gripper_closedness (mean finger width, linear, clipped). Uses
    |q0|+|q1| so it is robust to whether the two Panda finger joints are stored [+,-] or [+,+]."""
    q = np.asarray(qpos2, dtype=np.float64)
    width = 0.5 * (abs(float(q[0])) + abs(float(q[1])))
    denom = _LIBERO_GRIPPER_OPEN_W - _LIBERO_GRIPPER_CLOSED_W
    return float(np.clip((_LIBERO_GRIPPER_OPEN_W - width) / denom, 0.0, 1.0))


def _canonicalize_axisangle(aa) -> np.ndarray:
    """Map an axis-angle rotation vector to the shortest-arc equivalent (norm <= pi).

    robosuite mat2quat canonicalises w>=0, so LIBERO states never occupy norm>pi -- but the
    eval client's _quat2axisangle takes robot0_eef_quat as-is (no sign flip), so a w<0 pose
    lands at norm>pi, a region our (canonicalised) training states never saw. v and
    v*(1 - 2π/|v|) are the same rotation; the latter has norm 2π-|v| < π."""
    v = np.asarray(aa, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n > np.pi and n > 1e-9:
        v = v * (1.0 - 2.0 * np.pi / n)
    return v.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class LiberoStateToAxisEEF(_transforms.DataTransformFn):
    """Remap the LIBERO 8-D proprio state into our `state_eef` convention (inference only).

    [pos(3), axisangle(3), gripper_qpos(2)] -> [pos(3), canonical-axisangle(3), closedness, closedness].
    pos passes through (shared world frame); the axis-angle is canonicalised to norm<=pi to match
    our training states; the two raw finger positions collapse to the duplicated closedness scalar
    the model was trained on. No-op if `state` is absent."""

    def __call__(self, data: dict) -> dict:
        if "state" not in data:
            return data
        s = np.asarray(data["state"], dtype=np.float32)
        if s.shape[-1] != 8:
            return data
        closed = _libero_gripper_qpos_to_closedness(s[6:8])
        state = np.concatenate([s[:3], _canonicalize_axisangle(s[3:6]),
                                np.array([closed, closed], dtype=np.float32)]).astype(np.float32)
        return {**data, "state": state}


# robosuite OSC_POSE (LIBERO default controller, osc_pose.json) maps action [-1,1] ->
# output_max = [0.05 m]*3 + [0.5 rad]*3 with control_delta=true. Our Unnormalized action_eef
# is a RAW physical per-step delta (~±0.01 m / ±0.03 rad per our norm stats), so it must be
# divided by these scales to land in the controller's [-1,1] input space -- otherwise a 0.01 m
# delta reads as 0.01*0.05 = 0.5 mm and the arm is effectively frozen.
_OSC_POSE_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class AxisFrankaEEFLiberoOutputs(_transforms.DataTransformFn):
    """Serve-time EEF output for the LIBERO env. Slice to 7-D, then match robosuite OSC_POSE:
      pose (0:6): raw Δpos(m)/Δaxis-angle(rad) -> normalized [-1,1] via /output_max (osc_pose.json);
      gripper (6): our closedness [0,1] (0 open / 1 close) -> LIBERO [-1,1] (-1 open / +1 close),
                   i.e. 2c-1 (LIBERO_DUMMY_ACTION=[...,-1] confirms -1=open).
    Distinct from AxisFrankaEEFOutputs (AXIS-sim serving), which keeps raw deltas + closedness."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)[:, :7].copy()
        actions[:, :6] = np.clip(actions[:, :6] / _OSC_POSE_OUTPUT_MAX, -1.0, 1.0)
        actions[:, 6] = 2.0 * actions[:, 6] - 1.0
        return {"actions": actions}
