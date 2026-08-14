"""Which way up the LIBERO images are -- declared once, here, and enforced.

ONLINE TIER (openpi venv).

THE CONVENTION IS UPRIGHT (robot at the top of the frame), and three things must share it:

  * STAGE 1, the AXIS corpus (`derived224`), stores frames right-side-up and applies NO flip
    anywhere in its input path -- so stage 1 already trains upright, and it is the fixed point
    the other two have to meet.
  * THE EVAL CLIENT (`benchmarks_eval/run_libero_plus_eval.py`) turns MuJoCo's bottom-up render
    right-side-up with `obs["agentview_image"][::-1, ::-1]` (and the same for
    `robot0_eye_in_hand_image`) before it sends anything -- so inference is upright.
  * STAGE 2, therefore, must be upright too.

WHAT WENT WRONG, and why nothing caught it.  `physical-intelligence/libero` is a REPO ID, not a
build.  Physical Intelligence's own `convert_libero_data_to_lerobot.py` bakes the 180 degrees into
their copy, so it is already upright -- but it is LeRobot v2.0 and lerobot 0.4.4 refuses to load
it, so the box converted the LIBERO HDF5 demos itself WITHOUT that rotation, and
`axis_stage2.sh` points `HF_LEROBOT_HOME` at the homemade copy.  Same repo id, opposite
orientation.  Stage 2 trained upside-down; loss converged, every guard passed, every test was
green, because training on flipped images is perfectly self-consistent.  The disagreement was
only ever observable BETWEEN training and eval, and nothing in the pipeline had ever compared a
training observation with an eval observation.  The same stage-2 checkpoint scores 0/12 through
the client's rotation and 11/12 with the rotation removed.

MEASURED, not assumed (2026-08-14, on the box, from pixels).  Matching (task, first frame) rows
of the two builds, the homemade frame correlates with the PI frame at -0.07 / -0.25 / -0.43 in
the same orientation and at +0.99 / +0.97 / +0.96 after `[::-1, ::-1]`; wrist images likewise
(+0.94 / +0.92 / +0.87 flipped).  That is the whole basis of the `stored_orientation` column of
`KNOWN_BUILDS` below.

SO ORIENTATION IS DECLARED, NEVER INFERRED.  A LIBERO config states the orientation of the build
it reads and the rotation is DERIVED from that (`rotation_needed`), which makes "declared upright
but rotated anyway" unrepresentable -- there is no second switch to disagree with the
declaration.  And `check_dataset_build` refuses, at loader construction, to run a declared
orientation against a resolved dataset that is a DIFFERENT build.  That is the check which would
have caught this on day one: the day `HF_LEROBOT_HOME` was pointed at a 2,000-episode 128px copy
while the config still meant the 1,693-episode 256px one.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from typing_extensions import override

from openpi import transforms

# The two orientations any LIBERO build can store, relative to the world (not to MuJoCo's
# bottom-up framebuffer, which is an implementation detail of the renderer).
UPRIGHT = "upright"
INVERTED = "inverted"
ORIENTATIONS = (UPRIGHT, INVERTED)

# What stage 1 trained in and what the eval client sends. Stage 2 has no vote.
MODEL_ORIENTATION = UPRIGHT

# The two image keys of the LIBERO repack map. BOTH, always: rotating the base view and not the
# wrist view is a silent half-fix -- the model still gets a coherent-looking batch, and only the
# rollout number tells you, 45 GPU-hours later.
IMAGE_KEYS = ("observation/image", "observation/wrist_image")


def rotation_needed(declared_orientation: str) -> bool:
    """Whether a build stored `declared_orientation` needs a 180 degree turn to face the model."""
    if declared_orientation not in ORIENTATIONS:
        raise ValueError(
            f"unknown LIBERO image orientation {declared_orientation!r}; expected one of "
            f"{ORIENTATIONS}. This names the convention the DATASET stores, not the one the "
            f"model wants -- the model always wants {MODEL_ORIENTATION!r}."
        )
    return declared_orientation != MODEL_ORIENTATION


def _hw_axes(shape: tuple[int, ...], key: str) -> tuple[int, int]:
    """The (height, width) axes of an image array, or a refusal.

    LeRobot hands back float32 CHW (measured: `image` is a (3, 128, 128) torch.float32 tensor in
    [0, 1]); a fixture or an already-parsed frame is uint8 HWC. `[::-1, ::-1]` is only the right
    spelling for ONE of those -- on CHW it would flip the channel axis and the height axis, i.e.
    swap red and blue and mirror the image, which no test comparing "is it flipped" would
    distinguish from a correct rotation. So resolve the layout explicitly and refuse anything
    ambiguous rather than pick a spelling and hope.
    """
    if len(shape) != 3:
        raise ValueError(
            f"cannot rotate {key!r}: expected a 3-D image, got shape {shape}. A batched or "
            f"single-channel frame would silently rotate the wrong axes."
        )
    chw = shape[0] == 3
    hwc = shape[2] in (1, 3)
    if chw and hwc:
        raise ValueError(
            f"cannot rotate {key!r}: shape {shape} is ambiguous between CHW and HWC, so the "
            f"height/width axes cannot be identified. Rotating the wrong pair is invisible "
            f"downstream."
        )
    if chw:
        return 1, 2
    if hwc:
        return 0, 1
    raise ValueError(
        f"cannot rotate {key!r}: shape {shape} is neither CHW nor HWC (no axis of size 1 or 3)."
    )


def rot180(x: Any, key: str = "image") -> Any:
    """Turn one image 180 degrees about its centre, in whichever layout it arrived in.

    Type-preserving: torch tensors keep being torch tensors (`np.flip`'s negative strides are not
    something torch accepts back), numpy arrays stay numpy and stay contiguous.
    """
    h, w = _hw_axes(tuple(x.shape), key)
    if hasattr(x, "flip"):  # torch.Tensor -- negative-stride views are not supported there
        return x.flip((h, w))
    return np.ascontiguousarray(np.flip(np.asarray(x), axis=(h, w)))


@dataclasses.dataclass(frozen=True)
class Rotate180Images(transforms.DataTransformFn):
    """Turn the LIBERO observation images 180 degrees, TRAINING SIDE ONLY.

    Belongs in `DataConfig.repack_transforms`, which is the one group inference never runs
    (`policy_config.create_trained_policy` builds its chain from the `repack_transforms`
    ARGUMENT, and `serve_policy.py` passes none). That placement is the whole reason this fixes
    the mismatch instead of moving it: the eval client already rotates, so a rotation in
    `data_transforms` would apply twice at serve time and put us back where we started, upside
    down, with a different explanation.

    It runs AFTER `RepackTransform`, on the canonical `observation/...` keys, so a change to the
    dataset's own column names cannot silently make it a no-op on one key.
    """

    keys: tuple[str, ...] = IMAGE_KEYS

    @override
    def __call__(self, data: dict) -> dict:
        missing = [k for k in self.keys if k not in data]
        if missing:
            raise KeyError(
                f"Rotate180Images cannot rotate {missing}: the key is absent from the sample "
                f"(present: {sorted(data)}). Rotating only some of {list(self.keys)} would leave "
                f"the model a batch whose views disagree about which way is up, which is "
                f"self-consistent to train on and only shows up in a rollout score."
            )
        return {**data, **{k: rot180(data[k], k) for k in self.keys}}


# --- which build is on disk, and what convention it stores ---------------------------------------


@dataclasses.dataclass(frozen=True)
class LiberoBuild:
    """One conversion of the LIBERO demos, and the orientation it stores.

    Identified by (total_episodes, image height/width) because that is what a resolved
    `LeRobotDatasetMetadata` can tell us regardless of codebase version -- the repo id cannot, and
    the repo id being uninformative is exactly the bug.
    """

    label: str
    total_episodes: int
    image_hw: tuple[int, int]
    stored_orientation: str
    evidence: str


KNOWN_BUILDS: tuple[LiberoBuild, ...] = (
    LiberoBuild(
        label="physical-intelligence official (LeRobot v2.0)",
        total_episodes=1693,
        image_hw=(256, 256),
        stored_orientation=UPRIGHT,
        evidence=(
            "PI's convert_libero_data_to_lerobot.py rotates the HDF5 frames 180 degrees during "
            "conversion; confirmed against the box copy at /data/lerobot (v2.0, 1693 eps, 256px), "
            "whose frames correlate at +0.96..+0.99 with the homemade build's AFTER a flip"
        ),
    ),
    LiberoBuild(
        label="box-local HDF5 re-conversion (LeRobot v3.0, /data/lerobot_hdf5)",
        total_episodes=2000,
        image_hw=(128, 128),
        stored_orientation=INVERTED,
        evidence=(
            "converted from the raw LIBERO HDF5 with no rotation, because lerobot 0.4.4 rejects "
            "PI's v2.0 copy; measured 2026-08-14 -- matched (task, frame 0) rows correlate "
            "-0.07/-0.25/-0.43 with PI's in the same orientation and +0.99/+0.97/+0.96 flipped, "
            "wrist images +0.94/+0.92/+0.87 flipped"
        ),
    ),
)


def image_hw(info: dict) -> tuple[int, int]:
    """The (height, width) a LeRobot `meta/info.json` declares for its base image feature."""
    try:
        shape = info["features"]["image"]["shape"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"this LeRobot dataset declares no `image` feature, so it is not a LIBERO build this "
            f"orientation check knows how to identify (features: "
            f"{sorted((info or {}).get('features', {}))})"
        ) from e
    h, w = int(shape[0]), int(shape[1])
    return h, w


def identify_build(info: dict) -> LiberoBuild | None:
    """The registered build this dataset is, or None if it is one nobody has declared."""
    key = (int(info.get("total_episodes", -1)), image_hw(info))
    for build in KNOWN_BUILDS:
        if (build.total_episodes, build.image_hw) == key:
            return build
    return None


def check_dataset_build(*, repo_id: str, declared_orientation: str, info: dict) -> LiberoBuild:
    """Refuse to train a declared orientation against a dataset that is a different build.

    THE DAY-ONE CHECK. `HF_LEROBOT_HOME=/data/lerobot_hdf5` silently redirected
    `physical-intelligence/libero` from a 1,693-episode 256px upright build to a 2,000-episode
    128px inverted one; every other check in the pipeline is downstream of that resolution and so
    saw a perfectly consistent world. This one is not.
    """
    if declared_orientation not in ORIENTATIONS:
        raise ValueError(
            f"config declares LIBERO image orientation {declared_orientation!r}; expected one of "
            f"{ORIENTATIONS}"
        )
    build = identify_build(info)
    if build is None:
        known = ", ".join(f"{b.total_episodes} eps @ {b.image_hw[0]}px ({b.label})" for b in KNOWN_BUILDS)
        raise ValueError(
            f"the LeRobot dataset resolved for repo_id={repo_id!r} is {info.get('total_episodes')} "
            f"episodes at {image_hw(info)[0]}px, which matches no build declared in "
            f"openpi.training.libero_orientation.KNOWN_BUILDS ({known}). A repo id does not "
            f"determine which way up the frames are stored -- HF_LEROBOT_HOME does -- so this "
            f"dataset has to declare its convention before anything trains on it. Add a "
            f"LiberoBuild entry stating whether it is {UPRIGHT!r} or {INVERTED!r}, with the "
            f"measurement you based that on."
        )
    if build.stored_orientation != declared_orientation:
        rotates = "rotates the frames 180 degrees" if rotation_needed(declared_orientation) else "applies no rotation"
        raise ValueError(
            f"LIBERO orientation mismatch for repo_id={repo_id!r}: the config declares the "
            f"dataset is stored {declared_orientation!r} and therefore {rotates}, but the dataset "
            f"actually on disk is {build.label} ({build.total_episodes} episodes at "
            f"{build.image_hw[0]}px), which stores {build.stored_orientation!r} "
            f"({build.evidence}). Training would run at 180 degrees to the eval client, which "
            f"sends {MODEL_ORIENTATION!r} frames (run_libero_plus_eval.py flips MuJoCo's render "
            f"with [::-1, ::-1]) and to the AXIS stage-1 corpus, which is {MODEL_ORIENTATION!r} "
            f"with no flip in its input path. Fix ONE of: the config's declared orientation, or "
            f"HF_LEROBOT_HOME (which build gets resolved)."
        )
    return build
