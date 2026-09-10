"""PBC image geometry for the RLinf FR3 bench.

"PBC" = the in-house pi0.5 base `axis_pi05_droid_plainbc_v1` (AXIS franka_robotiq sim + real
DROID co-train, initialised from pi05_droid).  Its pretraining did NOT use openpi's default
`resize_with_pad` letterbox; it ran with `AxisFrankaInputs(center_crop=True)`:

    openpi fork WaterHyacinthInNANHU/openpi, branch box/server2-heldout-d8,
    src/openpi/policies/axis_franka_policy.py::_center_crop_square
        "Crop the WIDER axis to the shorter one, centred. A no-op on an already-square frame."
        applied to BOTH cameras identically, before ResizeImages(224, 224).
    AXIS-Bench branch myan/cotrain-phase, docs/reports/2026-08-24-cotrain-bc-audit.md #2:
        the plain-BC co-train arm (= this ckpt) had it on; DROID 180x320 -> 180x180 too.

i.e. for a WxH frame keep the central HxH square and let `ResizeImages(224, 224)` scale it.
No black bars, ~44 % of a 16:9 frame's width discarded.  This is the same geometry as RLinf's
`image_resize_mode: crop` (FrankaEnv) and the collect dashboard's `POLICY_VIEW_MODE=crop`.

Everything PBC-specific lives in this module so it cannot be confused with the pi05_droid
(letterbox) pipeline in `rlinf_franka_droid.py`:

  * `PbcCenterCropImages`  -- the transform.  It sits in `data_transforms.inputs` right after
    `DroidInputs`, so it runs at BOTH train time (dataset frames) and serve time (live camera
    frames).  A square input is a no-op, which is what makes train/serve parity hold by
    construction: the `*_pbc` dataset is already stored as 224x224 squares, the live ZED frame
    is 1280x720 and gets cropped to 720x720 here, then both go through the same
    `ResizeImages(224, 224)`.
  * `pbc_center_square`    -- the pure function, shared with the offline dataset builder
    (data_pipeline/on_labserver/pbc/make_pbc_dataset.py) so the two cannot drift.

The repack step is unchanged: the RLinf writer's column layout is the same for both pipelines,
so `RLinfFrankaDroidRepack` is reused.
"""

import dataclasses

import numpy as np

from openpi import transforms as _transforms

PBC_GEOMETRY = "center_square_full_height"  # recorded into the *_pbc dataset's meta/info.json


def pbc_center_square(image: np.ndarray) -> np.ndarray:
    """Full-height, horizontally centred square crop of an (H, W, C) or (..., H, W, C) image.

    Returns the input untouched (same object) when it is already square.  Pure numpy, no resize:
    the resize to 224 is `ResizeImages`' job (train and serve alike).
    """
    image = np.asarray(image)
    if image.ndim < 3:
        raise ValueError(f"expected (..., H, W, C), got shape {image.shape}")
    h, w = image.shape[-3], image.shape[-2]
    if h == w:
        return image
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[..., y0 : y0 + side, x0 : x0 + side, :]


POLICY_IMAGE_SIZE = 224


def pbc_center_square_224(image: np.ndarray, size: int = POLICY_IMAGE_SIZE) -> np.ndarray:
    """`pbc_center_square` followed by the SAME resize the training data was built with:
    ``cv2.resize(square, (224, 224))`` (INTER_LINEAR), as in
    ``tasl/tools/make_centercrop_dataset.py::crop224`` and RLinf ``FrankaEnv._crop_frame``.
    A ``size``x``size`` input is returned untouched, so at train time (dataset frames are
    already 224x224) this is the identity and at serve time a raw 1280x720 camera frame becomes
    exactly the pixels the checkpoint saw. Do NOT leave the resize to `ResizeImages`: its
    jax bilinear filter is antialiased and does not reproduce cv2's output.
    """
    sq = pbc_center_square(image)
    h, w = sq.shape[-3], sq.shape[-2]
    if h == size and w == size:
        return sq
    import cv2  # openpi depends on opencv-python; imported lazily to keep this module light

    if sq.ndim == 3:
        return cv2.resize(np.ascontiguousarray(sq), (size, size))
    flat = sq.reshape(-1, h, w, sq.shape[-1])
    out = np.stack([cv2.resize(np.ascontiguousarray(f), (size, size)) for f in flat])
    return out.reshape(*sq.shape[:-3], size, size, sq.shape[-1])


@dataclasses.dataclass(frozen=True)
class PbcCenterCropImages(_transforms.DataTransformFn):
    """Apply `pbc_center_square_224` to every image in `data["image"]` (after DroidInputs, before
    ResizeImages, which then sees a 224x224 square and is a no-op)."""

    def __call__(self, data: dict) -> dict:
        data["image"] = {k: pbc_center_square_224(v) for k, v in data["image"].items()}
        return data
