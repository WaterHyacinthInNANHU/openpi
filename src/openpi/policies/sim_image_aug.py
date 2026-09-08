"""MolmoBot-style photometric augmentation for SIM frames (arXiv 2603.16861 s4.4).

The paper: "We use image augmentation to improve our models' sim to real transfer.
Specifically, we use ColorJitter, GaussianBlur, RandomPosterize, RandomSharpness and
RandomGrayscale with different probabilities." The probabilities are unpublished; the values
here are conservative torchvision-style defaults and are recorded in this file, which is the
run record.

SIM/REAL GATING IS GEOMETRIC, decided per SAMPLE on the third-person frame: the graft sim
renders are 16:9 (640x360, both cameras) while the real 10-task frames are square 224x224 with
the center crop baked in. Gating on height != width keeps the real anchor byte-identical to the
droid stage-2 baseline pipeline WITHOUT needing a per-row domain bit -- which the untagged bc
control does not carry. A future SQUARE sim corpus would silently skip augmentation here: this
transform is scoped to the one-phase co-train arms and says so.

Runs BEFORE RepackTransform on the raw decoded corpus frames (so before the 224 resize).
Stochastic per presentation like dropout -- a per-worker RNG, deliberately outside the
byte-replayable row draw and the exact-resume machinery.
"""

import dataclasses
import os

import numpy as np
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps

from openpi import transforms as _transforms

_BASE_KEY = "observation.images.third_person"
_WRIST_KEY = "observation.images.wrist"

_rng: np.random.Generator | None = None


def _get_rng() -> np.random.Generator:
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(np.random.SeedSequence([os.getpid(), 0x51A6]))
    return _rng


def _to_pil(a: np.ndarray):
    """uint8/float, HWC/CHW -> (PIL RGB, restore_fn)."""
    chw = a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3)
    h = np.moveaxis(a, 0, -1) if chw else a
    if h.dtype == np.uint8:
        pil = Image.fromarray(h)

        def restore(x):
            out = np.asarray(x, dtype=np.uint8)
            return np.moveaxis(out, -1, 0) if chw else out
    else:
        f = np.asarray(h, dtype=np.float32)
        pil = Image.fromarray((np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8))

        def restore(x):
            out = (np.asarray(x, dtype=np.float32) / 255.0).astype(a.dtype)
            return np.moveaxis(out, -1, 0) if chw else out
    return pil, restore


@dataclasses.dataclass(frozen=True)
class SimImageAug(_transforms.DataTransformFn):
    p_jitter: float = 0.5
    p_blur: float = 0.2
    p_posterize: float = 0.1
    p_sharpness: float = 0.2
    p_grayscale: float = 0.05

    def _augment(self, pil, rng):
        if rng.random() < self.p_jitter:
            pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.8, 1.2))
            pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.8, 1.2))
            pil = ImageEnhance.Color(pil).enhance(rng.uniform(0.7, 1.3))
            hsv = np.array(pil.convert("HSV"), dtype=np.int16)
            hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-12, 13))) % 256
            pil = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        if rng.random() < self.p_blur:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.1, 1.0))))
        if rng.random() < self.p_posterize:
            pil = ImageOps.posterize(pil, 6)
        if rng.random() < self.p_sharpness:
            pil = ImageEnhance.Sharpness(pil).enhance(2.0)
        if rng.random() < self.p_grayscale:
            pil = pil.convert("L").convert("RGB")
        return pil

    def __call__(self, data: dict) -> dict:
        base = data.get(_BASE_KEY)
        if base is None:
            raise KeyError(
                f"{_BASE_KEY!r} is not in the sample; SimImageAug must run BEFORE "
                f"RepackTransform, which renames it."
            )
        b = np.asarray(base)
        hw = (b.shape[-2], b.shape[-1]) if (b.ndim == 3 and b.shape[0] in (1, 3)
                                            and b.shape[-1] not in (1, 3)) else (b.shape[0], b.shape[1])
        if hw[0] == hw[1]:
            return data  # square third-person frame == real sample: never touched
        rng = _get_rng()
        out = dict(data)
        for key in (_BASE_KEY, _WRIST_KEY):
            if key not in out:
                continue
            pil, restore = _to_pil(np.asarray(out[key]))
            out[key] = restore(self._augment(pil, rng))
        return out
