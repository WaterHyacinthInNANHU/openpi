"""Which way up stage 2 trains -- asserted on PIXELS, and against the eval client's own pixels.

WHY THIS FILE EXISTS AND WHY IT IS SHAPED LIKE THIS. Stage 2 trained upside-down for three runs
(~45 GPU-hours) with loss converging to 0.0118, every guard passing and every test green, because
training on flipped images is perfectly self-consistent. The mismatch was only ever observable
BETWEEN the training pipeline and the eval client, and nothing in this repo had ever put a
training observation and an eval observation in the same process. So:

  * every assertion here is on pixels, never on "a transform was constructed". A test that a
    `Rotate180Images` appears in a chain passes just as happily when it rotates the base image
    and not the wrist one, or rotates the channel axis instead of the height axis;
  * `test_training_and_eval_observations_agree_on_which_way_is_up` is the check that would have
    caught the original bug on day one. It runs ONE synthetic MuJoCo frame down BOTH paths -- the
    real training chain of the real registered stage-2 config, and the exact three lines
    `run_libero_plus_eval.py` uses -- and demands the model-facing arrays be equal.

Everything is a fixture rather than a download: `physical-intelligence/libero` is a third-party
dataset this suite must not require, and `repo_id="fake"` short-circuits `create_torch_dataset`
into `FakeDataset` before any of this would run.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from openpi import transforms as _transforms
# The EVAL CLIENT's own module, not `openpi.shared.image_tools`: `run_libero_plus_eval.py` does
# `from openpi_client import image_tools`, and the point of this file is to run the client's code,
# not a same-named cousin of it.
from openpi_client import image_tools
from openpi.policies import libero_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import libero_orientation as _orient

# The registered configs that read the box-local (inverted) build, and the ones that read
# Physical Intelligence's official (already-upright) copy. Pinned here BY NAME so a future config
# that reads a LIBERO dataset cannot join the registry without appearing in one of these lists --
# `test_every_libero_config_declares_its_datasets_orientation` fails until it does.
INVERTED_CONFIGS = (
    "pi05_libero_axisinit",
    "pi05_libero_axisinit_paper",
    "pi05_libero_axisinit_paper_cfg",
)
UPRIGHT_CONFIGS = (
    "pi0_libero",
    "pi0_libero_low_mem_finetune",
    "pi0_fast_libero",
    "pi0_fast_libero_low_mem_finetune",
    "pi05_libero",
    "pi05_axis_cfg_libero_serve",
)

STAGE2 = "pi05_libero_axisinit_paper"
STOCK = "pi05_libero"

RESIZE = 224


# --- fixtures ------------------------------------------------------------------------------------


def chiral_frame(seed: int, size: int = RESIZE) -> np.ndarray:
    """A uint8 HWC frame that is NOT equal to its own 180 degree rotation.

    Chirality is the whole point and is asserted, not assumed: a symmetric fixture (zeros, a
    constant, a centrally symmetric pattern) makes every orientation assertion in this file pass
    unconditionally, which is the exact failure mode -- a test that cannot fail -- this project
    has shipped before.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    # A saturated block in ONE corner, so a flip is unmissable even by eye in a failure dump.
    img[: size // 8, : size // 8] = 255
    img[-size // 8 :, -size // 8 :] = 0
    assert not np.array_equal(img, img[::-1, ::-1]), "fixture is centrally symmetric"
    return img


def as_lerobot(frame: np.ndarray) -> torch.Tensor:
    """The frame as LeRobot hands it back: float32 CHW in [0, 1].

    MEASURED on the box, not guessed: `LeRobotDataset[0]["image"]` is a (3, 128, 128)
    torch.float32 tensor with min 0.0 / max 1.0. This matters -- `[::-1, ::-1]` on a CHW array
    flips the CHANNEL axis and the height axis, which is a red/blue swap plus a mirror, and no
    "is it flipped?" assertion would tell that apart from a correct rotation.
    """
    return torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)).astype(np.float32) / 255.0)


def sample(base: np.ndarray, wrist: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    return {
        "image": as_lerobot(base),
        "wrist_image": as_lerobot(wrist),
        "state": torch.from_numpy(rng.random(8).astype(np.float32)),
        "actions": torch.from_numpy(rng.random((10, 7)).astype(np.float32)),
        "prompt": "pick up the black bowl and place it on the plate",
        "episode_index": torch.tensor(3),
        "frame_index": torch.tensor(11),
        "quality_presentation": torch.tensor(0),
    }


def train_images(config_name: str, base: np.ndarray, wrist: np.ndarray) -> dict[str, np.ndarray]:
    """The model-facing images of a sample, through THE REAL training chain.

    `training_transforms` is the function `transform_dataset` -- and therefore
    `create_torch_data_loader` -- calls, so this is not a reconstruction of the pipeline; it is
    the pipeline. Norm stats are skipped (the LIBERO stage-2 stats live only on the box) and they
    touch state/actions, never images.
    """
    cfg = _config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    data = sample(base, wrist)
    for transform in _data_loader.training_transforms(data_config, skip_norm_stats=True):
        data = transform(data)
    return {k: np.asarray(v) for k, v in data["image"].items()}


def eval_images(base: np.ndarray, wrist: np.ndarray) -> dict[str, np.ndarray]:
    """The model-facing images the EVAL CLIENT produces from the same raw MuJoCo frames.

    Lines 247-250 of `benchmarks_eval/run_libero_plus_eval.py`, verbatim, followed by the serve
    input chain -- which is `LiberoInputs` then `ResizeImages`, and NOTABLY NOT the repack group
    (`policy_config._input_chain` builds from the repack_transforms ARGUMENT, and
    `serve_policy.py` passes none). That asymmetry is exactly why the rotation belongs in the
    repack group: put it in `data_transforms` and it would fire here too, cancelling the client's
    own flip and leaving the model upside-down again.
    """
    img = np.ascontiguousarray(base[::-1, ::-1])
    wr = np.ascontiguousarray(wrist[::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, RESIZE, RESIZE))
    element = {
        "observation/image": img,
        "observation/wrist_image": wr,
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    cfg = _config.get_config(STAGE2)
    data = libero_policy.LiberoInputs(model_type=cfg.model.model_type)(element)
    data = _transforms.ResizeImages(RESIZE, RESIZE)(data)
    return {k: np.asarray(v) for k, v in data["image"].items()}


# --- the pixel claim, both directions --------------------------------------------------------


def test_stage2_images_are_the_raw_frames_rotated_180():
    """FLAG ON: what the model sees is exactly `raw[::-1, ::-1]`, on BOTH image keys.

    Both keys are asserted in one test on purpose. Rotating the base view and leaving the wrist
    view alone produces a batch that trains perfectly well and is wrong; the only place the two
    can be compared is here.
    """
    base, wrist = chiral_frame(1), chiral_frame(2)
    got = train_images(STAGE2, base, wrist)
    np.testing.assert_array_equal(
        got["base_0_rgb"], base[::-1, ::-1], err_msg="stage-2 base image is not the raw frame rotated 180"
    )
    np.testing.assert_array_equal(
        got["left_wrist_0_rgb"],
        wrist[::-1, ::-1],
        err_msg="stage-2 WRIST image is not the raw frame rotated 180 -- a half-fix",
    )
    # And the rotation is a real change, not a no-op on a symmetric fixture.
    assert not np.array_equal(got["base_0_rgb"], base)
    assert not np.array_equal(got["left_wrist_0_rgb"], wrist)


def test_upright_declaration_leaves_the_raw_frames_untouched():
    """FLAG OFF: a config declaring its build upright emits the raw frames unchanged."""
    base, wrist = chiral_frame(3), chiral_frame(4)
    got = train_images(STOCK, base, wrist)
    np.testing.assert_array_equal(got["base_0_rgb"], base)
    np.testing.assert_array_equal(got["left_wrist_0_rgb"], wrist)


def test_stock_pi05_libero_pipeline_is_untouched_by_the_rotation():
    """`pi05_libero` reads PI's already-rotated copy and is the only harness reference that works.

    Two claims: no rotation is wired into its chain at all (a `Rotate180Images` anywhere in the
    repack group would be one), and the chain is the same sequence of transforms it has always
    been.
    """
    cfg = _config.get_config(STOCK)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(t).__name__ for t in data_config.repack_transforms.inputs]
    assert names == ["RepackTransform"], f"stock pi05_libero repack group changed: {names}"
    assert not any(
        isinstance(t, _orient.Rotate180Images)
        for t in (*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs)
    )
    assert [type(t).__name__ for t in data_config.data_transforms.inputs] == ["LiberoInputs"]


# --- the durable half: training and eval, one frame, both paths ------------------------------


def test_training_and_eval_observations_agree_on_which_way_is_up():
    """THE CHECK THAT WAS MISSING.

    No review caught the original bug because every review was scoped to a code diff, and the bug
    was a disagreement between two pipelines that never met. This is the meeting.

    One synthetic MuJoCo render goes down both paths:
      * EVAL -- `obs[...][::-1, ::-1]` -> resize_with_pad -> uint8 -> serve input chain;
      * TRAIN -- stored the way the box-local converter stored it (straight from the HDF5, NO
        flip, float32 CHW), then the real `pi05_libero_axisinit_paper` training chain.
    Equality is EXACT, not statistical, because the fixture controls both sides: at 224 px both
    resizes are no-ops, and the uint8 -> float32/255 -> *255 -> uint8 round trip is exact for all
    256 values (verified), so orientation is the only thing left that can differ.

    Under the pre-fix pipeline the training side is the eval side rotated 180 and this fails; a
    statistic would have to be argued about, an array comparison does not.
    """
    base, wrist = chiral_frame(5), chiral_frame(6)
    train = train_images(STAGE2, base, wrist)
    serve = eval_images(base, wrist)
    for key, side in (("base_0_rgb", "agentview"), ("left_wrist_0_rgb", "wrist")):
        if not np.array_equal(train[key], serve[key]):
            flipped = np.array_equal(train[key], serve[key][::-1, ::-1])
            raise AssertionError(
                f"TRAIN/EVAL ORIENTATION DISAGREEMENT on the {side} image ({key}).\n"
                f"  training pipeline ({STAGE2}) emits: "
                f"{'the eval observation rotated 180 degrees' if flipped else 'a different image'}\n"
                f"  eval client (run_libero_plus_eval.py) emits: the MuJoCo render flipped with "
                f"[::-1, ::-1], i.e. {_orient.MODEL_ORIENTATION!r}\n"
                f"  the AXIS stage-1 corpus is {_orient.MODEL_ORIENTATION!r} with no flip in its "
                f"input path, so {_orient.MODEL_ORIENTATION!r} is the convention all three stages "
                f"must share.\n"
                f"  Fix the TRAINING side: LeRobotLiberoDataConfig.dataset_image_orientation must "
                f"name the convention the dataset on disk actually stores."
            )
    # The fixture had to be able to fail: the two sides are not trivially equal because both are
    # constants.
    assert not np.array_equal(serve["base_0_rgb"], base)


def test_the_parity_check_fires_when_one_side_is_flipped():
    """The parity check's own mutation test, run in-process: flip the eval side and it must fire.

    Without this, `test_training_and_eval_observations_agree_on_which_way_is_up` is a check whose
    sensitivity is asserted only by argument.
    """
    base, wrist = chiral_frame(7), chiral_frame(8)
    train = train_images(STAGE2, base, wrist)
    serve = eval_images(base, wrist)
    flipped = {k: v[::-1, ::-1] for k, v in serve.items()}
    assert np.array_equal(train["base_0_rgb"], serve["base_0_rgb"])
    assert not np.array_equal(train["base_0_rgb"], flipped["base_0_rgb"])
    assert not np.array_equal(train["left_wrist_0_rgb"], flipped["left_wrist_0_rgb"])


# --- the declaration, pinned ------------------------------------------------------------------


def test_every_libero_config_declares_its_datasets_orientation():
    """A repo id does not determine which way up the frames are; the config must say.

    Enumerating the registry (rather than checking the three configs we happen to care about) is
    what makes a FUTURE config that reads a LIBERO dataset state its convention: it lands in
    neither list and this fails.
    """
    registered = {
        c.name: c.data
        for c in _config._CONFIGS  # noqa: SLF001
        if isinstance(c.data, _config.LeRobotLiberoDataConfig)
    }
    assert set(registered) == set(INVERTED_CONFIGS) | set(UPRIGHT_CONFIGS), (
        "a LIBERO config appeared or vanished; add it to INVERTED_CONFIGS or UPRIGHT_CONFIGS "
        f"after establishing which build it reads. registry={sorted(registered)}"
    )
    for name in INVERTED_CONFIGS:
        assert registered[name].dataset_image_orientation == _orient.INVERTED, name
    for name in UPRIGHT_CONFIGS:
        assert registered[name].dataset_image_orientation == _orient.UPRIGHT, name


def test_known_builds_declare_the_measured_orientation():
    """Pins the two builds' identities and conventions to what was measured on the box.

    PI official: 1,693 episodes at 256 px, upright (their converter rotates).
    Box-local:   2,000 episodes at 128 px, inverted (converted from HDF5 with no flip; matched
                 (task, frame 0) rows correlate -0.07/-0.25/-0.43 unflipped and +0.99/+0.97/+0.96
                 flipped, wrist +0.94/+0.92/+0.87).
    """
    by_key = {(b.total_episodes, b.image_hw): b for b in _orient.KNOWN_BUILDS}
    assert by_key[(1693, (256, 256))].stored_orientation == _orient.UPRIGHT
    assert by_key[(2000, (128, 128))].stored_orientation == _orient.INVERTED
    assert _orient.MODEL_ORIENTATION == _orient.UPRIGHT
    assert _orient.rotation_needed(_orient.INVERTED) is True
    assert _orient.rotation_needed(_orient.UPRIGHT) is False


def _info(total_episodes: int, hw: int) -> dict:
    return {"total_episodes": total_episodes, "features": {"image": {"shape": [hw, hw, 3]}}}


def test_guard_refuses_the_box_build_under_an_upright_declaration():
    """The original bug, reproduced: `HF_LEROBOT_HOME` swapped the build under a config that
    still meant PI's copy. Must raise, and must name BOTH conventions."""
    with pytest.raises(ValueError) as e:
        _orient.check_dataset_build(
            repo_id="physical-intelligence/libero",
            declared_orientation=_orient.UPRIGHT,
            info=_info(2000, 128),
        )
    msg = str(e.value)
    assert "upright" in msg and "inverted" in msg
    assert "HF_LEROBOT_HOME" in msg


def test_guard_refuses_pi_official_under_an_inverted_declaration():
    """The mirror image, and the one that protects the FIX: running the corrected stage-2 config
    against PI's already-rotated copy would rotate it a second time."""
    with pytest.raises(ValueError, match="orientation mismatch"):
        _orient.check_dataset_build(
            repo_id="physical-intelligence/libero",
            declared_orientation=_orient.INVERTED,
            info=_info(1693, 256),
        )


def test_guard_refuses_a_build_nobody_declared():
    with pytest.raises(ValueError, match="matches no build declared"):
        _orient.check_dataset_build(
            repo_id="physical-intelligence/libero",
            declared_orientation=_orient.UPRIGHT,
            info=_info(1234, 224),
        )


def test_guard_accepts_each_build_under_its_own_declaration():
    assert _orient.check_dataset_build(
        repo_id="x", declared_orientation=_orient.INVERTED, info=_info(2000, 128)
    ).total_episodes == 2000
    assert _orient.check_dataset_build(
        repo_id="x", declared_orientation=_orient.UPRIGHT, info=_info(1693, 256)
    ).total_episodes == 1693


def test_the_loader_runs_the_guard(monkeypatch):
    """The guard is only worth anything if `create_torch_dataset` actually calls it."""

    class _Meta:
        info = _info(2000, 128)
        fps = 10
        tasks = None

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **k: _Meta())
    cfg = _config.get_config(STOCK)  # declares upright; the fake metadata is the inverted build
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    with pytest.raises(ValueError, match="orientation mismatch"):
        _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)


def test_non_libero_configs_are_not_subject_to_the_guard():
    """`libero_image_orientation` is None everywhere else, so nothing else changed behaviour."""
    cfg = _config.get_config("pi05_droid")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert getattr(data_config, "libero_image_orientation", None) is None


# --- the transform itself ---------------------------------------------------------------------


def test_rot180_finds_the_height_and_width_axes_in_both_layouts():
    """CHW and HWC must give the SAME picture. `[::-1, ::-1]` applied blindly to CHW would flip
    the channel axis -- a red/blue swap plus a mirror -- which looks flipped and is not."""
    hwc = chiral_frame(9, size=32)
    chw = hwc.transpose(2, 0, 1)
    np.testing.assert_array_equal(_orient.rot180(hwc), hwc[::-1, ::-1])
    np.testing.assert_array_equal(_orient.rot180(chw), hwc[::-1, ::-1].transpose(2, 0, 1))
    # channels untouched
    np.testing.assert_array_equal(_orient.rot180(chw)[0], hwc[::-1, ::-1, 0])


def test_rot180_preserves_torch_tensors():
    t = as_lerobot(chiral_frame(10, size=16))
    out = _orient.rot180(t)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(t).copy()[:, ::-1, ::-1])


def test_rot180_refuses_shapes_whose_axes_it_cannot_identify():
    with pytest.raises(ValueError, match="ambiguous"):
        _orient.rot180(np.zeros((3, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="expected a 3-D image"):
        _orient.rot180(np.zeros((4, 8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="neither CHW nor HWC"):
        _orient.rot180(np.zeros((8, 8, 4), np.uint8))


def test_rotate180images_refuses_a_missing_key():
    """A key rename upstream must be a crash, not a silently unrotated view."""
    with pytest.raises(KeyError, match="observation/wrist_image"):
        _orient.Rotate180Images()({"observation/image": np.zeros((8, 8, 3), np.uint8)})


def test_rotate180images_rotates_every_key_it_names():
    data = {
        "observation/image": chiral_frame(11, size=16),
        "observation/wrist_image": chiral_frame(12, size=16),
        "observation/state": np.arange(8),
    }
    out = _orient.Rotate180Images()(dict(data))
    for k in _orient.IMAGE_KEYS:
        np.testing.assert_array_equal(out[k], data[k][::-1, ::-1])
    np.testing.assert_array_equal(out["observation/state"], data["observation/state"])


def test_declared_orientation_must_be_one_of_the_two():
    with pytest.raises(ValueError, match="unknown LIBERO image orientation"):
        dataclasses.replace(
            _config.get_config(STAGE2).data, dataset_image_orientation="sideways"
        ).create(_config.get_config(STAGE2).assets_dirs, _config.get_config(STAGE2).model)
