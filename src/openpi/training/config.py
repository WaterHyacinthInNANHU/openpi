"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.axis_franka_policy as axis_franka_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.libero_orientation as libero_orientation
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # --- SLB variant bake-off (see openpi.training.slb_variant_sampler) ---
    # When slb_sidecar_root is set, the JAX loader restricts/weights rows to the
    # kept (attempt, window) set of the given WVM variant. Left None for all
    # non-SLB configs, which keeps the default full-shuffle behaviour.
    slb_variant: str = "vanilla"
    slb_task_id: int | None = None
    slb_sidecar_root: str | None = None
    slb_manifest_path: str | None = None
    # Local root of the (subfoldered) camera_fixed LeRobot dataset. When set, the
    # loader passes it as `root=` so a HF sub-dataset can be loaded by path.
    slb_dataset_root: str | None = None
    # AWR weighting: w = min(exp(tau*Delta), cap), WVM Eq E.7.
    #
    # tau=10 is the paper's value for its SHORTEST chunk length (Table E.5: H=10 -> tau=10,
    # H=50 -> tau=2, "set per task suite to account for the different per-frame value-change
    # scales induced by different chunk lengths"). Our H=16 sits near the H=10 end.
    #
    # An earlier tau=3 was justified as matching "our smoothed horizon Delta scale", but the
    # direction was backwards: our Delta is SMALLER than the paper's (their top-70% cut lands
    # at kappa=0.02-0.06; ours at ~0.001-0.002), so it needs a LARGER tau, not smaller.
    # Measured at tau=3 the weights were effectively uniform -- median weight 1.015, only
    # 1.7% reaching the cap, effective-sample-size ratio 0.96 -- i.e. the awr arm was
    # indistinguishable from vanilla. At tau=10, 12-15% reach the cap.
    awr_tau: float = 10.0
    awr_delta: float = 2.0
    # Map sidecar windows to dataset rows with the RENDER-ALIGNED rule
    # (round((t - t0)*fps) + RENDER_FRAME_OFFSET). The legacy round(t*fps) is measurably one
    # frame late for ~95% of windows -- see slb_variant_sampler.RENDER_FRAME_OFFSET, which
    # documents the qpos-based measurement. False only reproduces pre-2026-07-22 runs.
    slb_render_aligned_rows: bool = True

    # --- PSB/pretraining multi-task loader (see openpi.training.pretrain_sampler) ---
    # Distinct from the single-task SLB path above. When pretrain_roots_index is set, the
    # loader concatenates the per-task __droid8d LeRobot sub-datasets it names and restricts
    # rows to the non-idle sample-ranges at pretrain_ranges_path (openpi's own ranges format,
    # keyed "task_<id>--<episode_index>"). Left None for every non-pretraining config, which
    # keeps the single-dataset behaviour. These never coexist with the slb_* fields.
    pretrain_roots_index: str | None = None
    # AWR weights for the pretrain path: a JSON artifact keyed by `task_<id>--<episode>` ->
    # per-frame float, built offline by `axis.dataset.awr_weights`. Unset means uniform,
    # unweighted pretraining -- the plain-BC baseline. The two supervision arms differ from that
    # baseline, and from each other, ONLY in which file this points at.
    pretrain_awr_weights: str | None = None
    pretrain_ranges_path: str | None = None
    # A precomputed index schedule (`axis.dataset.build_index_schedule`): an int64
    # (num_train_steps, batch) block of flat dataset rows plus a meta block. When set, the loader
    # replays it verbatim -- row t IS the batch trained at step t -- instead of drawing rows
    # itself, which is what makes the `drop` (which rows) and `anneal` (in what order) arms
    # differ from the uniform baseline. Mutually exclusive with pretrain_awr_weights: the
    # schedule already encodes the supervision, so a run carrying both would be two arms at once.
    pretrain_schedule_path: str | None = None
    # "drop" / "anneal" for a named schedule arm, or None for every other config. Set by
    # AxisFrankaPretrainDataConfig.expected_mode and checked against the artifact's own
    # `meta["mode"]` where the sampler is built (data_loader.py): nothing else ties a config
    # NAME to the artifact it was actually launched with, so handing `pi05_axis_drop` an anneal
    # artifact would otherwise train anneal under the drop name and pass every other check.
    pretrain_expected_mode: str | None = None
    # A precomputed dense per-row quality tag (`axis.dataset.build_quality_labels`): uint8 over
    # the CONCATENATED corpus row space, 0 = dropped out (the unconditional branch), 1..5 =
    # `Quality: q`, 255 = not trainable. When set, the loader wraps the raw dataset so each sample
    # carries its tag and prepends `quality_conditioning.AxisQualityConditioning` to the transform
    # chain (via `wrap_and_transform`, which is the only supported way to assemble the two).
    #
    # It changes WHAT THE MODEL SEES, never WHICH ROWS ARE DRAWN -- which is why it is a separate
    # field from `pretrain_schedule_path`, why the two are mutually exclusive, and why a config
    # carrying it still falls through to the control's own `RowSampler`. That coverage neutrality
    # is this arm's distinguishing property against its round-2 siblings and is asserted rather
    # than assumed (config_cfg_arm_test.py, data_loader_cfg_test.py).
    pretrain_quality_path: str | None = None

    # --- LIBERO image orientation (see openpi.training.libero_orientation) ---
    # "upright" / "inverted": which way up the LIBERO build this config reads stores its frames.
    # Set by `LeRobotLiberoDataConfig` from its own declaration and carried here for ONE consumer
    # -- `create_torch_dataset`, which is the first and only place the declaration can be checked
    # against the dataset that `HF_LEROBOT_HOME` actually resolved. None for every non-LIBERO
    # config, which skips the check entirely.
    libero_image_orientation: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                if model_config.knowledge_insulation:
                    # KI needs the FAST action ids teacher-forced INTO the token stream, which
                    # is exactly what the pi0-FAST tokenizer already produces: the pi0.5 prompt
                    # (`Task: ..., State: ...;\n`) followed by an `Action:` + FAST-ids + `|`
                    # postfix, plus the causal `token_ar_mask` and the `token_loss_mask` the
                    # discrete cross-entropy is averaged over.
                    #
                    # PadStatesAndActions runs FIRST so the FAST ids describe the same padded
                    # chunk the flow-matching head regresses; otherwise the two halves of KI
                    # would supervise different action spaces.
                    return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            _transforms.PadStatesAndActions(model_config.action_dim),
                            _transforms.TokenizeFASTInputs(
                                _tokenizer.FASTTokenizer(model_config.max_token_len),
                            ),
                        ],
                    )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(self.norm_stats_dir(assets_dirs)),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def norm_stats_dir(self, assets_dirs: pathlib.Path) -> epath.Path | None:
        """The directory `create_base_config` will read norm stats from, without reading them.

        Split out of `_load_norm_stats` so a test can assert WHICH directory a config resolves
        to. `_load_norm_stats` swallows a miss and returns None, and the resulting failure
        surfaces much later as "Normalization stats not found" from `transform_dataset`, so
        nothing else notices when two configs that must share one stats directory stop doing so
        (see the round-2 index-schedule arms in `_axis_pretrain_config`).
        """
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        if asset_id is None:
            return None
        return epath.Path(self.assets.assets_dir or assets_dirs) / asset_id

    def _load_norm_stats(self, data_assets_dir: epath.Path | None) -> dict[str, _transforms.NormStats] | None:
        if data_assets_dir is None:
            return None
        try:
            norm_stats = _normalize.load(_download.maybe_download(str(data_assets_dir)))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    # WHICH WAY UP THE DATASET THIS CONFIG READS STORES ITS FRAMES -- "upright" or "inverted".
    #
    # Not a rotate/don't-rotate switch: the 180 degree turn is DERIVED from this
    # (`libero_orientation.rotation_needed`), so a config cannot declare the data upright and
    # rotate it anyway. Default "upright" means no rotation, which keeps every config that reads
    # Physical Intelligence's official copy -- `pi05_libero`, `pi0_libero`, `pi0_fast_libero` and
    # the serve configs -- byte-identical to what it was; PI's own converter already baked the
    # rotation in.
    #
    # "inverted" is for the stage-2 arms, which read the box-local re-conversion at
    # /data/lerobot_hdf5 (HF_LEROBOT_HOME in axis_stage2.sh). Same repo id, opposite convention,
    # because that copy was converted straight from the LIBERO HDF5 with no flip after lerobot
    # 0.4.4 rejected PI's v2.0 copy. This is a run-record field on purpose: reading a registry
    # entry six months from now must answer "which way up did this run train?" without going to
    # the box to look at a dataset.
    #
    # The model-facing convention is UPRIGHT for all three stages -- the AXIS stage-1 corpus
    # stores right-side-up with no flip anywhere in its input path, and the LIBERO-Plus eval
    # client flips MuJoCo's render before it sends anything. See `libero_orientation`.
    dataset_image_orientation: str = libero_orientation.UPRIGHT

    # STAGE-2 CFG TWIN ONLY; None for every other LIBERO arm, which must stay byte-identical.
    #
    # A CONSTANT π0.7 quality tag, not a per-row one: stage 2 finetunes on uniformly expert data,
    # so there is nothing to bin. The point is that the conditional branch is anchored where
    # inference asks for it (D9), while the two-level dropout inside the transform keeps the
    # unconditional branch trained. Unlike stage 1 there is no artifact and no dataset wrapper --
    # the dropout is a keyed hash of the row, a disclosed deviation from D5. See
    # `quality_conditioning.LiberoQualityConditioning`.
    quality_tag: int | None = None

    # The config NAME whose norm stats this one must read, or None to keep its own. Set on the
    # CFG twin so it shares `pi05_libero_axisinit_paper`'s stats rather than resolving to
    # `<assets_base_dir>/pi05_libero_axisinit_paper_cfg/...`, which does not exist anywhere -- the
    # LIBERO stage-2 norm stats were computed once, on the training box, and are published
    # nowhere (reports/portability_and_publish_plan.md). Without this the twin dies at loader
    # construction telling you to run `compute_norm_stats --config-name=<your-config>`, and doing
    # that would be a SECOND parity break: the CFG row must differ from the other stage-2 arms in
    # the conditioning ONLY, and this project has already retracted two conclusions that came
    # from norm stats quietly belonging to a different dataset than the one being trained on.
    #
    # A NAME, not a path, for the reason `AxisFrankaPretrainDataConfig.norm_stats_from` records at
    # length: `TrainConfig.assets_dirs` is `assets_base_dir / name`, so swapping the last
    # component follows an `--assets-base-dir` override instead of stranding this config on the
    # default base while the config it must match moves elsewhere.
    norm_stats_from: str | None = None

    @override
    def norm_stats_dir(self, assets_dirs: pathlib.Path) -> epath.Path | None:
        if self.norm_stats_from is None:
            return super().norm_stats_dir(assets_dirs)
        return super().norm_stats_dir(pathlib.Path(assets_dirs).parent / self.norm_stats_from)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        #
        # The CFG twin HEADS this group with its conditioning, and that position is load-bearing
        # in both directions: the transform needs `episode_index`/`frame_index` (which
        # RepackTransform drops) to key its dropout draw, and it needs to be the ONLY conditioning
        # entry in the chain or the tag lands twice. Unlike stage 1 there is no
        # `wrap_and_transform` here -- a constant tag needs no dataset wrapper, so the repack
        # group IS the right place, not the forbidden one.
        quality_inputs: list[_transforms.DataTransformFn] = []
        if self.quality_tag is not None:
            # Local import: `quality_conditioning` is only reached by the one twin, and keeping it
            # off this module's import path matches how the SLB factory reaches `slb_cfg`.
            from openpi.training import quality_conditioning

            quality_inputs = [
                quality_conditioning.LiberoQualityConditioning(q_ep=int(self.quality_tag))
            ]
        # THE ORIENTATION FIX, and it is deliberately in the repack group rather than in
        # `data_transforms`: inference never runs repack transforms (`serve_policy.py` passes
        # none), and the eval client ALREADY flips MuJoCo's render, so a rotation that also ran at
        # serve time would apply twice and leave us upside-down again with a new explanation.
        # It sits AFTER the repack, on the canonical `observation/...` keys, and rotates BOTH of
        # them -- see `libero_orientation.Rotate180Images`.
        rotate_inputs: list[_transforms.DataTransformFn] = []
        if libero_orientation.rotation_needed(self.dataset_image_orientation):
            rotate_inputs = [libero_orientation.Rotate180Images()]
        repack_transform = _transforms.Group(
            inputs=[
                *quality_inputs,
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                ),
                *rotate_inputs,
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Carried so `create_torch_dataset` can refuse a build that is not the one this
            # declaration is about. The transform above is derived from the SAME field, so the
            # check and the rotation cannot drift apart.
            libero_image_orientation=self.dataset_image_orientation,
        )


@dataclasses.dataclass(frozen=True)
class AxisFrankaSlbDataConfig(DataConfigFactory):
    """SLB fine-tuning over HF-rendered Franka data (pi0.5)."""

    variant: str = "vanilla"
    default_prompt: str | None = None
    task_id: int | None = None
    sidecar_root: str | None = None
    manifest_path: str | None = None
    dataset_root: str | None = None
    awr_tau: float = 10.0   # WVM Eq E.7 / Table E.5; see DataConfig.awr_tau
    awr_delta: float = 2.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        from openpi.policies import axis_franka_policy

        # SLB `cfg`: advantage-conditioned CFG, injected into the VLM's language stream
        # (RECAP / pi*0.6 style, mirroring RLinf's cfg_rl_openpi reference). Runs FIRST so
        # it still sees `episode_index` / `frame_index`, which RepackTransform drops.
        # Offline advantage only -- no critic, matching the reference's
        # add_value_head/use_critic_model=False.
        cfg_inputs: list[_transforms.DataTransformFn] = []
        # NB: these are the FACTORY's fields (`variant`, `task_id`, ...), not the DataConfig's
        # `slb_`-prefixed ones -- the prefixed spellings raised AttributeError on every call.
        if self.variant == "cfg" and self.sidecar_root and self.task_id is not None:
            from openpi.training import slb_cfg

            try:
                cfg_inputs = [slb_cfg.build_conditioning(
                    task_id=int(self.task_id),
                    sidecar_root=self.sidecar_root,
                    manifest_path=self.manifest_path,
                )]
            except Exception as exc:  # noqa: BLE001
                # Fail loudly: silently training `cfg` without conditioning would make it a
                # byte-for-byte duplicate of vanilla and quietly invalidate that arm.
                raise RuntimeError(f"SLB cfg conditioning unavailable for task {self.task_id}: {exc}") from exc

        repack_transform = _transforms.Group(
            inputs=cfg_inputs + [
                _transforms.RepackTransform(
                    {
                        "base_0_rgb": "observation.images.third_person",
                        "left_wrist_0_rgb": "observation.images.wrist",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[axis_franka_policy.AxisFrankaInputs()],
            outputs=[axis_franka_policy.AxisFrankaOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            slb_variant=self.variant,
            slb_task_id=self.task_id,
            slb_sidecar_root=self.sidecar_root,
            slb_manifest_path=self.manifest_path,
            slb_dataset_root=self.dataset_root,
            awr_tau=self.awr_tau,
            awr_delta=self.awr_delta,
        )


@dataclasses.dataclass(frozen=True)
class AxisFrankaPretrainDataConfig(DataConfigFactory):
    """pi0.5 pretraining over the FULL AXIS Franka corpus (~182 tasks, one demo/scene variant).

    Unlike the single-task SLB factory this trains over many tasks at once. The per-task
    __droid8d LeRobot sub-datasets are concatenated by the loader (roots named in
    ``roots_index``) and rows are restricted to the non-idle sample-ranges in ``ranges_path``.
    No SLB variant sidecars and no CFG conditioning. AWR IS supported here: set
    ``pretrain_awr_weights`` (or $AXIS_PRETRAIN_AWR_WEIGHTS) and the loader applies
    per-row weights via WVM Eq E.5.

    Norm stats are the config's OWN (computed by scripts/compute_norm_stats.py), NOT DROID's:
    a prior experiment showed DROID's stats do not fit this action/state distribution, and a
    full-weight run over a large corpus can learn its own scale. So this factory deliberately
    omits the ``assets=AssetsConfig(.../pi05_droid/assets, asset_id="droid")`` override the SLB
    arms carry -- assets default to the config's own dir.
    """

    roots_index: str | None = None
    ranges_path: str | None = None
    # Crop both cameras to square before the 224 resize. The 5k `camera_fixed` render stores
    # 640x360; without this a straight resize squashes 16:9 into 1:1. DECLARED per config, not
    # sniffed from the frame shape -- see `_center_crop_square`.
    image_center_crop: bool = False
    # Path to the AWR weights artifact, or None for uniform unweighted pretraining. This is the
    # ONLY field that differs between the plain-BC baseline and either supervision arm, which is
    # what makes the three runs a controlled comparison rather than three separate experiments.
    awr_weights: str | None = None
    # Path to an index-schedule artifact (`axis.dataset.build_index_schedule`), or None for the
    # loader's own uniform draw. NOT an env var, unlike round 1's $AXIS_PRETRAIN_AWR_WEIGHTS: a
    # checkpoint's arm must be recoverable from its config (name + this field) alone.
    schedule_path: str | None = None
    # True for the named schedule arms (`pi05_axis_drop`, `pi05_axis_anneal`): `create()` raises
    # if this is set but `schedule_path` is empty, so a launch that forgets the artifact flag
    # fails loudly instead of quietly training the plain control under the arm's name (the only
    # symptom would otherwise be the ABSENCE of the "index schedule" log line).
    schedule_required: bool = False
    # A named loss-reweighting arm that forgets --data.awr_weights trains the plain BC control
    # under the arm's name -- which is precisely what happened to the 5k eef awr arms. Fail loud.
    awr_required: bool = False
    # "drop" / "anneal" for a named schedule arm, or None otherwise. Reaches DataConfig as
    # `pretrain_expected_mode` and is checked against the artifact's own `meta["mode"]` in
    # data_loader.py -- binds the config NAME to the artifact's actual content, since nothing
    # else does (see DataConfig.pretrain_expected_mode).
    expected_mode: str | None = None
    # Path to a quality-tag artifact (`axis.dataset.build_quality_labels`), or None for the
    # untagged control. NOT an env var, for the same reason `schedule_path` is not: a checkpoint's
    # arm must be recoverable from its config (name + this field) alone -- and the SLB path's
    # `SLB_CFG_METADATA` switch is explicitly the pattern NOT to copy here.
    quality_path: str | None = None
    # True for `pi05_axis_cfg`: `create()` raises if this is set but `quality_path` is empty, so a
    # launch that forgets the artifact flag fails loudly instead of quietly training the plain
    # control under the arm's name (the only other symptom is an ABSENT log line). There is no
    # `expected_mode` twin: the artifact's own filename<->reward_id binding
    # (`QualityTags.check_reward_id`) is what separates `cfg_v2` from `cfg_phase`, which share
    # this one config name.
    quality_required: bool = False
    # STAGE-2 CONSTANT pi0.7 quality tag -- the heldout CFG finetune twins ONLY; None for every
    # other arm. DISTINCT FROM `quality_path`, which is stage 1's per-row artifact over the
    # concatenated pretrain corpus: a finetune on uniformly expert heldout demos has nothing to
    # bin, so the tag is a CONSTANT anchored at what inference asks for
    # (slb_cfg.INFER_QUALITY == 5) while the transform's own two-level dropout keeps the
    # unconditional branch trained -- the same mechanism, and the same transform
    # (`quality_conditioning.LiberoQualityConditioning`), as `LeRobotLiberoDataConfig.quality_tag`.
    # `create()` refuses to combine it with `quality_path` (two stages in one run), and with
    # `awr_weights` or `schedule_path` (two arms at once, and the schedule leaves no sampler slot
    # for the presentation counter the dropout is keyed on).
    #
    # NORM STATS: a tagged config must NEVER run compute_norm_stats -- the conditioning transform
    # heads the repack group and RAISES without the presentation counter that only the training
    # loader wires (`wrap_presentations`). Set `norm_stats_from` to the untagged twin: the stats
    # are over the same roots/ranges/columns and the tag never touches state or actions, so the
    # twin's stats ARE this arm's stats.
    quality_tag: int | None = None
    # The config NAME whose norm stats this one must read (round 1's, for the round-2 schedule
    # arms), or None to keep its own. A name, not a path, deliberately: `TrainConfig.assets_dirs`
    # is `assets_base_dir / name`, so resolving the sibling from the `assets_dirs` handed to
    # `create()` tracks any `--assets-base-dir` override. A literal `./assets/<round1>` would
    # instead pin the arms to the DEFAULT base while an overridden round 1 moved elsewhere --
    # silently leaving the arms on stats belonging to a different dataset, the exact failure this
    # binding exists to prevent (and one this project has already retracted two conclusions over).
    norm_stats_from: str | None = None
    default_prompt: str | None = None
    # Relative-EEF action space (LIBERO-Plus proxy benchmark): feed the baked `state_eef`(8) /
    # `action_eef`(7, robosuite OSC_POSE delta) columns and slice the output to 7. Default False
    # keeps the DROID-8D joint-velocity layout (for a future real-world checkpoint).
    eef_action: bool = False
    # Emit 9-D joint-position actions (7 joint targets + 2 finger widths) instead of the 8-D DROID
    # velocity slice. Must match the action space the eval controller is run in.
    joint9_action: bool = False

    @override
    def norm_stats_dir(self, assets_dirs: pathlib.Path) -> epath.Path | None:
        """Round-2 arms read a SIBLING config's stats dir, resolved from the base they were given.

        `assets_dirs` is `TrainConfig.assets_base_dir / TrainConfig.name`, so swapping the last
        component is the same computation round 1 does, under whatever base this run actually
        uses. See `norm_stats_from` for why this is not the literal path it replaces.
        """
        if self.norm_stats_from is None:
            return super().norm_stats_dir(assets_dirs)
        return super().norm_stats_dir(pathlib.Path(assets_dirs).parent / self.norm_stats_from)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        from openpi.policies import axis_franka_policy

        if self.schedule_required and not self.schedule_path:
            # A launch that omits --data.schedule_path trains the plain BC control under this
            # arm's name; the log line announcing the schedule simply never appears. Fail at
            # config-construction time instead of leaving that to be noticed later.
            raise ValueError(
                f"this is a named schedule arm (expected_mode={self.expected_mode!r}) but "
                f"schedule_path is not set. Pass --data.schedule_path=<artifact path> at launch "
                f"(the {self.expected_mode!r} artifact built by axis.dataset.build_index_schedule); "
                f"otherwise this run trains the plain BC control under the arm's name."
            )
        if self.schedule_path and not self.roots_index:
            # The schedule stores flat rows of the CONCATENATED multi-task dataset, which only
            # exists when roots_index names its parts. Without it the pretrain branch of the
            # loader never runs and the schedule would be silently ignored -- i.e. the arm would
            # train as the plain baseline under the arm's name.
            raise ValueError(
                "schedule_path is set but roots_index is not; the schedule indexes the "
                "concatenated pretrain dataset, so it is meaningless without one "
                "(set $AXIS_PRETRAIN_ROOTS_INDEX)."
            )
        if self.schedule_path and self.awr_weights:
            raise ValueError(
                "schedule_path and awr_weights are both set. The schedule already encodes the "
                "supervision (which rows, in what order); combining it with loss reweighting "
                "would make this run two arms at once."
            )
        if self.awr_required and not self.awr_weights:
            raise ValueError(
                "this is a named loss-reweighting arm but awr_weights is not set. Pass "
                "--data.awr_weights=<artifact path> at launch; otherwise this run trains the "
                "plain BC control under the arm's name -- which is exactly what happened to the "
                "5k eef awr arms (the named-config guard nulled the env var silently)."
            )
        if self.quality_required and not self.quality_path:
            # A launch that omits --data.quality_path trains the plain BC control under this arm's
            # name; the log line announcing the conditioning simply never appears.
            raise ValueError(
                "this is a named quality-conditioning arm but quality_path is not set. Pass "
                "--data.quality_path=<artifact path> at launch (the quality_<reward_id>.npz "
                "built by axis.dataset.build_quality_labels); otherwise this run trains the "
                "plain BC control under the arm's name."
            )
        if self.quality_path and not self.roots_index:
            # The tag array is dense over the CONCATENATED multi-task dataset, which only exists
            # when roots_index names its parts. Without it the pretrain branch of the loader never
            # runs, the artifact is silently ignored, and the arm trains as the plain baseline.
            raise ValueError(
                "quality_path is set but roots_index is not; the tag array is dense over the "
                "concatenated pretrain dataset, so it is meaningless without one "
                "(set $AXIS_PRETRAIN_ROOTS_INDEX)."
            )
        if self.quality_path and self.schedule_path:
            raise ValueError(
                "quality_path and schedule_path are both set. CFG's distinguishing property is "
                "coverage neutrality -- it draws exactly the rows the round-1 control draws, in "
                "the same order -- and a schedule replaces that draw entirely. Combining them "
                "would silently give up the one thing that makes this arm comparable to the "
                "control, and would make the run two arms at once."
            )
        if self.quality_path and self.awr_weights:
            raise ValueError(
                "quality_path and awr_weights are both set; conditioning and loss reweighting "
                "are two different mechanisms, so this run would be two arms at once."
            )
        if self.quality_tag is not None:
            if self.quality_path:
                raise ValueError(
                    "quality_tag (the stage-2 constant) and quality_path (stage 1's per-row "
                    "artifact) are both set; the two tag the same prompt from different "
                    "sources, and one run cannot be both stages."
                )
            if self.awr_weights:
                raise ValueError(
                    "quality_tag and awr_weights are both set; conditioning and loss "
                    "reweighting are two different mechanisms, so this run would be two arms "
                    "at once."
                )
            if self.schedule_path:
                raise ValueError(
                    "quality_tag and schedule_path are both set. The schedule replaces the "
                    "loader's draw entirely, which leaves no sampler slot for the presentation "
                    "counter the stage-2 dropout is keyed on."
                )
        state_col = "state_eef" if self.eef_action else "observation.state"
        action_col = "action_eef" if self.eef_action else "action"
        # NO ARTIFACT (stage-1) conditioning transform here, deliberately, and this is
        # load-bearing. `quality_conditioning.wrap_and_transform` -- the only supported way to
        # assemble the dataset wrapper and `AxisQualityConditioning`, and what the loader calls
        # -- PREPENDS the conditioning to the transform list it is handed, which is
        # `transform_dataset`'s and therefore starts with these very inputs. Heading the repack
        # group as well would append the tag TWICE ("...\nQuality: 5\nQuality: 5"): in range,
        # tokenizable, unmatched by any eval-time prompt, and silent. So with `quality_path` the
        # repack group stays exactly the control's -- asserted in config_cfg_arm_test.py.
        #
        # The CONSTANT-tag path (`quality_tag`, the heldout CFG finetune twins) is the OPPOSITE
        # case and mirrors `LeRobotLiberoDataConfig`: no artifact, no dataset wrapper, so the
        # head of the repack group IS the right place -- the transform needs `episode_index`/
        # `frame_index` and the presentation counter, all of which `RepackTransform` drops. The
        # guards above make the two paths mutually exclusive, so the double-tag route is closed.
        quality_inputs: list[_transforms.DataTransformFn] = []
        if self.quality_tag is not None:
            # Local import, matching LeRobotLiberoDataConfig: only the tagged twins reach this.
            from openpi.training import quality_conditioning

            quality_inputs = [
                quality_conditioning.LiberoQualityConditioning(q_ep=int(self.quality_tag))
            ]
        repack_transform = _transforms.Group(
            inputs=[
                *quality_inputs,
                _transforms.RepackTransform(
                    {
                        "base_0_rgb": "observation.images.third_person",
                        "left_wrist_0_rgb": "observation.images.wrist",
                        "state": state_col,
                        "actions": action_col,
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[axis_franka_policy.AxisFrankaInputs(center_crop=self.image_center_crop)],
            outputs=[
                axis_franka_policy.AxisFrankaEEFOutputs()
                if self.eef_action
                else (
                    axis_franka_policy.AxisFrankaJoint9Outputs()
                    if self.joint9_action
                    else axis_franka_policy.AxisFrankaOutputs()
                )
            ],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(action_col,),
            pretrain_roots_index=self.roots_index,
            pretrain_awr_weights=self.awr_weights,
            pretrain_ranges_path=self.ranges_path,
            pretrain_schedule_path=self.schedule_path,
            pretrain_expected_mode=self.expected_mode,
            pretrain_quality_path=self.quality_path,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _slb_freeze_filter(freeze_vision: bool):
    """Params to FREEZE for an SLB LoRA arm.

    The base LoRA recipe (get_freeze_filter) freezes LLM non-LoRA weights but leaves the
    ~400M SigLIP So400m image tower FULLY TRAINABLE (its params live under `.*img.*`, never
    matched by the `.*llm.*` freeze). On the 25-demo, per-episode appearance-randomized SLB
    set that overfits/corrupts the pretrained visual grounding -- diagnosed as the cause of
    uniform ~0-20% success with all-`timeout` failures and no variant separation. With
    `freeze_vision` we additionally freeze the entire image tower so only the LLM +
    action-expert LoRA adapt (the standard low-data VLA recipe), preserving visual grounding.
    """
    import openpi.shared.nnx_utils as nnx_utils

    base = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter()
    if not freeze_vision:
        return base
    return nnx.Any(base, nnx_utils.PathRegex(".*img.*"))

def _pretrain_freeze_filter(freeze_vision: bool):
    """Freeze the SigLIP vision ENCODER for a full-weight pretrain arm (the `_vfz` twins).

    Scope is MolmoBot-aligned: img/{Transformer,embedding,pos_embedding} freeze (412,442,352
    params) while img/head -- the projector; their conversion maps it to multi_modal_projector
    and their "vision_tower" freeze regex does not match it -- stays TRAINABLE (2,361,344
    params). A bare `.*img.*` (the `_slb_freeze_filter` tower regex) would over-freeze the
    projector, which is exactly the coverage difference this helper exists to encode.

    Unlike `_slb_freeze_filter` there is no LoRA base here: with `freeze_vision` unset this
    returns the `freeze_filter` field's default value (`nnx.Nothing()`), so a config built
    through this helper resolves identically to one built before the kwarg existed (proven by
    the before/after snapshot at patch time). That the matched params are actually nonzero is
    asserted by the param-count tripwire in scripts/train.py at startup, and the driver's
    bit-identity check asserts the complement: img/head MUST move -- a projector that never
    moves means the filter over-matched again.
    """
    if not freeze_vision:
        return nnx.Nothing()
    import openpi.shared.nnx_utils as nnx_utils

    return nnx_utils.PathRegex(".*img/(Transformer|embedding|pos_embedding).*")


# Knowledge insulation (arXiv:2505.23705), shared by every KI arm so the DROID, AXIS-pretrain
# and SLB twins are comparable. The FAST token budget is no longer a model field: the ids are
# spliced into the prompt, so `max_token_len` (250 for KI, see Pi0Config.__post_init__) is what
# bounds them.
_KI_MODEL_KWARGS = {"knowledge_insulation": True}


def _with_ki_twin(config: TrainConfig) -> list[TrainConfig]:
    """Returns `[config, config_with_knowledge_insulation]` for splicing into `_CONFIGS`.

    Built with `dataclasses.replace` so the twin provably differs from the base in the KI
    model fields and the name and NOTHING else -- the A/B stays clean even if the base config
    is later edited, and the base itself is returned untouched for in-flight runs.

    Only meaningful for TRAINABLE configs: `knowledge_insulation` is read exclusively by
    `compute_loss`, so an inference-only config (e.g. `pi05_droid`) would gain an untrained
    `discrete_action_head` and behave identically at sampling time.
    """
    # max_token_len is reset to None so Pi0Config.__post_init__ re-derives it: KI needs the
    # larger budget (250) to fit the spliced FAST postfix, and the base config has already
    # resolved its own default (200) by the time we get here.
    ki = dataclasses.replace(
        config,
        name=f"{config.name}_ki",
        model=dataclasses.replace(config.model, max_token_len=None, **_KI_MODEL_KWARGS),
    )
    return [config, ki]


def _axis_slb_config(
    task_id: int,
    variant: str,
    *,
    knowledge_insulation: bool = False,
    num_train_steps: int = 20_000,
    # Vision UNFROZEN by default: empirically, freezing the SigLIP tower made grasp success
    # WORSE in every condition (0/20-0/50 vs 1-2/20 unfrozen), with the gripper stuck ~90%
    # closed. Our robosuite renders are far from SigLIP's natural-image pretraining, so the
    # tower must ADAPT to the sim domain -- the opposite of the GR00T/pi0.5-KI freeze recipe,
    # which assumes a near-natural fine-tune domain. Keep False for AXIS sim-render data.
    freeze_vision: bool = False,
    name_suffix: str = "",
) -> TrainConfig:
    """One arm of the AXIS Franka SLB bake-off (pi0.5 LoRA, HF-rendered Franka dataset).

    `knowledge_insulation` adds a SECOND, parallel family of arms rather than changing the
    five existing ones: a sweep is running against those, so their configs must stay
    byte-identical. With the flag off this returns exactly what the five arms were before
    this helper existed.

    `num_train_steps` lets a caller pin the budget to a fixed number of EPOCHS over a small
    homogeneous demo set (e.g. 50 epochs over the 25-demo variant selection -> ~2.5k steps)
    instead of the 20k default. At a short budget the inherited warmup=1000 / decay=30_000
    cosine does not anneal (at ~2.5k steps LR is ~98% of peak) and warmup would eat ~40% of
    the run -- the exact failure the num_train_steps comment below warns about -- so we scale
    warmup to ~10% and decay over the whole run. At the 20k default the truncated cosine that
    matches the four reference recipes is kept unchanged.
    """
    # Budget-matched cosine: warmup ~10% of the run, decay over the WHOLE run so the LR
    # actually anneals to the floor at this fixed-epoch budget (the inherited default
    # warmup=1000/decay=30_000 neither warms nor anneals correctly at a few-thousand-step
    # run). Applied at every SLB budget now that we train a fixed number of epochs rather
    # than the old 20k truncated-cosine recipe.
    slb_lr = _optimizer.CosineDecaySchedule(
        warmup_steps=max(100, num_train_steps // 10),
        peak_lr=2.5e-5,
        decay_steps=num_train_steps,
        decay_lr=2.5e-6,
    )
    ki_kwargs = _KI_MODEL_KWARGS if knowledge_insulation else {}

    return TrainConfig(
        name=f"pi05_axis_slb_{task_id}_{variant}{name_suffix}" + ("_ki" if knowledge_insulation else ""),
        # LoRA fine-tune: the model itself must carry the LoRA gemma variants
        # (not just the freeze_filter), or training silently falls back to a
        # full fine-tune. Must match the freeze_filter variants below.
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            **ki_kwargs,
        ),
        data=AxisFrankaSlbDataConfig(
            repo_id="Devon018/Franka-Datasets-v2",
            variant=variant,
            # Per-task pilot config. The sidecar root defaults to the offline cache
            # convention; manifest + local dataset root come from env (keyed by
            # task_id) so the committed config carries no machine-specific paths.
            task_id=task_id,
            sidecar_root=os.path.join(
                os.environ.get("AXIS_DATALOADER_ROOT", os.path.expanduser("~/axis_dataloader_cache")),
                "sidecars",
            ),
            manifest_path=os.environ.get(f"SLB_MANIFEST_{task_id}"),
            dataset_root=os.environ.get(f"SLB_DATASET_ROOT_{task_id}"),
            base_config=DataConfig(prompt_from_task=True),
            # PER-TASK norm stats (NOT DROID's). Measured: AXIS 8-D velocity actions occupy
            # only 45-96% of DROID's normalized range (DROID over-scales each dim 1.3-2.2x),
            # so reusing DROID stats compresses the action signal the expert sees. We now run
            # compute_norm_stats per task and load those instead.
            #   * assets_dir=None + asset_id=None -> asset_id defaults to repo_id, and both
            #     compute_norm_stats (writes config.assets_dirs/repo_id) and the loader (reads
            #     assets_dirs/asset_id) resolve to the SAME per-CONFIG path
            #     openpi/assets/<config_name>/<repo_id>/norm_stats.json. They MATCH, so the run
            #     loads exactly what compute wrote.
            #   * To keep all five variants of a task on ONE normalization (no confound), the
            #     prep step computes stats on `vanilla` and COPIES the file into every variant
            #     config's assets dir. See sbatch/prep_slb_norm_stats.sbatch.
            # MUST run compute_norm_stats before training -- otherwise the stale 9-D
            # position-space norm_stats.json under openpi/assets/pi05_axis_slb_*/ would load
            # silently for this 8-D velocity dataset. The prep sbatch overwrites those.
            assets=AssetsConfig(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        # 20k is the field consensus for single-task pi0.5 finetuning, chosen
        # independently by four sources: openpi's own pi05_droid_finetune
        # ("a custom (smaller) DROID dataset" -- our exact setting, and we already
        # copy its init and norm-stats), RoboCasa's single-task pi05-og config,
        # RLinf's robotwin single-task SFT, and the one RoboTwin config that
        # overrides the default. The previous 30_000 was NOT a choice: it is the
        # TrainConfig default at :579 restated, and `git log -L` on that line shows
        # it was never revisited. RoboTwin inherits the same default the same way.
        #
        # Deliberately NOT changing the lr_schedule alongside it. The inherited
        # CosineDecaySchedule has decay_steps=30_000, so stopping at 20k ends at
        # ~34% of peak LR rather than the floor. Every one of those four sources
        # runs exactly this truncated cosine, so matching them is the defensible
        # choice; the truncation is a known divergence, not an oversight. (It WOULD
        # be a problem at a much shorter budget -- at 4k the LR is still 97.7% of
        # peak, i.e. no annealing at all.)
        num_train_steps=num_train_steps,
        # Keep ONLY the final checkpoint (max_to_keep=1 + no keep_period). At 5000 the
        # default kept 6 checkpoints/arm (~114 GB), and 10 arms exhausted the shared 10 TB
        # /bigdata group quota mid-run -> orbax EDQUOT -> 6 arms died. The eval only ever
        # loads the final step, so intermediates are pure quota cost here.
        keep_period=None,
        lr_schedule=slb_lr,
        # LeRobot v3.0 decodes video per item; the default 2 workers starve both
        # norm-stats and training. 8 matches --cpus-per-task=8 in the sbatch.
        num_workers=8,
        freeze_filter=_slb_freeze_filter(freeze_vision),
        ema_decay=None,
    )


# The held-out 20 benchmark. Baked in as a literal rather than read from the benchmark JSON,
# because a config module must import on a machine that has none of our artifacts -- a registry
# that silently shrinks to zero arms on a fresh box is the failure this avoids.
# Selection and rationale: docs/heldout_20_benchmark_spec.md. 813 is excluded (unevaluable).
HELDOUT_20_TASK_IDS = (
    809, 810, 811, 815, 866, 868, 929, 945, 953, 966,
    970, 1046, 1252, 1426, 1427, 1458, 1459, 1746, 1889, 1891,
)

# TRAINABLE SAMPLES in each task's 20-demo adaptation split -- the idle-FILTERED window total,
# not the raw episode length. This distinction is the whole budget.
#
# The loader restricts rows to the non-idle ranges, so raw frames are NOT what gets trained on, and
# retention is neither constant nor close to it: measured over these 20 tasks it runs from 35.7%
# (815) to 88.0% (1427), pooled 64.8%. Deriving the budget from raw frames -- which an earlier
# version of this table did -- would have handed task 1427 roughly 2.4x the real exposure of task
# 815 while both were labelled "50 epochs", and the gate would then have reported the difference as
# a difference in LEARNABILITY. That is precisely the confound an epoch budget exists to remove.
#
# Raw frame totals are kept alongside only so the retention is auditable; nothing derives from them.
#   * The four measured on the BUILT corpus (811/866/953/1889) are authoritative: they come from
#     the idle-filtered ranges the loader will actually draw, over the quality-ranked demos.
#   * The rest are still estimates from the pre-quality-ranking selection and MUST be re-measured
#     from each task's built ranges before its arm is launched -- ranking changed which demos are
#     selected, and 811 moved 1871 -> 1728 on that change alone.
HELDOUT_20_ADAPT_SAMPLES = {
    809: 6297, 810: 6395, 811: 1719, 815: 4066, 866: 5189,
    868: 2673, 929: 5490, 945: 8635, 953: 5486, 966: 5041,
    970: 4935, 1046: 5393, 1252: 4349, 1426: 10362, 1427: 9546,
    1458: 7291, 1459: 7395, 1746: 5097, 1889: 4750, 1891: 2832,
}

# Measured on the BUILT corpus rather than estimated. Only these are trustworthy today.
HELDOUT_SAMPLES_MEASURED = frozenset({809, 811, 866, 868, 945, 953, 966, 1426, 1459, 1889})

HELDOUT_20_ADAPT_RAW_FRAMES = {
    809: 8316, 810: 7593, 811: 2304, 815: 6291, 866: 7437,
    868: 3270, 929: 7533, 945: 11376, 953: 7530, 966: 6177,
    970: 7422, 1046: 9288, 1252: 6843, 1426: 12282, 1427: 10089,
    1458: 10836, 1459: 18294, 1746: 5670, 1889: 5949, 1891: 3447,
}

HELDOUT_GATE_EPOCHS = 50
HELDOUT_GATE_BATCH = 32
# Merged idle-filtered sample count over all 20 tasks (measured on the re-render).
# TASK 811's trainable-window count, and ONLY task 811's. Kept for the runs already reported
# against it; `heldout_epoch_steps` is what new runs should use.
HELDOUT_MULTITASK_SAMPLES = 1_597

def heldout_multitask_steps(epochs: int, batch_size: int = HELDOUT_GATE_BATCH) -> int:
    """DEPRECATED: `epochs` here means epochs OVER TASK 811, whatever data the run actually loads.

    `HELDOUT_MULTITASK_SAMPLES` is hardcoded to 811's 1,597 idle-filtered windows, so this returns a
    budget that is correct for 811 and wrong for everything else, silently:

        task 811   1,597 rows -> 998 steps -> 20.0 epochs   (eef7 scored 86%)
        task 1889  3,804 rows -> 998 steps ->  8.4 epochs   (eef7 scored 0/50)
        20 tasks  98,113 rows -> 998 steps ->  0.3 epochs

    A longer task silently receives proportionally less exposure, so per-task numbers produced with
    this helper are NOT like-for-like and cross-task comparisons using it are invalid.
    """
    return max(100, round(HELDOUT_MULTITASK_SAMPLES * epochs / batch_size))


def heldout_epoch_steps(epochs: int, ranges_path: str | None = None,
                        batch_size: int = HELDOUT_GATE_BATCH) -> int:
    """Steps for `epochs` passes over the samples THIS run will actually draw.

    Counts the idle-filtered windows in the run's own ranges index -- the same quantity the loader
    draws from, and the same convention `heldout_gate_steps` already uses per task. Falls back to
    HELDOUT_RANGES_ALL, which is what the heldout configs read.

    Raises rather than defaulting when the index cannot be read: a silent fallback budget is exactly
    how one arm of a comparison ends up trained differently from the rest, and the eval then reports
    a task as unlearnable when it was merely undertrained.
    """
    import json
    import os

    path = ranges_path or os.environ.get("HELDOUT_RANGES_ALL")
    if not path:
        raise ValueError(
            "heldout_epoch_steps needs a ranges index (arg or HELDOUT_RANGES_ALL); refusing to "
            "guess a budget, because guessing is how a task gets silently undertrained"
        )
    with open(path) as fh:
        ranges = json.load(fh)
    samples = sum(hi - lo for windows in ranges.values() for lo, hi in windows)
    if samples <= 0:
        raise ValueError(f"ranges index {path} declares no trainable windows")
    return max(100, round(samples * epochs / batch_size))



def heldout_gate_steps(task_id: int, *, epochs: int = HELDOUT_GATE_EPOCHS,
                       batch_size: int = HELDOUT_GATE_BATCH) -> int:
    """Steps for a fixed number of EPOCHS over this task's own TRAINABLE samples.

    Epochs are counted over idle-filtered windows, which is what the loader actually draws, and
    NOT over raw frames -- see HELDOUT_20_ADAPT_SAMPLES for why the difference is load-bearing.

    Raises on an unknown task rather than defaulting: a silent fallback budget would train one arm
    of a 20-arm gate differently from the rest, and the gate would then report a task as
    unlearnable when it was merely undertrained.
    """
    if task_id not in HELDOUT_20_ADAPT_SAMPLES:
        raise KeyError(
            f"task {task_id} is not in the held-out 20; no measured sample count exists for it, "
            f"so its epoch budget cannot be derived. Known: {sorted(HELDOUT_20_ADAPT_SAMPLES)}"
        )
    return max(100, round(HELDOUT_20_ADAPT_SAMPLES[task_id] * epochs / batch_size))



def _axis_heldout_multitask_config(*, num_train_steps: int, init_path: str | None = None,
                                   name: str = "pi05_axis_heldout_multitask",
                                   eef_action: bool = True,
                                   freeze_vision: bool = False,
                                   roots_index: str | None = None,
                                   ranges_path: str | None = None,
                                   quality_tag: int | None = None,
                                   norm_stats_from: str | None = None,
                                   awr_weights: str | None = None,
                                   awr_required: bool = False,
                                   schedule_path: str | None = None,
                                   schedule_required: bool = False,
                                   expected_mode: str | None = None,
                                   quality_path: str | None = None,
                                   quality_required: bool = False) -> TrainConfig:
    """One policy over all 20 held-out tasks -- the LIBERO-Plus-style protocol this suite replaces.

    Same recipe as `_axis_heldout_gate_config` (pi05_base init, LoRA, vision unfrozen, constant LR,
    own norm stats) but a single roots index spanning every task, so the model is fine-tuned once on
    all 400 demonstrations and then scored per task at eval. Reading the index from
    HELDOUT_ROOTS_ALL / HELDOUT_RANGES_ALL keeps the idle-frame filtering the gate inherits.
    """
    lr = _optimizer.CosineDecaySchedule(
        warmup_steps=100,
        peak_lr=2.5e-5,
        decay_steps=num_train_steps,
        decay_lr=2.5e-5,   # == peak_lr -> flat after warmup (openpi has no constant schedule)
    )
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            # 10, NOT openpi's default 16. The paper states action_horizon 10, and BOTH
            # stage-1 arms (pi05_axis_pretrain_{eef,d8}_paper_5k) and upstream pi05_libero use 10.
            # A finetune at 16 makes the pretrained action expert emit a chunk length it never
            # trained on, which costs exactly the transfer this benchmark measures -- and it biases
            # AGAINST the pretrained arm, so it would show up as "pretraining hurts" rather than as
            # a defect. 16 was inherited as a default here, never chosen.
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=AxisFrankaPretrainDataConfig(
            repo_id=_AXIS_PRETRAIN_REPO_ID,
            # A DECLARED index beats an env var, and not only for reproducibility:
            # `compute_norm_stats.py` takes ONLY --config-name and builds its loader from
            # this DataConfig, so whatever is named here is the population the statistics
            # are fitted on. Round 1 left this to HELDOUT_ROOTS_ALL, which a sibling driver
            # had set to the 20-task index -- so it trained on 10 tasks and normalised with
            # statistics from 20, including the 10 that are in stage-1 and the ~40% of
            # frames the idle filter drops. Same file train and serve, so nothing was
            # mismatched; it was simply the wrong population.
            roots_index=roots_index or os.environ.get("HELDOUT_ROOTS_ALL"),
            ranges_path=ranges_path or os.environ.get("HELDOUT_RANGES_ALL"),
            # The quality-tagged CFG finetune twins ONLY; None (a no-op) for every other arm.
            quality_tag=quality_tag,
            norm_stats_from=norm_stats_from,
            # Phase-reward stage-2 arms (qual_v2_{awrq,annealq,dropq,cfgq} + the _s3 twins) ONLY; the defaults
            # are inert for every other caller, so no existing config changes behavior.
            awr_weights=awr_weights,
            awr_required=awr_required,
            schedule_path=schedule_path,
            schedule_required=schedule_required,
            expected_mode=expected_mode,
            quality_path=quality_path,
            quality_required=quality_required,
            # stage-1 speaks 7-D relative EEF (state_eef 8-D / action_eef 7-D). Fine-tuning in any
            # other space forces the model to re-learn the action space and discards part of the
            # pretraining this benchmark exists to measure. The eval MUST run with
            # AXIS_EVAL_ACTION_SPACE=eef7; mismatched spaces score 0 on every task.
            eef_action=eef_action,
            # IMAGE GEOMETRY MUST MATCH STAGE-1, which passes center_crop=True. The default here
            # aspect-preserves and letterboxes: MEASURED 43.8% of every 224x224 tensor is black
            # padding, in the same place in every frame. A model pretrained on centre-cropped
            # frames and fine-tuned on letterboxed ones eats a constant full-width domain shift,
            # which costs precisely the pretraining this benchmark exists to measure -- and the
            # eval inherits the same transform, so both halves stay self-consistent and no guard
            # fires.
            image_center_crop=True,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            init_path or "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        batch_size=HELDOUT_GATE_BATCH,
        num_train_steps=num_train_steps,
        lr_schedule=lr,
        ema_decay=None,
        keep_period=None,
        num_workers=8,
        # freeze_vision=True is the ALREADY-DIAGNOSED fix for this exact symptom. The base
        # LoRA recipe freezes LLM non-LoRA weights but leaves the ~400M SigLIP tower fully
        # trainable, and `_slb_freeze_filter`'s own docstring records that this 'overfits/
        # corrupts the pretrained visual grounding -- diagnosed as the cause of uniform
        # ~0-20% success with all-timeout failures'. Round 1 of this benchmark reproduced
        # that signature precisely: floors at 0-8%, every non-success a timeout, on 49,801
        # frames from a SINGLE fixed viewpoint (camera pose ptp = 0 across all 200 episodes).
        freeze_filter=_slb_freeze_filter(freeze_vision=freeze_vision),
    )


def _axis_heldout_gate_config(task_id: int, *, num_train_steps: int) -> TrainConfig:
    """The LEARNABILITY GATE arm for one held-out benchmark task.

    WHAT THE GATE IS FOR. The benchmark's quality filters say the DEMONSTRATIONS succeed. They say
    nothing about whether a policy can learn the task from 20 of them, and the previous 10-task
    version floored: 4 of 6 evaluated tasks scored 0/10. A benchmark whose tasks are all
    unlearnable ranks nothing. So every candidate task is adapted here and evaluated, and kept only
    if `0.20 <= success <= 0.80` -- two-sided, because a task the reference already solves leaves
    no headroom to show improvement.

    WHY pi05_base AND NOT pi05_droid. The gate policy must share no pretraining with the arms under
    test, or the gate would select tasks that flatter them. `pi05_droid` was also rejected on
    evidence: it scored 0/50 on the current held-out set, so it would reject nearly every candidate
    for a reason that is about the policy rather than the task.

    THE RECIPE IS THE FROZEN SLB LoRA RECIPE, deliberately unchanged except for the init, so the
    gate measures the task rather than a new set of hyperparameters. Vision stays UNFROZEN: the
    measured result on this render domain is that freezing SigLIP makes grasp success worse in
    every condition (see `_axis_slb_config`), which is the opposite of the usual low-data recipe.

    NORM STATS ARE PER-TASK, computed on the 20-demo adaptation set itself. Reusing pretraining
    stats is the defect that produced two significant-but-wrong conclusions here before, both
    withdrawn; `AssetsConfig()` keeps compute and load resolving to the same per-config path.
    `compute_norm_stats` MUST run before training.

    THE BUDGET IS EPOCHS, NOT STEPS. The 20-demo sets differ 7x in length (3,222 frames for task
    811 against 23,484 for 1459), so a fixed step count would give tasks wildly unequal exposure
    and confound "unlearnable" with "undertrained". The caller passes a per-task step count derived
    from that task's own frame count.
    """
    lr = _optimizer.CosineDecaySchedule(
        warmup_steps=max(100, num_train_steps // 10),
        peak_lr=2.5e-5,
        decay_steps=num_train_steps,
        decay_lr=2.5e-6,
    )
    return TrainConfig(
        name=f"pi05_axis_heldout_gate_{task_id}",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            # 10, NOT openpi's default 16. The paper states action_horizon 10, and BOTH
            # stage-1 arms (pi05_axis_pretrain_{eef,d8}_paper_5k) and upstream pi05_libero use 10.
            # A finetune at 16 makes the pretrained action expert emit a chunk length it never
            # trained on, which costs exactly the transfer this benchmark measures -- and it biases
            # AGAINST the pretrained arm, so it would show up as "pretraining hurts" rather than as
            # a defect. 16 was inherited as a default here, never chosen.
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # THE ROOTS-INDEX PATH, not the single-dataset SLB one. The completed few-shot held-out
        # experiment expressed its 20-demo split as a one-task roots index plus a ranges file
        # (`axis.dataset.build_fewshot_splits`), and reusing that path is what makes this stage
        # inherit the IDLE-FRAME FILTERING and loader behaviour every prior run was measured
        # under. Pointing at a raw dataset root instead would silently train on the dwell frames
        # all of them excluded -- a recipe change invisible in a success rate afterwards.
        data=AxisFrankaPretrainDataConfig(
            repo_id=_AXIS_PRETRAIN_REPO_ID,
            roots_index=os.environ.get(f"HELDOUT_ROOTS_{task_id}"),
            ranges_path=os.environ.get(f"HELDOUT_RANGES_{task_id}"),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=num_train_steps,
        lr_schedule=lr,
        ema_decay=None,          # off for LoRA, per openpi's own LoRA examples
        keep_period=None,        # only the final checkpoint is ever evaluated
        num_workers=8,           # LeRobot v3.0 decodes video per item; 2 starves the loader
        freeze_filter=_slb_freeze_filter(freeze_vision=False),
    )


def _slb_available_task_ids() -> tuple[int, ...]:
    """SLB task_ids that have BOTH a manifest entry and converted LeRobot data on disk.

    The full SLB benchmark spans the tasks in slb_variant_homogeneous_800plus.json, but the
    per-task DROID-8D LeRobot dirs are uploaded/converted incrementally: only a task whose
    converted `*__droid8d` dir exists under AXIS_DATALOADER_ROOT can actually train. We
    register the five variants for exactly those tasks, so the config list grows automatically
    as conversions land -- without ever registering a dataless, un-trainable arm (which would
    only bloat get_config / difflib suggestions). Falls back to the always-present pilot pair
    (1644, 1645) if the manifest or data root cannot be read, so the core SLB configs never
    vanish on a machine that lacks the full offline cache.
    """
    fallback = (1644, 1645)
    # config.py -> openpi/src/openpi/training/; parents[4] is the outer dataset_preview repo.
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    manifest = pathlib.Path(
        os.environ.get(
            "SLB_TASKS_MANIFEST",
            repo_root / "datasets" / "manifests" / "slb_variant_homogeneous_800plus.json",
        )
    )
    try:
        candidates = [int(t["task_id"]) for t in json.loads(manifest.read_text())["tasks"]]
    except (OSError, ValueError, KeyError, TypeError):
        return fallback
    dl_root = (
        pathlib.Path(os.environ.get("AXIS_DATALOADER_ROOT", os.path.expanduser("~/axis_dataloader_cache")))
        / "hf_local"
        / "Franka-Datasets-v2"
    )
    available = tuple(
        tid for tid in candidates if any(dl_root.glob(f"Franka-{tid}-*/camera_fixed/*__droid8d"))
    )
    return available or fallback


def _axis_fullweight_speedtest(task_id: int = 1644, *, batch_size: int = 32, fsdp_devices: int = 4) -> TrainConfig:
    """FULL-WEIGHT pi0.5 fine-tune on real AXIS data -- built to MEASURE step throughput,
    not to produce a checkpoint.

    Differs from the LoRA SLB arms in exactly the two ways that matter for the 1M-trajectory
    full-weight plan:
      * gemma_2b / gemma_300m (NOT the _lora variants) and no freeze_filter -> all 3.35B
        parameters train, with full-size AdamW optimizer state (2x params).
      * fsdp_devices sharded across the 4 RTX Pro 6000 (Blackwell) so the optimizer state
        and params fit; batch_size is the GLOBAL batch (per-GPU = batch_size / fsdp_devices).

    It reuses the SLB data pipeline (one converted AXIS task) so data loading is realistic
    -- the point is s/step under a real dataloader, not synthetic data. num_train_steps is
    small on purpose; run ~200 steps and read the steady-state rate after compile+warmup.
    Env: set AXIS_DATALOADER_ROOT, SLB_MANIFEST_<task_id>, SLB_DATASET_ROOT_<task_id> as for
    the SLB arms.
    """
    return TrainConfig(
        name="pi05_axis_fullweight_speedtest",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16),  # full weight: no _lora
        data=AxisFrankaSlbDataConfig(
            repo_id="Devon018/Franka-Datasets-v2",
            variant="vanilla",
            task_id=task_id,
            sidecar_root=os.path.join(
                os.environ.get("AXIS_DATALOADER_ROOT", os.path.expanduser("~/axis_dataloader_cache")), "sidecars"
            ),
            manifest_path=os.environ.get(f"SLB_MANIFEST_{task_id}"),
            dataset_root=os.environ.get(f"SLB_DATASET_ROOT_{task_id}"),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets", asset_id="droid"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=200,       # a speed probe, not a training run
        batch_size=batch_size,     # GLOBAL; per-GPU = batch_size / fsdp_devices
        fsdp_devices=fsdp_devices,  # 4x RTX Pro 6000 Blackwell
        num_workers=8,
        save_interval=100_000,     # do not checkpoint during the probe
        ema_decay=None,
        # NO freeze_filter -> all params trainable (full weight).
    )


# Round 1's config name. The round-2 index-schedule arms bind their norm stats to ITS assets
# directory instead of resolving to their own (which does not exist) -- see `_axis_pretrain_config`.
_AXIS_ROUND1_NAME = "pi05_axis_pretrain_eef_paper"
_AXIS_PRETRAIN_REPO_ID = "Devon018/Franka-Datasets-v2"


def _axis_pretrain_config(
    *, num_train_steps: int = 100_000, batch_size: int = 32, knowledge_insulation: bool = False,
    eef: bool = False, paper: bool = False, name: str | None = None,
    schedule_required: bool = False, expected_mode: str | None = None,
    quality_required: bool = False, center_crop: bool = False,
    norm_stats_from_name: str | None = None, awr_required: bool = False,
    freeze_vision: bool = False,
) -> TrainConfig:
    """FULL-WEIGHT pi0.5 pretraining over the whole AXIS Franka corpus on the 8xA100 box.

    Multi-task: the loader concatenates the per-task __droid8d sub-datasets named in
    AXIS_PRETRAIN_ROOTS_INDEX and restricts rows to the non-idle ranges at
    AXIS_PRETRAIN_RANGES (both produced offline by axis_data.build_pretrain_datasets
    / pretrain_ranges). Paths come from env so the committed config carries no machine paths.

    Norm stats are OWN (run scripts/compute_norm_stats.py --config-name pi05_axis_pretrain
    before launch) -- NOT DROID's; see AxisFrankaPretrainDataConfig. Weight INIT is still the
    pi05_droid checkpoint. Full-weight (no _lora, no freeze_filter): all 3.35B params train,
    sharded over fsdp_devices=8; batch_size is GLOBAL (per-GPU = batch_size / 8).

    num_train_steps is a placeholder: recompute it as ceil(total_selected_windows / batch_size)
    * epochs against the FINAL downloaded corpus (see reports/fullweight_training_speed_estimate.md
    and reports/pretrain_dataloader_design.md) before the real run.

    `knowledge_insulation` emits a separately-named `_ki` twin rather than changing this config,
    so a run already launched against `pi05_axis_pretrain` keeps exactly the config it started
    with. Pretraining is where KI should matter most: this is a FULL-WEIGHT run over a large
    corpus, i.e. precisely the regime where the flow-matching gradient has enough steps to
    erode the VLM's pretrained semantics (arXiv:2505.23705).

    `paper` emits a `_paper` twin, for the same reason and by the same mechanism as `_ki`:
    the in-flight `pi05_axis_pretrain_eef` runs must keep the hyper-parameters they were
    launched with. The twin is stage 1 of the Axis-V1 recipe exactly as Appendix F Table 10
    states it -- action_horizon 10 (not 16), 10k warmup, 5e-5 CONSTANT after warmup, EMA
    0.999, 100k steps, GLOBAL batch 8 (1/GPU at fsdp_devices=8).

    DO NOT "FIX" THE WARMUP. 10,000 is a fixed absolute number in both tables, not a fraction
    of the budget -- CONFIRMED WITH THE AXIS-V1 AUTHOR (2026-08-11) for stage 1 (10k of 100k)
    and stage 2 (10k of 30k, which looks like a table typo and is not). The superficially
    sensible "scale the warmup to num_train_steps" edit takes this off-recipe. Note this is
    why `warmup_steps` is a literal here while the non-paper arm derives it from the budget.

    Everything Table 10 shares with the base arm is already true here and is NOT re-stated:
    init from pi05_base (the `eef` arm), full-model (no LoRA, no freeze_filter),
    OWN norm stats (no assets override -- except the round-2 schedule arms, which reuse round
    1's; see the `name` paragraphs below),
    action_dim padded to 32, and AdamW(b1=0.9, b2=0.95, eps=1e-8, wd=1e-10, clip=1.0), which
    is openpi's `_optimizer.AdamW` default field-for-field -- so the paper's optimizer block
    is reached by NOT passing an optimizer, and any future edit to those defaults is a
    deviation from the paper.

    The "constant LR" is expressed as CosineDecaySchedule(peak_lr == decay_lr, decay_steps
    well past the budget), which is how upstream openpi already spells a constant LR --
    `pi05_full_droid_finetune` (warmup 1k, 5e-5, decay_steps 1e6, decay_lr 5e-5, 100k steps)
    and `pi05_libero` both do exactly this. optax's `warmup_cosine_decay_schedule` computes
    `end + (peak - end) * cosine(...)`, so peak == end makes every post-warmup step exactly
    that value (measured: one unique post-warmup value, float32(5e-5)). No new schedule class
    is needed and the warmup ramp stays the one every other openpi config uses.

    `name` overrides the flag-composed name for the round-2 arms (`pi05_axis_drop`,
    `pi05_axis_anneal`, `pi05_axis_cfg`), which are not distinguished by any flag this factory
    takes -- their arm lives in `data.schedule_path` / `data.quality_path`, supplied per run.
    (`pi05_axis_cfg` is the coverage-neutral one: it carries no schedule, so it falls through to
    the control's own `RowSampler` and draws exactly the control's rows.)
    Passing it also switches the AWR weights off
    at the source: those arms inherit the launch environment, and reading
    $AXIS_PRETRAIN_AWR_WEIGHTS there would silently add round 1's loss reweighting on top of the
    schedule (`AxisFrankaPretrainDataConfig.create` refuses that combination anyway, so the
    alternative is a confusing hard failure on a box where the variable happens to be exported).

    Passing `name` ALSO pins norm stats to ROUND 1's assets directory (`<assets_base_dir>/
    pi05_axis_pretrain_eef_paper/Devon018/Franka-Datasets-v2`) instead of the arm's own -- by
    round 1's config NAME (`norm_stats_from`), resolved against whatever `assets_base_dir` the run
    uses, so an `--assets-base-dir` override moves the arms and round 1 together instead of
    stranding the arms on the old stats. Two reasons, both load-bearing:

    * Without it the arms do not run at all. `TrainConfig.assets_dirs` is
      `assets_base_dir / name`, so `pi05_axis_drop` would resolve to `./assets/pi05_axis_drop`,
      which does not exist; `_load_norm_stats` swallows the miss and `transform_dataset` then
      dies telling you to run `scripts/compute_norm_stats.py --config-name=<your-config>` -- an
      instruction that CANNOT be followed, because compute_norm_stats calls
      `config.data.create(config.assets_dirs, config.model)` with no way to pass
      `--data.schedule_path` and so trips `schedule_required` before computing anything.
    * Recomputing per-arm stats would be a SECOND parity break against round 1. These arms must
      differ from the round-1 control in exactly ONE thing -- the supervision the schedule
      encodes -- and this project has already retracted two conclusions that came from norm
      stats silently belonging to a different dataset than the one being trained on. So the
      sharing is explicit here and asserted in config_schedule_arms_test.py, not incidental.

    The paper's action space is NOT changed here: Appendix F's "7D joint-position action" is
    a typo for the LIBERO/robosuite OSC_POSE 7-D delta `[dpos(3), daxis-angle(3), gripper]`,
    which is what `eef=True` already feeds (axis_franka_policy / eef_math). Quantile
    normalisation is likewise already the pi0.5 default (`use_quantile_norm = model_type !=
    PI0`) and matches Appendix F's formula, so the only normalisation requirement -- own
    q01/q99 -- is met by this factory's deliberate lack of an `assets=` override.
    """
    return TrainConfig(
        name=name
        or (
            "pi05_axis_pretrain"
            + ("_eef" if eef else "")
            + ("_ki" if knowledge_insulation else "")
            + ("_paper" if paper else "")
        ),
        # Full weight: default (non-lora) gemma variants, no freeze_filter.
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=10 if paper else 16,
            **(_KI_MODEL_KWARGS if knowledge_insulation else {}),
        ),
        data=AxisFrankaPretrainDataConfig(
            repo_id=_AXIS_PRETRAIN_REPO_ID,
            # Round-2 schedule arms (name given) SHARE round 1's norm stats; every other arm
            # keeps the default (its own `assets_base_dir / name`). See the docstring's `name`
            # paragraphs: without this the arms cannot resolve norm stats at all, and
            # recomputing them per arm would break parity with the round-1 control. Expressed as
            # round 1's NAME, resolved against whatever `assets_base_dir` this run uses -- a
            # literal `./assets/<round1>` would silently stop following round 1 the moment
            # `--assets-base-dir` was passed. See `AxisFrankaPretrainDataConfig.norm_stats_from`.
            # ... and ONLY when this arm is in round 1's action space. Round 1's stats were
            # computed over state_eef/action_eef -- EEF positions, axis-angles, OSC deltas. A
            # droid8 arm's columns are joint ANGLES (+-2.4 rad) and joint VELOCITIES (p95 0.39
            # rad/s); normalising the latter by the former is silent, converges, and scales the
            # policy wrong. Every named config today passes eef=True, so this changes nothing
            # that exists -- it only stops a droid8 arm from inheriting statistics for columns
            # it does not have.
            # An explicit source wins: the d8 mechanism arms share the d8 BASELINE's stats
            # for the same reason the eef schedule arms share round 1's -- the comparison
            # requires identical normalisation, and per-arm stats would be a second
            # uncontrolled difference.
            norm_stats_from=(norm_stats_from_name if norm_stats_from_name
                             else (_AXIS_ROUND1_NAME if (name and eef) else None)),
            roots_index=os.environ.get("AXIS_PRETRAIN_ROOTS_INDEX"),
            ranges_path=os.environ.get("AXIS_PRETRAIN_RANGES"),
            # 640x360 corpora only; a no-op on square frames. Declared per config so a corpus
            # swap cannot change what an existing config means -- see _center_crop_square.
            image_center_crop=center_crop,
            # Schedule arms (name given) never take AWR weights from the environment; see the
            # docstring's `name` paragraph.
            awr_weights=None if name else os.environ.get("AXIS_PRETRAIN_AWR_WEIGHTS"),
            base_config=DataConfig(prompt_from_task=True),
            # relative-EEF (LIBERO-Plus proxy) feeds state_eef/action_eef; else DROID-8D joint-vel.
            eef_action=eef,
            # Named schedule arms only: require the artifact flag at launch, and bind this
            # config's name to the artifact's own meta["mode"] (see `AxisFrankaPretrainDataConfig`
            # and `DataConfig.pretrain_expected_mode`).
            schedule_required=schedule_required,
            expected_mode=expected_mode,
            # Named quality arm (`pi05_axis_cfg`) only: require the artifact flag at launch. No
            # `expected_mode` twin here -- the artifact's own filename<->reward_id binding
            # (quality_conditioning.QualityTags.check_reward_id) is what separates `cfg_v2` from
            # `cfg_phase`, which share this one config name.
            quality_required=quality_required,
            awr_required=awr_required,
            # NB: for every arm but the round-2 schedule ones the `assets` above is the empty
            # default -> own norm stats (compute_norm_stats), NOT pi05_droid's. See the class
            # docstring / reports/pretrain_dataloader_design.md.
        ),
        # EEF variant inits from pi05_base (its EEF control-mode head); the joint variant from
        # pi05_droid (DROID-8D joint-velocity). Only weights are reused; norm stats are ours.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"gs://openpi-assets/checkpoints/{'pi05_base' if eef else 'pi05_droid'}/params"
        ),
        num_train_steps=num_train_steps,
        lr_schedule=(
            # Appendix F Table 10: 10k warmup, then 5e-5 held CONSTANT. Spelled the way
            # upstream openpi spells a constant LR -- peak_lr == decay_lr with decay_steps
            # far beyond the budget (`pi05_full_droid_finetune`: warmup 1k, 5e-5, 1e6,
            # 5e-5 over 100k steps; `pi05_libero` likewise). See the docstring.
            _optimizer.CosineDecaySchedule(
                warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
            )
            if paper
            else _optimizer.CosineDecaySchedule(
                warmup_steps=max(1000, num_train_steps // 20),
                peak_lr=2.5e-5,
                decay_steps=num_train_steps,
                decay_lr=2.5e-6,
            )
        ),
        batch_size=batch_size,   # GLOBAL; per-GPU = batch_size / fsdp_devices
        fsdp_devices=8,          # 8x A100-80GB single node
        num_workers=8,
        # Appendix G initialises stage 2 from the EMA-SMOOTHED params, and openpi's
        # `checkpoints._split_params` writes ema_params (when set) as the `params` item -- so
        # EMA here is what makes <ckpt>/<step>/params the artifact stage 2 must consume.
        ema_decay=0.999 if paper else None,
        # Default: NO freeze_filter -> all params trainable (full weight). The `_vfz`
        # twins pass freeze_vision=True to freeze ONLY the SigLIP tower on top of the
        # otherwise-identical full-weight recipe; see _pretrain_freeze_filter.
        freeze_filter=_pretrain_freeze_filter(freeze_vision),
    )


def _axis_eef_libero_serve_config() -> TrainConfig:
    """INFERENCE-ONLY config to serve the `pi05_axis_pretrain_eef` checkpoint against the
    LIBERO-Plus client (benchmarks_eval/run_libero_plus_eval.py), which speaks the pi05_libero
    contract (observation/image, observation/wrist_image, observation/state 8-D, prompt).

    Differs from the train config `pi05_axis_pretrain_eef` (which reads parquet columns
    state_eef/action_eef): the LIBERO client's obs keys are repacked *inside* data_transforms
    (serve_policy.py passes no repack_transforms, so a DataConfig.repack_transforms would never
    run at inference), the LIBERO 8-D state is remapped into our `state_eef` convention
    (LiberoStateToAxisEEF: canonical axis-angle + gripper_qpos->closedness), and the 7-D output
    remaps the gripper from closedness [0,1] to LIBERO [-1,1] (AxisFrankaEEFLiberoOutputs).

    Model + asset_id match the train config exactly (Pi0Config action_dim=32/horizon=16;
    repo_id "Devon018/Franka-Datasets-v2"), so create_trained_policy loads OUR checkpoint's
    OWN norm stats from <ckpt>/assets/Devon018/Franka-Datasets-v2/norm_stats.json.
    Serve: scripts/serve_policy.py --env LIBERO --policy.config pi05_axis_eef_libero_serve
           --policy.dir <ckpt>."""
    serve_inputs = lambda model: _transforms.Group(  # noqa: E731 (GroupFactory is a callable)
        inputs=[
            _transforms.RepackTransform(
                {
                    "base_0_rgb": "observation/image",
                    "left_wrist_0_rgb": "observation/wrist_image",
                    "state": "observation/state",
                    "prompt": "prompt",
                }
            ),
            axis_franka_policy.LiberoStateToAxisEEF(),
            # NO center_crop here: this is a FREE function, so `self` does not exist (it
            # raised NameError at serve time), and the LIBERO client already sends square
            # pad-224 frames, so a crop would be a no-op regardless.
            axis_franka_policy.AxisFrankaInputs(),
        ],
        outputs=[axis_franka_policy.AxisFrankaEEFLiberoOutputs()],
    )
    return TrainConfig(
        name="pi05_axis_eef_libero_serve",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16),
        data=SimpleDataConfig(
            repo_id="Devon018/Franka-Datasets-v2",  # -> asset_id, loads our ckpt norm stats
            data_transforms=serve_inputs,
            base_config=DataConfig(prompt_from_task=True),
        ),
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    _axis_fullweight_speedtest(),
    _axis_pretrain_config(),
    # Knowledge-insulation twin: identical data/optimizer/budget, but the flow-matching
    # gradient is cut at the prefix KV and the VLM is trained by a FAST-token CE instead.
    _axis_pretrain_config(knowledge_insulation=True),
    # Relative-EEF variant (LIBERO-Plus proxy benchmark): state_eef/action_eef columns +
    # robosuite OSC_POSE 7-D delta action, init from pi05_base. Own norm stats.
    _axis_pretrain_config(eef=True),
    # Axis-V1 paper recipe, stage 1 (arXiv 2607.21588v1 Appendix F Table 10). Same data and
    # action space as the arm above; differs ONLY in the six optimisation fields the table
    # pins. batch_size is stated here, not defaulted: Table 10's batch is 8 GLOBAL, which at
    # fsdp_devices=8 is one sample per A100.
    #
    # SUPERVISION ARMS ARE OFF, AND THIS IS THE BC BASELINE. The experiment varies ONLY
    # supervision -- vanilla / +CFG / +AWR -- so the first run of both `_paper` configs must be
    # plain BC. Both are vanilla today because `slb_sidecar_root` is None, and that single field
    # gates the whole chain: it gates `slb_variant_sampler.build_sampler`, the only producer of
    # `weights_by_row`; which is the only thing that wraps the dataset in `WeightedRowDataset`;
    # which is the only writer of `slb_awr_loss.LOSS_WEIGHT_KEY`. With that key absent from the
    # batch, `DataLoaderImpl.__iter__` yields the original 2-tuple, `train_step` sees
    # `loss_weights=None`, and `slb_awr_loss.combine` returns literally `jnp.mean(chunked_loss)`
    # -- bit-for-bit the pre-SLB objective. Enforced by config_paper_test.py.
    #
    # WHERE EACH ARM WOULD ENTER, when the owner enables it (do NOT enable it here):
    #   +CFG  -- a prompt-tag transform PREPENDED to `repack_transforms.inputs`, ahead of the
    #            RepackTransform, in the relevant factory's `create()`
    #            (AxisFrankaPretrainDataConfig for stage 1, LeRobotLiberoDataConfig for stage 2).
    #            Order is load-bearing: RepackTransform drops `episode_index`/`frame_index`,
    #            which is what `slb_cfg.build_conditioning` keys the advantage label on. See
    #            AxisFrankaSlbDataConfig.create for the working precedent. Zero model change:
    #            the tag rides the prompt, so checkpoints stay interchangeable across arms.
    #   +AWR  -- a per-row weight reaching the batch under `slb_awr_loss.LOSS_WEIGHT_KEY`, i.e.
    #            a `WeightedRowDataset`-equivalent wrapper applied where the pretrain branch of
    #            `data_loader.create_torch_data_loader` builds its `RowSampler`. The loss half
    #            already exists and needs no edit: `combine()` applies WVM Eq E.5 the moment a
    #            weight is present. `DataConfig.awr_tau` (10.0) / `awr_delta` (2.0) carry the
    #            Eq E.7 constants.
    # STATUS: the AWR path now EXISTS and is what the onelayer_v3 arms train on. The pretrain
    # branch of `create_torch_data_loader` wraps its RowSampler in `StrictWeightedRowDataset`
    # whenever `DataConfig.pretrain_awr_weights` is set (fed from $AXIS_PRETRAIN_AWR_WEIGHTS).
    # The CFG path is still unbuilt: both factories still build their repack from a bare
    # RepackTransform, so no conditioning tag reaches the prompt.
    _axis_pretrain_config(eef=True, paper=True, batch_size=8, num_train_steps=100_000),
    # ---- the 5k `camera_fixed` corpus -------------------------------------------------------
    # Same recipe as the round-1 paper arm, one field different: the frames are 640x360 (16:9)
    # instead of square, so both cameras are centre-cropped to 360x360 before the 224 resize.
    # Without it a straight resize squashes every frame; with it the framing matches LIBERO's
    # square render. A SEPARATE config rather than a flag on the existing one, because the
    # existing one reads its corpus from the environment and the old corpora are NOT 16:9 --
    # v3 is 126x224, randcam 224x224 -- so cropping there would change what those runs meant
    # without anything recording it.
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_pretrain_eef_paper_5k"),
    #
    # THE SAME 5k CORPUS IN THE DROID-8D ACTION SPACE.
    # `eef=False` selects observation.state/action, which must be the CONVERTED corpus
    # (corpus_d8, roots_libero_5k_v2_d8.json): 8-D state [7 joint angles, gripper closedness]
    # and 8-D action [7 joint velocities rad/s, gripper closedness] at 15 fps. Pointing it at
    # the raw corpus is refused by the width guard in AxisFrankaInputs -- the raw columns are
    # 9-D (7 joints + 2 finger widths) and would train on absolute positions.
    #
    # WHY THIS ARM EXISTS. The AXIS-sim rollout harness applies 8-D joint velocity
    # (`q_cmd += a[:7]*dt`) and has no OSC path, so an eef7 checkpoint cannot be driven through
    # it at all. Measured by the held-out work: eef7 scored 1/50 on task 811 where droid8 scored
    # 49/50 through the SAME harness on the SAME demonstrations. The held-out stage-2 finetune
    # also runs discrete_state_input=True, so a droid8 stage-1 arm matches its successor in BOTH
    # action space and state convention -- unlike the LIBERO path, where stage 2 drops state
    # entirely and the pretrained state conditioning has nothing to attach to.
    #
    # Budget is unchanged at 20,605 steps: the conversion preserves row count exactly (asserted
    # per task at build time), so this is still 1.00 epoch over the same 1,319,784 rows, and the
    # row-indexed AWR / schedule / quality artifacts remain valid against it.
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_pretrain_d8_paper_5k"),
    # ---- DROID-8D twins of the mechanism configs. Identical to their eef siblings except
    # eef=False (droid8 columns) and norm stats pinned to the d8 baseline. The row-indexed
    # schedule/quality artifacts carry over unchanged: corpus_d8 preserves row count and order
    # exactly, so the same .npz files drive both action spaces.
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_drop_top_d8",
                          schedule_required=True, expected_mode="drop_top",
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k"),
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_anneal_d8",
                          schedule_required=True, expected_mode="anneal",
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k"),
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_cfg_d8",
                          quality_required=True,
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k"),
    # The AWR arm gets its own REQUIRED-FLAG config; weights ride --data.awr_weights, never the
    # env var a named config silently nulls.
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_awr_d8",
                          awr_required=True,
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k"),
    # ---- FROZEN-VISION (_vfz) twins. Identical to their unfrozen counterparts in every field
    # except freeze_vision=True (SigLIP tower frozen; ~400M params, logged and asserted nonzero
    # at startup by scripts/train.py). The bc twin pins norm stats to the unfrozen baseline's
    # assets dir EXPLICITLY: the unfrozen bc computed those stats, and recomputing them under
    # the vfz name would be a second uncontrolled difference between the arms being compared.
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_pretrain_d8_paper_5k_vfz",
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k",
                          freeze_vision=True),
    _axis_pretrain_config(eef=False, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_awr_d8_vfz",
                          awr_required=True,
                          norm_stats_from_name="pi05_axis_pretrain_d8_paper_5k",
                          freeze_vision=True),
    #
    # ROUND-2 INDEX-SCHEDULE ARMS. Both carry round 1's model, optimiser and budget (20,605 steps
    # = one epoch of the rung-5000 corpus, at GLOBAL batch 64) and differ from the `bc` control,
    # and from each other, ONLY in the schedule artifact handed to `data.schedule_path` at launch:
    #   drop    -- keep only the Delta >= 0 rows, drawn uniformly (which rows).
    #   anneal  -- keep every row, but ramp the draw toward the high-quality bins (in what order).
    # The path is a CONFIG FIELD, not an env var, so a checkpoint's arm is recoverable from the
    # config it was launched with. Two names rather than one because the checkpoint directory and
    # the run record are keyed by config name, and "which arm is this?" must not require reading a
    # command line back out of a log. The schedule is the whole difference, so both names pass
    # `schedule_required=True` here: `AxisFrankaPretrainDataConfig.create` now RAISES if
    # schedule_path is still unset at launch, and `expected_mode` ("drop"/"anneal") is asserted
    # against the artifact's own `meta["mode"]` in data_loader.py -- a launch that forgets the
    # flag, or hands the wrong artifact to the wrong name, fails loudly instead of silently
    # training the plain baseline (or the other arm) under this name.
    #
    # WARMUP IS NOT ROUND 1's HERE. `paper=True` keeps the literal 10,000-step warmup, and round 1
    # overrode it to 2,060 (10% of this budget, the fraction the paper uses) from its arm TOML.
    # Round 2's TOML must carry the same `lr_schedule.*` overrides or the arms are off parity with
    # the control they are compared against; a config-level default cannot enforce that, because
    # CLI overrides bypass it silently. See conf/experiments/onelayer_v3_stage1_arms.toml.
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_drop", schedule_required=True, expected_mode="drop"),
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_anneal", schedule_required=True, expected_mode="anneal"),
    # The SELECTIVE Filtered-BC cut. `pi05_axis_drop` above is WVM Eq E.6 at kappa = 0.0 (keep
    # Delta >= 0); this is the percentile variant of the same equation. It needs its own name and
    # its own `expected_mode` precisely because the two are the same MECHANISM at different
    # strengths -- sharing a mode string would let either artifact load under either name, which is
    # the confusion `expected_mode` exists to prevent.
    #
    # WHY THIS ARM EXISTS AT ALL: on this corpus 75.7% of rows are already advantaged, so
    # `pi05_axis_drop` shifts only ~1.32x of the gradient budget onto advantaged rows even as a
    # hard filter, and WVM's own top-70% would be nearly indistinguishable from it. The artifact
    # this name expects is built at `--keep-top-frac 0.50` (0.30 until 2026-08-18; see
    # index_schedule.DEFAULT_KEEP_TOP_FRAC for why 0.50, and why nothing below ~0.136 is
    # constructible from the weights). NOTHING VALIDATES THE FRACTION AT LOAD TIME, so an artifact
    # built at one value runs silently under a name documented at another -- state it in the arm's
    # spec rather than trusting this comment.
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_drop_top", schedule_required=True,
                          expected_mode="drop_top"),
    # THE CONTROL `pi05_axis_drop_top` IS UNINTERPRETABLE WITHOUT. It trains on 30% of the rows, so
    # measured against the full-data baseline it changes two things at once -- which rows, and how
    # many. This arm keeps the SAME NUMBER of rows drawn uniformly at random, so the only remaining
    # difference is the selection rule, i.e. whether the reward's ranking carries signal at all. If
    # drop_top does not beat this, no weighting function built on that reward will help either, at
    # any tau or delta -- which is the cheapest way to learn that.
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_drop_random", schedule_required=True,
                          expected_mode="drop_random"),
    #
    # ROUND-2 MECHANISM 3: pi0.7 quality conditioning. Same recipe and budget as the two arms
    # above (asserted against `pi05_axis_drop` itself in config_cfg_arm_test.py, so no
    # hyper-parameter can drift in unnoticed), differing from them only in the mechanism: the tag
    # rides the PROMPT instead of replacing the row draw.
    #
    # THE ONLY COVERAGE-NEUTRAL ROUND-2 ARM, and that is its distinguishing claim. It carries no
    # schedule and no AWR weights, so the pretrain branch of `create_torch_data_loader` falls
    # through to `RowSampler(rows, seed)` -- the round-1 control's own sampler, at the same seed
    # -- giving a byte-identical row sequence and 100% coverage, where its siblings each carry a
    # disclosed coverage asymmetry. `create()` REFUSES `quality_path` together with either of
    # those two fields rather than leaving the claim to a convention.
    #
    # Both CFG rewards (`cfg_v2`, `cfg_phase`) run under this ONE name; which one a run used is
    # in `data.quality_path`, bound to the artifact's own `meta["reward_id"]` by
    # `QualityTags.check_reward_id`. `quality_required=True` makes a launch that forgets the flag
    # fail loudly instead of training the plain BC control under this name.
    #
    # WARMUP IS NOT ROUND 1's HERE either -- see the schedule arms above; the arm TOML must carry
    # the same `lr_schedule.*` overrides.
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          center_crop=True, name="pi05_axis_cfg", quality_required=True),
    # INFERENCE-ONLY: serve the pi05_axis_pretrain_eef checkpoint to the LIBERO-Plus client.
    _axis_eef_libero_serve_config(),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # LIBERO fine-tune INITIALISED FROM our AXIS-pretrained EEF pilot (50k) instead of pi05_base.
    # Headline of the transfer experiment: does AXIS-Franka sim-pretraining improve LIBERO-Plus
    # robustness vs vanilla pi05_base init? Identical to pi05_libero except (a) weight init = our
    # 50k params (copied to UCR at /bigdata/jlilab/myan035/eef_ckpt_50000) and (b) box-safe
    # batch_size=64 (pi05_libero's 256 OOMs a single 8xA100-80GB node for full-weight pi0.5).
    # H=10 (LIBERO) despite our pilot's 16: the flow head params are per-action-dim, H is a
    # sequence length, so a 16->10 init transfers. LIBERO norm stats used (LeRobotLiberoDataConfig).
    TrainConfig(
        name="pi05_libero_axisinit",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            # 2026-08-17: now PI's OFFICIAL build (1,693 eps, 256x256, stored UPRIGHT), so no
            # rotation is needed and Rotate180Images drops out. Still a statement about the bytes
            # on disk rather than a recipe change -- what changed is which bytes. The box-local
            # 128px re-conversion this used to read is retired: the model input is 224x224, so
            # every one of its frames was UPSCALED 1.75x, and no transform recovers detail that
            # upscaling never added. The guard checks the declaration against total_episodes and
            # image shape, so a stale "inverted" here would refuse at loader construction.
            dataset_image_orientation="upright",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        # Env-driven so it works on the box (init = the pilot's own 50k output) or on UCR (a
        # copied checkpoint). On the box: AXIS_EEF_INIT_CKPT=/data/checkpoints/pi05_axis_pretrain_eef/axis_pretrain_eef/50000/params
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get("AXIS_EEF_INIT_CKPT", "/bigdata/jlilab/myan035/eef_ckpt_50000/params")
        ),
        num_train_steps=30_000,
        fsdp_devices=8,
    ),
    # Axis-V1 paper recipe, stage 2 (arXiv 2607.21588v1 Appendix G Table 11). A separately
    # named `_paper` twin of pi05_libero_axisinit, for the same reason as the `_ki` twins:
    # a run already launched against pi05_libero_axisinit keeps the schedule it started with.
    #
    # Differs from pi05_libero_axisinit in exactly two fields -- warmup 1,000 -> 10,000 and
    # decay_lr 5e-6 -> 5e-5, i.e. 5e-5 held CONSTANT after warmup (peak == decay collapses
    # optax's cosine leg to a constant).
    #
    # DO NOT "FIX" THE WARMUP. 10,000 of 30,000 steps is a third of training spent ramping,
    # which reads like a table typo -- it is not. CONFIRMED WITH THE AXIS-V1 AUTHOR
    # (2026-08-11): warmup is 10,000 for BOTH stages, deliberately. Scaling it to the step
    # budget (the "obvious" correction) would silently make this not the paper's recipe.
    #
    # The durable justification is upstream, not just the paper: openpi's OWN shipped
    # `pi05_libero` (registered just above in this list) is warmup 10_000 / peak 5e-5 / decay_steps
    # 1_000_000 / decay_lr 5e-5 / num_train_steps 30_000 / ema_decay 0.999 -- i.e. Table 11
    # IS openpi's LIBERO recipe on every axis except batch size. So this twin is the upstream
    # schedule verbatim; only batch differs (paper 64 vs upstream 256, chosen so 8xA100 holds
    # ~8 samples/GPU). If the warmup still looks wrong, compare against `pi05_libero` first.
    #
    # What pi05_libero_axisinit got wrong: decay_steps=30_000 with decay_lr=5e-6 is a REAL
    # cosine decay to a tenth of peak, disagreeing with BOTH the paper and upstream.
    #
    # Everything else Table 11 pins already matched and is unchanged: 30,000 steps, GLOBAL
    # batch 64, EMA 0.999, action_horizon 10, extra_delta_transform=False, LIBERO-specific
    # norm stats (LeRobotLiberoDataConfig -> physical-intelligence/libero assets), and
    # AdamW's openpi defaults (b1 0.9 / b2 0.95 / eps 1e-8 / wd 1e-10 / clip 1.0).
    #
    # WEIGHT INIT: Appendix G transfers MODEL PARAMS ONLY. `CheckpointWeightLoader` is exactly
    # that -- it reads `<stage1_ckpt>/<step>/params` and nothing else, so the optimizer state,
    # step counter and EMA buffer all start fresh (a `--resume` would not; do not use one).
    # And because stage 1 trains with ema_decay=0.999, `checkpoints._split_params` writes the
    # EMA params into that `params` item, so this loader consumes the EMA-smoothed weights
    # Appendix G asks for without any extra step.
    #
    # A DISTINCT env var from pi05_libero_axisinit's AXIS_EEF_INIT_CKPT, with NO usable
    # default: the 50k pilot checkpoint that variable points at does NOT qualify for this
    # recipe (action_horizon 16, trained without EMA), and silently initialising from it
    # would produce a plausible-looking run that is not the paper's. Unset, the path below
    # does not exist and the weight loader fails loudly at startup.
    TrainConfig(
        name="pi05_libero_axisinit_paper",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            # NOT a recipe change and NOT a deviation from Table 11 -- a statement about the bytes
            # on disk, and as of 2026-08-17 those bytes are PI's OFFICIAL build: 1,693 episodes,
            # 256x256, stored UPRIGHT, which is already the convention stage 1 and the eval client
            # use, so nothing rotates. It replaces the 128px re-conversion because the model input
            # is 224x224 and that build was therefore upscaled 1.75x on every frame.
            # CONSEQUENCE, stated rather than hidden: the training set goes 2,000 -> 1,693
            # episodes, so stage-2 numbers on this build are NOT comparable with the round-1
            # result (base_ctrl 86.1 > awr_v2 80.4 > bc 77.5), which used 128px/2,000 -- including
            # its control. Re-run the control here or report the two rounds separately.
            dataset_image_orientation="upright",
        ),
        batch_size=64,
        # Byte-identical to upstream `pi05_libero`'s schedule; see the note above.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "AXIS_EEF_PAPER_INIT_CKPT",
                "/unset/set-AXIS_EEF_PAPER_INIT_CKPT-to-a-pi05_axis_pretrain_eef_paper-step/params",
            )
        ),
        num_train_steps=30_000,
        fsdp_devices=8,
    ),
    # STAGE 2 ON ROUND 1's BUILD. A verbatim copy of `pi05_libero_axisinit_paper` differing in
    # exactly two fields: `name`, and the orientation this dataset stores.
    #
    # WHY IT EXISTS. The `_paper` config above reads PI's official build (1,693 eps, 256px,
    # UPRIGHT). Round 1's numbers -- base_ctrl 86.1 > awr_v2 80.4 > bc 77.5 -- were measured on a
    # DIFFERENT build: 2,000 episodes at 128px, stored INVERTED. Swapping build cost 24.9 points
    # on the control alone (61.2 vs 86.1), which is larger than every supervision effect we are
    # trying to measure, so round 3 returns to round 1's build to recover the headroom. Decision
    # taken 2026-08-19; the two builds' stage-2 numbers are NOT comparable with each other.
    #
    # THE POLICY STILL SEES UPRIGHT, which is the only thing that matters and the thing that was
    # got wrong once. `dataset_image_orientation` names what the BYTES ON DISK are, and the
    # rotation is DERIVED from it: `rotation_needed("inverted")` is True, so `Rotate180Images`
    # enters `repack_transforms` -- the one group inference never runs -- and the model is fed
    # upright frames, matching stage 1 and matching the eval client, which turns MuJoCo's
    # bottom-up render right-side-up itself. Declaring "upright" here instead would train this
    # 180 degrees to eval, converge beautifully, pass every guard, and score ~0.
    # `check_dataset_build` refuses at loader construction if the declaration disagrees with the
    # resolved dataset, keyed on (total_episodes, image_hw) rather than the repo id.
    #
    # THE COST, stated rather than hidden: the model input is 224x224, so every 128px frame is
    # UPSCALED 1.75x. That is exactly what the `_paper` config moved away from. We are taking it
    # back deliberately -- the leading explanation for round 1's higher scores is that this
    # upscale acted as accidental low-pass augmentation (measured 2.6-2.9x blurrier by Laplacian
    # variance), and round 3 is the run that tests whether the headroom comes back with it.
    TrainConfig(
        name="pi05_libero_axisinit_paper_r1data",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            dataset_image_orientation="inverted",
            # SHARE the _paper config's stats directory. `assets_dirs` is
            # `assets_base_dir / name`, so this config's NEW name would otherwise send it
            # looking in `assets/pi05_libero_axisinit_paper_r1data/...`, which nothing
            # provisions -- `conf/provision.toml` mirrors these to the `_paper` directory.
            # A NAME rather than a path so it follows an --assets-base-dir override.
            norm_stats_from="pi05_libero_axisinit_paper",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "AXIS_EEF_PAPER_INIT_CKPT",
                "/unset/set-AXIS_EEF_PAPER_INIT_CKPT-to-a-pi05_axis_pretrain_eef_paper-step/params",
            )
        ),
        num_train_steps=30_000,
        fsdp_devices=8,
    ),
    # STAGE 2 FOR THE CFG ARMS ONLY, and the ONE place round 2 breaks the "stage 2 is identical
    # across arms" rule that `conf/experiments/onelayer_v3_stage2_libero.toml` states.
    #
    # It is the mechanism, not an oversight. Stage 1's CFG checkpoint was trained with a
    # `Quality: q` tag in ~81% of its prompts; finetuning it on bare LIBERO prompts and then
    # evaluating with a tag (guidance needs one) would ask it about a prompt distribution stage 2
    # taught it to forget. So stage 2 keeps conditioning on -- at the CONSTANT top bin, because
    # LIBERO is uniformly expert data and that is what inference asks for (D9). The guidance
    # scale β is swept at EVAL, not here; nothing about β appears in this config.
    #
    # THE COST, which must be stated beside the result: the CFG row of the 4x2 table carries a
    # two-stage treatment where the drop/anneal/AWR rows carry a one-stage one.
    #
    # EVERY OTHER FIELD IS A VERBATIM COPY of pi05_libero_axisinit_paper above -- deliberately a
    # second literal, matching how `pi05_libero_axisinit` and its `_paper` twin are written, and
    # NOT `dataclasses.replace`: `config_cfg_stage2_test.py` diffs the two field by field and
    # asserts they differ in exactly {name, data}, which a `replace` would make vacuously true.
    # If you edit the parent's schedule/batch/EMA/budget, edit this too -- the test will say so.
    TrainConfig(
        name="pi05_libero_axisinit_paper_cfg",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            # THE TREATMENT, and the only one. 5 == `slb_cfg.INFER_QUALITY`, spelled as a literal
            # because a registry entry should be readable without chasing an import; the
            # transform's own default IS that constant, and config_cfg_stage2_test.py pins the
            # two equal so this literal cannot drift away from what inference asks for.
            quality_tag=5,
            # NOT a treatment difference either: the same statement about the same bytes on disk
            # as the parent's, and it moves WITH the parent to PI's official upright build. If
            # these two ever disagree the CFG arm would be the only stage-2 leg trained at a
            # different orientation, which config_cfg_stage2_test.py's field-by-field diff would
            # call a second treatment. See the parent for the measurement.
            dataset_image_orientation="upright",
            # NOT a treatment difference -- the opposite. This is what makes the twin READ the
            # parent's norm stats instead of resolving to its own (nonexistent) assets dir, so
            # the two stage-2 legs are normalised identically. See the field's own comment.
            norm_stats_from="pi05_libero_axisinit_paper",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "AXIS_EEF_PAPER_INIT_CKPT",
                "/unset/set-AXIS_EEF_PAPER_INIT_CKPT-to-a-pi05_axis_pretrain_eef_paper-step/params",
            )
        ),
        num_train_steps=30_000,
        fsdp_devices=8,
    ),
    # INFERENCE-ONLY config that serves the STAGE-2 CFG checkpoint above to the LIBERO-Plus
    # client, with the π0.7 quality tag selected BY NAME.
    #
    # WHY A CONFIG AND NOT A FLAG ON pi05_libero: round 2 permits no env-var mode switches, and
    # the existing serve-time π0.7 spelling is reachable only through `SLB_CFG_METADATA`. The arm
    # a checkpoint was served under has to be recoverable from the launch line, so the mode is
    # chosen by this name plus `--quality-tag`.
    #
    # IT MIRRORS `pi05_libero`, NOT `pi05_axis_eef_libero_serve`. That other serve config is
    # action_horizon=16 and belongs to the non-paper stage-1 arm; what this serves is a
    # HORIZON-10 stage-2 checkpoint. Cloning the 16 would emit chunks of the wrong length with no
    # error anywhere -- plausible rollout numbers that mean nothing. `model` and `data.repo_id`
    # are pinned equal to `pi05_libero_axisinit_paper_cfg`'s in policy_cfg_quality_test.py.
    #
    # A SECOND LITERAL, not `dataclasses.replace(pi05_libero, ...)`, for the same reason as the
    # stage-2 twin: a derived config makes the parity test vacuously true. If you edit
    # `pi05_libero`'s model or data, edit this too -- the test will say so.
    #
    # NO `quality_tag=` ON THE DATA CONFIG. That field wires `LiberoQualityConditioning` into
    # `repack_transforms`, which inference never runs (`serve_policy.py` passes no
    # repack_transforms), so it would be inert here AND it carries a dropout that has no meaning
    # at serve time. The serve-time tag comes from `policy_config._input_chain`, which also
    # builds the paired unconditional branch. Pinned by a test.
    #
    # Serve: scripts/serve_policy.py --env LIBERO --quality-tag 5 --guidance-scale <beta - 1>
    #        policy:checkpoint --policy.config pi05_axis_cfg_libero_serve --policy.dir <ckpt>
    # NOTE beta = 1 + guidance_scale; see conf/experiments/onelayer_v3_round2_cfg_eval.toml.
    TrainConfig(
        name="pi05_axis_cfg_libero_serve",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            # -> asset_id "physical-intelligence/libero", which is the key `checkpoints.save_assets`
            # wrote the stage-2 norm stats under, so `create_trained_policy` reads the served
            # CHECKPOINT's own stats rather than any config-local assets dir.
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
    ),
    #
    # Fine-tuning AXIS Franka SLB configs (pi0.5 LoRA, HF-rendered Franka dataset).
    #
    # Full SLB benchmark: the five variants of EVERY task in slb_variant_homogeneous_800plus.json
    # that has converted LeRobot data on disk. The task list is data-driven via
    # _slb_available_task_ids() -- it reads the manifest and keeps only tasks whose `*__droid8d`
    # dir exists, so the registry auto-expands as the collaborator finishes uploading/converting
    # (85 tasks total; 1424/1628/1644/1645 available at time of writing). No dataless arms are
    # registered. The homogeneous manifest fixes ONE scene_variant_index per task and selects 25
    # demos; per-(task,variant) sidecars + per-task 8-D norm stats gate each arm.
    #
    # Each arm also gets a `_ki` twin with knowledge insulation enabled (arXiv:2505.23705):
    # same data, same LoRA setup, but the flow-matching gradient is cut at the prefix KV and
    # the VLM is instead trained by a FAST-token cross-entropy. Separate names so the
    # in-flight non-KI sweep is untouched and the pair is a clean A/B.
    # Fixed 30k-step budget, matching openpi's example fine-tune configs (the TrainConfig
    # default and pi05_droid_finetune's regime) rather than an epoch-derived budget. Over the
    # 25-demo homogeneous set this is many passes; the budget-matched cosine (warmup 10%,
    # decay over the whole 30k) anneals to the floor. Same budget for all 5 variants of a task.
    *[
        _axis_slb_config(task_id, variant, knowledge_insulation=ki, num_train_steps=30_000)
        for task_id in _slb_available_task_ids()
        for variant in ("vanilla", "filt_bin", "top70", "awr", "cfg")
        for ki in (False, True)
    ],
    #
    # HELD-OUT 20 LEARNABILITY GATE (pi0.5 LoRA from pi05_base, 20 demos per task).
    # Not an experimental arm: this decides which candidate tasks are ADMITTED to the benchmark.
    # Kept if 0.20 <= dual-sim success <= 0.80. See docs/heldout_20_benchmark_spec.md.
    # Budget is 50 epochs over each task's own 20-demo set, so no task is judged unlearnable on a
    # shorter effective exposure than another.
    *[
        _axis_heldout_gate_config(task_id, num_train_steps=heldout_gate_steps(task_id))
        for task_id in HELDOUT_20_TASK_IDS
    ],
    _axis_heldout_multitask_config(num_train_steps=heldout_multitask_steps(5)),
    # AXIS stage-1 BC init. Same everything else, so a difference in score is the pretraining.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_multitask_steps(5),
        init_path="/disk/axis/stage1_ckpts/pi05_axis_pretrain_eef_paper_5k/libero5k_bc/20604/params",
        name="pi05_axis_heldout_multitask_bc",
    ),
    # ACTION-SPACE A/B. Identical data, demos, recipe and pi05_base init to
    # `pi05_axis_heldout_multitask`; only the action space differs (droid8 instead of eef7).
    # The eef7 pilot scored 1/50 on task 811 while the droid8 trust gate scored 49/50 on the
    # SAME task through the SAME harness, so this isolates the action space from the data.
    # Eval MUST run with AXIS_EVAL_ACTION_SPACE=droid8.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_multitask_steps(5),
        name="pi05_axis_heldout_multitask_d8",
        eef_action=False,
    ),
    # droid8 + AXIS stage-1 init. The joint-velocity BACKUP arm for Table 1: eef7 scores 0/50 on
    # every place-relative task while droid8 reaches 44% on 1889 from the same demonstrations.
    # NOTE the init is an eef7-pretrained checkpoint, so what transfers here is the vision/language
    # stack, not the action space -- see the module note where this arm is used.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_multitask_steps(5),
        init_path="/disk/axis/stage1_ckpts/pi05_axis_pretrain_eef_paper_5k/libero5k_bc/20604/params",
        name="pi05_axis_heldout_multitask_d8_bc",
        eef_action=False,
    ),
    # ROUND 2. Two defects from round 1, both fixed here, both declared rather than passed in:
    #   * VISION IS NOW FROZEN. Round 1's "LoRA" trained the 411M SigLIP tower (13.7% of params
    #     trainable, 88% of that vision) on a single-viewpoint corpus. See the freeze_filter note.
    #   * NORM STATS ARE FITTED ON THE DATA IT TRAINS ON. The index is declared here, so
    #     `compute_norm_stats --config-name pi05_axis_heldout_multitask_d8_r2` uses it. Its assets
    #     dir is keyed on this config name, so it cannot clobber round 1's statistics.
    # Recompute stats BEFORE training this arm; it does not inherit round 1's.
    _axis_heldout_multitask_config(
        # Same 20-epoch budget as round 1, but computed from the CLEAN index this arm
        # actually trains on rather than the 20-task one.
        num_train_steps=heldout_epoch_steps(
            20, "/disk/axis/render/splits_eef/clean10.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/clean10.ranges.json") else 1,
        name="pi05_axis_heldout_multitask_d8_r2",
        eef_action=False,
        freeze_vision=True,
        roots_index="/disk/axis/render/splits_eef/clean10.roots.json",
        ranges_path="/disk/axis/render/splits_eef/clean10.ranges.json",
    ),
    # ROUND 2, AXIS-PRETRAINED ARM. Identical to the base arm above except `init_path`, so a
    # difference in score is the pretraining and nothing else -- the horizon confound that
    # invalidated the earlier 86%-vs-58% comparison is gone (both stages are at 10 now).
    #
    # CAVEAT THAT MUST TRAVEL WITH ANY NUMBER FROM THIS ARM: stage-1 here speaks EEF7, while this
    # finetune trains droid8. The vision/language stack transfers; the ACTION HEAD RELEARNS the
    # space. Report it as "AXIS pretraining, different action space", never as action-space
    # transfer. Re-run against the droid8 stage-1 when it exists.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            20, "/disk/axis/render/splits_eef/clean10.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/clean10.ranges.json") else 1,
        name="pi05_axis_heldout_multitask_d8_r2_bc",
        init_path="/disk/axis/stage1_ckpts/pi05_axis_pretrain_eef_paper_5k/libero5k_bc/20604/params",
        eef_action=False,
        freeze_vision=True,
        roots_index="/disk/axis/render/splits_eef/clean10.roots.json",
        ranges_path="/disk/axis/render/splits_eef/clean10.ranges.json",
    ),
    # TASK-QUALIFICATION ARM. The same pinned recipe as _r2 (frozen vision, horizon 10, droid8,
    # own norm stats) over a WIDER index: the clean-8 plus the night's newly rendered candidates.
    # The index PATHS are fixed here; their CONTENTS are written by the candidate pipeline after
    # screening -- so the config is registered once and the task set is data, not code. Steps are
    # read from the SAME ranges file at import time; if the file grows, re-resolving the config
    # yields the matching budget. compute_norm_stats on THIS config name fits statistics on
    # exactly the tasks the arm trains, in its own assets dir.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            20, "/disk/axis/render/splits_eef/qual_v1.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v1.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v1",
        eef_action=False,
        # MEASURED 2026-08-24, round 2 vs round 1 on the identical eval: freezing the vision tower
        # collapsed every place task (1889 58%->0%, 953 34%->0%, 809 8%->2%) and helped only the
        # pick-only task (868 48%->52%). The SLB-era "freeze_vision fixes the 0-20% floor"
        # diagnosis does NOT transfer here: on this single-viewpoint rendered corpus the SigLIP
        # finetune IS the domain adaptation the place tasks need. Qualification uses the recipe
        # that put 3 tasks in band, which is round 1's: vision trainable.
        freeze_vision=False,
        roots_index="/disk/axis/render/splits_eef/qual_v1.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v1.ranges.json",
    ),
    # Benchmark v2: same pinned recipe as qual_v1 but over the TRUE-VARIANT re-rendered corpus
    # (the slot-swap defect fix, 2026-08-26 -- each attempt renders in the scene variant it was
    # collected on). Registered as a PAIR whose only difference is the init checkpoint, so the
    # base-vs-BC-pretrain comparison is one-thing-different by construction. 5 epochs is the
    # owner-pinned qualification budget.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2",
        eef_action=False,
        freeze_vision=False,
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bc",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k/"
                  "libero5k_d8_bc/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrvfz",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase_vfz/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bcvfz",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k/libero5k_d8_bc_vfz/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awr",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_v2/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrp",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfg",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_dropt",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/libero5k_d8_drop_top_v2/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_droptp",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/libero5k_d8_drop_top_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_anneal",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_anneal_d8/libero5k_d8_anneal_v2/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_annealp",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_anneal_d8/libero5k_d8_anneal_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgv2",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_v2/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),

    # GRANULARITY FACTORIAL (2026-09-01): three FT twins for the episode-vs-segment study.
    # cfgpst -- tagged FT on the SEGMENT-tag CFG pretrain (row-level quality_phase_seg.npz);
    # the cfgt registration pattern exactly, only the init moves.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgpst",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/"
                  "libero5k_d8_cfg_phase_seg/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2_cfg",
    ),
    # awrpep / droptpep -- plain FT twins for the EPISODE-granularity AWR and drop-top
    # pretrains (awr_weights_phase_ep.json / schedule_drop_top_phase_ep.npz); the awrp/droptp
    # registration patterns exactly, only the inits move.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrpep",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/"
                  "libero5k_d8_awr_phase_ep/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_droptpep",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/"
                  "libero5k_d8_drop_top_phase_ep/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
    ),
    # QUALITY-TAGGED twins of the two CFG finetune arms above: same demos, same 5-epoch budget,
    # same init -- the ONLY difference is that training prompts carry the constant "Quality: 5"
    # tag (with the stage-2 two-level dropout keeping the unconditional branch trained), so the
    # finetune preserves the conditioning channel the stage-1 CFG checkpoints were pretrained
    # with instead of finetuning it away on bare prompts. Serve with
    #   scripts/serve_policy.py --quality-tag 5 [--guidance-scale <beta - 1>]
    # -- the serve-time spelling (FixedQualityConditioning) and this train-time spelling
    # (LiberoQualityConditioning) are the SAME apply_metadata call, "\nQuality: 5".
    #
    # norm_stats_from names the UNTAGGED twin: the stats are computed over the same roots/
    # ranges/columns and the tag never touches state or actions, so the twin's stats ARE this
    # arm's stats -- and a tagged config must not run compute_norm_stats at all (its
    # conditioning transform heads the repack group and raises without the presentation counter
    # only the training loader wires).
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgt",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2_cfg",
    ),
    # NIGHT 13: bc-tagged control -- does tagged finetuning help WITHOUT a CFG pretrain?
    # bc 20604 init + constant quality_tag=5, the cfgt registration pattern exactly. Norm
    # stats from the untagged pi05_axis_heldout_qual_v2 twin (same roots/ranges/columns and
    # the tag never touches state or actions); a tagged config must not run
    # compute_norm_stats.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bct",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k/"
                  "libero5k_d8_bc/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),

    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgv2t",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_v2/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2_cfgv2",
    ),
    # QUALITY-AWARE PHASE FINETUNE ARMS (night 9). Same demos, same 5-epoch budget and pinned
    # recipe as the qual_v2 family, but each arm BOTH initializes from its phase-supervision
    # stage-1 checkpoint AND applies the same supervision during the finetune, from artifacts
    # derived on the held-out corpus itself (phase signals replayed per attempt in the TRUE
    # scene variant, row-aligned to the corpus -- see /disk/axis/heldout20/phase_arms/ and
    # /disk/axis/render/splits_eef/phase_signals_heldout/).
    #
    # NORM STATS come from the untouched pi05_axis_heldout_qual_v2 arm: same roots/ranges/
    # columns, and reweighting/scheduling/conditioning never touches state or actions, so the
    # sibling's stats ARE these arms' stats. These configs must NOT run compute_norm_stats.
    #
    # EVERY GUARD IS ON (awr_required / schedule_required+expected_mode / quality_required):
    # a launch that loses the artifact flag must refuse, not silently train the BC control
    # under the arm's name -- that failure has shipped twice in this project already.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrq",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        awr_weights="/disk/axis/render/splits_eef/awr_weights_phase_heldout.json",
        awr_required=True,
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_annealq",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_anneal_d8/libero5k_d8_anneal_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        schedule_path="/disk/axis/render/splits_eef/schedule_anneal_phase_heldout.npz",
        schedule_required=True,
        expected_mode="anneal",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_dropq",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/libero5k_d8_drop_top_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        schedule_path="/disk/axis/render/splits_eef/schedule_drop_top_phase_heldout.npz",
        schedule_required=True,
        expected_mode="drop_top",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/disk/axis/render/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/disk/axis/render/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgq",
        eef_action=False,
        freeze_vision=False,
        init_path="/disk/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_phase/20604/params",
        roots_index="/disk/axis/render/splits_eef/qual_v2.roots.json",
        ranges_path="/disk/axis/render/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        quality_path="/disk/axis/render/splits_eef/quality_phase_heldout.npz",
        quality_required=True,
    ),
    # SERVER-3 LOCAL twin of pi05_axis_heldout_qual_v2_cfgv2t: identical recipe, only the
    # paths differ (payload shipped to /home/mqd/axis/heldout_ft). Registered under its own
    # name so the server-2 entries above stay byte-inert on this box.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgv2t_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_v2/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2_cfgv2",
    ),
    # SERVER-3 LOCAL phase-reward arm finetunes (2026-08-28). The stage-1 phase-reward arms
    # were RETRAINED after the reward bug (old results void); these finetune the retrained
    # checkpoints, which land server-3-locally. UNTAGGED -- protocol-identical to the earlier
    # server-2 qual_v2 arms (same recipe, same data, no quality conditioning); the only
    # variable per arm is `init_path`, which is the treatment. `norm_stats_from` points at
    # the SHIPPED stats copy (assets/pi05_axis_heldout_qual_v2_cfgv2): every qual_v2 arm
    # trains on the identical roots/ranges, so those statistics are these arms' own by
    # construction, exactly as the cfgv2t_s3 twin above resolves them.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrp_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2_cfgv2",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgp_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_phase/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2_cfgv2",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_droptp_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/libero5k_d8_drop_top_phase/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2_cfgv2",
    ),
    # QUALITY-AWARE PHASE FINETUNE ARMS (night 9), SERVER-3 SHARE: awrq + dropq train here
    # (annealq + cfgq train on server 2). Each arm BOTH initializes from its phase-supervision
    # stage-1 checkpoint (the ORIGINALS live on this box; gen-3, mtime-checked) AND applies the
    # same supervision during the qual_v2 finetune, from artifacts derived on the held-out
    # corpus (built+guard-verified on server 2, shipped here md5-identical; see
    # /disk/axis/heldout20/night9_PLAN.md on server 2).
    #
    # NORM STATS come from the untagged pi05_axis_heldout_qual_v2 twin (assets copy shipped;
    # byte-identical to the _cfgv2 copy the other _s3 arms read): same roots/ranges/columns,
    # and reweighting/scheduling never touches state or actions. Do NOT run compute_norm_stats.
    #
    # EVERY GUARD IS ON (awr_required / schedule_required+expected_mode): a launch that loses
    # the artifact flag must refuse, not silently train the BC control under the arm's name.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrq_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        awr_weights="/home/mqd/axis/heldout_ft/splits_eef/awr_weights_phase_heldout.json",
        awr_required=True,
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_dropq_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_drop_top_d8/libero5k_d8_drop_top_phase/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
        schedule_path="/home/mqd/axis/heldout_ft/splits_eef/schedule_drop_top_phase_heldout.npz",
        schedule_required=True,
        expected_mode="drop_top",
    ),
    # FROZEN-VISION DECOMPOSITION (night 10), SERVER-3 SHARE: the bcvfz finetune trains here
    # while awrvfz trains on server 2. Pinned qual_v2 recipe unchanged (vision TRAINABLE at
    # finetune; the ONLY stage-1 variable is the frozen SigLIP tower of the init checkpoint,
    # libero5k_d8_bc_vfz, which lives LOCALLY on this box). Plain BC: no supervision kwargs.
    # Norm stats from the untagged qual_v2 twin's staged assets copy -- same roots/ranges/
    # columns; do NOT run compute_norm_stats.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bcvfz_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k_vfz/libero5k_d8_bc_vfz/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    # NIGHT 11 EPOCH-CURVE CELLS: plain qual_v2 finetunes from @6000-step stage-1 inits
    # (owner's visual-first hypothesis; each cell's 20604 twin already exists). awrp6k and
    # bc6k init from @6k-RESEED mini-pretrains (true intermediates were not kept; identical
    # recipe prefix modulo data-order seed, labeled honestly in status pushes).
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_awrp6k_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_awr_d8/libero5k_d8_awr_phase_6k/6000/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bcvfz6k_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k_vfz/libero5k_d8_bc_vfz/6000/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bc6k_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k/libero5k_d8_bc_6k/6000/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_annealp6k_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_anneal_d8/libero5k_d8_anneal_phase/6000/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    # NIGHT 12 (Track C): cfgv2t-exact analogue on the PHASE reward -- gen-3 cfg_phase
    # init (shipped from server 2; s3's own cfg_phase tree is STALE gen-2) + constant
    # quality_tag=5, mirroring server 2's pi05_axis_heldout_qual_v2_cfgt registration.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_cfgpt_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_cfg_d8/libero5k_d8_cfg_phase_gen3/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    # NIGHT 13: serve-side twin of server 2's pi05_axis_heldout_qual_v2_bct (bc 20604 init
    # + constant quality_tag=5). Registered here so the GUIDED cell can serve the shipped
    # checkpoint with a config whose data paths resolve on THIS box; init_path is inert at
    # serve time and mirrors server 2's registration.
    _axis_heldout_multitask_config(
        num_train_steps=heldout_epoch_steps(
            5, "/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json", HELDOUT_GATE_BATCH)
        if os.path.exists("/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json") else 1,
        name="pi05_axis_heldout_qual_v2_bct_s3",
        eef_action=False,
        freeze_vision=False,
        init_path="/home/mqd/axis/libero_5k_v2/ckpts/pi05_axis_pretrain_d8_paper_5k/"
                  "libero5k_d8_bc/20604/params",
        roots_index="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.s3.roots.json",
        ranges_path="/home/mqd/axis/heldout_ft/splits_eef/qual_v2.ranges.json",
        quality_tag=5,
        norm_stats_from="pi05_axis_heldout_qual_v2",
    ),
    # Vision-freeze A/B test: frozen SigLIP tower at a 7369-step (150-epoch) budget, matched
    # to the earlier UNFROZEN 7369-step vanilla (2/20) so the only difference is the frozen
    # image tower. If this lifts success well above 2/20, vision-tower overfitting on the
    # 25-demo appearance-randomized set was the primary cause of the uniform low scores.
    _axis_slb_config(1644, "vanilla", num_train_steps=7369, name_suffix="_vfz7k"),
    # Re-render A/B test: SAME 25 demos, but IMAGES re-rendered to ONE fixed eval-matched
    # appearance (via DATASET_ROOT_OVERRIDE at launch). Vision UNFROZEN + 7369 steps, so vs
    # the 2/20 unfrozen baseline the ONLY change is train visual variance (25 different looks
    # -> 1 consistent look). Tests the hypothesis that appearance variance, not demo count,
    # starves the policy.
    _axis_slb_config(1644, "vanilla", num_train_steps=7369, freeze_vision=False, name_suffix="_rr7k"),
    # Re-rendered 5-METHOD bake-off @ 75k steps, vision unfrozen, on the fixed-appearance data
    # (launch with CFG_SUFFIX=_rr75k + DATASET_ROOT_OVERRIDE + MANIFEST_OVERRIDE). Same 25 demos,
    # same per-(task,variant) sidecars (keyed by attempt_id), only IMAGES re-rendered. First
    # method comparison on consistent-appearance data. vanilla is the REQUIRED baseline: without
    # it at the same 75k step budget, "filt_bin/top70/awr beat plain BC" is unprovable.
    *[
        _axis_slb_config(1644, v, num_train_steps=75_000, name_suffix="_rr75k")
        for v in ("vanilla", "filt_bin", "top70", "awr", "cfg")
    ],
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    # `pi05_droid_finetune` plus its `_ki` twin (knowledge insulation, arXiv:2505.23705).
    # pi05-DROID was itself trained with KI, so a KI fine-tune keeps the recipe the base
    # checkpoint was produced under instead of reverting to a plain joint flow-matching
    # fine-tune. The head is new (absent from the released params) and starts from scratch --
    # see CheckpointWeightLoader.missing_regex.
    *_with_ki_twin(
        TrainConfig(
            # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
            # Here, we use LeRobot data format (like for all other fine-tuning examples)
            # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
            name="pi05_droid_finetune",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,  # pi05 is trained with 32-dim actions
                action_horizon=16,
            ),
            data=LeRobotDROIDDataConfig(
                # Replace with your custom DROID LeRobot dataset repo id.
                repo_id="your_hf_username/my_droid_dataset",
                base_config=DataConfig(prompt_from_task=True),
                assets=AssetsConfig(
                    # Important: reuse the original DROID norm stats during fine-tuning!
                    assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                    asset_id="droid",
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
            num_train_steps=20_000,
            batch_size=32,
        )
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
