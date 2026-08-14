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
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
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
    # "drop" / "anneal" for a named schedule arm, or None otherwise. Reaches DataConfig as
    # `pretrain_expected_mode` and is checked against the artifact's own `meta["mode"]` in
    # data_loader.py -- binds the config NAME to the artifact's actual content, since nothing
    # else does (see DataConfig.pretrain_expected_mode).
    expected_mode: str | None = None
    default_prompt: str | None = None
    # Relative-EEF action space (LIBERO-Plus proxy benchmark): feed the baked `state_eef`(8) /
    # `action_eef`(7, robosuite OSC_POSE delta) columns and slice the output to 7. Default False
    # keeps the DROID-8D joint-velocity layout (for a future real-world checkpoint).
    eef_action: bool = False

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
        state_col = "state_eef" if self.eef_action else "observation.state"
        action_col = "action_eef" if self.eef_action else "action"
        repack_transform = _transforms.Group(
            inputs=[
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
            inputs=[axis_franka_policy.AxisFrankaInputs()],
            outputs=[
                axis_franka_policy.AxisFrankaEEFOutputs()
                if self.eef_action
                else axis_franka_policy.AxisFrankaOutputs()
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

    `name` overrides the flag-composed name for the index-schedule arms (`pi05_axis_drop`,
    `pi05_axis_anneal`), which are not distinguished by any flag this factory takes -- their arm
    lives in `data.schedule_path`, supplied per run. Passing it also switches the AWR weights off
    at the source: those arms inherit the launch environment, and reading
    $AXIS_PRETRAIN_AWR_WEIGHTS there would silently add round 1's loss reweighting on top of the
    schedule (`AxisFrankaPretrainDataConfig.create` refuses that combination anyway, so the
    alternative is a confusing hard failure on a box where the variable happens to be exported).

    Passing `name` ALSO pins norm stats to ROUND 1's assets directory
    (`./assets/pi05_axis_pretrain_eef_paper/Devon018/Franka-Datasets-v2`) instead of the arm's
    own. Two reasons, both load-bearing:

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
            # recomputing them per arm would break parity with the round-1 control.
            assets=(
                AssetsConfig(
                    assets_dir=f"./assets/{_AXIS_ROUND1_NAME}", asset_id=_AXIS_PRETRAIN_REPO_ID
                )
                if name
                else AssetsConfig()
            ),
            roots_index=os.environ.get("AXIS_PRETRAIN_ROOTS_INDEX"),
            ranges_path=os.environ.get("AXIS_PRETRAIN_RANGES"),
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
        # NO freeze_filter -> all params trainable (full weight).
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
                          name="pi05_axis_drop", schedule_required=True, expected_mode="drop"),
    _axis_pretrain_config(eef=True, paper=True, batch_size=64, num_train_steps=20_605,
                          name="pi05_axis_anneal", schedule_required=True, expected_mode="anneal"),
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
