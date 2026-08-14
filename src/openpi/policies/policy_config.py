from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import slb_cfg as _slb_cfg
import openpi.transforms as transforms

# Eval has no advantage label, so the conditional branch always asks for the positive one.
# Mirrors slb_cfg.tag_for_label: 0 -> "\nAdvantage: positive".
CFG_POSITIVE_LABEL = 0


def _input_chain(
    *,
    repack_inputs: Sequence[transforms.DataTransformFn],
    default_prompt: str | None,
    data_config: _config.DataConfig,
    norm_stats: dict[str, transforms.NormStats],
    cfg_label: int | None,
    quality_tag: int | None = None,
    quality_drop_all: bool = False,
) -> list[transforms.DataTransformFn]:
    """Build the inference input chain, optionally carrying a fixed CFG advantage tag.

    Single source of truth for BOTH guidance branches: the conditional and unconditional
    chains are two calls that differ only in the tag argument (`cfg_label`, or
    `quality_drop_all` for the π0.7 route), so nothing except the prompt tag can drift
    between them. Guidance compares two velocities that are supposed to share everything but
    the prompt; if the chains were spelled out twice, any later edit to one would silently
    turn guidance into a comparison of two unrelated inputs. There is exactly ONE `return`
    below for that reason -- the branches choose a tag, never a chain.

    The tag sits directly after `InjectDefaultPrompt` -- as early as training applies it
    (see AxisFrankaSlbDataConfig, where SlbCfgConditioning heads the repack group) and
    still upstream of the tokenizer in `model_transforms`. `FixedCfgConditioning` no-ops
    when `prompt` is absent; both branches then no-op identically, so a missing prompt
    degrades guidance to plain conditional sampling rather than to a wrong tag.
    """
    cfg_tag: list[transforms.DataTransformFn]
    # π0.7 QUALITY CONDITIONING (the round-2 CFG arms). Selected by a NAMED serve config plus an
    # explicit flag -- never by an environment variable, because a checkpoint's arm must be
    # recoverable from what it was launched with. Checked FIRST so `SLB_CFG_METADATA` being
    # exported on the box cannot redirect this route into the SLB spelling, which appends a
    # `Mistake` token neither stage of this arm ever trained.
    if quality_tag is not None:
        if cfg_label is not None:
            raise ValueError(
                "quality_tag and cfg_label are both set: two conditioning schemes at once, and "
                "the served arm would not be recoverable from the config it was launched with"
            )
        # Local import: `quality_conditioning` imports `openpi.transforms` and is reached only by
        # this one route, matching how the training factories reach it.
        from openpi.training.quality_conditioning import FixedQualityConditioning

        cfg_tag = [
            FixedQualityConditioning(q_ep=int(quality_tag), drop_all=bool(quality_drop_all))
        ]
    elif quality_drop_all:
        # The unconditional half of a quality-guided pair, built without the tag it is the
        # counterpart OF. Silently this would be the plain chain -- identical to the conditional
        # one -- so (v_cond - v_uncond) == 0 and guidance vanishes at every β, i.e. a sweep that
        # reproduces the β = 1 cell at every scale and reports four distinct numbers for it.
        raise ValueError(
            "quality_drop_all=True with quality_tag=None: this is the unconditional branch of a "
            "pair whose conditional branch carries no tag, so the two chains would be identical "
            "and guidance would vanish at every scale."
        )
    # π0.7 metadata mode (env SLB_CFG_METADATA=1): serve with quality=max/mistake=no on the
    # conditional branch and the bare prompt on the unconditional branch (drop_all), matching
    # SlbCfgMetadataConditioning at train time. Else the legacy single advantage tag.
    elif cfg_label is None:
        cfg_tag = []
    elif _slb_cfg.metadata_enabled():
        cfg_tag = [_slb_cfg.FixedCfgMetadataConditioning(drop_all=(cfg_label == _slb_cfg.NULL_LABEL))]
    else:
        cfg_tag = [_slb_cfg.FixedCfgConditioning(label=cfg_label)]
    return [
        *repack_inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *cfg_tag,
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    cfg_conditioning: bool = False,
    guidance_scale: float = 0.0,
    quality_tag: int | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        cfg_conditioning: Enable the SLB `cfg` advantage tag at inference. The conditional
            branch gets "\\nAdvantage: positive" appended to the prompt and a second,
            otherwise-identical unconditional branch is built for classifier-free guidance.
            Off by default so every existing caller is byte-identical.
        guidance_scale: `w` in v = v_uncond + (1 + w) * (v_cond - v_uncond). Inference-only,
            so ONE checkpoint serves an entire sweep. w=0 short-circuits inside
            `Pi0.sample_actions` to plain conditional sampling; the kwarg is therefore only
            injected when w != 0, keeping models whose `sample_actions` has no
            `guidance_scale` parameter (pi0-FAST, the PyTorch path) working untouched.

            NOTE THE OFF-BY-ONE. π0.7's β is the TOTAL conditional weight, i.e. the `(1 + w)`
            above, so **β = 1 + guidance_scale** and this argument takes `β - 1`. Handing β
            straight through turns a β = 1.3 cell into β = 2.3 and nothing reports it, which is
            why `beta` is published in `policy_metadata` beside `guidance_scale`.
        quality_tag: π0.7 quality bin for the round-2 CFG arms (5 = what both stages condition
            on, 3 = the placebo cell). Appends `"\\nQuality: {q}"` to the conditional branch and
            builds the bare-prompt unconditional branch. A NAMED argument, never an environment
            variable: it takes precedence over the `SLB_CFG_METADATA` route -- which additionally
            appends a `Mistake` token this arm never trained -- so an exported variable on the
            box cannot change what a launched run is conditioning on.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    if quality_tag is not None and cfg_conditioning:
        raise ValueError(
            "quality_tag and cfg_conditioning are both set: two conditioning schemes at once, "
            "and the served arm would not be recoverable from the config it was launched with"
        )
    if guidance_scale != 0.0 and not cfg_conditioning and quality_tag is None:
        # Without the tag the two branches are the same input, (v_cond - v_uncond) == 0 and
        # guidance vanishes at every scale -- i.e. a β sweep that silently reproduces the
        # β = 1 cell four times over. That is exactly the failure this plumbing exists to
        # remove, so refuse it.
        raise ValueError(
            "guidance_scale != 0 requires cfg_conditioning=True or a quality_tag"
        )
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    chain_kwargs = {
        "repack_inputs": repack_transforms.inputs,
        "default_prompt": default_prompt,
        "data_config": data_config,
        "norm_stats": norm_stats,
    }
    input_transforms = _input_chain(
        **chain_kwargs,
        cfg_label=CFG_POSITIVE_LABEL if cfg_conditioning else None,
        quality_tag=quality_tag,
    )
    if quality_tag is not None:
        # Same call, same tag, one field apart -- see `_input_chain`'s docstring.
        uncond_transforms = _input_chain(
            **chain_kwargs, cfg_label=None, quality_tag=quality_tag, quality_drop_all=True
        )
    elif cfg_conditioning:
        uncond_transforms = _input_chain(**chain_kwargs, cfg_label=_slb_cfg.NULL_LABEL)
    else:
        uncond_transforms = None

    metadata = train_config.policy_metadata
    if cfg_conditioning or quality_tag is not None:
        # Served over a websocket, the sampling config is invisible to the rollout driver.
        # Publishing it lets the driver refuse to label a w=2 run as w=0 (or vice versa).
        sample_kwargs = dict(sample_kwargs or {})
        if guidance_scale != 0.0:
            sample_kwargs["guidance_scale"] = float(guidance_scale)
        metadata = {
            **(metadata or {}),
            "cfg_conditioning": True,
            "guidance_scale": float(guidance_scale),
        }
        if quality_tag is not None:
            metadata["quality_tag"] = int(quality_tag)
            # THE PAPER'S PARAMETER, recorded ALONGSIDE w rather than instead of it. β is what
            # the sweep and the write-up name; w is what openpi's arithmetic takes. Publishing
            # only one of them is how a β = 1.3 cell gets reported as β = 2.3 with every
            # artifact agreeing.
            metadata["beta"] = 1.0 + float(guidance_scale)

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        uncond_transforms=uncond_transforms,
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
