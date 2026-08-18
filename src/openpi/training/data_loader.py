from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.slb_awr_loss as _slb_awr_loss
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # A pretraining config trains over MANY tasks at once: concatenate the per-task __droid8d
    # sub-datasets named in the roots index. Row restriction (the sample-ranges filter) is
    # applied by the sampler in create_torch_data_loader, mirroring the SLB gate.
    if getattr(data_config, "pretrain_roots_index", None):
        from openpi.training import pretrain_dataset

        return pretrain_dataset.build_pretrain_concat_dataset(data_config, action_horizon)

    # An SLB config may point at a local subfoldered LeRobot dataset (a per-task
    # camera_fixed folder inside a larger HF repo); load it by path via `root=`.
    root = getattr(data_config, "slb_dataset_root", None)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)

    # THE DAY-ONE CHECK for LIBERO configs (None everywhere else, so nothing else is touched).
    # `HF_LEROBOT_HOME` decides which BUILD a repo id resolves to, and the two LIBERO builds
    # store frames 180 degrees apart. This is the first moment both facts -- what the config
    # declared and what is actually on disk -- exist in the same process.
    declared_orientation = getattr(data_config, "libero_image_orientation", None)
    if declared_orientation is not None:
        from openpi.training import libero_orientation

        libero_orientation.check_dataset_build(
            repo_id=repo_id,
            declared_orientation=declared_orientation,
            info=dataset_meta.info,
        )

    extra_kwargs = {}
    if root is not None:
        # Our re-rendered SLB videos carry sub-millisecond frame-timestamp jitter that
        # trips LeRobot v3.0's default tolerance_s=1e-4. Loosen to a quarter-frame
        # (well under the half-frame that would risk matching the wrong frame).
        extra_kwargs["tolerance_s"] = 0.25 / dataset_meta.fps
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **extra_kwargs,
    )

    if data_config.prompt_from_task:
        # LeRobot v3.0 exposes meta.tasks as a DataFrame (index=task string,
        # column "task_index"); PromptFromLeRobotTask wants {task_index: task}.
        tasks = {int(ti): str(task) for task, ti in dataset_meta.tasks["task_index"].items()}
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def training_transforms(
    data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> list[_transforms.DataTransformFn]:
    """The ordered transform list `transform_dataset` applies, as a list.

    Split out of `transform_dataset` for ONE caller: the quality-conditioning arm, which cannot
    use `transform_dataset` at all. `quality_conditioning.wrap_and_transform` has to build the
    wrapper and the transform TOGETHER (either alone is a bug, and one of the two is silent), so
    it needs this list rather than an already-transformed dataset. Returning the list -- instead
    of letting the arm assemble its own -- is what keeps the arm's chain identical to the
    control's below the head: a second spelling of these four groups would drift.
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    return TransformedDataset(dataset, training_transforms(data_config, skip_norm_stats=skip_norm_stats))


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    resuming: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        resuming: Whether this run is actually resuming from a checkpoint (not merely
            `config.resume` -- see scripts/train.py's `initialize_checkpoint_dir`, which also
            covers the "resume requested but no checkpoint exists yet" case). Only consulted by
            the index-schedule path, which must refuse to resume: see ScheduleSampler's module
            docstring and `_check_schedule_resume`.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        num_train_steps=config.num_train_steps,
        resuming=resuming,
    )


def _check_schedule_unsupported_on_pytorch(data_config: _config.DataConfig) -> None:
    """The index-schedule wiring lives only in the jax branch of `create_torch_data_loader`.

    `scripts/train_pytorch.py` calls this function's caller with `framework="pytorch"`; without
    this guard a schedule config would train there as the plain control, silently.
    """
    schedule_path = getattr(data_config, "pretrain_schedule_path", None)
    if schedule_path:
        raise ValueError(
            f"pretrain_schedule_path={schedule_path!r} is set, but the PyTorch training path "
            f"(scripts/train_pytorch.py, framework='pytorch') does not wire the index-schedule "
            f"sampler -- only the jax branch of create_torch_data_loader does. Training this "
            f"config through train_pytorch.py would silently drop the schedule and run the plain "
            f"BC control instead. Use scripts/train.py (framework='jax') for schedule arms."
        )


def _check_quality_unsupported_on_pytorch(data_config: _config.DataConfig) -> None:
    """The CFG arm is defined on the jax path only, and its central claim is why.

    Coverage neutrality -- "this arm draws exactly the rows the round-1 control draws, in the same
    order" -- is a statement about `RowSampler(plan_rows_from_roots(...), seed)`, which is built in
    the jax branch of `create_torch_data_loader` and nowhere else. Under `framework="pytorch"` the
    pretrain row plan is never consulted at all (that branch goes straight to a
    `DistributedSampler` over the whole concatenated dataset, idle frames included), so neither
    the arm nor its control draws that sequence and the comparison the experiment rests on simply
    does not hold. Nothing in a loss curve says so.

    `scripts/train_pytorch.py` calls this function's caller with `framework="pytorch"`.
    """
    quality_path = getattr(data_config, "pretrain_quality_path", None)
    if quality_path:
        raise ValueError(
            f"pretrain_quality_path={quality_path!r} is set, but the PyTorch training path "
            f"(scripts/train_pytorch.py, framework='pytorch') does not build the pretrain row "
            f"plan -- only the jax branch of create_torch_data_loader does. This arm's whole "
            f"claim is that it draws the round-1 control's rows in the control's order, and that "
            f"is not true here. Use scripts/train.py (framework='jax') for the CFG arm."
        )


def _check_stage2_quality_unsupported_on_pytorch(framework: str) -> None:
    """Stage 2's presentation counter reaches the worker through the SAMPLER's index, and the
    pytorch branch builds its own.

    `DistributedSampler` (or the plain shuffle that branch falls back to) draws 0..n-1, so
    `PresentationKeyedDataset` would decode presentation 0 for every sample of every epoch --
    which is exactly the fixed-partition bug the counter exists to remove, restored silently:
    the unconditional branch fit on one frozen 19.25% of rows, the conditional branch never
    seeing them, and a realized tagged fraction that still reads 0.8075.
    """
    if framework == "pytorch":
        raise ValueError(
            "this config carries the stage-2 quality conditioning (LeRobotLiberoDataConfig."
            "quality_tag), but the PyTorch training path (scripts/train_pytorch.py, "
            "framework='pytorch') builds its own sampler and cannot carry the presentation "
            "counter the dropout is keyed on. Every epoch would reuse presentation 0, i.e. a "
            "FIXED 19.25% row partition rather than a per-example dropout, with the realized "
            "rate still reading 0.8075. Use scripts/train.py (framework='jax')."
        )


def _check_schedule_unsupported_on_rlds(data_config: _config.DataConfig) -> None:
    """`create_rlds_data_loader` builds no sampler at all, so a schedule config would run uniform."""
    schedule_path = getattr(data_config, "pretrain_schedule_path", None)
    if schedule_path:
        raise ValueError(
            f"pretrain_schedule_path={schedule_path!r} is set, but this config also sets "
            f"rlds_data_dir, which routes create_data_loader to the RLDS branch -- and that "
            f"branch builds no sampler, so the precomputed (steps, batch) row block would be "
            f"ignored and the run would train the plain BC control under the arm's name. Use a "
            f"LeRobot/torch config (the jax branch of create_torch_data_loader) for schedule arms."
        )


def _check_quality_unsupported_on_rlds(data_config: _config.DataConfig) -> None:
    """Same refusal for the CFG arm: the RLDS branch never wraps the dataset with the tag."""
    quality_path = getattr(data_config, "pretrain_quality_path", None)
    if quality_path:
        raise ValueError(
            f"pretrain_quality_path={quality_path!r} is set, but this config also sets "
            f"rlds_data_dir, which routes create_data_loader to the RLDS branch -- and that "
            f"branch never calls quality_conditioning.wrap_and_transform, so every prompt would "
            f"stay bare and the arm would train as the plain BC control under its own name. Use a "
            f"LeRobot/torch config (the jax branch of create_torch_data_loader) for the CFG arm."
        )


def _bin_row_share(meta: dict) -> dict:
    """Per-bin share of the TRAINABLE rows, as the loader log's own reading of the artifact.

    `bin_row_counts` is absolute and only the tagged rows are counted in it, so a skewed corpus
    and a skewed dropout look the same in the raw counts. The share against `n_trainable` is the
    number the write-up quotes, and computing it here means the run record carries it rather than
    requiring the artifact to be reopened months later.
    """
    counts = meta.get("bin_row_counts") or {}
    total = int(meta.get("n_trainable") or 0)
    if not counts or total <= 0:
        return {}
    return {str(b): round(int(c) / total, 4) for b, c in sorted(counts.items(), key=lambda kv: int(kv[0]))}


def _check_quality_resume(quality_path: str, resuming: bool) -> None:
    """A resume restarts the row permutation at epoch 0 while the optimiser continues from step k.

    The CFG arm's ONLY claim is coverage neutrality: it draws exactly the rows the round-1 control
    draws, in the control's order, and differs from it in the prompt alone. That claim is a
    statement about `RowSampler(plan_rows_from_roots(...), seed)` -- and `RowSampler.__iter__`
    restarts its epoch counter at 0 every time the loader is built, while openpi checkpoints no
    loader position at all (`checkpoints.restore_state` drops its `data_loader` argument).

    So a resume at step k over an N-step budget trains `perm[0 : k*B]` and then `perm[0 : (N-k)*B]`
    -- a UNION of `max(k, N-k)*B` unique rows, not `N*B`. A resume at the midpoint of this 16-hour
    job halves coverage to 50% of the corpus while the run record, the TOML and the paper line all
    still say "100%, byte-identical to the control". Nothing else would show it: the loss curve,
    the step count, the checkpoint and every other guard are unchanged, and the sibling arms were
    blocked on exactly this quantity (37%).

    Note this is a DIFFERENT mechanism from `_check_schedule_resume`'s -- the schedule arms replay
    an artifact, this arm draws a permutation -- but the same consequence and the same cause: no
    loader position is checkpointed. Both refuse rather than warn.
    """
    if resuming:
        raise ValueError(
            f"config.resume is set together with pretrain_quality_path={quality_path!r}, but "
            f"openpi checkpoints do not save data-loader position (checkpoints.restore_state "
            f"drops its data_loader argument): RowSampler restarts its epoch counter at 0, so a "
            f"resume at step k over an N-step budget would draw perm[0:k*B] and then "
            f"perm[0:(N-k)*B] -- a union of max(k, N-k)*B unique rows instead of N*B. A resume at "
            f"the midpoint halves this arm's corpus coverage to ~50% while its whole claim is "
            f"that coverage is IDENTICAL to the round-1 control's. Restart the run clean instead "
            f"of resuming -- the same standing instruction the schedule arms carry in "
            f"conf/experiments/onelayer_v3_round2_cfg_arms.toml."
        )


def _check_stage2_quality_resume(resuming: bool) -> None:
    """A resume restarts `PresentationSampler` at presentation 0 while the optimiser continues
    from step k.

    Stage 2's claim is NOT coverage neutrality -- that is `_check_quality_resume`'s claim, about
    stage 1's `RowSampler`. Stage 2's claim is that dropout is drawn PER EXAMPLE rather than per
    row, because it runs ~5.7 epochs over the LIBERO corpus (30,000 x 64 samples over 338,575
    frames) and a row-keyed draw would fit the unconditional branch on a FIXED 19.25% partition
    instead: ~65k rows seen 5.7 times each, and the conditional branch never seeing them. That is
    exactly why `PresentationSampler` exists and why `LiberoQualityConditioning.dropped` takes a
    required presentation counter.

    That property depends on the counter actually advancing across the run. `PresentationSampler`
    restarts `self._presentation` at 0 every time it is constructed, and openpi checkpoints no
    loader position at all (`checkpoints.restore_state` drops its `data_loader` argument). So a
    resume rebuilds the sampler at presentation 0 and replays that pass's dropout pattern a second
    time: rows the first pass dropped are dropped again rather than re-drawn, rows it kept are kept
    again, and the later presentations the run's step budget was counting on are never reached.
    That partially reintroduces the fixed-partition bug the presentation counter exists to remove
    -- not a full reversion (only the replayed presentations repeat; presentations beyond the
    resume point are simply never trained), but a real and undisclosed degradation of the same
    property. Nothing in the loss curve, the checkpoint or the run record shows it.

    Same cause as `_check_quality_resume` and `_check_schedule_resume` -- no loader position is
    checkpointed -- and the same response: refuse rather than warn. `PresentationSampler`'s own
    docstring used to argue this was fine because stage 2's claim is not coverage neutrality; that
    is true but beside the point, since the presentation counter protects a different property
    that a resume degrades just the same.
    """
    if resuming:
        raise ValueError(
            "config.resume is set together with stage-2 presentation-keyed quality conditioning "
            "(LeRobotLiberoDataConfig.quality_tag), but openpi checkpoints no data-loader "
            "position (checkpoints.restore_state drops its data_loader argument): "
            "PresentationSampler restarts its presentation counter at 0 on every construction, so "
            "a resume would replay presentation 0's dropout pattern instead of continuing the "
            "sequence -- rows dropped the first time through are dropped again, rows kept are "
            "kept again, and the later presentations the run's step budget assumed are never "
            "reached. That partially reintroduces the fixed-partition bug the presentation "
            "counter exists to remove: over stage 2's ~5.7 epochs the unconditional branch would "
            "be skewed toward whichever rows the replayed presentation(s) happened to drop, with "
            "nothing in the loss curve to show it. Restart the run clean instead of resuming."
        )


def _check_schedule_resume(schedule_path: str, resuming: bool) -> None:
    """A resume replays the schedule from row 0 while the optimiser continues from step k.

    openpi checkpoints no data-loader position (`checkpoints.restore_state` drops its
    `data_loader` argument), so this is not merely "coverage is off" -- for the anneal arm,
    whose ramp only starts at `ramp_start_step ~= 0.85 * num_train_steps`, any resume before the
    last ~15% of training means the ramp is never reached and the arm silently degenerates into
    the uniform control.
    """
    if resuming:
        raise ValueError(
            f"config.resume is set together with pretrain_schedule_path={schedule_path!r}, but "
            f"openpi checkpoints do not save data-loader position (checkpoints.restore_state "
            f"drops its data_loader argument): a resume at step k would replay the schedule from "
            f"row 0 while the optimiser continues from k. For the anneal arm "
            f"(ramp_start_step ~= 0.85 * num_train_steps) this means the high-quality ramp is "
            f"never reached unless the resume happens in the last ~15% of training, silently "
            f"degenerating the arm into the plain control. Restart the run clean instead of "
            f"resuming -- this is the same standing instruction round 1 gives in "
            f"conf/experiments/onelayer_v3_stage1_arms.toml."
        )


def _check_schedule_mode(data_config: _config.DataConfig, sampler_meta: dict) -> None:
    """Bind the artifact's own `meta["mode"]` to the arm this config's name promises.

    Nothing else checks this: handing `pi05_axis_drop` an anneal artifact would train anneal
    under the drop name and pass every other check (batch size, dataset bounds, step budget all
    still line up), directly contradicting the rule that a checkpoint's arm be recoverable from
    its config name alone.
    """
    expected = getattr(data_config, "pretrain_expected_mode", None)
    if expected is None:
        return
    actual = sampler_meta.get("mode")
    if actual != expected:
        raise ValueError(
            f"schedule mode mismatch: this config expects mode={expected!r} (from "
            f"DataConfig.pretrain_expected_mode) but the schedule artifact's own meta reports "
            f"mode={actual!r}. Handing the {actual!r} artifact to the {expected!r} arm would "
            f"train {actual!r} under the {expected!r} name; pass the {expected!r} schedule "
            f"artifact instead."
        )


# The `mode`s `axis.dataset.index_schedule.MODES` can produce. Duplicated (three strings) rather
# than imported: that module is offline tier. Used only to recognise a `<mode>_<reward_id>.npz`
# artifact NAME, so a checkout drift here weakens `_check_schedule_reward_id` to a no-op on the
# unrecognised name rather than making it wrong.
_SCHEDULE_MODES = ("vanilla", "drop", "anneal")


def _check_schedule_reward_id(schedule_path: str, sampler_meta: dict) -> None:
    """Bind a `<mode>_<reward_id>.npz` filename to the artifact's own reward, since nothing else does.

    `_check_schedule_mode` cannot separate the two drop arms from each other: `drop_v2` and
    `drop_phase` both run under the config name `pi05_axis_drop` and both carry `meta["mode"] ==
    "drop"`, so only the artifact FILENAME in the TOML's `data.schedule_path` says which reward
    produced them -- and until this check, nothing compared that filename against the artifact's
    contents. A `drop_v2.npz` accidentally rebuilt from the phase weights would train under
    `exp_name=v3_5000_drop_v2`, pass every other guard, and differ only in one log line.

    Only names of the `<mode>_<reward_id>` shape are checked, recognised by their leading token
    being a schedule mode. Any other name (a staging path, `schedule.npz`, a dated file) makes no
    claim about its contents, so there is nothing to contradict and this is a no-op -- the check
    must not turn a naming convention into a requirement.
    """
    stem = pathlib.Path(schedule_path).stem
    # EVERY PUBLISHED ARTIFACT IS `schedule_<mode>_<reward>.npz`, so keying on the first token made
    # this a no-op exactly where it was needed: "schedule" is not a mode, the function returned, and
    # the one check standing between `drop_v2` and a file rebuilt from phase weights never ran. The
    # docstring described `<mode>_<reward>` while the builder wrote `schedule_<mode>_<reward>`, and
    # nothing compared the two. Strip the prefix the builder actually uses, then apply the same rule.
    if stem.startswith("schedule_"):
        stem = stem[len("schedule_"):]
    if stem.split("_", 1)[0] not in _SCHEDULE_MODES:
        return
    mode, reward_id = sampler_meta.get("mode"), sampler_meta.get("reward_id")
    if reward_id is None:
        raise ValueError(
            f"schedule {schedule_path} is named for a mode and reward but its meta carries no "
            f"'reward_id', so the name cannot be checked against the artifact. Rebuild it with "
            f"axis.dataset.build_index_schedule, which stamps the reward the weights came from."
        )
    expected = f"{mode}_{reward_id}"
    if stem != expected:
        raise ValueError(
            f"schedule {schedule_path} is named {stem!r} but its own meta reports mode={mode!r} "
            f"and reward_id={reward_id!r}, i.e. {expected!r}. The filename is the ONLY thing that "
            f"distinguishes the two rewards within one arm (both drop artifacts carry "
            f"mode='drop' and run under the same config name), so this run would record itself as "
            f"{stem!r} while training {expected!r}. Rebuild the artifact or launch the one the "
            f"name promises."
        )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    num_train_steps: int | None = None,
    resuming: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        num_train_steps: The training budget, when known. Consulted by the index-schedule path,
            which must refuse a budget longer than its artifact (see ScheduleSampler), and logged
            by the stage-2 quality path as the number of presentations the run will need.
        resuming: Whether this run is actually resuming from a checkpoint. Consulted by the
            index-schedule path, the stage-1 quality path and the stage-2 quality path, all three
            of which must refuse to resume: openpi checkpoints no loader position, so the row (or
            presentation) sequence would restart at 0 while the optimiser continued from step k.
            See `_check_schedule_resume`, `_check_quality_resume` and
            `_check_stage2_quality_resume`.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    quality_path = getattr(data_config, "pretrain_quality_path", None)
    # The STAGE-2 twin, read off the repack group rather than a second config flag -- see
    # `quality_conditioning.stage2_conditioning`. `None` for every other config, including stage 1.
    from openpi.training import quality_conditioning as _quality_conditioning

    stage2_conditioning = _quality_conditioning.stage2_conditioning(data_config)
    stage2_sampler = None
    if stage2_conditioning is not None:
        if quality_path:
            raise ValueError(
                f"this config sets both pretrain_quality_path={quality_path!r} (stage 1's per-row "
                f"artifact) and a stage-2 LiberoQualityConditioning in its repack group. The two "
                f"tag the same prompt from different sources; one run cannot be both stages."
            )
        _check_stage2_quality_unsupported_on_pytorch(framework)
        # RAISES. PresentationSampler restarts at presentation 0 on a resume while the optimiser
        # continues from step k, so a mid-run resume would replay one presentation's dropout
        # pattern instead of continuing the sequence -- see `_check_stage2_quality_resume`.
        # Checked first, before anything expensive.
        _check_stage2_quality_resume(resuming)
        # ONE call builds the wrapper and the sampler: drawn 0..n-1 by any other sampler, the
        # wrapper reports presentation 0 for every sample and stage 2 silently trains the fixed
        # 19.25% partition instead of a dropout. See `wrap_presentations`.
        dataset, stage2_sampler = _quality_conditioning.wrap_presentations(
            dataset, stage2_conditioning, seed=seed, shuffle=shuffle
        )
        logging.info(
            "stage-2 quality conditioning: q_ep=%d drop_whole=%.4f drop_component=%.4f "
            "target_tagged=%.4f seed=%d n_rows=%d shuffle=%s presentations_needed=%s",
            int(stage2_conditioning.q_ep), float(stage2_conditioning.drop_whole),
            float(stage2_conditioning.drop_component),
            (1.0 - float(stage2_conditioning.drop_whole))
            * (1.0 - float(stage2_conditioning.drop_component)),
            int(stage2_conditioning.seed), len(dataset), shuffle,
            "unknown" if num_train_steps is None
            else -(-num_train_steps * batch_size // len(dataset)),
        )
    if quality_path:
        # π0.7 quality conditioning. This REPLACES the `transform_dataset` call below rather than
        # wrapping around it, and that is the whole of the wiring's difficulty.
        #
        # The tag has to be in the sample when the FIRST transform runs: `RepackTransform` heads
        # the chain and rebuilds the dict from its structure map, dropping `quality` along with
        # everything else not in it. So the wrapper goes UNDER the transforms (not after them, the
        # way the AWR weight wrapper goes -- that weight rides to the batch under its own key and
        # never passes through a transform), and `AxisQualityConditioning` goes at the HEAD of the
        # transform list, ahead of the repack.
        #
        # Both of those are done by `wrap_and_transform`, which is the only supported way to
        # assemble the two objects: wrapper-without-transform injects a tag that the repack then
        # drops, leaving every prompt bare and the arm training as the plain BC control under its
        # own name -- silently. Hence no composition here, and NOTHING added to the repack group
        # in config.py either: `wrap_and_transform` prepends to a list that already begins with
        # `repack_transforms.inputs`, so an entry there too would tag twice.
        from openpi.training import quality_conditioning

        # RAISES. The row permutation this arm shares with the control restarts at epoch 0 on a
        # resume while the optimiser continues from step k, so a mid-run resume silently halves
        # the coverage the arm's entire claim rests on. Checked first, before anything expensive.
        _check_quality_resume(quality_path, resuming)
        tags = quality_conditioning.QualityTags(quality_path)
        # The filename is the only thing separating cfg_v2 from cfg_phase: one config name, one
        # artifact structure, two rewards. Checked before anything expensive happens.
        tags.check_reward_id(quality_path)
        # Binds the artifact to THIS corpus. `QualityTaggedDataset.__init__` binds it again (it is
        # the one place both lengths are in hand, so it cannot be forgotten); this call is here
        # only because it can name the roots index in the error, and a run that hits this is
        # reading somebody else's corpus.
        tags.check_dataset_rows(len(dataset), getattr(data_config, "pretrain_roots_index", None))
        # RAISES on overflow. PaligemmaTokenizer truncates from the right with only a warning, and
        # what it truncates is the `Action:` marker -- for this arm only, because only this arm
        # lengthens the prompt.
        margin = quality_conditioning.check_token_budget(tags.prompts, model_config.max_token_len)
        dataset = quality_conditioning.wrap_and_transform(
            dataset, tags, training_transforms(data_config, skip_norm_stats=skip_norm_stats)
        )
        logging.info(
            "quality conditioning %s: reward=%s n_rows=%s n_trainable=%s tagged=%s drop_whole=%s "
            "drop_component=%s bin_row_share=%s entropy_ratio=%s token_margin=%d config_hash=%s "
            "generator_sha256=%s seed=%s",
            quality_path, tags.reward_id, tags.meta.get("n_rows"),
            tags.meta.get("n_trainable"),
            tags.meta.get("realized_tagged_fraction"), tags.meta.get("realized_drop_whole"),
            tags.meta.get("realized_drop_component"), _bin_row_share(tags.meta),
            tags.meta.get("q_given_task_entropy_ratio"),
            margin, tags.meta.get("config_hash"),
            # WHICH builder source drew these tags. The schedule arms already log it; without it
            # here the run record cannot tell a re-derived artifact from the audited one, and
            # `config_hash` alone does not cover the generator.
            tags.meta.get("generator_sha256"), tags.meta.get("seed"),
        )
    else:
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        # The index-schedule wiring below lives ONLY in the jax branch; a schedule config run
        # through here would silently train as the plain control. See train_pytorch.py, which
        # is the only caller of this branch.
        _check_schedule_unsupported_on_pytorch(data_config)
        # ...and the CFG arm likewise: its coverage-neutrality claim is about the RowSampler this
        # branch never builds. See `_check_quality_unsupported_on_pytorch`.
        _check_quality_unsupported_on_pytorch(data_config)
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
        # Stage 2's presentation counter. Installed here and not above because this is where the
        # sampler lives; the wrapper it is paired with was built with it in one call.
        if stage2_sampler is not None:
            sampler = stage2_sampler
        # SLB variant bake-off: restrict/weight rows to the kept (attempt, window)
        # set of the configured WVM variant. Only engages when a sidecar root is
        # set, so all other JAX configs keep full-shuffle behaviour.
        if getattr(data_config, "slb_sidecar_root", None):
            from openpi.training import slb_variant_sampler

            sampler, weights_by_row = slb_variant_sampler.build_sampler(dataset, data_config, seed=seed)
            if weights_by_row is not None:
                # `awr` only (WVM Eq E.5): rows are drawn uniformly like every other
                # variant, and the per-row weight rides along in the sample dict so the
                # training loss can reweight. No other variant gets this wrapper, so
                # their batches keep exactly the keys they had before.
                dataset = slb_variant_sampler.WeightedRowDataset(dataset, weights_by_row)
        elif getattr(data_config, "pretrain_roots_index", None):
            # Pretraining: restrict the concatenated multi-task dataset to the non-idle
            # sample-ranges, drawn UNIFORMLY -- and, when an AWR weights artifact is configured,
            # carry a per-row Eq E.7 weight into the batch so the loss can reweight.
            #
            # Rows are drawn uniformly in BOTH of those cases. WVM Eq E.5 puts the weights in the
            # OBJECTIVE, not in the sampling distribution: weighted resampling has a different
            # estimator variance and, drawing with replacement, changes the epoch composition
            # independently of the weights. See slb_awr_loss's module docstring.
            #
            # The third case is the round-2 index-schedule arms: `pretrain_schedule_path` names a
            # precomputed (steps, batch) block of rows that replaces the draw entirely. It is
            # checked first because nothing else in this branch applies once the sampling has
            # been decided offline.
            from openpi.training import pretrain_dataset
            from openpi.training import slb_variant_sampler

            weights_path = getattr(data_config, "pretrain_awr_weights", None)
            schedule_path = getattr(data_config, "pretrain_schedule_path", None)
            if schedule_path:
                # Round-2 arms: WHICH rows (drop) and IN WHAT ORDER (anneal) were decided
                # offline, so there is nothing left to draw here -- the artifact is replayed
                # verbatim and its row t becomes the batch at step t. That also means the run's
                # coverage numbers are read off the artifact rather than reconstructed from an
                # RNG state.
                from openpi.training.schedule_sampler import ScheduleSampler

                if weights_path:
                    raise ValueError(
                        f"both an index schedule ({schedule_path}) and AWR weights "
                        f"({weights_path}) are configured; the schedule already encodes the "
                        f"supervision, so this run would be two arms at once"
                    )
                # openpi checkpoints no loader position (checkpoints.restore_state drops its
                # data_loader argument), so a resume at step k would replay the schedule from
                # row 0 while the optimiser continues from k -- see `_check_schedule_resume`.
                _check_schedule_resume(schedule_path, resuming)
                sampler = ScheduleSampler(schedule_path)
                # Bind the artifact's own content to the arm this config's NAME promises: nothing
                # else ties `pi05_axis_drop`/`pi05_axis_anneal` to the file handed to them at
                # launch, so a mismatched artifact would otherwise train silently under the wrong
                # name. See `DataConfig.pretrain_expected_mode`.
                _check_schedule_mode(data_config, sampler.meta)
                # ...and the mode alone cannot tell `drop_v2` from `drop_phase`: same config name,
                # same meta["mode"], different reward. The filename is the only thing that
                # separates them, so it is checked against the artifact's own reward_id.
                _check_schedule_reward_id(schedule_path, sampler.meta)
                sampler.check_batch(local_batch_size)
                # Not just a bounds check: this is what binds the artifact to THIS corpus, via
                # the row count the generator recorded. `roots_index` is passed so the error can
                # name both sides of the disagreement. See `ScheduleSampler.check_dataset_rows`.
                sampler.check_dataset_rows(len(dataset), data_config.pretrain_roots_index)
                if num_train_steps is not None:
                    sampler.check_num_train_steps(num_train_steps)
                logging.info(
                    "index schedule %s: mode=%s reward=%s steps=%d batch=%d keep_fraction=%s "
                    "unique_episodes=%s unique_frames=%s epochs=%.2f seed=%s config_hash=%s",
                    schedule_path, sampler.meta.get("mode"), sampler.meta.get("reward_id"),
                    sampler.total_steps, sampler.batch, sampler.meta.get("keep_fraction"),
                    sampler.meta.get("unique_episodes"), sampler.meta.get("unique_frames"),
                    sampler.meta.get("epochs_over_unique_rows", 0.0), sampler.meta.get("seed"),
                    sampler.meta.get("config_hash"),
                )
            elif weights_path:
                rows, weights = pretrain_dataset.plan_rows_and_weights_from_roots(
                    data_config.pretrain_roots_index,
                    data_config.pretrain_ranges_path,
                    weights_path,
                )
                # Dense over the WHOLE flat space; NaN everywhere the sampler will never go, so
                # a row reached by some other path raises rather than training unweighted.
                dense = np.full(len(dataset), np.nan, dtype=np.float32)
                dense[rows] = weights
                logging.info(
                    "pretrain AWR weights from %s: n_rows=%d mean=%.4f at_cap=%.1f%% at_one=%.1f%%",
                    weights_path, len(weights), float(weights.mean()),
                    100.0 * float(np.mean(weights >= weights.max() - 1e-9)),
                    100.0 * float(np.mean(weights == 1.0)),
                )
                dataset = slb_variant_sampler.StrictWeightedRowDataset(dataset, dense)
            else:
                rows = pretrain_dataset.plan_rows_from_roots(
                    data_config.pretrain_roots_index, data_config.pretrain_ranges_path
                )
            if not schedule_path:
                # The schedule IS the sampler; only the uniform arms plan their own rows.
                sampler = slb_variant_sampler.RowSampler(rows, seed=seed)

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    # The RLDS branch wires NONE of the round-2 mechanisms: no row plan, no schedule sampler, no
    # quality wrapper. `create_data_loader` picks it purely on `rlds_data_dir`, so a config that
    # set both that and an arm's artifact would train as the plain control under the arm's name --
    # the one remaining branch where that is still possible, now that the pytorch branch refuses.
    # Unreachable today (no round-2 config sets rlds_data_dir), which is exactly when a refusal is
    # cheap; it is the config that changes, not this file.
    _check_schedule_unsupported_on_rlds(data_config)
    _check_quality_unsupported_on_rlds(data_config)
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if _slb_awr_loss.LOSS_WEIGHT_KEY in batch:
                # SLB `awr` only: a third element carrying the per-row WVM Eq E.7 weight,
                # which train_step feeds into the Eq E.5 weighted loss. Absent for every
                # other config, which still yields the original 2-tuple.
                yield observation, batch["actions"], batch[_slb_awr_loss.LOSS_WEIGHT_KEY]
            else:
                yield observation, batch["actions"]
