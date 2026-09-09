from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

# The dataloader-position sidecar, written next to each orbax checkpoint (feat/loader-resume). It
# is a plain JSON file in the step directory, deliberately OUTSIDE the orbax item tree: orbax
# restores by named item, so a loose file alongside is invisible to it, and this keeps the resume
# machinery from depending on orbax internals. Absent for every checkpoint written before this
# feature, which is exactly the signal the resume guards use to fall back to their refusal.
LOADER_STATE_FILENAME = "loader_state.json"


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)
    save_loader_state(checkpoint_manager, data_loader, step)


def save_loader_state(
    checkpoint_manager: ocp.CheckpointManager,
    data_loader: _data_loader.DataLoader,
    step: int,
) -> None:
    """Persist the dataloader position as a JSON sidecar next to the step's orbax checkpoint.

    No-ops unless the loader is backed by a resumable sampler (RowSampler / ScheduleSampler); every
    other config writes nothing, so nothing about them changes. The offset stored is the
    AUTHORITATIVE `step * local_batch_size`, NOT the sampler's own yield count -- with prefetching
    workers the sampler runs ahead of the trained step, and taking the step makes the resumed
    position exact for any `num_workers`.

    Best-effort: a failure to write the sidecar is logged and swallowed rather than crashing
    training, because the only consequence is that a later resume falls back to the guard's refusal
    (the safe default), never a silent wrong-offset resume.
    """
    if jax.process_index() != 0:
        return
    sampler = _data_loader.resumable_sampler(data_loader)
    if sampler is None:
        return
    batch = getattr(data_loader, "local_batch_size", None)
    if not batch:
        logging.warning("save_loader_state: loader exposes no local_batch_size; skipping sidecar")
        return
    state = sampler.checkpoint_state(int(step) * int(batch))
    try:
        directory = epath.Path(checkpoint_manager.directory) / str(step)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / LOADER_STATE_FILENAME).write_text(json.dumps(state, indent=2, sort_keys=True))
        logging.info("save_loader_state: wrote %s for step %d: %s", LOADER_STATE_FILENAME, step, state)
    except OSError as e:
        logging.warning("save_loader_state: could not write loader sidecar for step %d: %s", step, e)


def load_loader_state(checkpoint_manager: ocp.CheckpointManager, step: int | None) -> dict | None:
    """Read the dataloader-position sidecar for `step`, or None if the checkpoint has none.

    A missing file is the expected case for any checkpoint written before feat/loader-resume, and
    it MUST read as None (not an error) so the resume guards can keep refusing on old checkpoints.
    Distinguishes "absent" (FileNotFoundError -> None) from a genuinely unreadable file (any other
    OSError propagates) -- the EACCES-swallow defect class this repo has been bitten by seven
    times: a permission error must not masquerade as "no sidecar", which would silently unblock a
    fallback resume onto the wrong rows.
    """
    if step is None:
        step = checkpoint_manager.latest_step()
    if step is None:
        return None
    path = epath.Path(checkpoint_manager.directory) / str(step) / LOADER_STATE_FILENAME
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    return json.loads(text)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
