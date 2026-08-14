"""Read a precomputed index schedule and hand the loader the exact rows for each step.

ONLINE TIER. There is deliberately NO randomness here: the arm's sampling was decided offline
(`axis.dataset.build_index_schedule`), so "what did step k train on?" is answered by reading row k
of a file, and samples-seen is a fact on disk rather than a reconstruction from RNG state.

That makes an exact resume EXPRESSIBLE (`rows_for_step(k)`) but does not by itself deliver one:
openpi checkpoints no data-loader position (`checkpoints.restore_state` drops its `data_loader`
argument), so a resumed run replays the schedule from row 0 while the optimiser continues. Until
a start offset is plumbed through `create_data_loader`, a died arm is restarted clean -- which is
already what conf/experiments/onelayer_v3_stage1_arms.toml tells you to do, for the same reason.

The artifact holds an int64 `(total_steps, batch)` block of flat dataset indices plus a JSON
`meta` string. Row t IS the batch trained at step t, which is why this class is itself the torch
`Sampler`: torch consumes a sampler as a flat index stream and cuts it into batches of
`batch_size`, so yielding the artifact row-major reproduces its batches one for one. Anything
that reshuffles, wraps, pads or reshapes destroys that correspondence and with it the arm's
audit trail -- hence `check_batch` and `rows_for_step` raise rather than adapt.
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np
import torch


class ScheduleSampler(torch.utils.data.Sampler[int]):
    """The offline schedule, replayed verbatim as a torch row sampler."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            rows = z["rows"]
            if not np.issubdtype(rows.dtype, np.integer):
                # `.astype(np.int64)` would silently TRUNCATE a float artifact (e.g. a row index
                # that was never rounded) rather than fail -- and a truncated row is a WRONG row,
                # indistinguishable from a correct one once training starts.
                raise ValueError(
                    f"schedule {self.path} stores rows as dtype {rows.dtype}, not an integer "
                    f"dtype. Casting would silently truncate; regenerate the artifact with an "
                    f"integer rows array instead."
                )
            self._rows = rows.astype(np.int64)
            self.meta = json.loads(str(z["meta"]))
        if self._rows.ndim != 2:
            raise ValueError(f"schedule {self.path} has shape {self._rows.shape}, expected 2-D")

    @property
    def total_steps(self) -> int:
        return int(self._rows.shape[0])

    @property
    def batch(self) -> int:
        return int(self._rows.shape[1])

    def check_batch(self, batch: int) -> None:
        if batch != self.batch:
            raise ValueError(
                f"schedule {self.path} was built for batch {self.batch}, but training requests "
                f"{batch}. The artifact IS the experiment's record of what was seen; regenerate it "
                f"rather than reshaping it here."
            )

    def check_num_train_steps(self, num_train_steps: int) -> None:
        """The run must not outlast its schedule.

        `rows_for_step` refuses to step past the end, but the torch loader never asks it: it just
        restarts an exhausted sampler, so a budget longer than the artifact replays the schedule
        from row 0 and quietly grants the extra pass this design exists to prevent. A budget
        SHORTER than the artifact is merely a truncated run, but it invalidates the coverage
        numbers (epochs, unique frames) the artifact reports, so it is logged loudly.
        """
        if num_train_steps > self.total_steps:
            raise ValueError(
                f"num_train_steps={num_train_steps} exceeds schedule {self.path} "
                f"({self.total_steps} steps). The loader would restart the schedule from row 0 "
                f"and train an extra pass nobody asked for; rebuild the artifact with "
                f"--total-steps {num_train_steps}, or shorten the run."
            )
        if num_train_steps < self.total_steps:
            logging.warning(
                "num_train_steps=%d is short of schedule %s (%d steps): the run will see only "
                "%.1f%% of the scheduled batches, so the artifact's epochs/unique-frames numbers "
                "do NOT describe it.",
                num_train_steps, self.path, self.total_steps,
                100.0 * num_train_steps / self.total_steps,
            )

    def check_dataset_rows(self, n_dataset_rows: int, roots_index: str | None = None) -> None:
        """The dataset must be the very corpus this schedule was built against.

        The artifact stores flat indices into the concatenated pretrain dataset, so a schedule
        built against a different corpus points at frames that are either absent (an IndexError
        deep inside a worker, hours in) or -- worse, if the corpus merely GREW or its task set
        shifted -- present but belonging to some other episode. Bounds alone cannot see that
        case: every index stays in range while every index means something else.

        So the binding is on SIZE, not on range: the generator records the concatenated row
        count it enumerated as `meta["n_rows"]`, and that number must equal `len(dataset)`
        exactly. Row identity in this corpus is positional (episodes concatenated in the roots
        index's order), so any insertion, removal or reorder that could move a row also moves
        the total -- with the sole exception of an equal-sized swap, which no operation on this
        pipeline performs. `roots_index` is passed in only to name the other side of the
        disagreement in the error; the sampler never reads it.

        The order below is deliberate: negative/out-of-range rows are corruption signatures with
        their own remedies, so they report themselves first; the size equality is the catch-all
        that closes the grew-silently case none of them can see.
        """
        lo = int(self._rows.min(initial=0))
        if lo < 0:
            # torch indexes a negative int as "from the end", so a negative row would silently
            # draw from the dataset TAIL instead of raising -- the opposite of the bounds check
            # below, and just as capable of training on the wrong frame.
            raise ValueError(
                f"schedule {self.path} contains a negative row index ({lo}). Torch would wrap "
                f"that to the dataset tail rather than raise; the artifact is corrupt or was "
                f"built against the wrong indexing convention -- rebuild it."
            )
        hi = int(self._rows.max(initial=-1))
        if hi >= n_dataset_rows:
            raise ValueError(
                f"schedule {self.path} draws row {hi}, which is outside the dataset "
                f"({n_dataset_rows} rows). The schedule was built against a different corpus; "
                f"rebuild it against this roots index rather than truncating it."
            )
        n_rows = self.meta.get("n_rows")
        if n_rows is None:
            raise ValueError(
                f"schedule {self.path} has no meta['n_rows'], so there is nothing to bind it to "
                f"the corpus it was built against -- an artifact predating that field cannot be "
                f"told apart from one built on a different corpus. Rebuild it with "
                f"axis.dataset.build_index_schedule."
            )
        if int(n_rows) != n_dataset_rows:
            source = roots_index or "the configured roots index"
            raise ValueError(
                f"corpus mismatch: schedule {self.path} was built against a dataset of "
                f"{int(n_rows)} rows (its own meta['n_rows']), but the dataset assembled from "
                f"{source} has {n_dataset_rows} rows. The schedule's indices are POSITIONS in "
                f"the concatenated corpus, so with a different row count they name different "
                f"frames -- and a corpus that merely grew would pass every bounds check while "
                f"training on the wrong episodes. Rebuild the schedule against this roots "
                f"index, or point the run at the roots index the schedule was built from."
            )

    def rows_for_step(self, t: int) -> np.ndarray:
        if not 0 <= t < self.total_steps:
            raise IndexError(
                f"step {t} is beyond the schedule ({self.total_steps} steps). Wrapping would give "
                f"the model an extra pass nobody asked for; extend num_train_steps deliberately."
            )
        return self._rows[t]

    def __len__(self) -> int:
        return self.total_steps * self.batch

    def __iter__(self):
        # Row-major, no generator, no epoch counter: the order is the artifact's.
        yield from self._rows.reshape(-1).tolist()
