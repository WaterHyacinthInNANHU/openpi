"""Dataloader-position resume: a checkpointed sampler must continue the row permutation at the
exact consumed offset, not restart at epoch 0.

The one test that matters is COVERAGE IDENTITY: the split (checkpoint + resume) run must emit the
SAME row sequence, byte for byte and in the same order, as a single uninterrupted run. That is the
property the CFG arm's whole claim rests on (it draws exactly the round-1 control's rows in the
control's order) and the property `_check_quality_resume` / `_check_schedule_resume` refuse to
break by restarting at epoch 0. These tests assert the feature makes that break unnecessary.

Covers exactly the two samplers whose guards forbid resume for a COVERAGE reason:
`slb_variant_sampler.RowSampler` (the stage-1 CFG arm and the uniform pretrain arms) and
`schedule_sampler.ScheduleSampler` (the drop/anneal arms). Stage-2's `PresentationSampler` is out
of scope: its guard protects per-example dropout, not coverage, and the task relaxes only the two
coverage guards.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.training.schedule_sampler import ScheduleSampler
from openpi.training.slb_variant_sampler import RowSampler


def _pull(sampler, n: int) -> list[int]:
    """Pull exactly `n` rows from a torch Sampler, re-`__iter__`-ing across epoch boundaries the
    way `TorchDataLoader.__iter__` does (a fresh `iter()` each time the previous one is exhausted).
    """
    out: list[int] = []
    while len(out) < n:
        for r in sampler:
            out.append(r)
            if len(out) >= n:
                return out[:n]
        # iterator exhausted (one epoch's worth) -> loop and re-iter, like the real loader
    return out[:n]


def _write_schedule(tmp_path, rows, n_rows=1000, name="s.npz"):
    p = tmp_path / name
    meta = {
        "mode": "anneal",
        "total_steps": int(rows.shape[0]),
        "batch": int(rows.shape[1]),
        "n_rows": int(n_rows),
    }
    np.savez(p, rows=rows.astype(np.int64), meta=np.array(json.dumps(meta)))
    return p


# ----------------------------------------------------------------------------------------------
# RowSampler
# ----------------------------------------------------------------------------------------------


def test_rowsampler_default_sequence_unchanged():
    """Backward compatibility: with no resume the emitted sequence is identical to before, so
    every existing (non-resumed) run is bit-for-bit unchanged."""
    rows = np.arange(50, 100, dtype=np.int64)
    a = RowSampler(rows, seed=7)
    # Two epochs' worth, pulled the way the loader pulls it.
    got = _pull(a, 2 * len(rows))
    # The original semantics: epoch e permutation = randperm(N, seed+e), mapped through rows.
    import torch

    expected = []
    for e in range(2):
        g = torch.Generator()
        g.manual_seed(7 + e)
        perm = torch.randperm(len(rows), generator=g)
        expected.extend(np.asarray(rows)[perm.numpy()].tolist())
    assert got == expected


def test_rowsampler_state_dict_round_trip():
    rows = np.arange(200, dtype=np.int64)
    a = RowSampler(rows, seed=3)
    _pull(a, 250)  # one full epoch (200) + 50 into the next
    st = a.state_dict()
    assert st["seed"] == 3
    assert st["n_rows"] == 200
    assert st["epoch"] == 1
    assert st["consumed_in_epoch"] == 50

    b = RowSampler(rows, seed=3)
    b.load_state_dict(st)
    assert b.state_dict() == st


def test_rowsampler_coverage_identity_within_single_epoch():
    """k and N-k both inside epoch 0: the split run must equal the single run exactly."""
    rows = np.arange(1000, 2000, dtype=np.int64)
    N = 700  # < len(rows): stays inside epoch 0
    k = 400

    ref = _pull(RowSampler(rows, seed=11), N)

    a = RowSampler(rows, seed=11)
    first = _pull(a, k)
    st = a.state_dict()
    assert st["epoch"] == 0 and st["consumed_in_epoch"] == k

    b = RowSampler(rows, seed=11)
    b.load_state_dict(st)
    rest = _pull(b, N - k)

    assert first + rest == ref


def test_rowsampler_coverage_identity_across_epoch_boundary():
    """k and N-k span >1 epoch: the resumed run must pick up mid-epoch and continue into the next
    epoch's fresh permutation, reproducing the single run exactly."""
    rows = np.arange(300, dtype=np.int64)  # epoch length 300
    N = 800  # spans epochs 0,1,2
    k = 350  # 300 (epoch 0) + 50 into epoch 1

    ref = _pull(RowSampler(rows, seed=5), N)

    a = RowSampler(rows, seed=5)
    first = _pull(a, k)
    st = a.state_dict()
    assert st["epoch"] == 1 and st["consumed_in_epoch"] == 50

    b = RowSampler(rows, seed=5)
    b.load_state_dict(st)
    rest = _pull(b, N - k)

    assert first + rest == ref
    # And full coverage of the spanned epochs: every row of epochs 0 and 1 appears.
    assert sorted((first + rest)[:600]) == sorted(rows.tolist() * 2)


def test_rowsampler_load_state_dict_rejects_wrong_seed_and_row_count():
    rows = np.arange(100, dtype=np.int64)
    a = RowSampler(rows, seed=1)
    _pull(a, 30)
    st = a.state_dict()

    with pytest.raises(ValueError, match="seed"):
        RowSampler(rows, seed=2).load_state_dict(st)
    with pytest.raises(ValueError, match="row"):
        RowSampler(np.arange(101, dtype=np.int64), seed=1).load_state_dict(st)


# ----------------------------------------------------------------------------------------------
# ScheduleSampler
# ----------------------------------------------------------------------------------------------


def test_schedulesampler_default_sequence_unchanged(tmp_path):
    rows = np.arange(400, dtype=np.int64).reshape(50, 8)
    s = ScheduleSampler(_write_schedule(tmp_path, rows))
    assert list(s) == rows.reshape(-1).tolist()


def test_schedulesampler_state_dict_round_trip(tmp_path):
    rows = np.arange(400, dtype=np.int64).reshape(50, 8)
    p = _write_schedule(tmp_path, rows)
    s = ScheduleSampler(p)
    _pull(s, 123)
    st = s.state_dict()
    assert st["total_steps"] == 50 and st["batch"] == 8 and st["consumed"] == 123

    t = ScheduleSampler(p)
    t.load_state_dict(st)
    assert t.state_dict() == st


def test_schedulesampler_coverage_identity(tmp_path):
    rows = np.arange(400, dtype=np.int64).reshape(50, 8)
    p = _write_schedule(tmp_path, rows)
    total = 50 * 8
    k = 17 * 8  # resume at step 17

    ref = list(ScheduleSampler(p))
    assert len(ref) == total

    a = ScheduleSampler(p)
    first = _pull(a, k)
    st = a.state_dict()

    b = ScheduleSampler(p)
    b.load_state_dict(st)
    rest = _pull(b, total - k)

    assert first + rest == ref


def test_schedulesampler_load_state_dict_rejects_shape_mismatch(tmp_path):
    rows = np.arange(400, dtype=np.int64).reshape(50, 8)
    st = ScheduleSampler(_write_schedule(tmp_path, rows)).state_dict()
    st_bad = {**st, "batch": 16}
    with pytest.raises(ValueError, match="batch"):
        ScheduleSampler(_write_schedule(tmp_path, rows, name="s2.npz")).load_state_dict(st_bad)


# ----------------------------------------------------------------------------------------------
# Guard relaxation: resume is allowed ONLY when a valid loader_state sidecar is present.
# ----------------------------------------------------------------------------------------------

import openpi.training.data_loader as _data_loader  # noqa: E402


def test_quality_resume_without_sidecar_still_raises():
    """Backward compatibility / safety hinge: an old checkpoint with no sidecar must still refuse
    to resume, so nothing silently regresses to the epoch-0 restart."""
    with pytest.raises(ValueError, match="resum"):
        _data_loader._check_quality_resume("quality_v2.npz", resuming=True, loader_state=None, seed=0)


def test_quality_resume_with_valid_sidecar_is_allowed():
    st = {"kind": "row", "seed": 43, "n_rows": 1319784, "num_samples": 1319784,
          "epoch": 0, "consumed_in_epoch": 1280000}
    _data_loader._check_quality_resume("quality_v2.npz", resuming=True, loader_state=st, seed=43)


def test_quality_resume_with_mismatched_seed_raises():
    st = {"kind": "row", "seed": 43, "n_rows": 1319784, "num_samples": 1319784,
          "epoch": 0, "consumed_in_epoch": 1280000}
    with pytest.raises(ValueError, match="seed"):
        _data_loader._check_quality_resume("quality_v2.npz", resuming=True, loader_state=st, seed=7)


def test_schedule_resume_without_sidecar_still_raises():
    with pytest.raises(ValueError, match="resum"):
        _data_loader._check_schedule_resume("s.npz", resuming=True, loader_state=None)


def test_schedule_resume_with_valid_sidecar_is_allowed():
    st = {"kind": "schedule", "total_steps": 50, "batch": 8, "n_rows": 1000, "consumed": 136}
    _data_loader._check_schedule_resume("s.npz", resuming=True, loader_state=st)


def test_schedule_resume_not_resuming_is_a_noop():
    _data_loader._check_schedule_resume("s.npz", resuming=False)  # legacy 2-arg call still works
    _data_loader._check_quality_resume("quality_v2.npz", resuming=False)  # legacy 2-arg call
