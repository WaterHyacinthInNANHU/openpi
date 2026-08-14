"""The sampler must reproduce the artifact EXACTLY -- it is the audit trail."""

import json

import numpy as np
import pytest

from openpi.training.schedule_sampler import ScheduleSampler


def _write(tmp_path, rows, meta=None):
    p = tmp_path / "s.npz"
    meta = meta or {"mode": "anneal", "total_steps": rows.shape[0], "batch": rows.shape[1]}
    np.savez(p, rows=rows.astype(np.int64), meta=np.array(json.dumps(meta)))
    return p


def test_rows_for_step_returns_the_artifact_row(tmp_path):
    rows = np.arange(40, dtype=np.int64).reshape(5, 8)
    s = ScheduleSampler(_write(tmp_path, rows))
    assert np.array_equal(s.rows_for_step(3), rows[3])
    assert s.total_steps == 5
    assert s.batch == 8


def test_resume_at_step_k_continues_from_row_k(tmp_path):
    rows = np.arange(40, dtype=np.int64).reshape(5, 8)
    s = ScheduleSampler(_write(tmp_path, rows))
    assert [s.rows_for_step(t).tolist() for t in range(2, 5)] == rows[2:].tolist()


def test_step_past_the_end_raises_rather_than_wrapping(tmp_path):
    """Wrapping would silently give the model a second pass nobody asked for."""
    rows = np.zeros((3, 4), dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    with pytest.raises(IndexError, match="beyond the schedule"):
        s.rows_for_step(3)


def test_batch_mismatch_raises(tmp_path):
    rows = np.zeros((3, 4), dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    with pytest.raises(ValueError, match="batch"):
        s.check_batch(8)


def test_iteration_is_the_artifact_flattened_in_order(tmp_path):
    """This is what makes the arm real: the loader's batch t must BE row t of the artifact.

    torch's DataLoader consumes the sampler as a flat index stream and cuts it into batches of
    `batch`, so yielding the artifact row-major reproduces the artifact's batches exactly.
    """
    rows = np.arange(40, dtype=np.int64).reshape(5, 8)
    s = ScheduleSampler(_write(tmp_path, rows))
    drawn = list(s)
    assert drawn == list(range(40))
    assert len(s) == 40
    assert [drawn[t * 8 : (t + 1) * 8] for t in range(5)] == rows.tolist()


def test_iteration_does_not_consume_the_schedule(tmp_path):
    """A second pass over the loader must replay the same rows, not an empty stream."""
    rows = np.arange(12, dtype=np.int64).reshape(3, 4)
    s = ScheduleSampler(_write(tmp_path, rows))
    assert list(s) == list(s)


def test_a_budget_longer_than_the_schedule_raises(tmp_path):
    """The torch loader restarts an exhausted sampler, so an over-long budget wraps silently."""
    rows = np.zeros((5, 4), dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    s.check_num_train_steps(5)
    with pytest.raises(ValueError, match="exceeds schedule"):
        s.check_num_train_steps(6)


def test_a_budget_shorter_than_the_schedule_warns_but_runs(tmp_path, caplog):
    rows = np.zeros((5, 4), dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    with caplog.at_level("WARNING"):
        s.check_num_train_steps(2)
    assert "short of schedule" in caplog.text


def test_row_beyond_the_dataset_raises(tmp_path):
    """A schedule built against a different corpus indexes frames this dataset does not have."""
    rows = np.array([[0, 1], [2, 99]], dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    s.check_dataset_rows(100)
    with pytest.raises(ValueError, match="outside the dataset"):
        s.check_dataset_rows(50)


def test_one_dimensional_artifact_is_rejected(tmp_path):
    p = tmp_path / "s.npz"
    np.savez(p, rows=np.arange(8, dtype=np.int64), meta=np.array(json.dumps({"mode": "drop"})))
    with pytest.raises(ValueError, match="expected 2-D"):
        ScheduleSampler(p)


def test_a_float_rows_array_is_rejected_rather_than_silently_truncated(tmp_path):
    """`.astype(np.int64)` truncates instead of raising -- a wrong row is not distinguishable
    from a right one once training starts, so a float artifact must fail at load time."""
    p = tmp_path / "s.npz"
    np.savez(p, rows=np.arange(8, dtype=np.float64).reshape(2, 4), meta=np.array(json.dumps({"mode": "drop"})))
    with pytest.raises(ValueError, match="integer"):
        ScheduleSampler(p)


def test_a_negative_row_index_raises(tmp_path):
    """Torch indexes a negative int as `from the end`, so a negative row would silently draw
    from the dataset tail instead of raising -- catch it here, at loader construction."""
    rows = np.array([[0, 1], [2, -1]], dtype=np.int64)
    s = ScheduleSampler(_write(tmp_path, rows))
    with pytest.raises(ValueError, match="negative"):
        s.check_dataset_rows(100)
