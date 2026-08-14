"""The roots index must survive being moved to another machine.

v1 stored ABSOLUTE per-task directories, so an index pinned itself to the box that built it:
shipping byte-identical shards to a second machine produced a file naming directories that
existed nowhere on it, and the corpus had to be re-indexed by hand before it would load. v2
stores relative roots plus a `base` that is itself relative to the index file, so the tree
resolves wherever it is copied.

The v1 tests are not legacy trivia: a training run is in flight against a v1 index, and
changing what an already-launched run reads is not a refactor.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from openpi.training.pretrain_dataset import CORPUS_ROOT_ENV, _ordered_roots


def _corpus(tmp_path: pathlib.Path, ids=(1000, 1003, 812)) -> pathlib.Path:
    derived = tmp_path / "derived224"
    for t in ids:
        (derived / f"task_{t}").mkdir(parents=True, exist_ok=True)
    (tmp_path / "work").mkdir(exist_ok=True)
    return derived


def _write_v2(tmp_path: pathlib.Path, ids=(1000, 1003, 812)) -> pathlib.Path:
    _corpus(tmp_path, ids)
    idx = tmp_path / "work" / "roots.json"
    idx.write_text(json.dumps({
        "version": 2,
        "base": "../derived224",
        "roots": {str(t): f"task_{t}" for t in ids},
    }))
    return idx


# --- v2: portable ---------------------------------------------------------------------------

def test_v2_resolves_relative_to_the_index_file(tmp_path):
    got = _ordered_roots(_write_v2(tmp_path))
    assert [t for t, _ in got] == [812, 1000, 1003]           # sorted by task id
    for task_id, root in got:
        assert pathlib.Path(root).is_dir()
        assert pathlib.Path(root).name == f"task_{task_id}"


def test_v2_survives_the_whole_tree_moving(tmp_path):
    """THE POINT OF v2. Copy the corpus somewhere else -- it must still resolve, with no edit
    to the index and no environment variable."""
    idx = _write_v2(tmp_path)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (tmp_path / "derived224").rename(moved / "derived224")
    (tmp_path / "work").rename(moved / "work")

    got = _ordered_roots(moved / "work" / "roots.json")
    assert len(got) == 3
    for _t, root in got:
        assert pathlib.Path(root).is_dir(), f"{root} did not follow the move"


def test_env_override_wins_over_base(tmp_path, monkeypatch):
    """For the case v2's `base` cannot cover: an index separated from its corpus."""
    idx = _write_v2(tmp_path)
    other = tmp_path / "second_copy"
    for t in (1000, 1003, 812):
        (other / f"task_{t}").mkdir(parents=True)
    monkeypatch.setenv(CORPUS_ROOT_ENV, str(other))

    for _t, root in _ordered_roots(idx):
        assert pathlib.Path(root).parent == other


def test_sorted_by_task_id_not_json_order(tmp_path):
    """The concat index space and the row planner both depend on this order; disagreeing would
    train on rows attributed to the wrong task."""
    _corpus(tmp_path, (1000, 812))
    idx = tmp_path / "work" / "roots.json"
    idx.write_text(json.dumps({
        "version": 2, "base": "../derived224",
        "roots": {"1000": "task_1000", "812": "task_812"},   # deliberately out of order
    }))
    assert [t for t, _ in _ordered_roots(idx)] == [812, 1000]


# --- v1: still read, because a live run depends on it ---------------------------------------

def test_v1_absolute_mapping_still_works(tmp_path):
    derived = _corpus(tmp_path)
    idx = tmp_path / "work" / "roots_v1.json"
    idx.write_text(json.dumps({str(t): str(derived / f"task_{t}") for t in (1000, 1003, 812)}))

    got = _ordered_roots(idx)
    assert [t for t, _ in got] == [812, 1000, 1003]
    for _t, root in got:
        assert pathlib.Path(root).is_dir()


def test_v1_and_v2_of_the_same_corpus_agree_exactly(tmp_path):
    """The migration must be a no-op for what actually gets trained on."""
    derived = _corpus(tmp_path)
    v1 = tmp_path / "work" / "v1.json"
    v1.write_text(json.dumps({str(t): str(derived / f"task_{t}") for t in (1000, 1003, 812)}))
    v2 = _write_v2(tmp_path)

    got_v1 = [(t, str(pathlib.Path(r).resolve())) for t, r in _ordered_roots(v1)]
    got_v2 = [(t, str(pathlib.Path(r).resolve())) for t, r in _ordered_roots(v2)]
    assert got_v1 == got_v2


def test_a_relative_root_without_a_base_is_refused(tmp_path):
    """A bare mapping is read as v1, so a relative value in one has no meaning. Silently
    resolving it against the cwd would produce roots that exist only when the job happens to
    start in the right directory."""
    idx = tmp_path / "roots.json"
    idx.write_text(json.dumps({"1000": "task_1000"}))
    with pytest.raises(ValueError, match="no 'base' key"):
        _ordered_roots(idx)
