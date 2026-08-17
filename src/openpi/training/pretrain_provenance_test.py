"""`check_artifact_provenance` had no test at all, and three ways to pass without checking.

It is the guard standing between an arm and training on another corpus's weights -- a failure that
converges, reports under the arm's name, and is invisible afterwards. It shipped uncovered, and an
audit reproduced all three holes:

  1. only `corpus_fingerprint` was compared; the `formula_fingerprint` was PRINTED, so a stamp whose
     formula fields contradicted its own fingerprint was accepted while the log said "provenance OK";
  2. an unreadable roots index made it `return` silently -- with a stamp present, which its own
     docstring calls always fatal;
  3. a legacy v1 index (no `base`) did the same.

The corpus fingerprint is monkeypatched: these tests are about the guard's CONTROL FLOW, and a real
fingerprint would need a corpus on disk, which would make them a slow integration test of something
else. `axis/dataset/tests` covers fingerprinting itself.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from openpi.training import pretrain_dataset as pd

CORPUS_FP = "sha256:deadbeef"


@pytest.fixture
def roots_index(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "corpus").mkdir()
    idx = tmp_path / "artifacts" / "roots.json"
    idx.parent.mkdir()
    idx.write_text(json.dumps({"version": 2, "base": "../corpus", "roots": {"1000": "task_1000"}}))
    return idx


@pytest.fixture(autouse=True)
def _fixed_fingerprint(monkeypatch):
    import axis.dataset.artifact_provenance as ap

    monkeypatch.setattr(ap, "corpus_fingerprint", lambda corpus: CORPUS_FP)


def _stamp(reward_id="v2", params=None, **overrides) -> dict:
    import axis.dataset.artifact_provenance as ap

    params = params if params is not None else {"tau": 10.0, "cap": 2.0, "horizon": 10}
    prov = {
        "corpus_fingerprint": CORPUS_FP,
        "formula_fingerprint": ap.formula_fingerprint(reward_id, **params),
        "formula_version": ap.FORMULA_VERSION,
        "reward_id": reward_id,
        "params": params,
    }
    prov.update(overrides)
    return {"provenance": prov, "weights": {}}


def test_a_correct_stamp_passes(roots_index, capsys):
    pd.check_artifact_provenance(_stamp(), roots_index)
    assert "provenance OK" in capsys.readouterr().out


def test_a_different_corpus_is_refused(roots_index):
    from axis.dataset.artifact_provenance import ProvenanceError

    with pytest.raises(ProvenanceError, match="DIFFERENT corpus"):
        pd.check_artifact_provenance(_stamp(corpus_fingerprint="sha256:other"), roots_index)


def test_a_formula_fingerprint_that_contradicts_its_own_fields_is_refused(roots_index):
    """HOLE 1. The stamp says reward v2 with these params; the fingerprint says otherwise."""
    from axis.dataset.artifact_provenance import ProvenanceError

    with pytest.raises(ProvenanceError, match="SELF-INCONSISTENT"):
        pd.check_artifact_provenance(_stamp(formula_fingerprint="sha256:notthisformula"), roots_index)


def test_a_half_written_formula_stamp_is_refused(roots_index):
    from axis.dataset.artifact_provenance import ProvenanceError

    with pytest.raises(ProvenanceError, match="INCOMPLETE formula stamp"):
        pd.check_artifact_provenance(_stamp(reward_id=None), roots_index)


def test_an_unreadable_roots_index_is_fatal_once_a_stamp_exists(tmp_path):
    """HOLE 2. Previously returned silently, leaving the run unverified and looking verified."""
    with pytest.raises(RuntimeError, match="cannot be read"):
        pd.check_artifact_provenance(_stamp(), tmp_path / "does_not_exist.json")


def test_a_legacy_index_without_base_is_fatal_once_a_stamp_exists(tmp_path):
    """HOLE 3. Same silent return for a v1 absolute-path index."""
    idx = tmp_path / "roots_v1.json"
    idx.write_text(json.dumps({"1000": "/abs/task_1000"}))
    with pytest.raises(RuntimeError, match="no 'base' key"):
        pd.check_artifact_provenance(_stamp(), idx)


def test_an_unstamped_artifact_is_still_allowed_through_with_a_warning(roots_index, capsys):
    """Deliberate: corpora predate stamping and refusing them would strand existing arms."""
    pd.check_artifact_provenance({"weights": {}}, roots_index)
    assert "no provenance stamp" in capsys.readouterr().out
