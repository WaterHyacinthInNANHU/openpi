"""AWR weights must land on the rows they were computed for, or the experiment is noise.

The two supervision arms differ ONLY in which weights file they read. If the join between the
offline artifact and the trainer's flat index space is wrong, each arm trains on a permutation of
its own weights -- which is neither arm, produces no error, and looks exactly like a normal run.
So the tests here are mostly about the JOIN, not the arithmetic.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from openpi.training import pretrain_sampler
from openpi.training.pretrain_dataset import plan_rows_and_weights_from_roots


def _corpus(tmp_path: pathlib.Path, spec: dict[int, list[int]]) -> pathlib.Path:
    """A fake corpus: {task_id: [episode lengths]} -> meta/episodes parquet per task."""
    import pandas as pd

    derived = tmp_path / "derived224"
    for tid, lengths in spec.items():
        d = derived / f"task_{tid}" / "meta" / "episodes" / "chunk-000"
        d.mkdir(parents=True)
        start = 0
        rows = []
        for ep, n in enumerate(lengths):
            rows.append({"episode_index": ep, "dataset_from_index": start, "length": n})
            start += n
        pd.DataFrame(rows).to_parquet(d / "file-000.parquet")
    idx = tmp_path / "roots.json"
    idx.write_text(json.dumps({
        "version": 2, "base": "derived224",
        "roots": {str(t): f"task_{t}" for t in spec},
    }))
    return idx


def _ranges(tmp_path, spec, frac=1.0):
    r = {}
    for tid, lengths in spec.items():
        for ep, n in enumerate(lengths):
            r[pretrain_sampler.episode_key(tid, ep)] = [[0, int(n * frac) - 1]]
    p = tmp_path / "ranges.json"
    p.write_text(json.dumps(r))
    return p


def _weights(tmp_path, spec, fn):
    w = {pretrain_sampler.episode_key(t, e): [float(fn(t, e, i)) for i in range(n)]
         for t, lengths in spec.items() for e, n in enumerate(lengths)}
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"version": 1, "reward_id": "test", "weights": w}))
    return p


SPEC = {800: [10, 12], 1000: [8], 900: [15]}     # deliberately NOT in sorted-string order


def test_weights_land_on_the_rows_they_belong_to(tmp_path):
    """THE test. Encode (task, episode, frame) in the weight itself, then decode it back from
    the flat array and check every row agrees."""
    idx, rng = _corpus(tmp_path, SPEC), _ranges(tmp_path, SPEC)
    wp = _weights(tmp_path, SPEC, lambda t, e, i: t * 10_000 + e * 100 + i)
    rows, weights = plan_rows_and_weights_from_roots(idx, rng, wp)

    # rebuild the same flat space independently and check the decode
    entries = []
    for tid in sorted(SPEC):                      # int order, as _ordered_roots does
        lengths = SPEC[tid]
        start, eps = 0, {}
        for ep, n in enumerate(lengths):
            eps[ep] = (start, n); start += n
        entries.append((tid, eps, start))
    subs, _ = pretrain_sampler.build_subdatasets(entries)
    decode = {}
    for sub in subs:
        for ep, (local, n) in sub.episodes.items():
            for i in range(n):
                decode[sub.global_base + local + i] = sub.task_id * 10_000 + ep * 100 + i

    assert len(rows) == len(weights)
    for r, w in zip(rows, weights):
        assert float(w) == pytest.approx(decode[int(r)]), f"row {r} got the wrong episode's weight"


def test_task_ids_are_ordered_AS_INTEGERS(tmp_path):
    """`"1000" < "800"` as strings. If the builder and the trainer disagree on this, every task's
    weights shift onto another task and nothing errors."""
    idx, rng = _corpus(tmp_path, SPEC), _ranges(tmp_path, SPEC)
    wp = _weights(tmp_path, SPEC, lambda t, e, i: t)
    rows, weights = plan_rows_and_weights_from_roots(idx, rng, wp)
    # the first planned row must belong to task 800, not task 1000
    assert float(weights[0]) == 800.0


def test_a_missing_episode_raises(tmp_path):
    """A partial artifact must not be filled in by a reader default -- that is how a row silently
    trains at weight 0 (contributing nothing) or 1 (unweighted)."""
    idx, rng = _corpus(tmp_path, SPEC), _ranges(tmp_path, SPEC)
    w = json.loads(_weights(tmp_path, SPEC, lambda t, e, i: 1.0).read_text())
    del w["weights"][pretrain_sampler.episode_key(900, 0)]
    p = tmp_path / "partial.json"; p.write_text(json.dumps(w))
    with pytest.raises(ValueError, match="no weights"):
        plan_rows_and_weights_from_roots(idx, rng, p)


def test_a_wrong_length_episode_raises(tmp_path):
    """Built against a different corpus: the lengths are the cheapest place to notice."""
    idx, rng = _corpus(tmp_path, SPEC), _ranges(tmp_path, SPEC)
    w = json.loads(_weights(tmp_path, SPEC, lambda t, e, i: 1.0).read_text())
    w["weights"][pretrain_sampler.episode_key(900, 0)] = [1.0] * 99
    p = tmp_path / "bad.json"; p.write_text(json.dumps(w))
    with pytest.raises(ValueError, match="different corpus"):
        plan_rows_and_weights_from_roots(idx, rng, p)


def test_weights_cover_exactly_the_planned_rows_under_partial_ranges(tmp_path):
    """The idle filter keeps only part of each episode. Weights are computed on the FULL episode
    curve (the action chunk spans real frames regardless of the filter), then indexed by the
    kept rows -- so coverage must hold after restriction, not before."""
    idx = _corpus(tmp_path, SPEC)
    rng = _ranges(tmp_path, SPEC, frac=0.5)
    wp = _weights(tmp_path, SPEC, lambda t, e, i: 1.0 + i / 100.0)
    rows, weights = plan_rows_and_weights_from_roots(idx, rng, wp)
    assert len(rows) == len(weights) > 0
    assert np.isfinite(weights).all()


def test_the_offline_key_matches_the_trainer_key():
    """The offline builder duplicates `episode_key` because it cannot import the online tier.
    Pin the two together so the duplication cannot drift."""
    import sys
    sys.path.insert(0, "/bigdata/jlilab/myan035/workspace/paper_writing/axis_ai/dataset_preview/.claude/worktrees/pretrain-dataconfig")
    from axis.dataset.awr_weights import episode_key as offline_key

    for t, e in ((800, 0), (1424, 17), (900, 3)):
        assert offline_key(t, e) == pretrain_sampler.episode_key(t, e)
