"""Guards the SLB cfg conditioning factory against caller/dataclass drift.

WHY THIS FILE EXISTS
    A cleanup pass removed `SlbCfgConditioning.train` but left `train=True` at the only
    call site in `build_conditioning`. Every model test still passed -- none of them
    construct the transform -- so the breakage was invisible until the arm would have
    started training ~22 h later. `AxisFrankaSlbDataConfig` catches the TypeError and
    re-raises loudly, so it would have failed fast rather than silently training as
    `vanilla`, but only after burning a queue slot.

    The point of these tests is that the FACTORY is exercised, not just the dataclass.
"""

from __future__ import annotations

import numpy as np
import pytest

import openpi.training.slb_cfg as slb_cfg


class _FakeSidecar:
    """Minimal stand-in exposing the public VariantSidecar surface build_cfg_labels uses."""

    variant = "cfg"

    def __init__(self, per_attempt, t_start):
        self._pa, self._ts = per_attempt, t_start

    def window_ids(self):
        return [(aid, w) for aid, lab in self._pa.items() for w in range(len(lab))]

    def label(self, attempt_id, window):
        return int(self._pa[int(attempt_id)][int(window)])

    def t_start(self, attempt_id):
        return np.asarray(self._ts[int(attempt_id)], dtype=float)


class _FakeRef:
    def __init__(self, episode_index, frame_count):
        self.episode_index, self.frame_count = episode_index, frame_count


class _FakeJoin:
    def __init__(self, mapping):
        self._m = mapping

    def episode_for(self, attempt_id):
        return self._m.get(int(attempt_id))


def test_build_conditioning_constructs_the_transform(monkeypatch):
    """The regression that motivated this file: factory kwargs must match the dataclass."""
    sidecar = _FakeSidecar({7: [0, 1]}, {7: [0.0, 1.0]})
    monkeypatch.setattr(slb_cfg, "build_cfg_labels", lambda *a, **k: {(0, 0): 0, (0, 15): 1})
    monkeypatch.setattr(
        "axis.dataset.sidecar_reader.VariantSidecar.load",
        classmethod(lambda cls, *a, **k: sidecar),
    )
    monkeypatch.setattr(
        "axis.episode.join_index.JoinIndex.from_manifest",
        classmethod(lambda cls, *a, **k: _FakeJoin({7: _FakeRef(0, 100)})),
    )
    t = slb_cfg.build_conditioning(task_id=1, sidecar_root="x", manifest_path="y")
    assert isinstance(t, slb_cfg.SlbCfgConditioning)
    assert t.cond_dropout == slb_cfg.DEFAULT_COND_DROPOUT


def test_build_conditioning_raises_when_no_labels(monkeypatch):
    """A cfg run without conditioning is byte-identical to vanilla, so an empty label map
    must fail loudly rather than put a duplicate arm in the bake-off."""
    monkeypatch.setattr(slb_cfg, "build_cfg_labels", lambda *a, **k: {})
    monkeypatch.setattr(
        "axis.dataset.sidecar_reader.VariantSidecar.load",
        classmethod(lambda cls, *a, **k: _FakeSidecar({}, {})),
    )
    monkeypatch.setattr(
        "axis.episode.join_index.JoinIndex.from_manifest",
        classmethod(lambda cls, *a, **k: _FakeJoin({})),
    )
    with pytest.raises(ValueError, match="no .*labels"):
        slb_cfg.build_conditioning(task_id=1, sidecar_root="x", manifest_path="y")


def test_build_conditioning_requires_a_manifest():
    with pytest.raises(ValueError, match="manifest"):
        slb_cfg.build_conditioning(task_id=1, sidecar_root="x", manifest_path=None)


def test_frame_index_is_clamped_into_the_episode():
    """A window whose t_start rounds past the last frame must key the LAST row, not a row
    the episode does not have -- otherwise the tail of every episode loses conditioning."""
    sidecar = _FakeSidecar({7: [0, 0]}, {7: [0.0, 99.0]})  # 99 s * 15 fps = frame 1485
    labels = slb_cfg.build_cfg_labels(sidecar, _FakeJoin({7: _FakeRef(3, 100)}), fps=15.0)
    assert max(fr for _, fr in labels) == 99


def test_windows_absent_from_the_join_are_skipped():
    labels = slb_cfg.build_cfg_labels(
        _FakeSidecar({7: [0]}, {7: [0.0]}), _FakeJoin({}), fps=15.0
    )
    assert labels == {}
