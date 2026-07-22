"""Row mapping: legacy vs render-aligned. Pins the measured +2 offset."""
import numpy as np
import openpi.training.slb_variant_sampler as S


class _Sidecar:
    variant = "vanilla"
    def __init__(self, t):
        self._t = np.asarray(t, float)
        self._per_attempt = {7: np.ones(len(self._t))}
    def keep_mask(self, aid): return np.ones(len(self._t), bool)
    def t_start(self, aid): return self._t
    def delta(self, aid, w): return 0.0


class _Ref:
    def __init__(self, ep): self.episode_index = ep


class _Join:
    def attempt_ids(self): return [7]
    def episode_for(self, aid): return _Ref(0)


def _rows(aligned, t):
    rows, _ = S.plan_indices({0: 1000}, _Sidecar(t), _Join(), "vanilla",
                             fps=15.0, ep_len={0: 500}, render_aligned_rows=aligned)
    return rows


def test_offset_constant_is_two():
    assert S.RENDER_FRAME_OFFSET == 2


def test_aligned_rule_is_relative_to_first_sim_time():
    """Traces mostly start at t=0.2s, so the absolute rule is offset by that head."""
    t = [0.2, 0.4, 0.6]
    assert list(_rows(True, t)) == [1002, 1005, 1008]


def test_legacy_rule_lands_one_frame_late():
    """round(0.2*15)=3 vs the true 2 -- reproduces the defect this flag fixes."""
    t = [0.2, 0.4, 0.6]
    assert list(_rows(False, t)) == [1003, 1006, 1009]
    assert (_rows(False, t) - _rows(True, t)).tolist() == [1, 1, 1]


def test_default_is_render_aligned():
    """The correct rule must be what you get without asking for it."""
    t = [0.2, 0.4, 0.6]
    rows, _ = S.plan_indices({0: 1000}, _Sidecar(t), _Join(), "vanilla",
                             fps=15.0, ep_len={0: 500})
    assert list(rows) == list(_rows(True, t))


def test_trace_starting_at_zero_agrees_with_legacy_plus_offset():
    t = [0.0, 0.2, 0.4]
    assert list(_rows(True, t)) == [1002, 1005, 1008]


def test_rows_stay_clamped_inside_the_episode():
    t = [0.0, 100.0]
    assert _rows(True, t).max() <= 1000 + 500 - 1
