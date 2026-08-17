"""Stage 2's CFG twin must differ from `pi05_libero_axisinit_paper` in exactly ONE treatment.

Stage 2 is the leg the experiment holds fixed across arms -- `conf/experiments/
onelayer_v3_stage2_libero.toml` says so in its first line -- so every field that moves here is a
confound the CFG row carries and the other five arms do not. This file therefore diffs the twin
against its parent FIELD BY FIELD rather than asserting a handful of values, so a future edit to
the parent that does not reach the twin fails loudly. That diff is only load-bearing because the
twin is a second literal in `_CONFIGS`; a `dataclasses.replace` of the parent would make it pass
vacuously, which is why the registry entry is written out in full.

The `LiberoQualityConditioning` behaviour tested here is the maths and the guards. That the arm is
LIVE -- that a real stage-2 batch's tokenized prompt decodes to `Quality: 5` exactly once -- is
`data_loader_cfg_stage2_test.py`, because construction-level assertions cannot see an inert arm
(round 1 shipped one three times, and Task 6's first real failure was an arm that built, ran, and
produced the control's prompts).
"""

from __future__ import annotations

import dataclasses

import pytest

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import slb_cfg
from openpi.training.quality_conditioning import DROP_COMPONENT_STAGE2
from openpi.training.quality_conditioning import DROP_WHOLE_STAGE2
from openpi.training.quality_conditioning import N_BINS
from openpi.training.quality_conditioning import PRESENTATION_KEY
from openpi.training.quality_conditioning import LiberoQualityConditioning

PARENT = "pi05_libero_axisinit_paper"
TWIN = "pi05_libero_axisinit_paper_cfg"

# 1 - 0.85*0.95. Held equal to `axis.dataset.quality_labels.TAGGED_FRACTION`, which the offline
# suite asserts against the same arithmetic; the two tiers cannot import each other.
TAGGED_FRACTION = (1.0 - DROP_WHOLE_STAGE2) * (1.0 - DROP_COMPONENT_STAGE2)


def _row(ep: int, fr: int, prompt: str = "pick up the black bowl", presentation: int = 0) -> dict:
    """A sample as the transform sees it -- including the presentation counter
    `PresentationKeyedDataset` injects, which the dropout is keyed on alongside the row."""
    return {
        "prompt": prompt,
        "episode_index": ep,
        "frame_index": fr,
        PRESENTATION_KEY: presentation,
    }


# --- the twin is registered and is its parent in every field but the treatment -------------------


def test_the_twin_is_registered():
    assert _config.get_config(TWIN).name == TWIN


def test_the_twin_differs_from_its_parent_in_exactly_the_name_and_the_data():
    """Catches a warmup, LR, EMA, batch, budget, fsdp or weight-loader difference sneaking in --
    each of which would make the CFG row incomparable to the other five arms for a reason nothing
    in a training log, a checkpoint or the write-up would report."""
    parent, twin = _config.get_config(PARENT), _config.get_config(TWIN)
    differing = {
        f.name for f in dataclasses.fields(parent) if getattr(parent, f.name) != getattr(twin, f.name)
    }
    assert differing == {"name", "data"}, f"unexpected stage-2 differences: {differing}"


def test_the_twins_data_factory_differs_only_in_the_tag_and_the_stats_binding():
    """`quality_tag` is the treatment. `norm_stats_from` is the opposite of a treatment: it is
    what makes the twin READ the parent's norm stats instead of resolving to
    `<assets_base_dir>/pi05_libero_axisinit_paper_cfg/...`, which exists nowhere -- the LIBERO
    stage-2 stats were computed once on the training box and are published nowhere. Recomputing
    them per-arm would be the second parity break, so the sharing is asserted, not incidental."""
    parent, twin = _config.get_config(PARENT).data, _config.get_config(TWIN).data
    differing = {
        f.name for f in dataclasses.fields(parent) if getattr(parent, f.name) != getattr(twin, f.name)
    }
    assert differing == {"quality_tag", "norm_stats_from"}, f"unexpected data differences: {differing}"


def test_the_twin_reads_the_parents_norm_stats_directory(tmp_path):
    """The binding, resolved the way a run resolves it -- under whatever `--assets-base-dir` is in
    force, not the default. Catches the twin quietly acquiring its own stats directory."""
    parent, twin = _config.get_config(PARENT), _config.get_config(TWIN)
    assert twin.data.norm_stats_dir(tmp_path / TWIN) == parent.data.norm_stats_dir(tmp_path / PARENT)
    assert PARENT in str(twin.data.norm_stats_dir(tmp_path / TWIN))


def test_the_parent_is_still_untouched_by_the_new_fields():
    """The five non-CFG arms must keep the untagged stage 2 exactly as it was."""
    for name in (PARENT, "pi05_libero", "pi05_libero_axisinit"):
        data = _config.get_config(name).data
        assert data.quality_tag is None
        assert data.norm_stats_from is None


# --- the tag reaches the pipeline, at the head of the repack group -------------------------------


def _repack_inputs(cfg) -> list:
    return list(cfg.data.create(cfg.assets_dirs, cfg.model).repack_transforms.inputs)


def test_the_twin_heads_the_repack_group_with_the_conditioning():
    """THE INERT-ARM TEST at the config seam: a field added to the factory but never forwarded
    into `create()` would leave every other assertion here green and train the plain control.

    Position matters as much as presence -- `RepackTransform` drops `episode_index`/`frame_index`,
    so a conditioning entry AFTER it could never key its dropout draw.
    """
    twin = _config.get_config(TWIN)
    inputs = _repack_inputs(twin)
    assert isinstance(inputs[0], LiberoQualityConditioning)
    assert isinstance(inputs[1], _transforms.RepackTransform)
    # Exactly ONE conditioning entry, still the claim this length check was protecting: two of
    # them give `"...\nQuality: 5\nQuality: 5"`. Spelled by COUNTING the conditioning rather than
    # by the group's length, which is what let this survive the group gaining and then losing
    # `Rotate180Images`: it was added 2026-08-14 for the box-local inverted build and dropped
    # 2026-08-17 when stage 2 moved to PI's official UPRIGHT build, where no rotation is needed.
    assert sum(isinstance(t, LiberoQualityConditioning) for t in inputs) == 1
    assert [type(t).__name__ for t in inputs] == [
        "LiberoQualityConditioning",
        "RepackTransform",
    ]


def test_the_twin_tags_the_quality_inference_asks_for():
    """D9: stage 2 is uniformly expert data, so the conditional branch is anchored at the constant
    value CFG conditions on at eval. A twin tagging some other bin would train a branch no eval
    prompt can reach."""
    twin = _config.get_config(TWIN)
    assert twin.data.quality_tag == slb_cfg.INFER_QUALITY
    assert _repack_inputs(twin)[0].q_ep == slb_cfg.INFER_QUALITY


def test_the_twin_keeps_stage_ones_dropout_rates():
    """Both stages must train the unconditional branch at the same marginal, or they disagree
    about how often the model sees a bare prompt for no stated reason."""
    head = _repack_inputs(_config.get_config(TWIN))[0]
    assert (head.drop_whole, head.drop_component) == (DROP_WHOLE_STAGE2, DROP_COMPONENT_STAGE2)


def test_the_parent_repacks_exactly_what_it_repacked_before():
    """The other half of the inert test: the untagged parent carries NO conditioning, and repacks
    exactly what `pi05_libero` repacks. Catches a transform that tags unconditionally, which would
    silently condition all five other arms too.

    From 2026-08-14 to 2026-08-17 the parent's group also carried `Rotate180Images`, because both
    stage-2 legs read the box-local build, which stored frames 180 degrees from the eval client.
    Stage 2 now reads PI's official UPRIGHT build, so `rotation_needed()` is False and the group is
    upstream's again, exactly. The rotation was on BOTH legs while it existed and is on NEITHER
    now, so the "conditioning is the only difference" claim never depended on it.
    """
    parent, upstream = _config.get_config(PARENT), _config.get_config("pi05_libero")
    inputs = _repack_inputs(parent)
    assert not any(isinstance(t, LiberoQualityConditioning) for t in inputs)
    assert isinstance(inputs[0], _transforms.RepackTransform)
    assert inputs[:1] == _repack_inputs(upstream), "the repack MAP drifted from upstream's"
    assert [type(t).__name__ for t in inputs] == ["RepackTransform"]


def test_the_rest_of_the_pipeline_is_the_parents():
    """The conditioning must be the ONLY pipeline difference: same data transforms, same model
    transforms, same action-sequence keys, same base DataConfig. Catches the delta-action flag or
    the prompt source drifting in alongside the tag."""
    parent, twin = _config.get_config(PARENT), _config.get_config(TWIN)
    p = parent.data.create(parent.assets_dirs, parent.model)
    t = twin.data.create(twin.assets_dirs, twin.model)
    assert p.data_transforms == t.data_transforms
    # The model transforms are `ModelTransformFactory()(model_config)` on both sides, and the
    # top-level diff above already pins `model` equal -- so they are equal by construction. Only
    # their SHAPE is compared here, because `TokenizePrompt` holds a `PaligemmaTokenizer` whose
    # instances compare by identity and would make an `==` fail for no reason that matters.
    assert [type(x).__name__ for x in p.model_transforms.inputs] == [
        type(x).__name__ for x in t.model_transforms.inputs
    ]
    assert parent.model == twin.model
    assert p.action_sequence_keys == t.action_sequence_keys
    assert p.prompt_from_task == t.prompt_from_task is True
    # ...and the repack group differs by exactly the one prepended entry.
    assert list(t.repack_transforms.inputs)[1:] == list(p.repack_transforms.inputs)


# --- the transform's guards ----------------------------------------------------------------------


@pytest.mark.parametrize("q", [0, -1, N_BINS + 1, 255])
def test_a_tag_outside_the_bins_is_refused_at_construction(q):
    """`quality_tag=0` reads like "off" and would instead emit `Quality: 0` -- a sixth condition
    the tokenizer has never seen and no eval prompt can match. Refused when the config is BUILT,
    not on the first batch of a 15-hour run."""
    with pytest.raises(ValueError, match="not a bin"):
        LiberoQualityConditioning(q_ep=q)


def test_a_tag_outside_the_bins_is_refused_through_the_registered_factory(tmp_path):
    """The construction guard must be REACHED from the config seam, not merely defined."""
    twin = _config.get_config(TWIN)
    with pytest.raises(ValueError, match="not a bin"):
        dataclasses.replace(twin.data, quality_tag=0).create(tmp_path, twin.model)


def test_zero_dropout_is_refused():
    """Stage 1's counterpart is `QualityTags`' refusal of an artifact with no NO_TAG rows: with
    no dropout the model never learns p(a | no tag) and guidance -- which subtracts an
    unconditional forward pass -- is undefined. Nothing online would notice."""
    with pytest.raises(ValueError, match="unconditional branch"):
        LiberoQualityConditioning(drop_whole=0.0, drop_component=0.0)


@pytest.mark.parametrize("kwargs", [{"drop_whole": 1.0}, {"drop_component": 1.2}, {"drop_whole": -0.1}])
def test_a_dropout_rate_outside_the_unit_interval_is_refused(kwargs):
    """A rate of 1 drops every row, leaving the CONDITIONAL branch untrained while the config
    still says the arm is conditioned."""
    with pytest.raises(ValueError, match="outside"):
        LiberoQualityConditioning(**kwargs)


def test_the_transform_raises_without_the_row_keys():
    """`RepackTransform` drops `episode_index`/`frame_index`, so a transform placed after it would
    key every row identically -- i.e. tag all of them or none of them, silently, for a whole run.
    Raising is the only way that shows up before 15 hours are spent."""
    t = LiberoQualityConditioning()
    with pytest.raises(KeyError, match="episode_index"):
        t({"prompt": "p", "frame_index": 3})
    with pytest.raises(KeyError, match="frame_index"):
        t({"prompt": "p", "episode_index": 3})


def test_the_transform_raises_without_the_presentation_counter():
    """Without the counter the draw is a pure function of the row, which over stage 2's ~5.7
    epochs is a FIXED 19.25% partition: the unconditional branch fit on one frozen ~65k rows seen
    5.7 times each, the conditional branch never seeing them. The realized tagged fraction still
    reads 0.8075 under that bug, so a raise is the only signal there is."""
    with pytest.raises(KeyError, match="quality_presentation"):
        LiberoQualityConditioning()({"prompt": "p", "episode_index": 1, "frame_index": 2})


# --- the presentation counter: the same row, dropped differently on different passes -------------


STAGE2_PRESENTATIONS = 6  # ceil(30_000 * 64 / 338_575) -- the real stage-2 budget's epoch count


def test_a_row_is_dropped_on_some_presentations_and_kept_on_others():
    """THE FIXED-PARTITION TEST. Keyed on the row alone, every row's fate is decided once and
    repeated for all ~5.7 epochs -- two disjoint datasets rather than a dropout. Over the real
    presentation count a nontrivial share of rows must flip, and both directions must occur.
    """
    t = LiberoQualityConditioning(seed=0)
    flipped = 0
    rows = [(e, f) for e in range(60) for f in range(20)]
    for ep, fr in rows:
        fates = {t.dropped(ep, fr, p) for p in range(STAGE2_PRESENTATIONS)}
        flipped += len(fates) == 2
    # Independent Bernoulli(0.1925) over 6 presentations: P(not all equal) = 1 - .1925^6 - .8075^6
    # = ~0.723. A row-keyed draw gives exactly 0.
    assert 0.60 < flipped / len(rows) < 0.85, f"{flipped}/{len(rows)} rows ever change fate"


def test_the_presentation_actually_moves_the_draw():
    """Catches a counter that is threaded through every signature and never mixed into the key --
    under which the argument exists, the tests read plausibly, and the bug is untouched."""
    t = LiberoQualityConditioning(seed=0)
    a = [t.dropped(e, f, 0) for e in range(40) for f in range(10)]
    b = [t.dropped(e, f, 1) for e in range(40) for f in range(10)]
    assert a != b
    assert sum(x != y for x, y in zip(a, b, strict=True)) > 10


def test_every_presentation_realizes_the_same_marginal():
    """The counter must not skew the rate it varies: each pass over the corpus still trains the
    unconditional branch at 1 - 0.85*0.95."""
    t = LiberoQualityConditioning(seed=0)
    for p in range(STAGE2_PRESENTATIONS):
        tagged = sum(not t.dropped(e, f, p) for e in range(120) for f in range(100))
        assert abs(tagged / 12_000 - TAGGED_FRACTION) < 0.02, f"presentation {p}"


def test_the_draw_is_still_recomputable_from_the_seed_alone():
    """The D5 property the counter must not cost: a fresh instance of the same config, given the
    row and the presentation, answers identically. Nothing is carried on the object."""
    t = LiberoQualityConditioning(seed=0)
    for p in range(STAGE2_PRESENTATIONS):
        assert [t.dropped(e, 7, p) for e in range(100)] == [
            LiberoQualityConditioning(seed=0).dropped(e, 7, p) for e in range(100)
        ]


def test_adjacent_presentations_of_one_row_are_not_correlated_through_the_key():
    """The counter goes through the SAME SHA-256 mix as the row keys, for the same reason:
    consecutive presentations are adjacent keys, and a weak mix there would make a row's fate
    change in runs rather than independently."""
    t = LiberoQualityConditioning(seed=0)
    flips = sum(
        t.dropped(e, f, 0) != t.dropped(e, f, 1) for e in range(100) for f in range(10)
    )
    # Independent Bernoulli(0.1925) draws differ ~2*p*(1-p) = 31% of the time.
    assert 200 < flips < 420, f"{flips} flips in 1000 pairs looks structured"


def test_the_presentation_counter_does_not_disturb_the_seed():
    """Two seeds must still disagree at every presentation -- catches a key that collapses the
    seed into the counter (e.g. concatenated without a separator)."""
    for p in (0, 3):
        a = [LiberoQualityConditioning(seed=0).dropped(e, f, p) for e in range(40) for f in range(10)]
        b = [LiberoQualityConditioning(seed=1).dropped(e, f, p) for e in range(40) for f in range(10)]
        assert sum(x != y for x, y in zip(a, b, strict=True)) > 10, p


def test_the_transform_reads_the_presentation_the_dataset_injected():
    """End to end through `__call__`: the tag on a row must follow the counter in its sample, not
    a constant. Uses whichever presentations disagree for a chosen row."""
    t = LiberoQualityConditioning(seed=0)
    ep, fr = next(
        (e, f)
        for e in range(60)
        for f in range(20)
        if len({t.dropped(e, f, p) for p in range(STAGE2_PRESENTATIONS)}) == 2
    )
    for p in range(STAGE2_PRESENTATIONS):
        tagged = "Quality: 5" in t(_row(ep, fr, presentation=p))["prompt"]
        assert tagged is (not t.dropped(ep, fr, p)), (ep, fr, p)


def test_the_transform_raises_without_a_prompt():
    """`prompt_from_task=False`, or a repack that does not surface `prompt`. Appending to a
    synthesised empty prompt would condition the arm on a bare `Quality: 5`."""
    with pytest.raises(KeyError, match="prompt"):
        LiberoQualityConditioning()({"episode_index": 1, "frame_index": 2})


def test_tagging_an_already_tagged_prompt_raises():
    """DOUBLE TAG: `"...\\nQuality: 5\\nQuality: 5"` is in range, tokenizes cleanly, is logged
    nowhere, and is matched by no eval-time prompt."""
    t = LiberoQualityConditioning()
    tagged = t(_row(*_a_tagged_row()))
    assert "Quality: 5" in tagged["prompt"]
    with pytest.raises(ValueError, match="TWICE"):
        t({**tagged, "episode_index": 0, "frame_index": 0})


# --- the dropout: a pure function of (seed, episode, frame) --------------------------------------


def _a_tagged_row() -> tuple[int, int]:
    """The first `(ep, fr)` the default transform does NOT drop, so tag assertions are not
    hostage to the hash. Asserted non-empty by its callers' use."""
    t = LiberoQualityConditioning()
    for ep in range(50):
        for fr in range(50):
            if not t.dropped(ep, fr, 0):
                return ep, fr
    raise AssertionError("no undropped row in 2500 tries; the dropout is not ~19%")


def test_the_dropout_is_a_pure_function_of_the_row_and_the_seed():
    """THE D5 DEVIATION, pinned. Stage 1 bakes its dropout into an artifact; stage 2 has none and
    cannot get one without a second offline pipeline over a third-party dataset. What D5 protects
    is that the realized dropout be reconstructable from the run record WITHOUT any worker's RNG
    state -- so it must be a keyed hash, and two passes over one row must agree.

    Catches a per-call or per-batch RNG, under which the same row would land differently on two
    passes and the realized dropout of a finished run would be unreconstructable.
    """
    t = LiberoQualityConditioning(seed=0)
    for ep, fr in ((3, 17), (0, 0), (11, 240)):
        assert t(_row(ep, fr)) == t(_row(ep, fr))
        assert t.dropped(ep, fr, 0) == t.dropped(ep, fr, 0)
    # ...and a fresh instance of the same config agrees, i.e. nothing is carried on the object.
    assert [t.dropped(e, 0, 0) for e in range(200)] == [
        LiberoQualityConditioning(seed=0).dropped(e, 0, 0) for e in range(200)
    ]


def test_the_seed_actually_moves_the_draw():
    """Catches a seed that is stored, documented and never mixed in -- under which every run of
    every arm would share one dropout pattern while the config claimed otherwise. Asserted as a
    disagreement over many rows, not one, because any single row may agree by chance."""
    a = [LiberoQualityConditioning(seed=0).dropped(e, f, 0) for e in range(40) for f in range(10)]
    b = [LiberoQualityConditioning(seed=1).dropped(e, f, 0) for e in range(40) for f in range(10)]
    assert a != b
    assert sum(x != y for x, y in zip(a, b, strict=True)) > 10


def test_the_frame_index_actually_moves_the_draw():
    """Catches a draw keyed on the episode alone, which would drop whole episodes at a time --
    a ~19% marginal that is nothing like π0.7's per-sample dropout, and invisible in any rate."""
    t = LiberoQualityConditioning()
    assert len({t.dropped(7, fr, 0) for fr in range(200)}) == 2


def test_adjacent_rows_are_not_correlated_through_the_key():
    """The reason the key goes through SHA-256 rather than an integer product: consecutive frames
    of one episode are exactly the adjacent keys, and a weak mix there would drop them in runs."""
    flips = sum(
        LiberoQualityConditioning().dropped(4, fr, 0) != LiberoQualityConditioning().dropped(4, fr + 1, 0)
        for fr in range(999)
    )
    # Independent Bernoulli(0.1925) neighbours flip ~2*p*(1-p) = 31% of the time; a key that
    # leaks structure gives long identical runs (few flips) or perfect alternation (many).
    assert 200 < flips < 420, f"{flips} flips in 999 adjacent pairs looks structured"


def test_the_realized_marginal_matches_stage_ones():
    """Catches a collapsed ONE-level dropout here while stage 1 keeps two, which would train the
    two stages' unconditional branches at different rates -- 0.85 vs 0.8075 -- for no stated
    reason and with nothing online to show it."""
    t = LiberoQualityConditioning(seed=0)
    tagged = sum(
        "Quality: 5" in t(_row(e, f))["prompt"] for e in range(200) for f in range(100)
    )
    assert abs(tagged / 20_000 - TAGGED_FRACTION) < 0.02


def test_a_dropped_row_gets_the_bare_prompt_not_quality_zero():
    """The bare prompt IS the unconditional branch. `Quality: 0` would be a sixth condition."""
    t = LiberoQualityConditioning()
    dropped = next((e, f) for e in range(50) for f in range(50) if t.dropped(e, f, 0))
    out = t(_row(*dropped))
    assert out["prompt"] == "pick up the black bowl"
    assert "Quality" not in out["prompt"]


def test_a_tagged_row_carries_the_marker_exactly_once_and_keeps_the_instruction():
    ep, fr = _a_tagged_row()
    out = LiberoQualityConditioning()(_row(ep, fr))
    assert out["prompt"] == "pick up the black bowl\nQuality: 5"
    assert out["prompt"].count("Quality:") == 1
    # The tag is appended, not substituted: the LIBERO instruction must survive intact.
    assert out["prompt"].startswith("pick up the black bowl")


def test_the_row_keys_are_read_the_way_lerobot_supplies_them():
    """LeRobot hands these back as 0-d tensors/arrays, not python ints. A transform that indexed
    them directly, or compared them as objects, would key every row on the same value."""
    import numpy as np

    t = LiberoQualityConditioning()
    ep, fr = _a_tagged_row()
    boxed = {"prompt": "pick up the black bowl",
             "episode_index": np.asarray([ep], dtype=np.int64),
             "frame_index": np.asarray(fr, dtype=np.int64),
             PRESENTATION_KEY: np.asarray(0, dtype=np.int64)}
    assert t(boxed)["prompt"] == t(_row(ep, fr))["prompt"]
