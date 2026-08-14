"""The stage-2 CFG twin, observed through the real loader and decoded back to text.

Construction-level assertions cannot see an inert arm. Task 6's first real failure was a stage-1
arm that built, ran to completion, and produced the CONTROL's prompts -- visible only by decoding
`observation.tokenized_prompt`. Stage 2 reaches the model through a different mechanism (a repack
group entry, not `wrap_and_transform`), so it needs its own version of that test rather than
inheriting confidence from stage 1's.

Three properties, none of them visible in a loss curve:

(1) THE ARM IS LIVE, per row. Every row the loader drew must carry `Quality: 5` exactly when the
    keyed dropout says it should -- not merely "about 81% of prompts are tagged", which a draw
    keyed on the wrong field would also produce.
(2) THE TAG LANDS EXACTLY ONCE. Two conditioning entries in one chain give
    `"...\\nQuality: 5\\nQuality: 5"`: in range, tokenizable, matched by no eval-time prompt.
(3) THE PARENT IS UNCHANGED. The same corpus through `pi05_libero_axisinit_paper` must yield zero
    tagged prompts, or the treatment leaked into the five arms stage 2 is supposed to hold fixed.

The dataset is a toy in the LIBERO repack map's own shape. The real
`physical-intelligence/libero` is a third-party download that this suite must not require, and
`repo_id="fake"` would short-circuit `create_torch_dataset` into `FakeDataset` before any of this
ran -- the bypass that made an earlier fixture test nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.quality_conditioning import LiberoQualityConditioning

PARENT = "pi05_libero_axisinit_paper"
TWIN = "pi05_libero_axisinit_paper_cfg"

FRAMES_PER_EPISODE = 32
N_EPISODES = 24
N_ROWS = FRAMES_PER_EPISODE * N_EPISODES
BATCH = 16
PROMPTS = [
    "pick up the black bowl and place it on the plate",
    "open the top drawer of the cabinet",
]


class _Toy(torch.utils.data.Dataset):
    """A LIBERO-shaped corpus that RECORDS the rows it was asked for.

    Carries `episode_index`/`frame_index` as 0-d tensors, which is how LeRobot hands them back --
    a transform that used them as python ints would key every row on the same value here, exactly
    as it would on the box.
    """

    def __init__(self, seen: list):
        self.seen = seen

    def __len__(self) -> int:
        return N_ROWS

    def __getitem__(self, i):
        i = int(i)
        self.seen.append(i)
        rng = np.random.default_rng(i)
        return {
            "image": np.zeros((224, 224, 3), np.uint8),
            "wrist_image": np.zeros((224, 224, 3), np.uint8),
            "state": rng.random(8).astype(np.float32),
            "actions": rng.random((10, 7)).astype(np.float32),
            # The prompt VARIES by row: a tag appended to a single constant instruction would
            # decode identically no matter which row it landed on.
            "prompt": PROMPTS[i % len(PROMPTS)],
            "episode_index": torch.tensor(i // FRAMES_PER_EPISODE),
            "frame_index": torch.tensor(i % FRAMES_PER_EPISODE),
        }


@pytest.fixture
def toy(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(
        _data_loader, "create_torch_dataset", lambda *_a, **_k: _Toy(seen)
    )
    return seen


@pytest.fixture(scope="module")
def tok():
    return _tokenizer.PaligemmaTokenizer(200)


def _decode(observation, i, tok) -> str:
    ids = np.asarray(observation.tokenized_prompt)[i]
    mask = np.asarray(observation.tokenized_prompt_mask)[i].astype(bool)
    return tok._tokenizer.decode([int(t) for t in ids[mask]])  # noqa: SLF001


def _drive(config_name: str, *, num_batches: int = 8):
    """The REGISTERED config, through the real loader. `shuffle=False` and `num_workers=0` make
    the fetch order the row order, so a decoded prompt can be tied back to its own row."""
    cfg = _config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    return list(
        _data_loader.create_torch_data_loader(
            data_config,
            cfg.model,
            action_horizon=cfg.model.action_horizon,
            batch_size=BATCH,
            num_batches=num_batches,
            num_workers=0,
            seed=0,
            shuffle=False,
            skip_norm_stats=True,
        )
    )


def _expected_tagged(row: int) -> bool:
    """What the keyed dropout says about this row, computed independently of the loader."""
    return not LiberoQualityConditioning(q_ep=5).dropped(
        row // FRAMES_PER_EPISODE, row % FRAMES_PER_EPISODE
    )


# --- the arm is live -----------------------------------------------------------------------------


def test_every_tag_lands_on_exactly_the_row_the_dropout_names(toy, tok):
    """THE ANTI-INERT TEST. Catches the field never forwarded into `create()`, the transform
    placed after `RepackTransform` (where the row keys are gone), a draw keyed on the wrong
    field, and a tag appended to the wrong sample.

    Per-row exactness, not a fraction: a uniform ~81% tagged is also what a draw keyed on the
    episode alone, or one shifted by a row, would produce.
    """
    batches = _drive(TWIN)
    n_tagged = n_bare = 0
    for b, (observation, _actions) in enumerate(batches):
        rows = toy[b * BATCH : (b + 1) * BATCH]
        assert len(rows) == BATCH
        for i, row in enumerate(rows):
            text = _decode(observation, i, tok)
            if _expected_tagged(row):
                assert "Quality: 5" in text, f"row {row} should be tagged, got {text!r}"
                n_tagged += 1
            else:
                assert "Quality" not in text, f"row {row} is dropped out but was tagged: {text!r}"
                n_bare += 1
    # Neither branch may be empty, or half of the assertion above never ran.
    assert n_tagged > 0, f"nothing was tagged in {len(batches)} batches"
    assert n_bare > 0, "no dropped-out row was drawn: the unconditional branch went unchecked"


def test_the_instruction_survives_beside_the_tag(toy, tok):
    """Catches a tag that REPLACES the prompt rather than extending it -- which would train an
    arm with no task text at all and still decode a plausible `Quality: 5`."""
    seen_both = set()
    for observation, _actions in _drive(TWIN, num_batches=4):
        for i in range(BATCH):
            text = _decode(observation, i, tok)
            matching = [p for p in PROMPTS if p in text]
            assert matching, f"no LIBERO instruction survived in {text!r}"
            seen_both.update(matching)
    assert len(seen_both) == len(PROMPTS), "one of the two instructions never appeared"


def test_the_tag_appears_exactly_once_in_every_prompt(toy, tok):
    """THE DOUBLE-TAG TEST. `"...\\nQuality: 5\\nQuality: 5"` is in range, tokenizes cleanly, and
    is matched by no eval-time prompt; counting occurrences is the only place it is observable."""
    for observation, _actions in _drive(TWIN):
        for i in range(BATCH):
            assert _decode(observation, i, tok).count("Quality") <= 1


def test_the_parent_tags_nothing_through_the_same_loader(toy, tok):
    """The other half of the anti-inert test, and the parity claim at its most concrete: stage 2
    for the other five arms must be byte-identical to what it was. Catches a transform that tags
    unconditionally, or a `quality_tag` default that is not None."""
    for observation, _actions in _drive(PARENT):
        for i in range(BATCH):
            assert "Quality" not in _decode(observation, i, tok)


def test_two_runs_of_the_arm_tag_identically(toy, tok):
    """The D5 deviation, end to end: the realized dropout of a run is a property of the rows and
    the seed, so a second pass over the same rows must tag the same ones. Catches a per-batch RNG
    that only shows up across runs."""
    first = [
        _decode(o, i, tok) for o, _ in _drive(TWIN, num_batches=6) for i in range(BATCH)
    ]
    toy.clear()
    second = [
        _decode(o, i, tok) for o, _ in _drive(TWIN, num_batches=6) for i in range(BATCH)
    ]
    assert first == second
    assert any("Quality: 5" in t for t in first)


def test_the_realized_tagged_fraction_is_stage_ones(toy, tok):
    """A full pass over the toy corpus, so this is the realized rate rather than a sample of it."""
    tagged = sum(
        "Quality: 5" in _decode(o, i, tok)
        for o, _ in _drive(TWIN, num_batches=N_ROWS // BATCH)
        for i in range(BATCH)
    )
    assert abs(tagged / N_ROWS - 0.8075) < 0.05
