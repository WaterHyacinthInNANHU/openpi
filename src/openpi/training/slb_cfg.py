"""RECAP-style classifier-free guidance conditioning for the SLB `cfg` variant.

WHY CONDITION THE VLM (not the action expert)
    The other three non-vanilla variants act on WHICH rows are sampled (filt_bin/top70
    drop rows, awr reweights them). `cfg` keeps every row and instead tells the model
    whether the chunk it is imitating was good or bad, so one network learns both the
    conditional and unconditional distributions and inference can extrapolate away from
    "bad".

    The condition is injected into the VLM's language stream, matching the RECAP /
    pi*0.6 style of advantage-conditioned policies, rather than into the action expert's
    adaRMS. Consequences that matter here:
      * ZERO model change -- no new parameters, no altered checkpoint structure, so the
        existing LoRA fine-tune setup is untouched and checkpoints stay interchangeable
        with the other three variants.
      * pi0.5 already serialises discretised state into the prompt
        (`discrete_state_input=True`), so a quality tag travels a path PaliGemma is
        pretrained to read and the LLM LoRA adapters can adapt to.
      * Guidance at inference is two forward passes that differ only by the tag.
    Conditioning the action expert instead would add randomly-initialised parameters to
    a 3B model being fine-tuned on ~100 demos, and would leave the VLM -- the part that
    actually reads the scene -- unaware of the quality signal.

LABELS
    The sidecar already stores a per-window label from variant_selector:
        0 = positive (delta >= 70th percentile of the task's windows)
        1 = negative (everything else)
    We condition ONLY on 0; 1 is treated as unconditional (positive-only, per the RLinf
    reference). A further `cond_dropout` fraction of the positives is also dropped to
    unconditional so p(a) is learned from data that includes good chunks -- without that
    dropout there is no unconditional branch to guide away from and CFG degenerates to
    plain conditional BC.

NO VALUE MODEL
    The RLinf reference runs this with add_value_head/use_critic_model/use_reward_model all
    False -- it is offline SFT on advantage-TAGGED data, not actor-critic. Our advantage
    comes from the deterministic smoothed reward curve (see adv_proxy), so we need no
    critic either; the two setups line up without modification.

Runs in the openpi venv (numpy only here; no jax needed).
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from typing_extensions import override

import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

# Tag appended to the prompt for the CONDITIONAL branch.
#
# The exact spelling "\nAdvantage: positive" is the de-facto standard across every
# advantage-conditioned VLA implementation surveyed -- RLinf (cfg_model.py
# TokenizePromptWithGuidance), Evo-RL (rl/acp_tags.py) and OpenTau (modeling_pi0.py) all
# emit this identical string. Matching it costs nothing and makes our `cfg` arm directly
# comparable to published results instead of using a private convention. Safe to adopt now
# because no cfg checkpoint has been trained yet (the running sweep is the other 4 arms).
ADVANTAGE_KEY = "Advantage"
POSITIVE_TAG = "positive"
NULL_LABEL = 2  # unconditional: no tag at all

# Fraction of positive rows dropped to unconditional so the p(a) branch is learned.
# 0.1 matches the RLinf reference (unconditional_prob: 0.1) and the usual CFG default.
DEFAULT_COND_DROPOUT = 0.1


def tag_for_label(label: int) -> str | None:
    """Prompt suffix for a condition label, or None for the unconditional case.

    POSITIVE-ONLY, matching RLinf's CFG-RL reference for openpi pi0.5
    (examples/offline_rl/config/cfg_rl_openpi.yaml: positive_only_conditional: true,
    guidance_type: "positive"). Only high-advantage chunks carry a tag; the sidecar's
    negative label (1) maps to unconditional. CFG needs p(a|good) and p(a) to extrapolate
        p_guided = p(a|good) + w * (p(a|good) - p(a));
    an explicit "bad" condition would instead ask the model to MODEL bad behaviour, which
    is both unnecessary and a use of capacity we cannot afford on ~100 demos. That is why
    there is no negative branch here.
    """
    return POSITIVE_TAG if int(label) == 0 else None


def apply_tag(prompt: str, label: int) -> str:
    """Append the advantage tag. Null returns the prompt unchanged -- that IS the
    unconditional prompt, so conditional and unconditional inputs differ only by the tag."""
    tag = tag_for_label(label)
    return prompt if tag is None else f"{prompt}\n{ADVANTAGE_KEY}: {tag}"


@dataclasses.dataclass(frozen=True)
class FixedCfgConditioning(_transforms.DataTransformFn):
    """Apply a FIXED condition at inference, independent of any label.

    Eval has no advantage label -- that is what we are trying to predict -- so the
    conditional branch simply asks for the positive one (`label=0`), which is the whole
    point of advantage conditioning, and the unconditional branch (`label=NULL_LABEL`)
    leaves the prompt bare. Pair the two through Policy(uncond_transforms=...) so the
    branches differ only in the tag.
    """

    label: int = 0

    @override
    def __call__(self, data: dict) -> dict:
        prompt = data.get("prompt")
        if prompt is None:
            return data
        return {**data, "prompt": apply_tag(str(prompt), self.label)}


@dataclasses.dataclass(frozen=True)
class SlbCfgConditioning(_transforms.DataTransformFn):
    """Append a quality tag to the prompt, from a per-window CFG label.

    Keyed by (episode_index, frame_index) rather than the flat row index: both are raw
    LeRobot columns present on every item, so the map can be built from the sidecar alone
    and needs no dataset metadata (episode_from / ep_len) that only exists after the
    dataset is constructed. Windows absent from the map are left UNCONDITIONED rather
    than guessed -- a wrong label teaches the opposite quality signal, which is worse
    than no signal.

    Must run BEFORE RepackTransform (which drops unmapped keys, including episode_index)
    and therefore also before TokenizePrompt.
    """

    labels: dict[tuple[int, int], int]
    cond_dropout: float = DEFAULT_COND_DROPOUT
    seed: int = 0

    @override
    def __call__(self, data: dict) -> dict:
        prompt = data.get("prompt")
        if prompt is None:
            return data
        ep, fr = data.get("episode_index"), data.get("frame_index")
        if ep is None or fr is None:
            return data
        key = (
            int(np.asarray(ep).reshape(-1)[0]),
            int(np.asarray(fr).reshape(-1)[0]),
        )
        label = self.labels.get(key)
        if label is None:
            return data
        if self.cond_dropout > 0.0:
            # Deterministic per window: the same window always drops (or not), so a rerun
            # with the same seed reproduces the run exactly.
            rng = np.random.default_rng(
                (int(self.seed) << 32) ^ ((key[0] * 1_000_003 + key[1]) & 0xFFFFFFFF)
            )
            if rng.random() < self.cond_dropout:
                label = NULL_LABEL
        return {**data, "prompt": apply_tag(str(prompt), label)}


def build_cfg_labels(sidecar, join_index, fps: float) -> dict[tuple[int, int], int]:
    """Map (episode_index, frame_index) -> CFG label {0=pos, 1=neg}.

    frame_index uses the SAME round(t_start * fps) the sampler uses for its row offset,
    so a labelled window is exactly the window the sampler draws. If the two ever
    diverge, rows would carry a neighbouring window's label.

    Read through VariantSidecar's PUBLIC api rather than its `_per_attempt` map: `label()`
    additionally enforces variant == "cfg", so a sidecar of the wrong variant fails here
    instead of silently yielding zeros that look like "everything is positive".

    The frame index is CLAMPED to the episode, mirroring slb_variant_sampler.plan_indices.
    Without the clamp a window whose t_start rounds past the last frame would key a row the
    episode does not have, and that label would simply never match -- silently dropping the
    conditioning for the tail of every episode.
    """
    out: dict[tuple[int, int], int] = {}
    for aid, w in sidecar.window_ids():
        ref = join_index.episode_for(aid)
        if ref is None:
            continue
        t_start = sidecar.t_start(aid)
        if w >= len(t_start):
            continue
        frame = int(round(float(t_start[w]) * fps))
        if ref.frame_count:
            frame = min(frame, int(ref.frame_count) - 1)
        out[(int(ref.episode_index), frame)] = sidecar.label(aid, w)
    return out


def build_conditioning(
    *,
    task_id: int,
    sidecar_root: str,
    manifest_path: str | None,
    fps: float = 15.0,
    cond_dropout: float = DEFAULT_COND_DROPOUT,
    seed: int = 0,
) -> SlbCfgConditioning:
    """Load the cfg sidecar + join index and build the conditioning transform.

    Raises rather than degrading: a `cfg` run without conditioning is byte-for-byte
    identical to `vanilla`, so a silent fallback would put a duplicate arm in the bake-off
    and make it look like CFG had no effect.
    """
    from benchmarks.dataloader.join_index import JoinIndex
    from benchmarks.dataloader.sidecar_reader import VariantSidecar

    if not manifest_path:
        raise ValueError("slb_manifest_path is required to join attempts to episodes")
    sidecar = VariantSidecar.load(int(task_id), "cfg", sidecar_root=sidecar_root)
    join_index = JoinIndex.from_manifest(manifest_path)
    labels = build_cfg_labels(sidecar, join_index, fps)
    if not labels:
        raise ValueError("cfg sidecar produced no (episode, frame) labels")
    n_pos = sum(1 for v in labels.values() if int(v) == 0)
    logger.info(
        "SLB cfg conditioning: task=%s windows=%d positive=%d (%.1f%%) dropout=%.2f",
        task_id, len(labels), n_pos, 100.0 * n_pos / max(1, len(labels)), cond_dropout,
    )
    return SlbCfgConditioning(labels=labels, cond_dropout=cond_dropout, train=True, seed=seed)
