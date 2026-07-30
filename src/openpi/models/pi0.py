import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        # KI half 2: a discrete (FAST) action-token cross-entropy keeps training the VLM once
        # the action-expert gradient path is cut. It runs through the TIED LM head
        # (`gemma.Module.decode`), so knowledge insulation allocates NO parameters and KI
        # checkpoints stay interchangeable with stock pi0.5.
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_fast_loss_weight = config.ki_fast_loss_weight

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "*b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if self.knowledge_insulation and obs.token_ar_mask is not None:
                # Under KI the prompt carries a teacher-forced FAST action postfix, which must
                # be CAUSAL so the discrete loss is genuine next-token prediction rather than
                # each position seeing its own answer. `token_ar_mask` (from `FASTTokenizer`)
                # is 0 on the image/text prefix and 1 on that postfix. It is per-example, so
                # the whole ar_mask becomes [b, s] here rather than the usual [s]; every
                # consumer of it broadcasts, so this is safe.
                image_part = jnp.zeros((obs.token_ar_mask.shape[0], len(ar_mask)), dtype=bool)
                ar_mask = jnp.concatenate([image_part, obs.token_ar_mask.astype(bool)], axis=1)
            else:
                # full attention between image and language inputs
                ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        if isinstance(ar_mask, list):
            ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        if self.knowledge_insulation:
            prefix_out, suffix_out = self._forward_insulated(
                prefix_tokens, prefix_mask, prefix_ar_mask,
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond,
                self.fast_span_mask(observation, prefix_mask.shape[1]),
            )
        else:
            # one big forward pass of prefix + suffix at once
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if self.knowledge_insulation:
            # Broadcasting over the action horizon leaves the mean unchanged, so the CE
            # enters the total loss with exactly ki_fast_loss_weight.
            loss = loss + self.ki_fast_loss_weight * self._fast_token_loss(prefix_out, observation)[:, None]
        return loss

    def fast_span_mask(self, obs: _model.Observation, prefix_len: int) -> at.Bool[at.Array, "b p"]:
        """True at prefix positions holding the teacher-forced FAST action postfix.

        The action expert must be blind to these (see `_forward_insulated`). `token_loss_mask`
        marks exactly the `Action:` + FAST-ids + `|` postfix, and is in PROMPT coordinates, so
        it is left-padded here by the number of image tokens to reach prefix coordinates.
        """
        if obs.token_loss_mask is None:
            return jnp.zeros((obs.state.shape[0], prefix_len), dtype=bool)
        n_image = prefix_len - obs.token_loss_mask.shape[1]
        image_part = jnp.zeros((obs.token_loss_mask.shape[0], n_image), dtype=bool)
        return jnp.concatenate([image_part, obs.token_loss_mask.astype(bool)], axis=1)

    def _forward_insulated(
        self, prefix_tokens, prefix_mask, prefix_ar_mask,
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond,
        fast_span_mask,
    ):
        """Prefix and suffix in two passes, with the VLM's KV detached between them.

        The joint single-pass forward lets the flow-matching loss backprop through the
        suffix's attention into every VLM activation. That is what KI exists to stop: the
        gradient of a continuous action loss is what degrades the pretrained
        representations. Splitting the forward gives us a seam to cut.

        The cut is on the KV CACHE, not on `prefix_tokens`. Detaching the input embeddings
        only protects the vision tower and embedder -- the LLM layers processing the prefix
        would still receive action gradients through attention. Detaching the cache severs
        every path, matching TensorAuto/OpenTau (`past_key_values[...]["key_states"].detach()`)
        and, more literally still, cijerezg/lerobot, which splits K/V by query owner so VLM
        queries keep the gradient while action queries get a detached copy.

        `fast_span_mask` marks the teacher-forced FAST action tokens now living in the prefix.
        The expert is masked OFF them, because they are the ground-truth answer: letting the
        flow-matching branch attend to them would leak the label (and they are absent at
        inference, so the branch would also see a different world at test time). The paper:
        "we set the attention mask A such that no discrete FAST action token can attend to
        continuous action tokens and vice-versa." OpenTau does the same by subtracting the
        discrete-action span from `num_cross_att_tokens`.

        The expert's POSITION ids are offset by the count of tokens it can actually see, not by
        the full prefix length, for the same train/inference-consistency reason.
        """
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

        # What the expert is allowed to attend to: valid prefix tokens MINUS the FAST span.
        visible_prefix_mask = jnp.logical_and(prefix_mask, jnp.logical_not(fast_span_mask))

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_part = einops.repeat(visible_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_part, suffix_attn_mask], axis=-1)
        positions = jnp.sum(visible_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
        )
        return prefix_out, suffix_out

    def _fast_token_loss(self, prefix_out, observation) -> at.Float[at.Array, " b"]:
        """Shift-by-one cross-entropy on the teacher-forced FAST action tokens.

        This is the ONLY signal training the VLM once the action gradient is cut, so a missing
        target is a hard error: silently dropping it would turn `knowledge_insulation` into
        plain backbone freezing while still reporting itself as KI.

        Mirrors `pi0_fast.compute_loss`: each position predicts the NEXT token, so logits are
        shifted left against targets shifted right, and the loss is averaged over the masked
        positions only. Logits come from the TIED embedding table (`gemma.Module.decode`), so
        no parameters are added -- the same choice cijerezg/lerobot and open-gigaai/giga-brain-0
        make.
        """
        if observation.token_loss_mask is None or observation.tokenized_prompt is None:
            raise ValueError(
                "knowledge_insulation=True requires a FAST-tokenized prompt (tokenized_prompt "
                "+ token_ar_mask + token_loss_mask) from the data pipeline: with the "
                "action-expert gradient path cut, the VLM would otherwise receive no gradient "
                "at all, which is backbone freezing rather than knowledge insulation."
            )
        # The prefix is [image tokens, prompt tokens]; only the prompt part has token targets.
        n_prompt = observation.tokenized_prompt.shape[1]
        prompt_out = prefix_out[:, -n_prompt:]

        logits = self.PaliGemma.llm(prompt_out[:, :-1], method="decode")
        logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        targets = observation.tokenized_prompt[:, 1:]
        picked = jnp.take_along_axis(logp, targets[..., None].astype(jnp.int32), axis=-1)[..., 0]

        loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)
        return -jnp.sum(picked * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        guidance_scale: float = 0.0,
        uncond_observation: _model.Observation | None = None,
    ) -> _model.Actions:
        """Sample an action chunk, optionally with classifier-free guidance.

        CFG (used by the SLB `cfg` variant, see training/slb_cfg.py) needs `uncond_observation`
        to be identical to `observation` except for the tokenized prompt, which omits the
        quality tag. The guided velocity is
            v = v_uncond + (1 + guidance_scale) * (v_cond - v_uncond)
        so guidance_scale=0 reproduces plain conditional sampling exactly.

        It must be applied at EVERY denoising step: the flow ODE is integrated, so mixing
        only the final actions of two separate rollouts is not guidance, it is an average of
        two different trajectories. We therefore run both branches as one batch-doubled
        forward -- cond and uncond share x_t, so the only thing that differs is the prefix.

        Inference-only; no parameters are added, so checkpoints stay interchangeable with
        the other SLB variants.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Both branches are stacked along the batch axis and run through one forward pass,
        # rather than two sequential passes, so guidance costs ~1 extra batch not ~2x latency.
        do_cfg = bool(guidance_scale != 0.0 and uncond_observation is not None)
        if do_cfg:
            uncond_observation = _model.preprocess_observation(None, uncond_observation, train=False)
            observation = jax.tree.map(
                lambda c, u: jnp.concatenate([c, u], axis=0), observation, uncond_observation
            )
        eff_bs = 2 * batch_size if do_cfg else batch_size

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            # Both branches denoise the SAME x_t; they differ only through the prefix they
            # attend to, which is what makes the two velocities comparable.
            x_in = jnp.concatenate([x_t, x_t], axis=0) if do_cfg else x_t
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_in, jnp.broadcast_to(time, eff_bs)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                eff_bs,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if do_cfg:
                v_cond, v_uncond = jnp.split(v_t, 2, axis=0)
                v_t = v_uncond + (1.0 + guidance_scale) * (v_cond - v_uncond)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
