"""A synthetic stand-in for the paper's alignment tasks.

The paper scores generations with pretrained reward models on Beaver, Summary and
Assistant. None of that fits in a unit test, so the objectives here are closed
form: a prompt is a short token sequence, the model emits a vector, and objective
i rewards how close that vector is to target t_i(x).

Two things are copied from the real setting because the method needs them.

  Conflict. The targets t_i disagree, so no single output maximizes every
  objective and the achievable rewards form a genuine trade-off.

  Prompt dependence. Prompts fall into two regimes, "safe" and "risky" in the
  paper's Beaver example, and the targets are drawn differently in each. The
  best merging coefficient therefore depends on the prompt, which is the
  non-constancy premise Lemma 5 needs and what Appendix A measures as the
  preference-coefficient gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import ModelConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Task:
    cfg: ModelConfig
    n_objectives: int
    feature_embed: jnp.ndarray  # (vocab, d_feat), fixed random prompt featurizer
    weights: jnp.ndarray  # (n_regimes, n_objectives, d_feat, n_out)

    def tree_flatten(self):
        return (self.feature_embed, self.weights), (self.cfg, self.n_objectives)

    @classmethod
    def tree_unflatten(cls, static, children):
        cfg, n_objectives = static
        return cls(cfg=cfg, n_objectives=n_objectives, feature_embed=children[0], weights=children[1])


def make_task(key, cfg, n_objectives=2, d_feat=8, alignment=0.85):
    """Build the target functions.

    In regime 0 the objectives are drawn close to a shared direction with
    `alignment`, so the trade-off is mild and almost any merge does well. In
    regime 1 they are drawn independently, so they pull hard against each other.
    The optimal merge therefore differs between the two regimes.
    """
    k_embed, k_base, k_obj = jax.random.split(key, 3)
    feature_embed = jax.random.normal(k_embed, (cfg.vocab, d_feat))
    base = jax.random.normal(k_base, (d_feat, cfg.n_out))
    raw = jax.random.normal(k_obj, (2, n_objectives, d_feat, cfg.n_out))
    aligned = alignment * base + (1.0 - alignment) * raw[0]
    weights = jnp.stack([aligned, raw[1]], axis=0)
    return Task(cfg=cfg, n_objectives=n_objectives, feature_embed=feature_embed, weights=weights)


def sample_prompts(key, task, n):
    """Draw n prompts uniformly over the vocabulary."""
    return jax.random.randint(key, (n, task.cfg.seq_len), 0, task.cfg.vocab)


def regimes(tokens):
    """Which regime a prompt belongs to, read off its first token."""
    return tokens[:, 0] % 2


def targets(task, tokens):
    """t_i(x) for every objective i and prompt x, shape (n_objectives, B, n_out)."""
    features = jnp.mean(task.feature_embed[tokens], axis=1)
    per_regime = jnp.tanh(jnp.einsum("bf,rifo->ribo", features, task.weights))
    return jnp.take_along_axis(
        per_regime, regimes(tokens)[None, None, :, None], axis=0
    )[0]


def rewards(task, tokens, outputs):
    """Raw reward vector, one entry per objective. Higher is better."""
    t = targets(task, tokens)
    return -jnp.mean((outputs[None] - t) ** 2, axis=(1, 2))


def per_prompt_rewards(task, tokens, outputs):
    """Raw rewards kept per prompt, shape (n_objectives, B)."""
    t = targets(task, tokens)
    return -jnp.mean((outputs[None] - t) ** 2, axis=-1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Normalizer:
    lo: jnp.ndarray
    hi: jnp.ndarray

    def __call__(self, raw):
        return (raw - self.lo) / (self.hi - self.lo)

    def tree_flatten(self):
        return (self.lo, self.hi), None

    @classmethod
    def tree_unflatten(cls, static, children):
        del static
        return cls(lo=children[0], hi=children[1])


def normalizer_from_experts(expert_rewards):
    """Map each objective to [0, 1] using the pure experts as the two anchors.

    Section 4.2 normalizes all rewards to [0, 1] per objective before computing
    utilities and hypervolume. expert_rewards[i] is the raw reward vector of the
    gate pinned to expert i, so expert i scores 1 on objective i and the worst
    expert scores 0. Merged gates can land outside [0, 1]; nothing downstream
    assumes otherwise.
    """
    raw = jnp.asarray(expert_rewards)
    hi = jnp.diag(raw)
    lo = jnp.min(raw, axis=0)
    return Normalizer(lo=lo, hi=jnp.maximum(hi, lo + 1e-6))
