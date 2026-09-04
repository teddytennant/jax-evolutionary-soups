"""The three stages of Section 3.1, end to end on the toy task.

  1. train one expert per objective as a LoRA adapter over a frozen backbone,
  2. evolve the gating networks into a Pareto front (Algorithm 1),
  3. select from that front by user preference at inference time (Eq. 1).

Stage 1 stands in for the paper's SFT plus PPO pipeline: a short supervised
pretrain of the backbone, then gradient descent on each expert's adapter against
its own objective with the backbone frozen. Everything else is the paper's.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import task as task_mod
from .entmax import DEFAULT_ALPHA
from .evolution import evolve
from .gating import constant_coeff_fn, coeff_fn, init_population
from .hypervolume import hypervolume, nondominated, reference_point
from .model import ModelConfig, forward, init_backbone, init_experts


@dataclass
class Setup:
    cfg: ModelConfig
    task: task_mod.Task
    backbone: dict
    experts: list
    normalizer: task_mod.Normalizer
    expert_rewards: np.ndarray  # raw reward matrix, row i is expert i alone
    ref: np.ndarray


def uniform_coeffs(n_experts):
    return jnp.full((n_experts,), 1.0 / n_experts)


def one_hot(i, n):
    return jnp.eye(n)[i]


def run_model(backbone, experts, tokens, lam=None, gate=None, mode="per_layer", alpha=DEFAULT_ALPHA):
    """Forward pass under either a pinned lambda or a gating network."""
    fn = constant_coeff_fn(lam) if gate is None else coeff_fn(gate, mode, alpha)
    return forward(backbone, experts, tokens, fn)


@jax.jit
def _sweep_shared(backbone, experts, task, normalizer, tokens, grid):
    def one(lam):
        y = run_model(backbone, experts, tokens, lam=lam)
        return normalizer(task_mod.rewards(task, tokens, y))

    return jax.vmap(one)(grid)


@jax.jit
def _sweep_per_layer(backbone, experts, task, normalizer, tokens, lams):
    def one(lam):
        def fn(h_attn, prompt_embedding, layer):
            del prompt_embedding
            return jnp.broadcast_to(lam[layer], h_attn.shape[:-1] + lam.shape[-1:])

        y = forward(backbone, experts, tokens, fn)
        return normalizer(task_mod.rewards(task, tokens, y))

    return jax.vmap(one)(lams)


def _sft(key, cfg, task, backbone, experts, batch, steps, lr):
    """Supervised pretrain of the backbone on the mean of the objectives.

    Experts are zero at init, so this trains the shared base only. After it the
    backbone is frozen for good, as in Rewarded Soups. Prompts are redrawn every
    step, so the backbone learns the task rather than memorizing a fixed set.
    """
    lam = uniform_coeffs(cfg.n_experts)

    def loss(params, tokens):
        out = run_model(params, experts, tokens, lam=lam)
        return -jnp.mean(task_mod.rewards(task, tokens, out))

    opt = optax.adam(lr)
    state = opt.init(backbone)

    @jax.jit
    def step(params, state, tokens):
        grads = jax.grad(loss)(params, tokens)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state

    for k in jax.random.split(key, steps):
        backbone, state = step(backbone, state, task_mod.sample_prompts(k, task, batch))
    return backbone


def _train_expert(key, cfg, task, backbone, experts, index, batch, steps, lr):
    """Stage 1: push expert `index` toward objective `index` alone, backbone frozen."""
    lam = one_hot(index, cfg.n_experts)
    others = list(experts)

    def loss(adapter, tokens):
        merged = list(others)
        merged[index] = adapter
        out = run_model(backbone, merged, tokens, lam=lam)
        return -task_mod.rewards(task, tokens, out)[index]

    opt = optax.adam(lr)
    adapter = experts[index]
    state = opt.init(adapter)

    @jax.jit
    def step(adapter, state, tokens):
        grads = jax.grad(loss)(adapter, tokens)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(adapter, updates), state

    for k in jax.random.split(key, steps):
        adapter, state = step(adapter, state, task_mod.sample_prompts(k, task, batch))
    return adapter


def build_setup(
    seed=0,
    n_objectives=2,
    cfg=None,
    batch=256,
    sft_steps=400,
    expert_steps=400,
    lr=3e-2,
):
    """Stage 1. Returns the frozen backbone, the trained experts, and the scaling."""
    cfg = cfg or ModelConfig(n_experts=n_objectives)
    if cfg.n_experts != n_objectives:
        raise ValueError("this setup uses one expert per objective, as the paper does")
    key = jax.random.PRNGKey(seed)
    k_task, k_back, k_exp, k_sft, k_train, k_data = jax.random.split(key, 6)
    task = task_mod.make_task(k_task, cfg, n_objectives)
    backbone = init_backbone(k_back, cfg)
    experts = init_experts(k_exp, cfg)

    backbone = _sft(k_sft, cfg, task, backbone, experts, batch, sft_steps, lr)
    experts = [
        _train_expert(k, cfg, task, backbone, experts, i, batch, expert_steps, lr)
        for i, k in enumerate(jax.random.split(k_train, cfg.n_experts))
    ]

    tokens = task_mod.sample_prompts(k_data, task, 512)

    raw = np.stack(
        [
            np.asarray(
                task_mod.rewards(
                    task, tokens, run_model(backbone, experts, tokens, lam=one_hot(i, cfg.n_experts))
                )
            )
            for i in range(cfg.n_experts)
        ]
    )
    return Setup(
        cfg=cfg,
        task=task,
        backbone=backbone,
        experts=experts,
        normalizer=task_mod.normalizer_from_experts(raw),
        expert_rewards=raw,
        ref=reference_point(n_objectives),
    )


@functools.partial(jax.jit, static_argnames=("mode", "alpha"))
def _population_rewards(backbone, experts, task, gates, tokens, mode, alpha):
    def one(gate):
        out = forward(backbone, experts, tokens, coeff_fn(gate, mode, alpha))
        return task_mod.rewards(task, tokens, out)

    return jax.vmap(one)(gates)


def population_fitness(setup, gates, tokens, mode="per_layer", alpha=DEFAULT_ALPHA):
    """Normalized objective vectors f(g) for a stacked population of gates."""
    raw = _population_rewards(
        setup.backbone, setup.experts, setup.task, gates, tokens, mode, alpha
    )
    return np.asarray(setup.normalizer(raw))


def fixed_coefficient_front(setup, tokens, n_points=11):
    """The Rewarded Soups baseline: one lambda per preference, shared everywhere.

    Sweeps the simplex on the same grid the paper evaluates baselines on, 11
    points for two objectives and 21 for three, and returns the normalized
    rewards of every sampled merge.
    """
    from .preference import sample_preferences

    grid = jnp.asarray(sample_preferences(setup.cfg.n_experts, n_points))
    return np.asarray(
        _sweep_shared(
            setup.backbone, setup.experts, setup.task, setup.normalizer, tokens, grid
        )
    )


def per_layer_reachable(setup, tokens, key, n_samples=64):
    """Sample R_per-layer, the reward region of independent per-layer lambdas.

    Lemma 6 says R_single is the diagonal of this set, so it is contained in it.
    Used to check the containment without any evolution in the way.
    """
    lams = jax.random.dirichlet(
        key, jnp.ones(setup.cfg.n_experts), (n_samples, setup.cfg.n_layers)
    )
    return np.asarray(
        _sweep_per_layer(
            setup.backbone, setup.experts, setup.task, setup.normalizer, tokens, lams
        )
    )


def evolve_gates(
    setup,
    seed=0,
    mode="per_layer",
    population_size=16,
    generations=15,
    hidden=32,
    alpha=DEFAULT_ALPHA,
    sigma=0.25,
    mutation_rate=0.3,
    chunk_size=64,
    selector="greedy_hvc",
    fixed_chunk=None,
    bias_scale=1.5,
    weight_scale=0.1,
):
    """Stage 2, Algorithm 1, for one gate class."""
    key = jax.random.PRNGKey(seed)
    k_init, k_evolve, k_chunk = jax.random.split(key, 3)
    d_in = setup.cfg.d_model
    population = init_population(
        k_init, population_size, d_in, setup.cfg.n_experts, hidden, bias_scale, weight_scale
    )

    if fixed_chunk is None:
        def chunk_fn(k):
            return task_mod.sample_prompts(k, setup.task, chunk_size)
    else:
        chunk_fn = None

    def fitness_fn(gates, chunk):
        tokens = fixed_chunk if chunk is None else chunk
        return population_fitness(setup, gates, tokens, mode, alpha)

    return evolve(
        k_evolve,
        population,
        fitness_fn,
        setup.ref,
        generations=generations,
        population_size=population_size,
        sigma=sigma,
        mutation_rate=mutation_rate,
        selector=selector,
        chunk_fn=chunk_fn,
    )


def front_hypervolume(setup, result, tokens, mode="per_layer", alpha=DEFAULT_ALPHA):
    """Re-score an evolved front on a held-out set and take its hypervolume."""
    scores = population_fitness(setup, result.params, tokens, mode, alpha)
    return hypervolume(scores[nondominated(scores)], setup.ref), scores
