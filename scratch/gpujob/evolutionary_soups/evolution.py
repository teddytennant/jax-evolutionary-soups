"""Algorithm 1: evolve the gating networks, keep the front by greedy HVC.

Only the gate MLP moves. The backbone and the expert LoRA adapters stay frozen
for the whole run, which is what makes evolution affordable in the paper and
what makes it fast here.

Each generation samples a fresh chunk D_t, breeds P offspring from the retained
set by per-weight uniform crossover and Gaussian mutation, scores the pooled
parents and offspring on D_t, drops the dominated ones, and rebuilds S_t
additively by marginal hypervolume contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .hypervolume import (
    crowding_distance_selection,
    greedy_hvc_selection,
    hypervolume,
    nondominated,
)


@dataclass
class GenerationLog:
    """What happened at generation t, all scored on that generation's chunk D_t."""

    generation: int
    pool_size: int
    front_size: int
    retained: int
    hv_retained: float  # HV_t(S_t)
    hv_previous: float  # HV_t(S_{t-1}), the same chunk, the previous retained set


@dataclass
class EvolutionResult:
    params: dict  # the evolved front P_hat, stacked on the leading axis
    scores: np.ndarray  # (|P_hat|, n) rewards on the final chunk
    history: list = field(default_factory=list)
    initial: dict = None  # the population generation 1 started from, S_0

    @property
    def size(self):
        return len(self.scores)


def crossover_mutate(key, parents, n_children, sigma, mutation_rate):
    """Reproduction, Algorithm 1 line 4.

    Uniform per-weight crossover between two parents drawn at random, then
    Gaussian noise N(0, sigma^2) on a random subset of the weights.
    """
    leaves, treedef = jax.tree_util.tree_flatten(parents)
    n_parents = leaves[0].shape[0]
    k_a, k_b, k_rest = jax.random.split(key, 3)
    pick_a = jax.random.randint(k_a, (n_children,), 0, n_parents)
    pick_b = jax.random.randint(k_b, (n_children,), 0, n_parents)
    keys = jax.random.split(k_rest, 3 * len(leaves))
    children = []
    for j, leaf in enumerate(leaves):
        a, b = leaf[pick_a], leaf[pick_b]
        take_a = jax.random.bernoulli(keys[3 * j], 0.5, a.shape)
        child = jnp.where(take_a, a, b)
        touched = jax.random.bernoulli(keys[3 * j + 1], mutation_rate, a.shape)
        noise = jax.random.normal(keys[3 * j + 2], a.shape) * sigma
        children.append(child + jnp.where(touched, noise, 0.0))
    return jax.tree_util.tree_unflatten(treedef, children)


def _stack(a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), a, b)


def _index(tree, idx):
    idx = jnp.asarray(idx)
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def select(scores, size, ref, selector="greedy_hvc"):
    """Algorithm 1 lines 7 to 12: non-dominated filter, then greedy HVC.

    `selector="crowding"` swaps in NSGA-II's crowding distance, the ablation of
    Section 4.3. Returns indices into `scores`.
    """
    front = nondominated(scores)
    keep = min(size, len(front))
    sub = np.asarray(scores)[front]
    if selector == "greedy_hvc":
        chosen = greedy_hvc_selection(sub, keep, ref)
    elif selector == "crowding":
        chosen = crowding_distance_selection(sub, keep)
    else:
        raise ValueError(f"unknown selector {selector!r}")
    return [int(front[i]) for i in chosen]


def evolve(
    key,
    population,
    fitness_fn,
    ref,
    generations=20,
    population_size=None,
    sigma=0.15,
    mutation_rate=0.3,
    selector="greedy_hvc",
    chunk_fn=None,
):
    """Run Algorithm 1 and return the evolved front P_hat = S_T.

    fitness_fn(stacked_gate_params, chunk) returns an array of shape
    (n_individuals, n_objectives) of normalized rewards. chunk_fn(key) draws the
    fresh chunk D_t; pass None to score every generation on the same data, which
    is the noise-free case Theorem 3's monotonicity claim is stated on.
    """
    ref = np.asarray(ref, dtype=float)
    leaves = jax.tree_util.tree_leaves(population)
    if population_size is None:
        population_size = leaves[0].shape[0]
    retained = population
    history = []
    scores = None
    for t in range(1, generations + 1):
        key, k_chunk, k_breed = jax.random.split(key, 3)
        chunk = None if chunk_fn is None else chunk_fn(k_chunk)
        offspring = crossover_mutate(
            k_breed, retained, population_size, sigma, mutation_rate
        )
        pool = _stack(retained, offspring)
        pool_scores = np.asarray(fitness_fn(pool, chunk), dtype=float)
        n_parents = jax.tree_util.tree_leaves(retained)[0].shape[0]
        chosen = select(pool_scores, population_size, ref, selector)
        history.append(
            GenerationLog(
                generation=t,
                pool_size=len(pool_scores),
                front_size=len(nondominated(pool_scores)),
                retained=len(chosen),
                hv_retained=hypervolume(pool_scores[chosen], ref),
                hv_previous=hypervolume(pool_scores[:n_parents], ref),
            )
        )
        retained = _index(pool, chosen)
        scores = pool_scores[chosen]
    if scores is None:  # generations == 0
        scores = np.asarray(fitness_fn(retained, None), dtype=float)
    return EvolutionResult(
        params=retained, scores=scores, history=history, initial=population
    )
