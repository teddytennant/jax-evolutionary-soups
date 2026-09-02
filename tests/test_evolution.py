"""Algorithm 1 on a deterministic toy, and Theorem 3's monotonicity claim."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evolutionary_soups.evolution import crossover_mutate, evolve, select
from evolutionary_soups.hypervolume import hypervolume, nondominated, reference_point

REF = reference_point(2)
CENTRE = jnp.array([0.7, 0.0])


def toy_fitness(params, chunk=None):
    """Two Gaussian objectives pulling in opposite directions.

    f_i(z) = exp(-||z -/+ c||^2), both in (0, 1], maximized. The trade-off curve
    between them is non-convex, which is the region linear scalarization cannot
    reach and evolutionary search can.
    """
    z = params["z"]
    f1 = jnp.exp(-jnp.sum((z - CENTRE) ** 2, axis=-1))
    f2 = jnp.exp(-jnp.sum((z + CENTRE) ** 2, axis=-1))
    scores = jnp.stack([f1, f2], axis=-1)
    if chunk is not None:  # a fresh chunk moves every score a little
        scores = scores + 0.02 * jax.random.normal(chunk, scores.shape)
    return np.asarray(scores)


def toy_population(key, size=12):
    return {"z": jax.random.normal(key, (size, 2))}


def test_hypervolume_climbs_but_the_per_chunk_bound_is_not_tight():
    """Theorem 3, second inequality: HV_t(S_t) >= HV_t(S_{t-1}).

    Scored on the same data every generation, so the only thing that can move
    hypervolume is selection. The parents stay in the pool, so S_{t-1} is always
    available to be re-picked and the theorem says the retained set can never do
    worse. It does, occasionally, by a fraction of a percent: greedy HVC is not
    an exact subset selector even for two objectives (see
    test_greedy_hvc_is_not_exact_in_two_dimensions), so when the whole pool is
    non-dominated and larger than P the greedy prefix can fall short of the
    parents it replaced. The claim holds in the direction that matters and fails
    at the fourth decimal.
    """
    key = jax.random.PRNGKey(0)
    result = evolve(
        key,
        toy_population(jax.random.PRNGKey(1)),
        toy_fitness,
        REF,
        generations=25,
        sigma=0.2,
        mutation_rate=0.5,
    )
    deltas = np.array([g.hv_retained - g.hv_previous for g in result.history])
    final = result.history[-1].hv_retained
    assert final > result.history[0].hv_previous
    assert np.mean(deltas < -1e-12) <= 0.2
    assert -deltas.min() < 0.005 * final


def test_evolution_extends_the_front():
    key = jax.random.PRNGKey(0)
    start = toy_population(jax.random.PRNGKey(1))
    before = hypervolume(toy_fitness(start), REF)
    result = evolve(key, start, toy_fitness, REF, generations=25, sigma=0.2, mutation_rate=0.5)
    after = hypervolume(result.scores, REF)
    assert after > before
    assert len(nondominated(result.scores)) == result.size  # the front stays clean


def test_chunk_noise_breaks_pathwise_monotonicity_but_not_the_trend():
    """What survives the fresh chunk D_t of Algorithm 1 line 3.

    The per-chunk comparison still holds by construction, since S_{t-1} and S_t
    are scored on the same chunk. The comparison across generations does not: the
    scores move under the chunk, so hypervolume measured on noiseless fitness
    goes down as often as a bad chunk says it should. Theorem 3 is stated in the
    per-chunk form, and the proof says so ("monotonicity holds in expectation
    rather than pathwise" across generations).
    """
    size = 12
    start = toy_population(jax.random.PRNGKey(3), size)
    trace = []

    def watched(params, chunk):
        parents = jax.tree_util.tree_map(lambda x: x[:-size], params)
        trace.append(hypervolume(toy_fitness(parents), REF))
        return toy_fitness(params, chunk)

    result = evolve(
        jax.random.PRNGKey(2),
        start,
        watched,
        REF,
        generations=25,
        population_size=size,
        sigma=0.2,
        mutation_rate=0.5,
        chunk_fn=lambda k: k,
    )
    per_chunk = np.array([g.hv_retained - g.hv_previous for g in result.history])
    assert np.mean(per_chunk < -1e-12) <= 0.2

    trace = np.array(trace)
    assert np.any(np.diff(trace) < 0)  # the path wobbles once the chunk moves
    assert hypervolume(toy_fitness(result.params), REF) > trace[0]  # the trend does not


def test_crossover_takes_every_weight_from_one_parent_or_the_other():
    parents = {"w": jnp.stack([jnp.zeros((4, 3)), jnp.ones((4, 3))])}
    children = crossover_mutate(
        jax.random.PRNGKey(0), parents, n_children=8, sigma=0.0, mutation_rate=0.0
    )
    values = np.asarray(children["w"])
    assert values.shape == (8, 4, 3)
    assert np.all((values == 0.0) | (values == 1.0))
    assert values.min() == 0.0 and values.max() == 1.0  # both parents contribute


def test_mutation_touches_a_subset_of_the_weights():
    parents = {"w": jnp.zeros((3, 20, 20))}
    none = crossover_mutate(
        jax.random.PRNGKey(0), parents, 4, sigma=1.0, mutation_rate=0.0
    )
    assert np.all(np.asarray(none["w"]) == 0.0)
    some = crossover_mutate(
        jax.random.PRNGKey(0), parents, 4, sigma=1.0, mutation_rate=0.25
    )
    touched = np.mean(np.asarray(some["w"]) != 0.0)
    assert 0.2 < touched < 0.3
    assert np.std(np.asarray(some["w"])[np.asarray(some["w"]) != 0.0]) == pytest.approx(
        1.0, rel=0.15
    )


def test_selection_keeps_only_non_dominated_candidates():
    scores = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4], [0.2, 0.2]])
    chosen = select(scores, 3, REF)
    assert len(chosen) == 3
    assert set(chosen) <= set(nondominated(scores).tolist())
    small = select(scores[:2], 5, REF)
    assert len(small) == 2  # min(P, |F_t|), Algorithm 1 line 9
