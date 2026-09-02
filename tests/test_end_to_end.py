"""The whole pipeline on the toy task: Theorem 2, measured.

One expert per objective over a shared frozen backbone, then the same
evolutionary budget spent on each of the three gate classes, then the fronts
compared by hypervolume on prompts none of them ever saw.
"""

import numpy as np
import pytest

from evolutionary_soups import experiment as X
from evolutionary_soups.hypervolume import hypervolume, nondominated
from evolutionary_soups.preference import preference_selection, selection_utility

POPULATION = 20
GENERATIONS = 30
HIDDEN = 16
CHUNK = 256


@pytest.fixture(scope="module")
def fronts(setup, held_out):
    """Evolve each gate class from the same initial population and score it."""
    out = {}
    for mode in ("per_layer", "single", "fixed"):
        result = X.evolve_gates(
            setup,
            seed=1,
            mode=mode,
            population_size=POPULATION,
            generations=GENERATIONS,
            hidden=HIDDEN,
            chunk_size=CHUNK,
        )
        hv, scores = X.front_hypervolume(setup, result, held_out, mode)
        out[mode] = (hv, scores, result)
    return out


@pytest.fixture(scope="module")
def baseline(setup, held_out):
    """Rewarded Soups: one shared lambda per preference, 21 of them for n = 3."""
    scores = X.fixed_coefficient_front(setup, held_out, 21)
    return hypervolume(scores[nondominated(scores)], setup.ref), scores


def test_evolved_front_dominates_the_fixed_coefficient_soup(fronts, baseline):
    hv_soup, _ = baseline
    hv_evolved = fronts["per_layer"][0]
    assert hv_evolved > hv_soup


def test_per_layer_gating_beats_both_ablations(fronts):
    """Theorem 2: f(G_fixed) and f(G_single) are strict subsets of f(G).

    Same population size, same generations, same initial population, same chunks.
    The only difference is what the gate is allowed to condition on.
    """
    hv_per_layer = fronts["per_layer"][0]
    assert hv_per_layer > fronts["single"][0]
    assert hv_per_layer > fronts["fixed"][0]


def test_the_front_is_a_front(fronts, setup):
    for mode, (hv, scores, result) in fronts.items():
        assert hv > 0.0
        assert len(scores) == result.size
        front = scores[nondominated(scores)]
        assert len(front) >= 2, mode
        assert np.all(front.max(axis=0) > setup.ref)


def test_preference_control_moves_along_the_evolved_front(fronts, setup):
    """Eq. (1) at the corners of the simplex should not keep returning one gate."""
    _, scores, _ = fronts["per_layer"]
    front = scores[nondominated(scores)]
    utility = selection_utility("linear")
    picks = [
        preference_selection(front, mu, utility) for mu in np.eye(setup.task.n_objectives)
    ]
    assert len(set(picks)) > 1
    for j, pick in enumerate(picks):
        assert front[pick, j] == pytest.approx(front[:, j].max())


def test_evolution_improved_on_its_starting_population(fronts):
    for mode, (_, _, result) in fronts.items():
        first, last = result.history[0], result.history[-1]
        assert last.hv_retained > first.hv_previous, mode
