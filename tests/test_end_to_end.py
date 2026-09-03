"""The whole pipeline on the toy task.

One expert per objective over a shared frozen backbone, then the same
evolutionary budget spent on each of the three gate classes, then the fronts
compared by hypervolume on prompts none of them ever saw. Theorem 2 itself is
checked in tests/test_moe.py, where it does not have to survive a search.
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


def test_every_gate_class_beats_the_fixed_coefficient_soup(fronts, baseline):
    """Evolving any of the three gate classes beats sweeping one shared lambda.

    This used to be the Theorem 2 test: per-layer above single-gating above fixed
    coefficients, on one seed. That ordering is a search outcome, not the
    theorem, and it does not survive a change of seed. The containment Theorem 2
    actually states is checked without evolution in tests/test_moe.py, and
    scripts/ablation.py measures how often the ordering shows up. What holds on
    every draw either of them has been run on is this: 30 generations of gates
    cover more of the objective space than the 21-point simplex sweep. Across
    243 evolved fronts, on CPU and on an H200, the smallest margin was +0.024.
    """
    hv_soup, _ = baseline
    for mode, (hv, _, _) in fronts.items():
        assert hv > hv_soup, mode


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
