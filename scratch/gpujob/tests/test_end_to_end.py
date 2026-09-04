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


def test_evolution_improved_on_its_starting_population(fronts, setup, held_out):
    """Score S_0 and S_T on the same prompts. The generation log cannot do that.

    This used to take two numbers straight out of result.history: the last
    generation's hv_retained against the first generation's hv_previous. Every
    generation draws its own chunk D_t, so those two hypervolumes come from
    different samples of 256 prompts, and the sample moves them further than
    thirty generations of evolution do. Hold one population fixed and score it
    on 40 chunks: a starting population gives sd 0.091, the front it evolves
    into gives 0.076, and the gap between their means is 0.15. So the logged
    difference came out negative on 39 of 150 runs, and not for one gate class
    more than another: 8/50 per_layer, 16/50 single, 15/50 fixed. An H200 drew
    0.352 against 0.364 on fixed and that is the one that got reported.

    The search is fine. Held-out prompts are the same 512 before and after, and
    on them every gate class gains on every draw: 25 evolution seeds against two
    stage-1 setups, 150 runs, +0.059 at the smallest and +0.130 on average.
    """
    for mode, (hv, _, result) in fronts.items():
        start = X.population_fitness(setup, result.initial, held_out, mode)
        hv_start = hypervolume(start[nondominated(start)], setup.ref)
        assert hv > hv_start, mode


def test_the_generation_log_holds_up_where_theorem_3_states_it(fronts):
    """HV_t(S_t) >= HV_t(S_{t-1}), which is the one comparison the log can make.

    Both sides are the same chunk D_t, so this is the form the theorem is stated
    in. It still dips, because greedy HVC is not an exact subset selector, and
    it dips most for the fixed gate: over 50 runs of 30 generations each, 0.3%
    of generations for per_layer, 0.7% for single and 6.7% for fixed, worst
    single run 5 of 30. Why fixed is the one that trips it, I do not know. The
    size is the part that matters and it stays small: the worst dip over those
    150 runs was 0.62% of the final hypervolume.
    """
    for mode, (_, _, result) in fronts.items():
        deltas = np.array([g.hv_retained - g.hv_previous for g in result.history])
        assert np.mean(deltas < -1e-12) <= 0.3, mode
        assert -deltas.min() < 0.02 * result.history[-1].hv_retained, mode
