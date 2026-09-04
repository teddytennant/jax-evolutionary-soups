"""Eq. (1), the inference-time selection over the evolved front."""

import numpy as np
import pytest

from evolutionary_soups.preference import (
    linear_utility,
    preference_selection,
    sample_preferences,
    selection_utility,
    tchebyshev_utility,
)

FRONT = np.array([[1.0, 0.0], [0.8, 0.35], [0.55, 0.6], [0.3, 0.85], [0.0, 1.0]])


def test_selection_matches_brute_force():
    linear = selection_utility("linear")
    for mu in sample_preferences(2, 11):
        chosen = preference_selection(FRONT, mu, linear)
        best = max(range(len(FRONT)), key=lambda i: linear_utility(FRONT[i], mu))
        assert chosen == best
        assert linear_utility(FRONT[chosen], mu) >= max(
            linear_utility(r, mu) for r in FRONT
        ) - 1e-12


def test_extreme_preferences_pick_the_extreme_members():
    """mu = e_j maximizes objective j alone, so it must select that end of the front."""
    for j in range(2):
        mu = np.eye(2)[j]
        chosen = preference_selection(FRONT, mu, selection_utility("linear"))
        assert chosen == int(np.argmax(FRONT[:, j]))


def test_selection_costs_one_utility_evaluation_per_front_member():
    calls = []

    def counting(reward, mu):
        calls.append(reward)
        return linear_utility(reward, mu)

    preference_selection(FRONT, np.array([0.5, 0.5]), counting)
    assert len(calls) == len(FRONT)


def test_tchebyshev_reaches_a_point_linear_utility_never_picks():
    """The non-convex case of Section 2, why Eq. (1) also carries the Tchebyshev form.

    The middle point sits under the line joining the two ends, so no weighted sum
    ever ranks it first. Its worst-case distance to the ideal point is the
    smallest of the three, so the Tchebyshev utility takes it at mu = [0.5, 0.5].
    """
    front = np.array([[1.0, 0.0], [0.45, 0.45], [0.0, 1.0]])
    linear = selection_utility("linear")
    tcheb = selection_utility("tchebyshev")
    picked_by_linear = {
        preference_selection(front, mu, linear) for mu in sample_preferences(2, 21)
    }
    assert 1 not in picked_by_linear
    assert preference_selection(front, np.array([0.5, 0.5]), tcheb) == 1


def test_tchebyshev_is_a_distance_to_the_ideal_point():
    mu = np.array([0.3, 0.7])
    assert tchebyshev_utility(np.array([1.0, 1.0]), mu) == pytest.approx(0.0)
    assert tchebyshev_utility(np.array([1.0, 0.0]), mu) == pytest.approx(0.7)
    assert tchebyshev_utility(np.array([0.0, 1.0]), mu, ideal=np.array([1.0, 1.0])) == (
        pytest.approx(0.3)
    )


@pytest.mark.parametrize("n,count", [(2, 11), (3, 21)])
def test_preference_grid_lies_on_the_simplex(n, count):
    grid = sample_preferences(n, count)
    assert grid.shape == (count, n)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0.0)
    assert len({tuple(np.round(row, 6)) for row in grid}) == count
