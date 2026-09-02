"""Hypervolume (Definition 3), Delta_HV (Eq. 3), and greedy HVC selection."""

import itertools

import numpy as np
import pytest

from evolutionary_soups.hypervolume import (
    crowding_distance_selection,
    dominates,
    greedy_hvc_selection,
    hv_contribution,
    hypervolume,
    nondominated,
    reference_point,
)

REF2 = reference_point(2)
REF3 = reference_point(3)


def monte_carlo_hv(points, ref, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    top = np.max(points, axis=0)
    sample = rng.uniform(ref, top, size=(n, len(ref)))
    covered = np.any(np.all(sample[:, None, :] <= points[None, :, :], axis=2), axis=1)
    return float(np.prod(top - ref) * covered.mean())


def test_two_dimensional_closed_form():
    # One box, then two boxes overlapping in the corner nearest the reference.
    assert hypervolume(np.array([[1.0, 0.0]]), REF2) == pytest.approx(1.1 * 0.1)
    pts = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert hypervolume(pts, REF2) == pytest.approx(1.1 * 0.1 + 0.1 * 1.1 - 0.1 * 0.1)
    pts = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    expected = 0.6 * 0.6 + (1.1 - 0.6) * 0.1 + 0.1 * (1.1 - 0.6)
    assert hypervolume(pts, REF2) == pytest.approx(expected)


@pytest.mark.parametrize("n_obj,seed", [(2, 0), (2, 1), (3, 2), (3, 3)])
def test_matches_monte_carlo(n_obj, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(9, n_obj))
    ref = reference_point(n_obj)
    assert hypervolume(pts, ref) == pytest.approx(monte_carlo_hv(pts, ref, seed=seed), rel=0.01)


def test_monotone_and_blind_to_dominated_points():
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.0, 1.0, size=(6, 3))
    base = hypervolume(pts, REF3)
    grown = hypervolume(np.vstack([pts, rng.uniform(0.0, 1.0, size=(3, 3))]), REF3)
    assert grown >= base - 1e-12
    dominated = pts[0] - 0.05
    assert dominates(pts[0], dominated)
    assert hypervolume(np.vstack([pts, dominated]), REF3) == pytest.approx(base)


def test_points_below_the_reference_contribute_nothing():
    assert hypervolume(np.array([[-0.5, 0.5]]), REF2) == 0.0
    assert hypervolume(np.zeros((0, 2)), REF2) == 0.0


def test_contribution_of_a_member_is_zero():
    """Eq. (3) with g already in S."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.0, 1.0, size=(5, 2))
    for i in range(len(pts)):
        assert hv_contribution(pts[i], pts, REF2) == pytest.approx(0.0, abs=1e-12)
    outside = np.array([1.5, 1.5])
    assert hv_contribution(outside, pts, REF2) > 0.0


def clustered_front():
    """A front with one dense cluster and two isolated extremes.

    The extremes are thin slivers, so they carry little volume, while the knee
    and its two neighbours carry most of it. Crowding distance hands both
    extremes an infinite score and then splits the cluster; hypervolume does not.
    """
    return np.array(
        [
            [1.00, 0.01],
            [0.72, 0.66],
            [0.70, 0.70],
            [0.66, 0.72],
            [0.01, 1.00],
        ]
    )


def test_greedy_hvc_beats_crowding_distance():
    pts = clustered_front()
    assert len(nondominated(pts)) == len(pts)
    greedy = hypervolume(pts[greedy_hvc_selection(pts, 3, REF2)], REF2)
    crowding = hypervolume(pts[crowding_distance_selection(pts, 3)], REF2)
    assert greedy > crowding


def test_greedy_hvc_spreads_its_first_picks():
    """A candidate overlapping the retained set contributes little, so it waits."""
    pts = clustered_front()
    picks = greedy_hvc_selection(pts, 3, REF2)
    assert picks[0] == 2  # the knee, the single largest box
    assert set(picks) == {0, 2, 4}  # then the two extremes, not the cluster


def test_greedy_hvc_is_not_exact_in_two_dimensions():
    """Lemma 8 claims greedy hypervolume subset selection is exact for n <= 3.

    It is not, and the counterexample is small: the point with the largest single
    box is the wrong first pick when the two extremes barely overlap each other.
    Greedy is committed to it, so it ends up below the best pair. The gap is
    small but it is why Theorem 3's per-chunk monotonicity can dip.
    """
    pts = np.array([[10.0, 1.0], [1.0, 10.0], [3.2, 3.2]])
    ref = np.zeros(2)
    greedy = hypervolume(pts[greedy_hvc_selection(pts, 2, ref)], ref)
    best = max(
        hypervolume(pts[list(c)], ref) for c in itertools.combinations(range(3), 2)
    )
    assert greedy == pytest.approx(17.04)
    assert best == pytest.approx(19.0)
    assert greedy < best


def test_greedy_hvc_is_usually_optimal():
    """Over random fronts the greedy choice matches exhaustive search almost always."""
    rng = np.random.default_rng(11)
    losses = 0
    trials = 0
    for _ in range(600):
        pts = rng.uniform(0.0, 1.0, size=(9, 2))
        pts = pts[nondominated(pts)]
        if len(pts) < 4:
            continue
        trials += 1
        greedy = hypervolume(pts[greedy_hvc_selection(pts, 3, REF2)], REF2)
        best = max(
            hypervolume(pts[list(c)], REF2)
            for c in itertools.combinations(range(len(pts)), 3)
        )
        if greedy < best - 1e-12:
            losses += 1
    assert trials > 120
    assert losses / trials < 0.05


def test_nondominated_filter():
    pts = np.array([[1.0, 0.0], [0.5, 0.5], [0.4, 0.4], [0.0, 1.0], [0.5, 0.5]])
    keep = nondominated(pts)
    assert set(keep) == {0, 1, 3, 4}  # duplicates do not dominate each other
    assert dominates(np.array([0.5, 0.5]), np.array([0.4, 0.4]))
    assert not dominates(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
