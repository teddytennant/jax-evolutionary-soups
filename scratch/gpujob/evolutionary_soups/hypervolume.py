"""Pareto dominance, the hypervolume indicator, and greedy HVC selection.

Definitions 1 to 3 and Eq. (3) of the paper, plus Algorithm 1 lines 8 to 12.
All of this is set arithmetic on small reward matrices, so it runs in numpy on
the host rather than in JAX. Every objective is maximized and the reference
point is the paper's r = [-0.1]^n.
"""

from __future__ import annotations

import numpy as np

REFERENCE_POINT = -0.1  # Section 2, hypervolume reference r = [-0.1]^n


def reference_point(n_objectives, value=REFERENCE_POINT):
    return np.full(n_objectives, float(value))


def dominates(a, b):
    """Definition 1: a > b if a is at least as good everywhere and better once."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a >= b) and np.any(a > b))


def nondominated_mask(points):
    """Definition 2: mask of the points no other point dominates."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    m = len(pts)
    keep = np.ones(m, dtype=bool)
    for i in range(m):
        if not keep[i]:
            continue
        ge = np.all(pts >= pts[i], axis=1)
        gt = np.any(pts > pts[i], axis=1)
        if np.any(ge & gt):
            keep[i] = False
    return keep


def nondominated(points):
    """Indices of the non-dominated front, Algorithm 1 line 7."""
    return np.flatnonzero(nondominated_mask(points))


def _hv2d(pts, ref):
    """Exact two-objective hypervolume by a single sweep, the recursion base case."""
    order = np.argsort(-pts[:, 0], kind="stable")
    xs, ys = pts[order, 0], pts[order, 1]
    prev = np.empty_like(ys)
    prev[0] = ref[1]
    if len(ys) > 1:
        prev[1:] = np.maximum.accumulate(ys)[:-1]
    return float(np.sum((xs - ref[0]) * np.maximum(ys - prev, 0.0)))


def hypervolume(points, ref):
    """Definition 3: volume of the union of boxes [f(g), r] over the set.

    Dimension sweep. Sort on the last objective, then for each prefix of the
    sorted points multiply the slab depth by the hypervolume of the prefix
    projected to the remaining objectives, down to a two-objective sweep. Exact
    for any number of objectives and fast enough for the population sizes here.
    """
    ref = np.asarray(ref, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0:
        return 0.0
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    if pts.shape[1] == 1:
        return float(pts[:, 0].max() - ref[0])
    if pts.shape[1] == 2:
        return _hv2d(pts, ref)
    order = np.argsort(-pts[:, -1], kind="stable")
    pts = pts[order]
    edges = np.append(pts[1:, -1], ref[-1])
    total = 0.0
    for k in range(len(pts)):
        depth = pts[k, -1] - edges[k]
        if depth <= 0.0:
            continue
        total += depth * hypervolume(pts[: k + 1, :-1], ref[:-1])
    return float(total)


def hv_contribution(point, points, ref):
    """Eq. (3): Delta_HV(g, S) = HV(S union {g}) - HV(S)."""
    point = np.asarray(point, dtype=float).reshape(1, -1)
    if len(points) == 0:
        return hypervolume(point, ref)
    joined = np.concatenate([np.atleast_2d(points), point], axis=0)
    return hypervolume(joined, ref) - hypervolume(points, ref)


def greedy_hvc_selection(points, size, ref):
    """Algorithm 1 lines 8 to 12: build S additively by largest Delta_HV.

    Starts from the empty set and repeatedly adds argmax_g Delta_HV(g, S) until
    |S| reaches `size`. A candidate whose box is already covered contributes
    nothing, so selection is pushed toward regions the retained set misses.
    Ties go to the lower index, which keeps the result deterministic. Returns
    the selected indices in the order they were picked. The gain below is
    hv_contribution with HV(S) carried along instead of recomputed.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    size = int(min(size, len(pts)))
    chosen = []
    current = np.zeros((0, pts.shape[1]))
    current_hv = 0.0
    remaining = list(range(len(pts)))
    while len(chosen) < size:
        gains = [
            hypervolume(np.concatenate([current, pts[i : i + 1]]), ref) - current_hv
            for i in remaining
        ]
        best = int(np.argmax(gains))
        idx = remaining.pop(best)
        chosen.append(idx)
        current = np.concatenate([current, pts[idx : idx + 1]])
        current_hv = current_hv + gains[best]
    return chosen


def crowding_distance_selection(points, size):
    """NSGA-II crowding distance selection, the ablation of Section 4.3.

    Deb et al. (2002). Ranks by the sum over objectives of the normalized gap
    between a point's two neighbours, boundary points getting infinity. It
    measures Euclidean spacing, with no tie to the volume the set covers, which
    is the gap greedy HVC is meant to close.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    m, n = pts.shape
    size = int(min(size, m))
    distance = np.zeros(m)
    for j in range(n):
        order = np.argsort(pts[:, j], kind="stable")
        span = pts[order[-1], j] - pts[order[0], j]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if span <= 0:
            continue
        for k in range(1, m - 1):
            distance[order[k]] += (pts[order[k + 1], j] - pts[order[k - 1], j]) / span
    return list(np.argsort(-distance, kind="stable")[:size])
