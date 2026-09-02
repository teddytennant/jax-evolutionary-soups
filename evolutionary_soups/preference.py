"""Inference-time preference selection, Eq. (1) and Theorem 1.

The user preference mu never touches the gating network. It enters once, at the
end, to pick which member of the evolved front P_hat to run. That is an O(|P_hat|)
lookup over stored reward vectors with no cost in the forward pass.
"""

from __future__ import annotations

import numpy as np


def linear_utility(reward, mu):
    """u_lin_mu(r) = sum_i mu_i r_i, optimal when the front is convex."""
    return float(np.dot(np.asarray(mu, dtype=float), np.asarray(reward, dtype=float)))


def tchebyshev_utility(reward, mu, ideal=None):
    """u_tch_mu(r) = max_i mu_i |r_i - r*_i|, the worst-case gap to the ideal point.

    Lower is better, so preference_selection negates it before the arg max.
    """
    reward = np.asarray(reward, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ideal = np.ones_like(reward) if ideal is None else np.asarray(ideal, dtype=float)
    return float(np.max(mu * np.abs(reward - ideal)))


def selection_utility(kind="linear", ideal=None):
    """Eq. (1) needs a single arg max, so Tchebyshev enters negated (Theorem 1, Note)."""
    if kind == "linear":
        return lambda reward, mu: linear_utility(reward, mu)
    if kind == "tchebyshev":
        return lambda reward, mu: -tchebyshev_utility(reward, mu, ideal)
    raise ValueError(f"unknown utility {kind!r}")


def preference_selection(front_rewards, mu, utility=None):
    """Eq. (1): g_mu = arg max over g in P_hat of u_mu(f(g)).

    Evaluates the utility once per front member and returns the winning index,
    so the cost is |P_hat| utility evaluations and nothing else.
    """
    utility = selection_utility("linear") if utility is None else utility
    rewards = np.atleast_2d(np.asarray(front_rewards, dtype=float))
    values = [utility(r, mu) for r in rewards]
    return int(np.argmax(values))


def sample_preferences(n_objectives, n_points):
    """The evaluation grid of Section 4.1: 11 preferences for n = 2, 21 for n = 3."""
    if n_objectives == 2:
        w = np.linspace(0.0, 1.0, n_points)
        return np.stack([w, 1.0 - w], axis=1)
    if n_objectives == 3:
        k = 0
        while (k + 1) * (k + 2) // 2 < n_points:
            k += 1
        grid = [
            (i / k, j / k, (k - i - j) / k)
            for i in range(k + 1)
            for j in range(k + 1 - i)
        ]
        return np.asarray(grid[:n_points], dtype=float)
    raise ValueError("preference grids are implemented for 2 and 3 objectives")
