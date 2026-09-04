"""alpha-entmax, the simplex map used by the gating network in Eq. (2).

The paper writes lambda^(l) = entmax_alpha(MLP_gate(h_ATTN^(l))) and relies on
one property of that operator: unlike softmax it can return exact 0 and exact 1,
so lambda_i = 1 recovers expert i's output and the merged hidden state can sit on
the Pareto boundary instead of strictly inside it (Section 3.2, Theorem 2).

alpha-entmax (Peters et al., 2019) is the argmax of <p, z> plus the Tsallis
entropy H_alpha over the simplex. Its solution has the closed form

    p_i = [(alpha - 1) z_i - tau]_+ ^ (1 / (alpha - 1)),

with the threshold tau fixed by sum_i p_i = 1. alpha = 1 is softmax and alpha = 2
is sparsemax. There is no closed form for tau at general alpha, so we bisect on
the bracket [max(x) - 1, max(x) - (1/d)^(alpha-1)] with x = (alpha - 1) z, which
brackets the root because sum_i p_i is decreasing in tau and is >= 1 at the low
end and <= 1 at the high end. The paper's experiments use alpha = 1.2.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

DEFAULT_ALPHA = 1.2  # Appendix C, "fixed_alpha": 1.2
DEFAULT_N_ITER = 50


def _p_of_tau(x, tau, alpha):
    """[x - tau]_+ ^ (1 / (alpha - 1)), the per-coordinate solution map."""
    z = jnp.maximum(x - tau, 0.0)
    return jnp.where(z > 0, jnp.where(z > 0, z, 1.0) ** (1.0 / (alpha - 1.0)), 0.0)


def entmax_threshold(logits, alpha=DEFAULT_ALPHA, n_iter=DEFAULT_N_ITER):
    """Bisect for the threshold tau of alpha-entmax along the last axis.

    Returns tau with a trailing axis of size 1, so that
    sum_i [(alpha - 1) logits_i - tau]_+ ^ (1/(alpha-1)) = 1 up to the bisection
    tolerance. Only defined for alpha > 1.
    """
    if alpha <= 1.0:
        raise ValueError("entmax_threshold needs alpha > 1; alpha = 1 is softmax")
    d = logits.shape[-1]
    x = logits * (alpha - 1.0)
    top = jnp.max(x, axis=-1, keepdims=True)
    tau = top - 1.0  # sum p >= 1 here, since the largest coordinate alone is 1
    span = (1.0 / d) ** (alpha - 1.0)
    width = (1.0 - span) * jnp.ones_like(tau)  # the bracket is this wide

    def halve(_, carry):
        tau, width = carry
        width = width * 0.5
        mid = tau + width
        excess = jnp.sum(_p_of_tau(x, mid, alpha), axis=-1, keepdims=True) - 1.0
        return jnp.where(excess >= 0, mid, tau), width

    tau, _ = jax.lax.fori_loop(0, n_iter, halve, (tau, width))
    return tau


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _entmax(logits, alpha, n_iter):
    if alpha == 1.0:
        return jax.nn.softmax(logits, axis=-1)
    tau = entmax_threshold(logits, alpha, n_iter)
    p = _p_of_tau(logits * (alpha - 1.0), tau, alpha)
    return p / jnp.sum(p, axis=-1, keepdims=True)


@_entmax.defjvp
def _entmax_jvp(alpha, n_iter, primals, tangents):
    """Jacobian of alpha-entmax (Peters et al., 2019, Prop. 2).

    With s_i = p_i^(2 - alpha) on the support and 0 off it,
    dp/dz = diag(s) - s s^T / sum(s), which is symmetric. At alpha = 1 this is
    the softmax Jacobian diag(p) - p p^T and at alpha = 2 the sparsemax one.
    """
    (logits,), (dlogits,) = primals, tangents
    p = _entmax(logits, alpha, n_iter)
    safe = jnp.where(p > 0, p, 1.0)
    s = jnp.where(p > 0, safe ** (2.0 - alpha), 0.0)
    weighted = jnp.sum(s * dlogits, axis=-1, keepdims=True) / jnp.sum(
        s, axis=-1, keepdims=True
    )
    return p, s * (dlogits - weighted)


def entmax(logits, alpha=DEFAULT_ALPHA, n_iter=DEFAULT_N_ITER):
    """alpha-entmax over the last axis, the operator in Eq. (2).

    alpha must be a Python float, it is static. alpha = 1.0 dispatches to
    softmax; alpha > 1 solves for the threshold by bisection and renormalizes so
    the output sums to 1 in floating point. Exact zeros stay exactly zero.
    """
    return _entmax(logits, float(alpha), int(n_iter))


def sparsemax(logits):
    """Sparsemax, alpha-entmax at alpha = 2, by the sort-based closed form.

    Kept as an independent reference for the bisection at alpha = 2.
    """
    d = logits.shape[-1]
    srt = jnp.sort(logits, axis=-1)[..., ::-1]
    csum = jnp.cumsum(srt, axis=-1)
    k = jnp.arange(1, d + 1, dtype=logits.dtype)
    support = srt * k > (csum - 1.0)
    size = jnp.sum(support, axis=-1, keepdims=True)
    tau = (jnp.take_along_axis(csum, size - 1, axis=-1) - 1.0) / size
    return jnp.maximum(logits - tau, 0.0)
