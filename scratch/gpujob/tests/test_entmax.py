"""alpha-entmax, the operator Eq. (2) puts on the gate logits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evolutionary_soups.entmax import entmax, entmax_threshold, sparsemax

ALPHAS = [1.0, 1.05, 1.2, 1.5, 2.0, 3.0]


def logits(seed=0, shape=(6, 5), scale=2.0):
    return jax.random.normal(jax.random.PRNGKey(seed), shape) * scale


@pytest.mark.parametrize("alpha", ALPHAS)
def test_lands_on_the_simplex(alpha):
    p = np.asarray(entmax(logits(), alpha))
    assert np.all(p >= 0.0)
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-12)


def test_alpha_one_is_softmax():
    z = logits(1)
    assert np.allclose(entmax(z, 1.0), jax.nn.softmax(z, axis=-1), atol=1e-12)


def test_approaches_softmax_from_above():
    z = logits(1)
    near = np.asarray(entmax(z, 1.0 + 1e-6))
    assert np.allclose(near, jax.nn.softmax(z, axis=-1), atol=1e-4)


def test_alpha_two_is_sparsemax():
    z = logits(2)
    assert np.allclose(entmax(z, 2.0), sparsemax(z), atol=1e-12)


def test_only_alpha_above_one_gives_exact_zeros():
    z = logits(3, scale=3.0)
    soft = np.asarray(entmax(z, 1.0))
    assert np.all(soft > 0.0)
    for alpha in (1.2, 1.5, 2.0):
        p = np.asarray(entmax(z, alpha))
        assert np.any(p == 0.0), f"alpha={alpha} produced no exact zero"


def test_reaches_a_vertex_exactly():
    """lambda_i = 1 exactly, the claim Section 3.2 makes about entmax."""
    z = jnp.array([12.0, 0.0, -1.0, 0.5])
    for alpha in (1.2, 1.5, 2.0):
        p = np.asarray(entmax(z, alpha))
        assert np.array_equal(p, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(np.asarray(entmax(z, 1.0)) > 0.0)


@pytest.mark.parametrize("alpha", [1.2, 1.5, 2.0, 3.0])
def test_threshold_solves_its_own_fixed_point(alpha):
    """sum_i [(alpha-1) z_i - tau]_+ ^ (1/(alpha-1)) = 1 at the bisected tau."""
    z = logits(4)
    tau = entmax_threshold(z, alpha)
    x = np.asarray(z) * (alpha - 1.0)
    p = np.maximum(x - np.asarray(tau), 0.0) ** (1.0 / (alpha - 1.0))
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-10)


@pytest.mark.parametrize("alpha", [1.0, 1.2, 1.5, 2.0])
def test_gradient_matches_finite_differences(alpha):
    z = np.asarray(logits(5, shape=(5,), scale=1.0))
    w = np.asarray(jax.random.normal(jax.random.PRNGKey(6), (5,)))

    def f(x):
        return jnp.sum(w * entmax(x, alpha))

    grad = np.asarray(jax.grad(f)(jnp.asarray(z)))
    eps = 1e-6
    fd = np.empty_like(z)
    for i in range(len(z)):
        step = np.zeros_like(z)
        step[i] = eps
        fd[i] = (f(jnp.asarray(z + step)) - f(jnp.asarray(z - step))) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-6)


@pytest.mark.parametrize("alpha", [1.0, 1.2, 1.5, 2.0])
def test_jacobian_is_symmetric(alpha):
    z = logits(7, shape=(5,), scale=1.0)
    jac = np.asarray(jax.jacfwd(lambda x: entmax(x, alpha))(z))
    assert np.allclose(jac, jac.T, atol=1e-12)
    assert np.allclose(jac.sum(axis=0), 0.0, atol=1e-12)  # the output stays on the simplex


def test_batches_independently():
    z = logits(8, shape=(3, 7))
    p = np.asarray(entmax(z, 1.5))
    rows = np.stack([np.asarray(entmax(row, 1.5)) for row in z])
    assert np.allclose(p, rows, atol=1e-14)
