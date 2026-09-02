"""The gating network of Section 3.2 and the two classes it strictly contains.

A 2-layer MLP, d -> 256 -> N, whose parameters are shared across all transformer
layers and which reads that layer's attention-sublayer hidden state. Its output
passes through alpha-entmax to land on the simplex over the N experts.

Three gate classes, matching Theorem 2 and the ablation of Section 4.3:

  "per_layer"  G,          lambda^(l) = entmax(MLP(h_ATTN^(l))), one per layer
  "single"     G_single,   one lambda from the prompt embedding, reused at every layer
  "fixed"      G_fixed,    one lambda, the same for every layer and every input

G_fixed and G_single sit inside G as parameter subsets: zero the first layer
weights and you get a constant gate, feed the same hidden state at every layer
and you get a single gate. Theorem 2 says the containment of their images in
objective space is strict.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .entmax import DEFAULT_ALPHA, entmax

MODES = ("per_layer", "single", "fixed")
PAPER_HIDDEN = 256  # the paper's d -> 256 -> N gate


def init_gate(key, d_model, n_experts, hidden=PAPER_HIDDEN, bias_scale=1.5, weight_scale=0.1):
    """Random initialization of one individual, Algorithm 1 line 1.

    Two deliberate choices, neither of which the paper pins down. The output bias
    is drawn wide, so entmax lands near the corners of the simplex and the
    starting population already spans the merging space instead of clustering on
    the uniform mix. The weights are drawn small, so an individual starts close
    to a constant gate, the class G_fixed that G contains, and evolution has to
    grow context dependence rather than start from noise.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (d_model, hidden)) * (weight_scale / jnp.sqrt(d_model)),
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, n_experts)) * (weight_scale / jnp.sqrt(hidden)),
        "b2": jax.random.normal(k3, (n_experts,)) * bias_scale,
    }


def init_population(
    key, size, d_model, n_experts, hidden=PAPER_HIDDEN, bias_scale=1.5, weight_scale=0.1
):
    """A stacked population of `size` gates, leading axis is the individual."""
    keys = jax.random.split(key, size)
    return jax.vmap(
        lambda k: init_gate(k, d_model, n_experts, hidden, bias_scale, weight_scale)
    )(keys)


def gate_logits(params, h):
    """MLP_gate(h). The paper does not name the hidden activation; tanh here."""
    return jnp.tanh(h @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


def coefficients(params, h, mode="per_layer", alpha=DEFAULT_ALPHA):
    """lambda = entmax_alpha(MLP_gate(h)) for the given gate class."""
    if mode == "fixed":
        logits = jnp.broadcast_to(params["b2"], h.shape[:-1] + params["b2"].shape)
    else:
        logits = gate_logits(params, h)
    return entmax(logits, alpha)


def coeff_fn(params, mode="per_layer", alpha=DEFAULT_ALPHA):
    """Build the callable that model.forward asks for at each layer."""

    def fn(h_attn, prompt_embedding, layer):
        del layer
        if mode == "per_layer":
            return coefficients(params, h_attn, "per_layer", alpha)
        if mode == "single":
            lam = coefficients(params, prompt_embedding, "single", alpha)
            return jnp.broadcast_to(lam[:, None, :], h_attn.shape[:-1] + lam.shape[-1:])
        if mode == "fixed":
            return coefficients(params, h_attn, "fixed", alpha)
        raise ValueError(f"unknown gate mode {mode!r}, expected one of {MODES}")

    return fn


def constant_coeff_fn(lam):
    """A gate pinned to a given lambda, the fixed-coefficient soup of Rewarded Soups."""

    def fn(h_attn, prompt_embedding, layer):
        del prompt_embedding, layer
        return jnp.broadcast_to(lam, h_attn.shape[:-1] + lam.shape[-1:])

    return fn
