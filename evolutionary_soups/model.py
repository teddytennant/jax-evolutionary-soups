"""A toy transformer with N LoRA experts over one shared frozen FFN, Eq. (2).

The paper's MoE keeps a single frozen backbone and loads the experts as LoRA
increments over the feed-forward sublayer of every transformer layer. At layer l
the merged output is

    o^(l) = FFN^(l)(h_ATTN^(l)) + sum_i lambda_i^(l) Delta_i^(l)(h_ATTN^(l)),

with lambda^(l) coming from the gating network (see gating.py). Here the model is
a few thousand parameters wide and the "reward models" are synthetic, so the
whole thing runs on CPU. Nothing about the merging arithmetic changes with scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 24
    seq_len: int = 8
    d_model: int = 16
    d_ff: int = 32
    n_layers: int = 3
    n_out: int = 4
    lora_rank: int = 4
    n_experts: int = 2


def _normal(key, shape, scale):
    return jax.random.normal(key, shape) * scale


def init_backbone(key, cfg):
    """Frozen base model: embeddings, L attention plus FFN blocks, a readout."""
    keys = jax.random.split(key, 3 + 4 * cfg.n_layers)
    it = iter(keys)
    s = 1.0 / jnp.sqrt(cfg.d_model)
    params = {
        "embed": _normal(next(it), (cfg.vocab, cfg.d_model), 1.0),
        "pos": _normal(next(it), (cfg.seq_len, cfg.d_model), 0.1),
        "readout": _normal(next(it), (cfg.d_model, cfg.n_out), s),
        "readout_bias": jnp.zeros((cfg.n_out,)),
        "layers": [],
    }
    for _ in range(cfg.n_layers):
        params["layers"].append(
            {
                "wqkv": _normal(next(it), (3, cfg.d_model, cfg.d_model), s),
                "wo": _normal(next(it), (cfg.d_model, cfg.d_model), s),
                "w_in": _normal(next(it), (cfg.d_model, cfg.d_ff), s),
                "b_in": jnp.zeros((cfg.d_ff,)),
                "w_out": _normal(next(it), (cfg.d_ff, cfg.d_model), 1.0 / jnp.sqrt(cfg.d_ff)),
                "b_out": jnp.zeros((cfg.d_model,)),
            }
        )
    return params


def init_experts(key, cfg):
    """N LoRA increments over the shared FFN, one pair of factors per layer.

    B is zero at init, so every expert starts as the identity on the backbone,
    which is the usual LoRA convention and the starting point of the paper's
    per-objective SFT plus PPO stage.
    """
    keys = jax.random.split(key, cfg.n_experts * cfg.n_layers)
    experts = []
    for i in range(cfg.n_experts):
        layers = []
        for l in range(cfg.n_layers):
            k = keys[i * cfg.n_layers + l]
            layers.append(
                {
                    "a": _normal(k, (cfg.d_model, cfg.lora_rank), 1.0 / jnp.sqrt(cfg.d_model)),
                    "b": jnp.zeros((cfg.lora_rank, cfg.d_model)),
                }
            )
        experts.append(layers)
    return experts


def _layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _attention(x, layer):
    q, k, v = jnp.einsum("btd,hde->hbte", x, layer["wqkv"])
    scores = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bts,bsd->btd", weights, v) @ layer["wo"]


def expert_delta(expert_layer, x):
    """Delta_i^(l)(x), expert i's LoRA increment over the shared frozen FFN."""
    return (x @ expert_layer["a"]) @ expert_layer["b"]


def merged_ffn(layer, expert_layers, coeffs, h_attn):
    """Eq. (2): shared FFN output plus the lambda-weighted expert increments.

    coeffs has shape (batch, tokens, N) and is applied per token and per layer,
    which is what "coefficients are computed independently at each layer" means
    once the gate reads a per-token hidden state.
    """
    x = _layer_norm(h_attn)
    base = jax.nn.gelu(x @ layer["w_in"] + layer["b_in"]) @ layer["w_out"] + layer["b_out"]
    deltas = jnp.stack([expert_delta(e, x) for e in expert_layers], axis=-2)
    return base + jnp.einsum("btn,btnd->btd", coeffs, deltas)


def forward(backbone, experts, tokens, coeff_fn):
    """Run the MoE. coeff_fn(h_attn, prompt_embedding, layer) -> (B, T, N)."""
    h = backbone["embed"][tokens] + backbone["pos"][: tokens.shape[1]]
    prompt_embedding = jnp.mean(h, axis=1)
    for l, layer in enumerate(backbone["layers"]):
        h_attn = h + _attention(_layer_norm(h), layer)
        coeffs = coeff_fn(h_attn, prompt_embedding, l)
        expert_layers = [e[l] for e in experts]
        h = h_attn + merged_ffn(layer, expert_layers, coeffs, h_attn)
    pooled = jnp.mean(_layer_norm(h), axis=1)
    return pooled @ backbone["readout"] + backbone["readout_bias"]
