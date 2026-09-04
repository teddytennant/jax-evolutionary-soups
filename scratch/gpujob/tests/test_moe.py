"""The merging arithmetic of Eq. (2) and the mechanism behind Theorem 2."""

import jax
import jax.numpy as jnp
import numpy as np

from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod
from evolutionary_soups.gating import coeff_fn, coefficients, init_gate
from evolutionary_soups.hypervolume import hypervolume, nondominated
from evolutionary_soups.model import (
    ModelConfig,
    forward,
    init_backbone,
    init_experts,
    merged_ffn,
)

CFG = ModelConfig(n_experts=3)


def toy_model(seed=0):
    """A backbone and three experts with non-zero LoRA factors."""
    k_back, k_exp, k_b = jax.random.split(jax.random.PRNGKey(seed), 3)
    backbone = init_backbone(k_back, CFG)
    experts = init_experts(k_exp, CFG)
    keys = jax.random.split(k_b, CFG.n_experts * CFG.n_layers).reshape(
        CFG.n_experts, CFG.n_layers, -1
    )
    experts = [
        [
            {"a": e["a"], "b": jax.random.normal(jax.random.wrap_key_data(keys[i, l]), e["b"].shape) * 0.5}
            for l, e in enumerate(layers)
        ]
        for i, layers in enumerate(experts)
    ]
    return backbone, experts


def hidden_states(seed=1, batch=4):
    return jax.random.normal(jax.random.PRNGKey(seed), (batch, CFG.seq_len, CFG.d_model))


def test_vertex_coefficients_recover_one_expert_exactly():
    """"lambda_i = 1 recovers expert i's output exactly" (Section 3.2), bit for bit."""
    backbone, experts = toy_model()
    layer, expert_layers = backbone["layers"][0], [e[0] for e in experts]
    h = hidden_states()
    for i in range(CFG.n_experts):
        lam = jnp.broadcast_to(jnp.eye(CFG.n_experts)[i], h.shape[:-1] + (CFG.n_experts,))
        merged = merged_ffn(layer, expert_layers, lam, h)
        alone = merged_ffn(layer, [expert_layers[i]], lam[..., i : i + 1], h)
        assert np.array_equal(np.asarray(merged), np.asarray(alone))


def test_entmax_gating_attains_a_vertex_and_softmax_never_does():
    """Theorem 2's mechanism: entmax can sit on the boundary, softmax cannot."""
    backbone, experts = toy_model()
    layer, expert_layers = backbone["layers"][0], [e[0] for e in experts]
    h = hidden_states()
    gate = init_gate(jax.random.PRNGKey(2), CFG.d_model, CFG.n_experts, hidden=8)
    gate = {**gate, "w1": gate["w1"] * 0.0, "b2": jnp.array([9.0, 0.0, 0.0])}

    def pure(i):
        lam = jnp.broadcast_to(jnp.eye(CFG.n_experts)[i], h.shape[:-1] + (CFG.n_experts,))
        return np.asarray(merged_ffn(layer, expert_layers, lam, h))

    sparse = np.asarray(coefficients(gate, h, "per_layer", 1.5))
    assert np.array_equal(sparse, np.broadcast_to(np.eye(3)[0], sparse.shape))
    merged = np.asarray(merged_ffn(layer, expert_layers, jnp.asarray(sparse), h))
    assert np.array_equal(merged, pure(0))

    dense = np.asarray(coefficients(gate, h, "per_layer", 1.0))
    assert np.all(dense > 0.0)
    soft_merged = np.asarray(merged_ffn(layer, expert_layers, jnp.asarray(dense), h))
    hull = np.stack([pure(i) for i in range(CFG.n_experts)])
    for vertex in hull:
        assert not np.allclose(soft_merged, vertex)
    assert np.all(soft_merged <= hull.max(axis=0) + 1e-9)
    assert np.all(soft_merged >= hull.min(axis=0) - 1e-9)


def test_coefficients_vary_across_layers_and_prompts():
    """Lemma 5 and Lemma 6 need lambda to move with the input and with the layer."""
    backbone, experts = toy_model()
    gate = init_gate(jax.random.PRNGKey(3), CFG.d_model, CFG.n_experts, hidden=8, weight_scale=2.0)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (6, CFG.seq_len), 0, CFG.vocab)
    seen = []

    def watch(h_attn, prompt_embedding, layer):
        lam = coefficients(gate, h_attn, "per_layer", 1.2)
        seen.append(np.asarray(lam))
        return lam

    forward(backbone, experts, tokens, watch)
    assert len(seen) == CFG.n_layers
    assert not np.allclose(seen[0], seen[1])  # per layer
    assert not np.allclose(seen[0][0], seen[0][1])  # per prompt
    assert not np.allclose(seen[0][0, 0], seen[0][0, 1])  # per token

    single = coeff_fn(gate, "single", 1.2)
    lam = np.asarray(single(jnp.zeros((6, CFG.seq_len, CFG.d_model)), jnp.ones((6, CFG.d_model)), 0))
    assert np.allclose(lam, lam[:, :1])  # one coefficient, reused everywhere


def test_experts_are_specialists(setup):
    """Each expert should win its own objective, or the trade-off is not a trade-off."""
    assert np.array_equal(np.argmax(setup.expert_rewards, axis=0), np.arange(3))


def test_the_fixed_gate_class_is_a_parameter_subset_of_the_per_layer_one():
    """G_fixed sits inside G exactly: zero the output weights and G is constant.

    This is the half of Theorem 2's containment that holds as an identity here,
    so it is worth pinning down rather than inferring from a hypervolume. Note
    that the other half does not: G_single reads the prompt embedding, which a
    per-layer gate never sees, so no choice of per-layer weights reproduces a
    single gate's lambda. See the README.
    """
    gate = init_gate(jax.random.PRNGKey(11), CFG.d_model, CFG.n_experts, hidden=8)
    flattened = {**gate, "w2": jnp.zeros_like(gate["w2"])}
    h = hidden_states(seed=12)
    constant = np.asarray(coefficients(gate, h, "fixed"))
    assert np.array_equal(np.asarray(coefficients(flattened, h, "per_layer")), constant)
    assert np.allclose(constant, constant[0, 0])


def test_per_layer_region_strictly_contains_the_single_gating_one(setup, held_out):
    """Lemma 6: R_single = {r(lambda)} is the diagonal of R_per-layer, and is smaller.

    No evolution involved. Sweep the simplex with one shared lambda, then sample
    independent lambdas per layer, and compare the hypervolume the two regions
    cover. The second set contains the first, so the comparison only asks whether
    the extra freedom buys anything. This is the claim Theorem 2 makes, and it is
    the one that reproduces: +0.043 to +0.185 on 25 stage-1 draws, on CPU and on
    an H200, never zero or negative.
    """
    shared = X.fixed_coefficient_front(setup, held_out, 21)
    per_layer = X.per_layer_reachable(setup, held_out, jax.random.PRNGKey(5), 256)
    both = np.vstack([shared, per_layer])
    hv_shared = hypervolume(shared[nondominated(shared)], setup.ref)
    hv_both = hypervolume(both[nondominated(both)], setup.ref)
    assert hv_both > hv_shared


def test_forward_shapes(setup, held_out):
    lam = jnp.full((setup.cfg.n_experts,), 1.0 / setup.cfg.n_experts)
    out = X.run_model(setup.backbone, setup.experts, held_out[:8], lam=lam)
    assert out.shape == (8, setup.cfg.n_out)
    reward = task_mod.rewards(setup.task, held_out[:8], out)
    assert reward.shape == (3,)
