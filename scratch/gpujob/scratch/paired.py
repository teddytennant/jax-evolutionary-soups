"""Evolve from a stage-1 setup loaded off disk, so both backends see the same experts."""
import argparse, json, pickle, sys, time
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod

ap = argparse.ArgumentParser()
ap.add_argument("--blob", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--modes", default="fixed,per_layer,single")
a = ap.parse_args()

b = pickle.load(open(a.blob, "rb"))
task = task_mod.Task(cfg=b["cfg"], n_objectives=b["n_objectives"],
                     feature_embed=jnp.asarray(b["task_feature_embed"]),
                     weights=jnp.asarray(b["task_weights"]))
setup = X.Setup(cfg=b["cfg"], task=task,
                backbone=jax.tree_util.tree_map(jnp.asarray, b["backbone"]),
                experts=jax.tree_util.tree_map(jnp.asarray, b["experts"]),
                normalizer=task_mod.normalizer_from_experts(b["expert_rewards"]),
                expert_rewards=b["expert_rewards"], ref=b["ref"])
held = jnp.asarray(b["held_out"])

rows = []
for mode in a.modes.split(","):
    for seed in range(1, a.seeds + 1):
        r = X.evolve_gates(setup, seed=seed, mode=mode, population_size=20,
                           generations=30, hidden=16, chunk_size=256)
        hv = X.front_hypervolume(setup, r, held, mode)[0]
        rows.append(dict(mode=mode, seed=seed, backend=jax.default_backend(),
                         first_prev=r.history[0].hv_previous,
                         last_ret=r.history[-1].hv_retained,
                         hv_final_held=hv))
        print(f"# paired {mode} e{seed} last_ret={r.history[-1].hv_retained!r} "
              f"first_prev={r.history[0].hv_previous!r} held={hv!r}", file=sys.stderr, flush=True)
        open(a.out, "w").write(json.dumps(rows))
