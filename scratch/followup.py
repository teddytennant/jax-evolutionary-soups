"""The other single-seed search-outcome claims in test_end_to_end.py / test_moe.py.

  test_the_front_is_a_front:                        len(nondominated(held-out)) >= 2
  test_preference_control_moves_along_the_evolved_front: len(set(picks)) > 1
"""
import argparse, json, sys, time
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod
from evolutionary_soups.gating import init_population
from evolutionary_soups.hypervolume import hypervolume, nondominated
from evolutionary_soups.preference import preference_selection, selection_utility

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=25)
ap.add_argument("--setup-seeds", type=int, nargs="+", default=[0])
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = []
for s_seed in a.setup_seeds:
    setup = X.build_setup(seed=s_seed, n_objectives=3)
    held = task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
    spec = bool(np.array_equal(np.argmax(setup.expert_rewards, axis=0), np.arange(3)))
    print(f"# setup {s_seed} specialists={spec} rewards[0]={setup.expert_rewards[0].tolist()}",
          file=sys.stderr, flush=True)
    util = selection_utility("linear")
    for mode in ("per_layer", "single", "fixed"):
        for seed in range(1, a.seeds + 1):
            r = X.evolve_gates(setup, seed=seed, mode=mode, population_size=20,
                               generations=30, hidden=16, chunk_size=256)
            scores = X.population_fitness(setup, r.params, held, mode)
            front = scores[nondominated(scores)]
            start = X.population_fitness(setup, init_population(
                jax.random.split(jax.random.PRNGKey(seed), 3)[0], 20,
                setup.cfg.d_model, setup.cfg.n_experts, 16, 1.5, 0.1), held, mode)
            picks = [preference_selection(front, mu, util) for mu in np.eye(3)]
            rows.append(dict(
                setup_seed=s_seed, mode=mode, seed=seed, specialists=spec,
                front_size=int(len(front)), n_picks=int(len(set(picks))),
                size=int(r.size),
                hv_final_held=hypervolume(front, setup.ref),
                hv_start_held=hypervolume(start[nondominated(start)], setup.ref),
                first_prev=r.history[0].hv_previous,
                last_ret=r.history[-1].hv_retained,
                per_chunk=[h.hv_retained - h.hv_previous for h in r.history],
            ))
            print(f"# s{s_seed} {mode} e{seed} front={len(front)} picks={len(set(picks))} "
                  f"held {rows[-1]['hv_final_held']:.4f} vs {rows[-1]['hv_start_held']:.4f} "
                  f"({rows[-1]['hv_final_held']-rows[-1]['hv_start_held']:+.4f})",
                  file=sys.stderr, flush=True)
            open(a.out, "w").write(json.dumps(rows))
