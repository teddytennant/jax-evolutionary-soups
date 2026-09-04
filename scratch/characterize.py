"""Characterize test_evolution_improved_on_its_starting_population.

Two versions of the same question, over many evolve seeds:

  logged:   last.hv_retained  vs  first.hv_previous   (what the test asserts,
            two different chunks, 256 prompts each)
  held-out: HV(final front)   vs  HV(initial population), both on the same 512
            held-out prompts
"""
import argparse, json, sys, time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod
from evolutionary_soups.gating import init_population
from evolutionary_soups.hypervolume import hypervolume, nondominated

POPULATION, GENERATIONS, HIDDEN, CHUNK = 20, 30, 16, 256


def initial_population(setup, seed):
    """Exactly what evolve_gates starts from for this seed."""
    k_init, _, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
    return init_population(k_init, POPULATION, setup.cfg.d_model,
                           setup.cfg.n_experts, HIDDEN, 1.5, 0.1)


def hv_on(setup, gates, tokens, mode):
    s = X.population_fitness(setup, gates, tokens, mode)
    return hypervolume(s[nondominated(s)], setup.ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--setup-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--modes", default="per_layer,single,fixed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for s_seed in args.setup_seeds:
        t0 = time.time()
        setup = X.build_setup(seed=s_seed, n_objectives=3)
        held_out = task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
        print(f"# setup {s_seed} in {time.time()-t0:.1f}s on {jax.devices()} "
              f"expert_rewards[0]={setup.expert_rewards[0].tolist()}",
              file=sys.stderr, flush=True)
        for mode in args.modes.split(","):
            hv_init = hv_on(setup, initial_population(setup, 0), held_out, mode)  # placeholder
            for seed in range(args.start, args.start + args.seeds):
                t = time.time()
                r = X.evolve_gates(setup, seed=seed, mode=mode,
                                   population_size=POPULATION, generations=GENERATIONS,
                                   hidden=HIDDEN, chunk_size=CHUNK)
                hv_final = X.front_hypervolume(setup, r, held_out, mode)[0]
                hv_start = hv_on(setup, initial_population(setup, seed), held_out, mode)
                rows.append(dict(
                    setup_seed=s_seed, mode=mode, seed=seed,
                    first_prev=r.history[0].hv_previous,
                    first_ret=r.history[0].hv_retained,
                    last_ret=r.history[-1].hv_retained,
                    hv_start_held=hv_start,
                    hv_final_held=hv_final,
                    retained=[h.retained for h in r.history],
                    front=[h.front_size for h in r.history],
                    hv_ret_traj=[h.hv_retained for h in r.history],
                    hv_prev_traj=[h.hv_previous for h in r.history],
                    secs=time.time() - t,
                ))
                print(f"# s{s_seed} {mode} e{seed} "
                      f"logged {r.history[-1].hv_retained:.5f} vs {r.history[0].hv_previous:.5f} "
                      f"({r.history[-1].hv_retained - r.history[0].hv_previous:+.5f}) | "
                      f"held-out {hv_final:.5f} vs {hv_start:.5f} "
                      f"({hv_final - hv_start:+.5f}) [{time.time()-t:.0f}s]",
                      file=sys.stderr, flush=True)
                open(args.out, "w").write(json.dumps(rows))


main()
