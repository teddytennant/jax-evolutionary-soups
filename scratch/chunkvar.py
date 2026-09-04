"""How much of the logged hypervolume is just the chunk?

Score one fixed set of gates on many independent 256-prompt chunks, the same
size Algorithm 1 draws each generation, and look at the spread.
"""
import sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod
from evolutionary_soups.gating import init_population
from evolutionary_soups.hypervolume import hypervolume, nondominated

setup = X.build_setup(seed=0, n_objectives=3)
print("expert_rewards[0]", setup.expert_rewards[0].tolist(), file=sys.stderr)

for mode in ("per_layer", "single", "fixed"):
    for label, gates in (
        ("initial pop (evolve seed 1)",
         init_population(jax.random.split(jax.random.PRNGKey(1), 3)[0], 20,
                         setup.cfg.d_model, setup.cfg.n_experts, 16, 1.5, 0.1)),
        ("evolved front (evolve seed 1)",
         X.evolve_gates(setup, seed=1, mode=mode, population_size=20,
                        generations=30, hidden=16, chunk_size=256).params),
    ):
        hvs = []
        for i in range(40):
            chunk = task_mod.sample_prompts(jax.random.PRNGKey(1000 + i), setup.task, 256)
            s = X.population_fitness(setup, gates, chunk, mode)
            hvs.append(hypervolume(s[nondominated(s)], setup.ref))
        hvs = np.array(hvs)
        print(f"{mode:>9}  {label:<30}  mean {hvs.mean():.4f}  sd {hvs.std(ddof=1):.4f} "
              f"min {hvs.min():.4f}  max {hvs.max():.4f}  range {hvs.max()-hvs.min():.4f}",
              flush=True)
