"""Longer run of the same pipeline, sized for an accelerator.

Three objectives, three experts, a bigger population and more generations than
the tests use. Prints the JAX backend and the evolved front's hypervolume next
to the fixed-coefficient soup baseline. Runs on CPU too.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolutionary_soups import experiment as X  # noqa: E402
from evolutionary_soups import task as task_mod  # noqa: E402
from evolutionary_soups.hypervolume import hypervolume, nondominated  # noqa: E402
from evolutionary_soups.preference import (  # noqa: E402
    preference_selection,
    selection_utility,
)

N_OBJECTIVES = 3
POPULATION = 40
GENERATIONS = 40
CHUNK = 512
HIDDEN = 32


def main():
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}")
    start = time.time()

    setup = X.build_setup(seed=0, n_objectives=N_OBJECTIVES, batch=512, sft_steps=600, expert_steps=600)
    print(f"experts trained in {time.time() - start:.1f}s")
    print("expert reward matrix, row i is expert i on its own objective and the others:")
    print(np.round(setup.expert_rewards, 4))

    held_out = task_mod.sample_prompts(jax.random.PRNGKey(1234), setup.task, 1024)
    baseline = X.fixed_coefficient_front(setup, held_out, 21)
    hv_baseline = hypervolume(baseline[nondominated(baseline)], setup.ref)

    t0 = time.time()
    result = X.evolve_gates(
        setup,
        seed=1,
        mode="per_layer",
        population_size=POPULATION,
        generations=GENERATIONS,
        hidden=HIDDEN,
        chunk_size=CHUNK,
    )
    evolve_seconds = time.time() - t0
    hv, scores = X.front_hypervolume(setup, result, held_out)
    front = scores[nondominated(scores)]

    dips = sum(1 for g in result.history if g.hv_retained < g.hv_previous - 1e-12)
    print(f"{GENERATIONS} generations, population {POPULATION}, in {evolve_seconds:.1f}s")
    print(f"front size {len(front)}, generations that lost hypervolume on their own chunk: {dips}")
    print(f"hypervolume, fixed-coefficient soup (21 preferences): {hv_baseline:.4f}")
    print(f"hypervolume, evolved per-layer gates:                 {hv:.4f}")
    print(f"gain: {hv - hv_baseline:+.4f}")

    utility = selection_utility("linear")
    print("Eq. (1) selection at the simplex corners:")
    for mu in np.eye(N_OBJECTIVES):
        i = preference_selection(front, mu, utility)
        print(f"  mu {mu} -> {np.round(front[i], 3)}")
    print(f"total {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
