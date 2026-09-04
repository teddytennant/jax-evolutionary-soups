"""Run the three stages once and print the front, the baseline, and Eq. (1)."""

from __future__ import annotations

import argparse

import jax
import numpy as np

from . import experiment as X
from . import task as task_mod
from .hypervolume import hypervolume, nondominated
from .preference import preference_selection, sample_preferences, selection_utility


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objectives", type=int, default=2)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=16, help="gate width, 256 in the paper")
    ap.add_argument("--chunk", type=int, default=256, help="D_c, prompts scored per generation")
    args = ap.parse_args(argv)

    setup = X.build_setup(seed=args.seed, n_objectives=args.objectives)
    held_out = task_mod.sample_prompts(jax.random.PRNGKey(1234), setup.task, 512)
    n_grid = 11 if args.objectives == 2 else 21

    baseline = X.fixed_coefficient_front(setup, held_out, n_grid)
    hv_baseline = hypervolume(baseline[nondominated(baseline)], setup.ref)

    result = X.evolve_gates(
        setup,
        seed=args.seed + 1,
        mode="per_layer",
        population_size=args.population,
        generations=args.generations,
        hidden=args.hidden,
        chunk_size=args.chunk,
    )
    hv, scores = X.front_hypervolume(setup, result, held_out)
    front = scores[nondominated(scores)]

    print(f"experts {setup.cfg.n_experts}  objectives {args.objectives}  front {len(front)}")
    print("evolved front (normalized rewards, held-out prompts):")
    for row in front[np.lexsort(front.T)]:
        print("  " + "  ".join(f"{v:+.3f}" for v in row))
    print(f"hypervolume, fixed-coefficient soup: {hv_baseline:.4f}")
    print(f"hypervolume, evolved per-layer gates: {hv:.4f}")

    print("preference selection, Eq. (1):")
    linear = selection_utility("linear")
    tcheb = selection_utility("tchebyshev")
    for mu in sample_preferences(args.objectives, n_grid):
        i = preference_selection(front, mu, linear)
        j = preference_selection(front, mu, tcheb)
        mu_str = " ".join(f"{v:.2f}" for v in mu)
        print(
            f"  mu [{mu_str}] linear -> {np.round(front[i], 3)}"
            f"  tchebyshev -> {np.round(front[j], 3)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
