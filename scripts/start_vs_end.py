"""Did the evolution improve on the population it started from, and how do you tell?

Two ways to ask, on the same runs.

The generation log has hv_retained and hv_previous for every generation, but each
one is measured on that generation's own chunk D_t. Comparing the last
generation's number with the first generation's compares two samples of
`--chunk` prompts as much as it compares two populations, and the sample is the
bigger term.

The held-out set does not move, so scoring S_0 and S_T on it leaves only the
thing being asked about.

    python scripts/start_vs_end.py --setup-seeds 0 1 --evolve-seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolutionary_soups import experiment as X  # noqa: E402
from evolutionary_soups import task as task_mod  # noqa: E402
from evolutionary_soups.hypervolume import hypervolume, nondominated  # noqa: E402

MODES = ("per_layer", "single", "fixed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--evolve-seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--objectives", type=int, default=3)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--float32", action="store_true",
                    help="run in float32; the tests and the README use float64")
    args = ap.parse_args(argv)

    if not args.float32:
        jax.config.update("jax_enable_x64", True)

    print(f"jax {jax.__version__}  backend {jax.default_backend()}  "
          f"dtype {'float32' if args.float32 else 'float64'}")
    print(f"budget: population {args.population}, {args.generations} generations, "
          f"gate width {args.hidden}, chunk {args.chunk}")
    print()
    print(f"{'setup':>5} {'evolve':>7} {'mode':>10} {'logged':>9} {'held-out':>9}")

    gaps = {mode: {"logged": [], "held": []} for mode in MODES}
    for s_seed in args.setup_seeds:
        start = time.time()
        setup = X.build_setup(seed=s_seed, n_objectives=args.objectives)
        held_out = task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
        print(f"  (stage 1 for setup seed {s_seed} in {time.time() - start:.0f}s)")
        for e_seed in args.evolve_seeds:
            for mode in MODES:
                result = X.evolve_gates(
                    setup,
                    seed=e_seed,
                    mode=mode,
                    population_size=args.population,
                    generations=args.generations,
                    hidden=args.hidden,
                    chunk_size=args.chunk,
                )
                hv_end = X.front_hypervolume(setup, result, held_out, mode)[0]
                s0 = X.population_fitness(setup, result.initial, held_out, mode)
                hv_start = hypervolume(s0[nondominated(s0)], setup.ref)
                logged = result.history[-1].hv_retained - result.history[0].hv_previous
                gaps[mode]["logged"].append(logged)
                gaps[mode]["held"].append(hv_end - hv_start)
                print(f"{s_seed:>5} {e_seed:>7} {mode:>10} {logged:>+9.4f}"
                      f" {hv_end - hv_start:>+9.4f}")

    print()
    for mode in MODES:
        for name in ("logged", "held"):
            v = gaps[mode][name]
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"{mode:>10} {name:>7}: mean {statistics.mean(v):+.4f}  sd {sd:.4f}"
                  f"  min {min(v):+.4f}  max {max(v):+.4f}"
                  f"  negative {sum(1 for x in v if x <= 0)}/{len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
