"""How often does per-layer gating actually win the Section 4.3 ablation?

Theorem 2 is a statement about reachable sets, and tests/test_moe.py checks it
that way, by sampling the regions with no evolution in between. This script asks
the other question: given the same evolutionary budget, does the bigger gate
class come back with a bigger front?

That is a search outcome, not a theorem, and it turns on two draws. The evolution
seed picks the initial population and the chunks, and moves the margin a little.
The setup seed picks the experts, and moves it a lot, because stage 1 is 800 Adam
steps and lands somewhere different every time the arithmetic is split
differently. Sweep both before quoting a margin, and read the last section of
the README before quoting one from an accelerator.

    python scripts/ablation.py --setup-seeds 0 1 2 --evolve-seeds 0 1 2 3 4
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

MODES = ("per_layer", "single", "fixed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--evolve-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--objectives", type=int, default=3)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--float32", action="store_true",
                    help="run in float32; the tests and the numbers in the README use float64")
    args = ap.parse_args(argv)

    if not args.float32:
        jax.config.update("jax_enable_x64", True)

    print(f"jax {jax.__version__}  backend {jax.default_backend()}  "
          f"dtype {'float32' if args.float32 else 'float64'}")
    print(f"budget: population {args.population}, {args.generations} generations, "
          f"gate width {args.hidden}, chunk {args.chunk}")
    print()
    print(f"{'setup':>5} {'evolve':>7} {'per_layer':>10} {'single':>9} {'fixed':>9}"
          f" {'vs single':>10} {'vs fixed':>9}")

    d_single, d_fixed = [], []
    for s_seed in args.setup_seeds:
        start = time.time()
        setup = X.build_setup(seed=s_seed, n_objectives=args.objectives)
        held_out = task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
        print(f"  (stage 1 for setup seed {s_seed} in {time.time() - start:.0f}s)")
        for e_seed in args.evolve_seeds:
            hv = {}
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
                hv[mode] = X.front_hypervolume(setup, result, held_out, mode)[0]
            a = hv["per_layer"] - hv["single"]
            b = hv["per_layer"] - hv["fixed"]
            d_single.append(a)
            d_fixed.append(b)
            print(f"{s_seed:>5} {e_seed:>7} {hv['per_layer']:>10.4f} {hv['single']:>9.4f}"
                  f" {hv['fixed']:>9.4f} {a:>+10.4f} {b:>+9.4f}")

    print()
    for name, values in (("vs single", d_single), ("vs fixed", d_fixed)):
        n = len(values)
        sd = statistics.stdev(values) if n > 1 else 0.0
        wins = sum(1 for v in values if v > 0)
        print(f"per-layer {name}: mean {statistics.mean(values):+.4f}  sd {sd:.4f}"
              f"  min {min(values):+.4f}  max {max(values):+.4f}  wins {wins}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
