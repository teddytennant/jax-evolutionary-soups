import json, sys, statistics as st
rows = json.load(open(sys.argv[1]))
print(f"{len(rows)} runs from {sys.argv[1]}\n")
setups = sorted({x.get("setup_seed", 0) for x in rows})
for s in setups:
    for mode in ("per_layer", "single", "fixed"):
        r = [x for x in rows if x["mode"] == mode and x.get("setup_seed", 0) == s]
        if not r:
            continue
        logged = [x["last_ret"] - x["first_prev"] for x in r]
        held = [x["hv_final_held"] - x["hv_start_held"] for x in r]
        for name, v in (("logged (old assertion)", logged), ("held-out (same prompts)", held)):
            sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f"setup {s} {mode:>9} {name:<24} n={len(v):2d} mean {st.mean(v):+.4f} "
                  f"sd {sd:.4f} min {min(v):+.4f} max {max(v):+.4f} "
                  f"fails {sum(1 for x in v if x <= 0)}/{len(v)}")
        if "front_size" in r[0]:
            print(f"setup {s} {mode:>9} {'front size / picks':<24} "
                  f"front min {min(x['front_size'] for x in r)} max {max(x['front_size'] for x in r)} | "
                  f"picks min {min(x['n_picks'] for x in r)} "
                  f"runs with picks<2: {sum(1 for x in r if x['n_picks'] < 2)}/{len(r)}")
        if "per_chunk" in r[0]:
            allpc = [d for x in r for d in x["per_chunk"]]
            neg = [d for d in allpc if d < -1e-12]
            worst = min(allpc)
            finals = [x["last_ret"] for x in r]
            print(f"setup {s} {mode:>9} {'per-chunk HV_t(S_t)-HV_t(S_t-1)':<24} "
                  f"n={len(allpc)} negative {len(neg)} ({100*len(neg)/len(allpc):.1f}%) "
                  f"worst {worst:+.6f} = {abs(worst)/st.mean(finals)*100:.3f}% of mean final")
        if "retained" in r[0]:
            print(f"setup {s} {mode:>9} {'retained per gen':<24} "
                  f"min {min(min(x['retained']) for x in r)} max {max(max(x['retained']) for x in r)}")
        print()
