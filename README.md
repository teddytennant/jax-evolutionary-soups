# jax-evolutionary-soups

JAX implementation of "Evolutionary Soups: Evolving Mixture-of-Experts for
Multi-Objective LLM Alignment" (arXiv:2608.29978).

## install

uv pip install -e .

## run

    python -m evolutionary_soups   # train experts, evolve the gates, select by preference
    python scripts/smoke.py        # three objectives, bigger population, prints the backend
    python scripts/ablation.py     # the Section 4.3 ablation, swept over seeds
    python scripts/start_vs_end.py # what evolution gained over its starting population

## what is not the paper

There is no LLM here. The paper trains LLaMA-2-7B experts with SFT and PPO and
scores generations with reward models on Beaver, Summary and Assistant. In this
repo the backbone is a few thousand parameters and the objectives are synthetic
functions built to conflict, so no number here compares to a number there. The
method is present: per-layer alpha-entmax gating on the attention hidden state
(Eq. 2), greedy hypervolume contribution selection (Algorithm 1), preference
selection over the evolved front (Eq. 1).

## claims that did not survive the tests

Lemma 8 says greedy hypervolume subset selection is exact for three or fewer
objectives. It is not, and the counterexample in tests/test_hypervolume.py has
three points in two dimensions. That is also why Theorem 3's
HV_t(S_t) >= HV_t(S_{t-1}) dips. On the toy in tests/test_evolution.py that is
one or two generations in twenty five. On the real pipeline it turns on the gate
class: over 50 runs of 30 generations, 0.3% of generations for per-layer, 0.7%
for single gating, 6.7% for fixed, worst single run 5 of 30. I have not chased
down why fixed is the one that trips it. Every dip is small either way. The
worst over those 150 runs was 0.62% of the final hypervolume.

The ablation ordering is the other one, and it is worse than I first wrote. What
Theorem 2 states is a containment of reward regions, and that part holds. Sample
the two regions directly, with no evolution in the way, and the per-layer region
covers more hypervolume than the shared-lambda sweep every time: 25 sets of
experts, on CPU and on an H200, gaps from +0.043 to +0.185, none of them zero or
negative. tests/test_moe.py checks it that way.

What does not hold is the step from there to "so evolving the bigger gate class
gives a bigger front". That is a search outcome and it is decided by which
experts stage 1 happens to land on, not by the gate class:

    experts trained on CPU, evolution on CPU   39 runs, 6 draws   39/39, mean +0.049
    the same experts, evolution on an H200     48 runs, 3 draws   48/48, mean +0.055
    experts trained on the H200, same code     60 runs, 10 draws  45/60, mean +0.034

A draw is one call to `build_setup`, and the runs inside it differ only in the
evolution seed. Two of the ten GPU draws come out negative on average, and
`build_setup(seed=0)` alone produced draws averaging -0.004, +0.062 and +0.105.
So the number the old test asserted, per-layer above single-gating on one seed,
was a coin flip with a bias, and it is gone. scripts/ablation.py is what
measures it now.

The third one was smaller and it was mine, not the paper's.
`test_evolution_improved_on_its_starting_population` compared the last
generation's `hv_retained` with the first generation's `hv_previous`, both read
straight off `result.history`. Algorithm 1 draws a fresh chunk D_t every
generation, so those two hypervolumes are measured on different samples of 256
prompts, and the sample moves them further than thirty generations of evolution
do. Hold one population fixed and score it on 40 chunks: a starting population
gives sd 0.091 and the front it evolves into gives sd 0.076, against a gap
between their means of 0.15. So the comparison came out negative on 39 of 150
runs, and not for one gate class more than another: 8/50 per-layer, 16/50 single
gating, 15/50 fixed. An H200 hit it on the fixed gate first, 0.352 against
0.364, which is what got reported. Nothing about that gate class is special
here.

The improvement itself is real, it just needs the same prompts on both sides.
The held-out set does not move, and on it the evolved front covers more
hypervolume than the population it grew out of on every draw: 25 evolution seeds
against two stage-1 setups, 150 runs, +0.059 at the smallest and +0.130 on
average. `EvolutionResult` carries the starting population now so the test can
score it, and scripts/start_vs_end.py prints the two comparisons side by side.

Not a backend question either. Evolving from one pickled stage-1 setup gave
hypervolumes identical to the last bit at 16 threads and at 4, so the search is
not sensitive to how the arithmetic is split. Stage 1 still is, which is the
next section.

## why the numbers move between runs

Given the same experts the two backends agree completely. Loading the
CPU-trained stage 1 onto an H200 reproduced the CPU hypervolumes over 24 paired
runs to a worst relative difference of 5.6e-16, which is two ulp of float64.
Nothing in the evolution, the entmax solve or the hypervolume arithmetic is
backend-sensitive.

Stage 1 is the part that moves. It is 800 Adam steps, and a reduction that gets
split differently changes the last bits of the result, which 800 steps later is a
visibly different set of experts. Both backends do this, but they differ in what
it takes to trigger it.

On CPU the split follows the thread count, so a run repeats as long as that does.
Two processes pinned to four cores gave `build_setup(seed=0)` first-row expert
rewards of -0.0446958008125150 twice, to the last digit. Six cores gave
-0.0451205681497302, three times. That is enough to move the ablation: on the
seed the old test fixed, six cores put per-layer 0.015 hypervolume below
single-gating on the same machine where four cores put it 0.041 above. The test
was not passing on CPU, it was passing on one thread count.

On GPU nothing has to change. One job on one node ran `build_setup(seed=0)` three
times back to back and got -0.0486, -0.0459 and -0.0463, whose per-layer minus
single-gating margins are -0.012, +0.064 and +0.011. Two more runs in the same
job with `XLA_FLAGS=--xla_gpu_autotune_level=0` gave -0.0481 and -0.0476, so
autotuning is not the whole story.

The tests run in float64, which conftest.py turns on, and scripts/ablation.py
does the same so its numbers line up with theirs. `python -m evolutionary_soups`
and scripts/smoke.py stay in float32, where on NVIDIA the matmuls run at TF32 by
default. I have not measured what that does to any of the numbers here.
