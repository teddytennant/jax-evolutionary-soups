# jax-evolutionary-soups

JAX implementation of "Evolutionary Soups: Evolving Mixture-of-Experts for
Multi-Objective LLM Alignment" (arXiv:2608.29978).

## install

uv pip install -e .

## run

    python -m evolutionary_soups   # train experts, evolve the gates, select by preference
    python scripts/smoke.py        # three objectives, bigger population, prints the backend

## what is not the paper

There is no LLM here. The paper trains LLaMA-2-7B experts with SFT and PPO and
scores generations with reward models on Beaver, Summary and Assistant. In this
repo the backbone is a few thousand parameters and the objectives are synthetic
functions built to conflict, so no number here compares to a number there. The
method is present: per-layer alpha-entmax gating on the attention hidden state
(Eq. 2), greedy hypervolume contribution selection (Algorithm 1), preference
selection over the evolved front (Eq. 1).

Two claims did not survive the tests. Lemma 8 says greedy hypervolume subset
selection is exact for three or fewer objectives. It is not, and the
counterexample in tests/test_hypervolume.py has three points in two dimensions.
That is also why Theorem 3's HV_t(S_t) >= HV_t(S_{t-1}) dips on one or two
generations in twenty five, each time by well under a percent. The ablation
ordering, per-layer above single-gating and above fixed coefficients, holds on
the seed the tests fix and on the seeds I tried with three objectives; on an
easier two-objective draw, single-gating beat per-layer once in nine seed pairs.
