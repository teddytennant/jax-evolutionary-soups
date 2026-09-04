"""test_moe.py::test_experts_are_specialists over stage-1 draws."""
import sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from evolutionary_soups import experiment as X

ok = 0
for s in range(15):
    setup = X.build_setup(seed=s, n_objectives=3)
    arg = np.argmax(setup.expert_rewards, axis=0)
    good = bool(np.array_equal(arg, np.arange(3)))
    ok += good
    margin = float(np.min(np.diag(setup.expert_rewards) - np.max(
        setup.expert_rewards - 1e9 * np.eye(3), axis=0)))
    print(f"setup {s}: argmax {arg.tolist()} specialists={good} min_margin={margin:+.4f}",
          flush=True)
print(f"specialists on {ok}/15 stage-1 draws")
