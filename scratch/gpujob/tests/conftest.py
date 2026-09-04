import jax

# The threshold solve, the finite-difference gradient checks and the exact
# hypervolume comparisons are all tighter than float32 can carry.
jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from evolutionary_soups import experiment as X  # noqa: E402
from evolutionary_soups import task as task_mod  # noqa: E402

N_OBJECTIVES = 3


@pytest.fixture(scope="session")
def setup():
    """Stage 1 once for the whole session: frozen backbone, three trained experts."""
    return X.build_setup(seed=0, n_objectives=N_OBJECTIVES)


@pytest.fixture(scope="session")
def held_out(setup):
    """Prompts the experts were never trained on and no chunk ever contained."""
    return task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
