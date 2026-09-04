import pickle, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from evolutionary_soups import experiment as X
from evolutionary_soups import task as task_mod

setup = X.build_setup(seed=0, n_objectives=3)
held = task_mod.sample_prompts(jax.random.PRNGKey(7), setup.task, 512)
blob = dict(
    cfg=setup.cfg, task_feature_embed=np.asarray(setup.task.feature_embed),
    task_weights=np.asarray(setup.task.weights), n_objectives=setup.task.n_objectives,
    backbone=jax.tree_util.tree_map(np.asarray, setup.backbone),
    experts=jax.tree_util.tree_map(np.asarray, setup.experts),
    expert_rewards=np.asarray(setup.expert_rewards),
    ref=np.asarray(setup.ref),
    held_out=np.asarray(held),
)
with open(sys.argv[1], "wb") as f:
    pickle.dump(blob, f)
print("expert_rewards[0]", setup.expert_rewards[0].tolist())
