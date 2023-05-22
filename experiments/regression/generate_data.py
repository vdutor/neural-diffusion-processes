#%%
import sys; sys.path.append('.')
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#%%

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from itertools import product
from tqdm import tqdm
from data import get_batch, DATASETS, TASKS, _DATASET_CONFIGS
# %%

BATCH_SIZE = 1024
DATASET_SIZE = int(2**14)
SEED = 0
key = jax.random.PRNGKey(SEED)

for dataset, task in product(DATASETS, TASKS):
    for input_dim in range(1, _DATASET_CONFIGS[dataset].max_input_dim + 1):
        print(dataset, task, input_dim)

        jitted_get_batch = jax.jit(lambda k: get_batch(k, batch_size=BATCH_SIZE, name=dataset, task=task, input_dim=input_dim))

        batches = []
        for i in tqdm(range(DATASET_SIZE // BATCH_SIZE)):
            key, bkey = jax.random.split(key)
            batch = jitted_get_batch(bkey)
            batches.append(batch)

        x_context = jnp.concatenate([b.x_context for b in batches], axis=0)
        y_context = jnp.concatenate([b.y_context for b in batches], axis=0)
        x_target = jnp.concatenate([b.x_target for b in batches], axis=0)
        y_target = jnp.concatenate([b.y_target for b in batches], axis=0)
        print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)

        np.savez(
            os.getcwd() + f"/data/{dataset}_{input_dim}_{task}.npz",
            x_context=x_context,
            y_context=y_context,
            x_target=x_target,
            y_target=y_target,
        )
#%%

# %%
