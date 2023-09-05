import sys; sys.path.append('.')
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from itertools import product
from tqdm import tqdm
from data import get_batch, DATASETS, TASKS, _DATASET_CONFIGS

DRYRUN = False
PLOT = False
BATCH_SIZE = 4
DATASET_SIZE = {"training": int(2**14), "interpolation": 128}
SEED = 0
key = jax.random.PRNGKey(SEED)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(6 + 2, 3, figsize=(8, 20))
axes = np.array(axes).flatten()
j = 0

for dataset, task in product(DATASETS, TASKS):
    for input_dim in range(1, _DATASET_CONFIGS[dataset].max_input_dim + 1):
        print(dataset, task, input_dim)

        jitted_get_batch = jax.jit(lambda k: get_batch(k, batch_size=BATCH_SIZE, name=dataset, task=task, input_dim=input_dim))

        batches = []
        for i in tqdm(range(DATASET_SIZE[task] // BATCH_SIZE)):
            key, bkey = jax.random.split(key)
            batch = jitted_get_batch(bkey)
            batches.append(batch)

        x_context = jnp.concatenate([b.x_context for b in batches], axis=0)
        y_context = jnp.concatenate([b.y_context for b in batches], axis=0)
        x_target = jnp.concatenate([b.x_target for b in batches], axis=0)
        y_target = jnp.concatenate([b.y_target for b in batches], axis=0)
        mask_target = jnp.concatenate([b.mask_target for b in batches], axis=0)
        mask_context = jnp.concatenate([b.mask_context for b in batches], axis=0)
        print(f"{dataset} {input_dim} {task}")
        print(x_context.shape, y_context.shape, x_target.shape, y_target.shape, mask_target.shape, mask_context.shape)

        if not DRYRUN:
            np.savez(
                os.getcwd() + f"/data/{dataset}_{input_dim}_{task}.npz",
                x_context=x_context,
                y_context=y_context,
                x_target=x_target,
                y_target=y_target,
                mask_target=mask_target,
                mask_context=mask_context,
            )

        if PLOT:
            num_context = mask_context.shape[1] - jnp.count_nonzero(mask_context, axis=1, keepdims=True)
            num_context = jnp.ravel(num_context)
            num_target = mask_target.shape[1] - jnp.count_nonzero(mask_target, axis=1, keepdims=True)
            num_target = jnp.ravel(num_target)
            axes[j].hist(num_context, bins=20, label="context")
            axes[j].hist(num_target, bins=20, label="target")
            axes[j].set_title(f"{dataset} {input_dim} {task}", fontsize=8)
            if j == 0:
                axes[j].legend()
            j+=1

if PLOT:
    plt.savefig("num_data.png")