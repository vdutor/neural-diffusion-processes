from functools import partial
import numpy as np

from typing import Mapping, Tuple
import os
import string
import random
import datetime
import pathlib
import jax
import haiku as hk
import jax.numpy as jnp
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import optax
from functools import partial
from dataclasses import asdict

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from neural_diffusion_processes.types import Dataset, Batch, Rng
from neural_diffusion_processes.gp import predict

# enable jax float64
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)




def get_data(
    dataset: str,
    input_dim: int = 1,
    train: bool = True,
    batch_size: int = 1024,
    num_epochs: int = 1,
) -> Dataset:
    task = "training" if train else "interpolation"    
    data = np.load(f"data/{dataset}_{input_dim}_{task}.npz")
    ds = tf.data.Dataset.from_tensor_slices({
        "x_target": data["x_target"].astype(np.float64),
        "y_target": data["y_target"].astype(np.float64),
        "x_context": data["x_context"].astype(np.float64),
        "y_context": data["y_context"].astype(np.float64),
        "mask_context": data["mask_context"].astype(np.float64),
        "mask_target": data["mask_target"].astype(np.float64),
    })
    if train:
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(seed=42, buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return map(lambda d: Batch(**d), ds)


from data import _DATASET_FACTORIES
import argparse


if __name__ == "__main__":
    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="se")
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument('--fullcov', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    fullcov = args.fullcov
    dataset = args.dataset
    dim_x = args.dim_x
    true_gp = _DATASET_FACTORIES[dataset](list(range(dim_x)))


    @jax.jit
    @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
    def eval_conditional_gp(x_test, y_test, x_context, y_context, mask_context):
        x_context = x_context + mask_context[:, None] * 1e3
        post = predict(true_gp.prior, true_gp.params, (x_context, y_context), diag=not fullcov)(x_test)
        ll =  post.log_prob(y_test.squeeze()) / len(x_test)
        mse = jnp.mean((post.mean() - y_test.squeeze()) ** 2)
        num_context = len(x_context) - jnp.count_nonzero(mask_context)
        return {"mse": mse, "ll": ll, "nc": num_context}


    ds_test = get_data(
        dataset,
        input_dim=dim_x,
        train=False,
        batch_size=4,
        num_epochs=1,
    )
    metrics = {"mse": [], "ll": [], "nc": []}
    for i, batch in enumerate(ds_test):
        m = eval_conditional_gp(batch.x_target, batch.y_target, batch.x_context, batch.y_context, batch.mask_context)
        for k, v in m.items():
            metrics[k].append(v)


    def err(v):
        return 1.96 * jnp.std(v) / jnp.sqrt(len(v))

    summary_stats = [
        ("mean", jnp.mean),
        ("std", jnp.std),
        ("err", err)
    ]
    metrics = {f"{k}_{n}": s(jnp.stack(v).ravel()) for k, v in metrics.items() for n, s in summary_stats}

    print(
        {
            "model": "gp" if fullcov else "gp_diag",
            "data": "eq" if dataset == "se" else dataset,
            "dim_x": dim_x,
            "loglik_mean": float(metrics["ll_mean"]),
            "loglik_error": float(metrics["ll_err"]),
        }, end=",\n"
    )