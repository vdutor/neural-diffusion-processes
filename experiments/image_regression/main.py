from __future__ import annotations
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
import datasets
import optax
from functools import partial
from dataclasses import asdict

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools import writers
from ml_tools import actions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.types import Dataset, Batch, Rng, ndarray
from neural_diffusion_processes.model import AttentionModel
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion

from config import Config


EXPERIMENT = "eval-May29-test"
EXPERIMENT_NAME = None
DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
HERE = pathlib.Path(__file__).parent
LOG_DIR = 'logs'


def get_experiment_name(config: Config):
    del config  # Currently not used but name could be based on config
    global EXPERIMENT_NAME

    if EXPERIMENT_NAME is None:
        letters = string.ascii_lowercase
        id = ''.join(random.choice(letters) for i in range(4))
        EXPERIMENT_NAME = f"{DATETIME}_{id}"

    return EXPERIMENT_NAME



def get_experiment_dir(config: Config, output: str = "root", exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)
    root = HERE / LOG_DIR / EXPERIMENT / experiment_name

    if output == "root":
        dir_ = root
    elif output == "plots":
        dir_ = root / "plots"
    elif output == "tensorboard":
        # All tensorboard logs are stored in a single directory
        # Run tensorboard with:
        # tensorboard --logdir logs/{EXPERIMENT-NAME}/tensorboard
        dir_ = HERE / LOG_DIR / EXPERIMENT / "tensorboard" / experiment_name
    else:
        raise ValueError("Unknown output: %s" % output)

    dir_.mkdir(parents=True, exist_ok=exist_ok)
    return dir_



def get_image_grid_inputs(size: int, a=2):
    x1 = np.linspace(a, -a, size)
    x2 = np.linspace(-a, a, size)
    x1, x2 = np.meshgrid(x2, x1)
    return np.stack([x1.ravel(), x2.ravel()], axis=-1)


def get_rescale_function_fwd_and_inv(config: Config):
    if config.dataset == "mnist":
        mean = jnp.zeros((1,)).reshape(1, 1)
        std = jnp.ones((1,)).reshape(1, 1)
    elif config.dataset == "celeba32":
        mean = np.array([0.5066832 , 0.4247095 , 0.38070202]).reshape(1, 3)
        std = np.array([0.30913046, 0.28822428, 0.2866247]).reshape(1, 3)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")
    
    def fwd(y):
        """y: [N, H*W, C] """
        return (y - mean[None]) / std[None]
    
    def inv(y):
        """y: [N, H*W, C] """
        y = y * std[None] + mean[None]
        y = jnp.clip(y, 0., 1.)
        return y
    
    return fwd, inv


def get_image_data(
    config: Config,
    train: bool = True,
    batch_size: int = 1024,
    num_epochs: int = 1,
) -> Dataset:
    size = config.image_size
    output_dim = config.output_dim
    input_ = get_image_grid_inputs(size)
    rescale = get_rescale_function_fwd_and_inv(config)[0]

    def preprocess(batch) -> Mapping[str, ndarray]:
        y = batch["image"]
        batch_size = len(y)
        x = tf.repeat(input_[None, ...], batch_size, axis=0)
        y = tf.image.resize(y[..., None] if output_dim == 1 else y, (size, size))
        y = tf.cast(y, tf.float32) / 255.  # rescale
        y = tf.reshape(y, (batch_size, size*size, output_dim))
        y = rescale(y)
        return dict(
            x_target=x,
            y_target=y,
        )

    images_dataset = datasets.load_dataset(config.hf_dataset_name)
    images_dataset.set_format('tensorflow')
    images_dataset = images_dataset.select_columns("image")
    subset = "train" if train else "test"
    ds = images_dataset[subset].to_tf_dataset(batch_size=batch_size, shuffle=True, drop_remainder=True)
    if train:
        ds = ds.repeat(count=num_epochs)
    ds = ds.map(preprocess)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return map(lambda d: Batch(**d), ds)


config: Config = setup_config(Config)
key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)

rescale, rev_rescale = get_rescale_function_fwd_and_inv(config)
ds = get_image_data(config, train=True, batch_size=config.batch_size, num_epochs=config.num_epochs)
batch_init = next(ds)
print(batch_init.x_target.shape, batch_init.y_target.shape)
n = int(config.batch_size ** 0.5)
fig, axes = plt.subplots(5, 5, figsize=(n, n), sharex=True, sharey=True)
axes = np.array(axes).reshape(-1)
im_shape = (config.image_size, config.image_size, config.output_dim)
images_batch = rev_rescale(batch_init.y_target)
for i in range(len(axes)):
    axes[i].imshow(images_batch[i].reshape(im_shape))
    axes[i].set_xticks([])
    axes[i].set_yticks([])
fig.savefig(str(get_experiment_dir(config, "plots") / "data.png"))

ys = []
for i in range(100):
    batch = next(ds)
    ys.append(batch.y_target)

ys = np.concatenate(ys, axis=0)
print("data mean", jnp.mean(ys, axis=[0,1]))
print("data std", jnp.std(ys, axis=[0,1]))

@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask):
    model = AttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
        output_dim=config.output_dim,
        sparse=config.network.sparse_attention,
    )
    return model(x, y, t, mask)


@jax.jit
def net(params, t, yt, x, mask, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs
    return network.apply(params, t[None], yt[None], x[None], mask[None])[0]


def loss_fn(params, batch: Batch, key):
    net_with_params = partial(net, params)
    kwargs = dict(num_timesteps=config.diffusion.timesteps, loss_type=config.loss_type)
    return ndp.process.loss(process, net_with_params, batch, key, **kwargs)


learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=config.optimizer.init_lr,
    peak_value=config.optimizer.peak_lr,
    warmup_steps=config.steps_per_epoch * config.optimizer.num_warmup_epochs,
    decay_steps=config.steps_per_epoch * config.optimizer.num_decay_epochs,
    end_value=config.optimizer.end_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask=batch.mask_target
    )
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )


@jax.jit
def ema_update(decay, ema_params, new_params):
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params
    return jax.tree_map(_ema, ema_params, new_params)


@jax.jit
def update_step(state: TrainingState, batch: Batch) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    new_params_ema = ema_update(config.optimizer.ema_rate, state.params_ema, new_params)
    new_state = TrainingState(
        params=new_params,
        params_ema=new_params_ema,
        opt_state=new_opt_state,
        key=new_key,
        step=state.step + 1
    )
    metrics = {
        'loss': loss_value,
        'step': state.step,
        'data_seen': (state.step + 1) * batch.x_target.shape[0],
    }
    return new_state, metrics


@jax.jit
def sample_prior(state: TrainingState, key: Rng, x: jnp.array):
    net_with_params = partial(net, state.params_ema)
    y0 = process.sample(key, x, mask=None, model_fn=net_with_params, output_dim=config.output_dim)
    return x, y0


@jax.jit
def sample_conditional(state: TrainingState, key: Rng):
    x = jnp.linspace(-2, 2, 57)[:, None]
    xc = jnp.array([-1., 0., 1.]).reshape(-1, 1)
    yc = jnp.array([0., -1., 1.]).reshape(-1, 1)
    net_with_params = partial(net, state.params_ema)
    y0 = process.conditional_sample(key, x, mask=None, x_context=xc, y_context=yc, mask_context=None, model_fn=net_with_params)
    return x, y0, xc, yc


def plots(state: TrainingState, key: Rng):
    if config.input_dim != 1: return {}  # only plot for 1D inputs
    # prior
    fig_prior, ax = plt.subplots()
    x = jnp.linspace(-2, 2, 60)[:, None]
    x, y0 = jax.vmap(lambda k: sample_prior(state, k, x))(jax.random.split(key, 10))
    ax.plot(x[...,0].T, y0[...,0].T, color="C0", alpha=0.5)

    # conditional
    fig_cond, ax = plt.subplots()
    x, y0, xc, yc = jax.vmap(lambda k: sample_conditional(state, k))(jax.random.split(key, 10))
    ax.plot(x[...,0].T, y0[...,0].T, "C0", alpha=0.5)
    ax.plot(xc[...,0].T, yc[...,0].T, "C3o")
    return {"prior": fig_prior, "conditional": fig_cond}


def plot_prior_image(state: TrainingState, key: Rng):
    fig, axes = plt.subplots(3, 3, figsize=(5, 5), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    x = get_image_grid_inputs(config.image_size)
    x, y0 = jax.vmap(lambda k: sample_prior(state, k, x))(jax.random.split(key, len(axes)))
    y0 = rev_rescale(y0)
    im_shape = (config.image_size, config.image_size, config.output_dim)
    for i in range(len(axes)):
        axes[i].imshow(y0[i].reshape(im_shape))
        axes[i].set_xticks([]); axes[i].set_yticks([]);

    return {"prior": fig}


state = init(batch_init, jax.random.PRNGKey(config.seed))

experiment_dir_if_exists = pathlib.Path(config.restore)
if (experiment_dir_if_exists / "checkpoints").exists():
    index = state_utils.find_latest_checkpoint_step_index(str(experiment_dir_if_exists))
    if index is not None:
        state = state_utils.load_checkpoint(state, str(experiment_dir_if_exists), step_index=index)
        print("Restored checkpoint at step {}".format(state.step))


exp_root_dir = get_experiment_dir(config)
local_writer = writers.LocalWriter(str(exp_root_dir), flush_every_n=100)
tb_writer = writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))
aim_writer = writers.AimWriter(EXPERIMENT)
writer = writers.MultiWriter([aim_writer, tb_writer, local_writer])
writer.log_hparams(asdict(config))

ds_train: Dataset = get_image_data(
    config,
    batch_size=config.batch_size,
    num_epochs=config.num_epochs,
    train=True,
)

actions = [
    actions.PeriodicCallback(
        every_steps=10,
        callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
    ),
    actions.PeriodicCallback(
        every_steps=config.total_steps // 100,
        callback_fn=lambda step, t, **kwargs: writer.write_figures(step, plot_prior_image(kwargs["state"], kwargs["key"]))
    ),
    actions.PeriodicCallback(
        every_steps=config.total_steps // 25,
        callback_fn=lambda step, t, **kwargs: state_utils.save_checkpoint(kwargs["state"], exp_root_dir, step)
    ),
]

steps = range(state.step + 1, config.total_steps + 1)
progress_bar = tqdm.tqdm(steps)

for step, batch_init in zip(progress_bar, ds_train):
    if step < state.step: continue  # wait for the state to catch up in case of restarts

    state, metrics = update_step(state, batch_init)
    metrics["lr"] = learning_rate_schedule(state.step)

    for action in actions:
        action(step, t=None, metrics=metrics, state=state, key=key)

    if step % 100 == 0:
        progress_bar.set_description(f"loss {metrics['loss']:.2f}")


print("EVALUATION")
import numpy as np

net_with_params = partial(net, state.params_ema)

@partial(jax.vmap, in_axes=(0, None, None, None, None))
@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
@jax.jit
def sample_n_conditionals(key, x_test, x_context, y_context, mask_context):
    return process.conditional_sample(
        key, x_test, mask=None, x_context=x_context, y_context=y_context, mask_context=mask_context, model_fn=net_with_params)


KEEP = 33  # random number
NOT_KEEP = 44  # random number


def get_context_mask(key, config: Config, context_type: str | float = "horizontal") -> ndarray:
    x = get_image_grid_inputs(config.image_size)
    if context_type == "horizontal":
        condition = x[..., 1] > 0.0
    elif context_type == "vertical":
        condition = x[..., 0] < 0.0
    elif isinstance(context_type, float):
        p = context_type
        condition = jax.random.uniform(key, shape=(len(x),)) <= p
    else:
        raise ValueError(f"Unknown context type {context_type}")

    return jnp.where(
        condition,
        KEEP * jnp.ones_like(x[..., 0]),
        NOT_KEEP * jnp.ones_like(x[..., 0]),
    )

get_idx_keep = lambda x: jnp.where(x == KEEP, jnp.ones(x.shape, dtype=bool), jnp.zeros(x.shape, dtype=bool))

rescale, rev_rescale = get_rescale_function_fwd_and_inv(config)
ds_test: Dataset = get_image_data(
    config,
    batch_size=config.eval.batch_size,
    num_epochs=1,
    train=False,
)
n_samples = 6

batch0 = next(ds_test)
x_target = get_image_grid_inputs(config.image_size)
im_shape = (config.image_size, config.image_size, config.output_dim)

PERCENTAGES = [0.05, 0.1, 0.2, .4, .8, .95]
data = []

for i, percentage in enumerate(PERCENTAGES):

    key, ckey = jax.random.split(key)
    context_mask = get_context_mask(ckey, config, percentage)
    num_context_points = jnp.where(context_mask == KEEP, jnp.ones_like(context_mask), jnp.zeros_like(context_mask)).sum()
    print(percentage, num_context_points, num_context_points / len(context_mask))

    key, skey = jax.random.split(key)
    samples = sample_n_conditionals(
        jax.random.split(skey, n_samples),
        batch0.x_target,
        batch0.x_target[:, get_idx_keep(context_mask)],
        batch0.y_target[:, get_idx_keep(context_mask)],
        jnp.zeros_like(batch0.x_target[:, get_idx_keep(context_mask)][..., 0]),
    )  # [num_samples, batch_size, num_points, output_dim]
    samples = jax.vmap(rev_rescale)(samples)
    target = rev_rescale(batch0.y_target)

    m = jnp.mean(samples, axis=0)  # [batch_size, num_points, output_dim]
    mse = jnp.mean((m - target)**2, axis=[1, 2])  # [batch_size]
    data.append({"per": percentage, "mse_mean": jnp.mean(mse), "mse_std": jnp.std(mse)})
    print(data[-1])


import pandas as pd
df = pd.DataFrame(data)
print(df)
df.to_csv(f"eval_{config.dataset}.csv")
