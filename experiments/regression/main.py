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

import neural_diffusion_processes as ndp
from neural_diffusion_processes.types import Dataset, Batch, Rng
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import cosine_schedule, GaussianDiffusion
from neural_diffusion_processes.utils.config import setup_config
from neural_diffusion_processes.utils.state import TrainingState
from neural_diffusion_processes.utils import writers
from neural_diffusion_processes.utils import actions
from config import Config


EXPERIMENT = "regression-May23"
DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
HERE = pathlib.Path(__file__).parent
LOG_DIR = 'logs'


def get_experiment_name(config: Config):
    del config  # Currently not used but name could be based on config
    letters = string.ascii_lowercase
    id = ''.join(random.choice(letters) for i in range(4))
    return f"{DATETIME}_{id}"


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


def get_gp_data(
    dataset: str,
    input_dim: int = 1,
    train: bool = True,
    batch_size: int = 1024,
    num_epochs: int = 1,
) -> Dataset:
    task = "training" if train else "interpolation"    
    data = np.load(f"data/{dataset}_{input_dim}_{task}.npz")
    ds = tf.data.Dataset.from_tensor_slices({
        "x_target": data["x_target"].astype(np.float32),
        "y_target": data["y_target"].astype(np.float32),
        "x_context": data["x_context"].astype(np.float32),
        "y_context": data["y_context"].astype(np.float32),
    })
    if train:
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(seed=42, buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return map(lambda d: Batch(**d), ds)



config: Config = setup_config(Config)

if ('mnist' in config.dataset) or ('celeba' in config.dataset):
    ds_train: Dataset = get_image_data(
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
    )
else:
    ds_train: Dataset = get_gp_data(
        config.dataset,
        input_dim=1,
        train=True,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
    )

batch0 = next(ds_train)
key = jax.random.PRNGKey(config.seed)

beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)


@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask):
    model = BiDimensionalAttentionModel(
        n_layers=config.network.n_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
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
        'step': state.step
    }
    return new_state, metrics


@jax.jit
def sample_prior(state: TrainingState, key: Rng, x: jnp.array):
    print("compiling sample_prior")
    key, ykey, bkey  = jax.random.split(key, 3)
    yT = jax.random.normal(ykey, (len(x), 1))
    net_with_params = partial(net, state.params_ema)
    y0 = process.sample(bkey, yT, x, mask=None, model_fn=net_with_params)
    return x, y0


def plot_prior(state: TrainingState, key: Rng):
    fig, ax = plt.subplots()
    x = jnp.linspace(-2, 2, 60)[:, None]
    x, y0 = jax.vmap(lambda k: sample_prior(state, k, x))(jax.random.split(key, 10))
    ax.plot(x[...,0].T, y0[...,0].T, color="C0", alpha=0.5)
    return {"prior": fig}


state = init(batch0, jax.random.PRNGKey(config.seed))


exp_root_dir = get_experiment_dir(config)
local_writer = writers.LocalWriter(str(exp_root_dir), flush_every_n=100)
tb_writer = writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))
# aim_writer = writers.AimWriter(EXPERIMENT)
writer = writers.MultiWriter([tb_writer, local_writer])
writer.log_hparams(asdict(config))


actions = [
    actions.PeriodicCallback(
        every_steps=10,
        callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
    ),
    actions.PeriodicCallback(
        every_steps=config.total_steps // 5,
        callback_fn=lambda step, t, **kwargs: writer.write_figures(step, plot_prior(kwargs["state"], kwargs["key"]))
    )
    # ml_tools.actions.PeriodicCallback(
    #     every_steps=None if is_smoketest(config) else num_steps // 4,
    #     callback_fn=lambda step, t, **kwargs: [
    #         cb(step, t, **kwargs) for cb in task_callbacks
    #     ]
    # ),
    # ml_tools.actions.PeriodicCallback(
    #     every_steps=num_steps//10,
    #     callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(kwargs["state"], exp_root_dir, step)
    # ),
]


progress_bar = tqdm.tqdm(list(range(1, config.total_steps + 1)), miniters=1)

for step, batch in zip(progress_bar, ds_train):
    if step < state.step: continue  # wait for the state to catch up in case of restarts

    state, metrics = update_step(state, batch)
    metrics["lr"] = learning_rate_schedule(state.step)

    for action in actions:
        action(step, t=None, metrics=metrics, state=state, key=key)

    if step % 100 == 0:
        progress_bar.set_description(f"loss {metrics['loss']:.2f}")