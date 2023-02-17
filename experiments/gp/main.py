from typing import Any, Tuple, Mapping, Iterator
from absl import app

import functools
import tqdm
import pathlib
import equinox as eqx
import jaxkern
import jax
import jax.numpy as jnp
import pandas as pd
import optax
import datetime
import matplotlib.pyplot as plt

from ml_collections import config_flags
from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)

from ml_tools import config_utils
import ml_tools

from neural_diffusion_processes.types import RNGKey
import neural_diffusion_processes as ndp

try:
    from .config import Config
except:
    from config import Config


_DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
_HERE = pathlib.Path(__file__).parent
_LOG_DIR = 'logs'


_CONFIG = config_flags.DEFINE_config_dict("config", config_utils.to_configdict(Config()))


class TrainingState(eqx.Module):
    network: eqx.Module
    opt_state: optax.OptState
    key: RNGKey
    step: int


def get_experiment_name(config: Config):
    return f"{_DATETIME}_{config_utils.get_id(config)}"


def get_experiment_dir(config: Config, output: str = "root", exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)

    if output == "root":
        dir_ = _HERE / _LOG_DIR / experiment_name
    elif output == "plots":
        dir_ = _HERE / _LOG_DIR / experiment_name / "plots"
    elif output == "tensorboard":
        dir_ = _HERE / _LOG_DIR / "tensorboard" / experiment_name
    else:
        raise ValueError("Unknown output: %s" % output)

    dir_.mkdir(parents=True, exist_ok=exist_ok)
    return dir_


def get_kernel(kernel_type: str) -> jaxkern.base.AbstractKernel:
    if kernel_type == "matern12":
        return jaxkern.stationary.Matern12(active_dims=list(range(1)))
    elif kernel_type == "matern32":
        return jaxkern.stationary.Matern32(active_dims=list(range(1)))
    elif kernel_type == "matern52":
        return jaxkern.stationary.Matern52(active_dims=list(range(1)))
    elif kernel_type == "rbf":
        return jaxkern.stationary.RBF(active_dims=list(range(1)))
    elif kernel_type == "white":
        return jaxkern.stationary.White(active_dims=list(range(1)))
    else:
        raise NotImplementedError("Unknown kernel: %s" % kernel_type)


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


def main(_):
    config = config_utils.to_dataclass(Config, _CONFIG.value)

    path = get_experiment_dir(config, 'root') / 'config.yaml'
    with open(str(path), 'w') as f:
        f.write(config_utils.to_yaml(config))

    key = jax.random.PRNGKey(config.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    beta = ndp.sde.LinearBetaSchedule()
    sde = ndp.sde.SDE(beta)

    ####### prepare data
    data_kernel = get_kernel(config.data.kernel)
    data = ndp.data.get_gp_data(
        next(key_iter),
        data_kernel,
        num_samples=config.data.num_samples,
        num_points=config.data.num_points,
        params=config.data.hyperparameters,
    )
    dataloader = ndp.data.dataloader(
        data,
        batch_size=config.optimization.batch_size,
        key=next(key_iter)
    )
    batch0 = next(dataloader)
    
    plt.plot(batch0.function_inputs[..., 0].T, batch0.function_outputs[..., 0].T, ".")
    plt.savefig(str(get_experiment_dir(config, "plots") / "data.png"))

    ####### Forward haiku model
    network = ndp.bidimensional_attention_model.BiDimensionalAttentionScoreModel(
        num_bidim_attention_blocks=config.network.num_bidim_attention_layers,
        hidden_dim=config.network.hidden_dim,
        num_heads=config.network.num_heads,
        key=next(key_iter)
    )

    def loss_fn(network: eqx.Module, batch: ndp.data.DataBatch, key: RNGKey):
        # Network awkwardly requires a batch dimension for the inputs
        network = lambda t, yt, x, *, key: network(t[None], yt[None], x[None], key=key)[0]
        return ndp.sde.loss(sde, network, batch, key)

    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-3, warmup_steps=1000, decay_steps=config.optimization.num_steps, end_value=1e-4
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))

    @eqx.filter_jit
    def update_step(state: TrainingState, batch: ndp.data.DataBatch) -> Tuple[TrainingState, Mapping[str, Any]]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
        loss_value, grads = loss_and_grad_fn(state.network, batch, loss_key)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_network = eqx.apply_updates(state.network, updates)
        new_state = TrainingState(
            network=new_network,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1
        )
        metrics = {
            'loss': loss_value,
            'step': state.step
        }
        return new_state, metrics


    state = TrainingState(
        network=network,
        opt_state=opt_state,
        key=next(key_iter),
        step=0
    )

    progress_bar = tqdm.tqdm(list(range(1, config.optimization.num_steps + 1)), miniters=1)
    exp_root_dir = get_experiment_dir(config)
    
    ########## Plotting
    # x_plt = jnp.linspace(-1, 1, 100)[:, None]
    # x_context = jnp.array([-0.5, 0.2, 0.4]).reshape(-1, 1)
    # y_context = x_context * 0.0

    # @jax.jit
    # def plot_reverse(key, params):
    #     net_ = functools.partial(net, params)
    #     return ndp.sde.reverse_solve(sde, net_, x_plt, key=key)

    # @jax.jit
    # def plot_cond(key, params):
    #     net_ = functools.partial(net, params)
    #     return ndp.sde.conditional_sample(sde, net_, x_context, y_context, x_plt, key=key)

    # def plots(state: TrainingState, key) -> Mapping[str, plt.Figure]:
    #     fig_reverse, ax = plt.subplots()
    #     out = jax.vmap(plot_reverse, in_axes=[0, None])(jax.random.split(key, 100), state.params)
    #     ax.plot(x_plt, out[:, -1, :, 0].T, "C0", alpha=.3)

    #     fig_cond, ax = plt.subplots()
    #     key = jax.random.PRNGKey(0)
    #     samples = jax.vmap(plot_cond, in_axes=[0, None])(jax.random.split(key, 100), state.params)
    #     ax.plot(x_plt, samples[..., 0].T, "C0", alpha=.3)
    #     ax.plot(x_context, y_context, "ko")
        
    #     return {
    #         "reverse": fig_reverse,
    #         "conditional": fig_cond,
    #     }

    #############

    local_writer = ml_tools.writers.LocalWriter(exp_root_dir, flush_every_n=100)
    tb_writer = ml_tools.writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))
    writer = ml_tools.writers.MultiWriter([tb_writer])

    action1 = ml_tools.actions.PeriodicCallback(
        every_steps=1,
        callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
    )
    action2 = ml_tools.actions.PeriodicCallback(
        every_steps=500,
        callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(kwargs["state"], exp_root_dir, step)
    )
    # action3 = ml_tools.actions.PeriodicCallback(
    #     every_steps=500,
    #     callback_fn=lambda step, t, **kwargs: writer.write_figures(step, plots(kwargs["state"], kwargs["key"]))
    # )
    actions = [action1, action2]

    for step, batch, key in zip(progress_bar, dataloader, key_iter):
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")

if __name__ == "__main__":
    app.run(main)