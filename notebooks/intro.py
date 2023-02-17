# %%
# In this notebook we cover the basics of Neural Diffusion Process (NDP) models.
# We'll cover:
#  - the forward and reverse SDE formulation,
#  - learning the score (gradient of the log prob) from data,
#  - conditional sampling.

# Neural Diffusion Processes are a particular class of score-based generative models, where
# one parameterises the score through the function values *and* the function inputs.
# %%
from typing import Iterator, Tuple
import jaxkern

import pandas as pd
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)


from jaxtyping import Array
from matplotlib import cm

from check_shapes import check_shapes, check_shape

from neural_diffusion_processes.data import get_gp_data, dataloader, DataBatch
from neural_diffusion_processes.sde import ScalarControlTerm
from neural_diffusion_processes.types import RNGKey

# %%

# A NDP defines a stochastic differential equation (SDE) on the function values `y` of the form
# $$
#  dy = - 0.5 \beta_t y_t dt + \sqrt{ \beta_t (1 - \exp(\int_{s=0}^t \beta_t ds))} dw,
# $$
# where `dw` is Brownian motion, and `dt` is an infinitesimal small step forward in time.
# First term is called the drift, the second the diffusion. We use `diffrax` to run this process from
# time 0 to 1. Notice that the 'noise' that is added to the function values is independent of their location.

t0, t1 = 1e-5, 1.0
dt = 1.0e-3
beta0, beta1 = 0.0, 20.0
plot_num_timesteps = 5

beta_schedule = lambda t: beta0 + t * (beta1 - beta0)
int_beta_schedule = lambda t: beta0 * t + t**2 * (beta1 - beta0) / 2.0


@check_shapes("t: []", "yt: [N, 1]", "return: [N, 1]")
def drift(t, yt, x):
    del x
    return -0.5 * beta_schedule(t) * yt  # [N, 1]


@check_shapes("t: []", "yt: [N, 1]", "return: []")
def diffusion(t, yt, x):
    del x, yt
    return jnp.sqrt(beta_schedule(t))



def forward_sde_process(key, y0) -> dfx.Solution:
    """Uses diffrax to run the SDE forward from 0 to 1. Stores a couple of intermediate results."""
    shape = jax.ShapeDtypeStruct(y0.shape, y0.dtype)
    bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt / 2.0, shape=shape, key=key)
    terms = dfx.MultiTerm(dfx.ODETerm(drift), ScalarControlTerm(diffusion, bm))
    ts = jnp.linspace(t0, t1, plot_num_timesteps)
    ts = t0 + (t1 - t0) * (jnp.exp(ts) - jnp.exp(t0)) / (jnp.exp(t1) - jnp.exp(t0))
    saveat = dfx.SaveAt(ts=ts)
    return dfx.diffeqsolve(
        terms,
        solver=dfx.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        adjoint=dfx.NoAdjoint(),
        saveat=saveat,
    )


SEED = 0
key = jax.random.PRNGKey(SEED)
x = jnp.linspace(-1, 1, 30)[:, None]
y0 = jnp.stack([jnp.sin(2 * x),  jnp.sin(-2 * x), jnp.sin(5 * x)])
num_samples = len(y0)
key, *keys = jax.random.split(key, num=num_samples+1)
out = jax.vmap(forward_sde_process)(jnp.stack(keys), y0)

# viridis = cm.get_cmap("viridis", 12)

fig, axes = plt.subplots(plot_num_timesteps, num_samples, sharex=True, sharey=True)
for t in range(plot_num_timesteps):
    for i in range(num_samples):
        axes[t, i].plot(x, out.ys[i, t], 'o-', ms=2)
        axes[t, 0].set_ylabel(f"t = {out.ts[0, t]:.2f}")


# We start from a function sample (in our case just some sinusoidal data) and notice how
# we slowly corrupt the function values. At time $t=1$ the function values are $N(0,1)$.
# %%

# A cool property for the above SDE is that we can directly compute the mean and covariance 
# for a given time.

@check_shapes("t: []", "y0: [N, 1]", "return[0]: [N, 1]", "return[1]: [N, 1]")
def marginal_distribution(t, y0):
    """Mean and std. dev of p(yt | y0)."""
    int_beta_t = int_beta_schedule(t)
    mean = jnp.exp(-0.5 * int_beta_t) * y0
    var = jnp.maximum(1. - jnp.exp(-int_beta_t), 1e-5)
    std = jnp.sqrt(var)
    return mean, std * jnp.ones_like(y0)


# In the plot above, we will overlay the amalytical mean and (diagonal) std. dev.

fig, axes = plt.subplots(plot_num_timesteps, num_samples, sharex=True, sharey=True)
for t_index in range(plot_num_timesteps):
    for sample_index in range(num_samples):
        t = out.ts[0, t_index]
        mean, std = marginal_distribution(t, out.ys[sample_index, 0])
        axes[t_index, sample_index].plot(x, out.ys[sample_index, t_index], 'o-', ms=2)
        axes[t_index, sample_index].plot(x, mean, 'k--', alpha=.5)
        lo, up = (v.flatten() for v in (mean - 2 * std, mean + 2 * std))
        axes[t_index, sample_index].fill_between(x.flatten(), lo, up, alpha=.2, color='k')
        axes[t_index, 0].set_ylabel(f"t = {t:.2f}")
        axes[t_index, sample_index].set_ylim(-2.5, 2.5)

# %%

# The reverse SDE has the following form

# which requires the unknown score

# We can learn this score from data.


# %%

# %%

# Let's load some data
kernel = jaxkern.RBF(active_dims=list(range(1)))
key, dkey = jax.random.split(key)
data = get_gp_data(key, kernel=kernel, num_samples=10_000)

# %%

# A simple loader that chunks the data (function inputs and outputs) into batches. The iterator
# yields objects of type `DataBatch` which is a convenient way of explicitly keeping track of the data.


BATCH_SIZE = 32
key, dlkey = jax.random.split(key)
dataset_iter = dataloader(data, batch_size=BATCH_SIZE, key=dlkey)
batch0 = next(dataset_iter)

# A batch contains `BATCH_SIZE` different function draws (both the inputs and outputs are random).
# The inputs are distributed uniformly with [-1, 1] and the outputs are draws from a GP.

# sorted_indices = jnp.argsort(batch.function_inputs, axis=1).flatten()
plt.plot(batch0.function_inputs[:, :, 0].T, batch0.function_outputs[:, :, 0].T, '.');

# %%

# We implement the loss for a single function draw (x, y) which has been perturbed for `t` time as:


@check_shapes("t: []", "y: [num_points, output_dim]", "x: [num_points, input_dim]", "return: []")
def single_loss(network: eqx.Module, key: RNGKey, t: Array, y: Array, x: Array):
    weight = 1 - jnp.exp(-int_beta_schedule(t))
    ekey, nkey = jax.random.split(key)
    mean, std = marginal_distribution(t, y)
    eps = jax.random.normal(ekey, mean.shape)
    yt = mean + std * eps
    score = std * network(t[None], yt[None], x[None], key=nkey)[0]  # ScoreModel assumes batch dim
    objective = eps
    # objective = eps / std
    return weight * jnp.mean(jnp.sum((objective - score) ** 2, -1), -1)

import functools

def loss(network: eqx.Module, batch: DataBatch, key: RNGKey):
    key, tkey = jax.random.split(key)
    batch_size = len(batch)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=t0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)

    keys = jax.random.split(key, batch_size)
    x, y = batch.function_inputs, batch.function_outputs
    batch_loss_fn = functools.partial(single_loss, network)
    batch_loss_fn = jax.vmap(batch_loss_fn)
    error = batch_loss_fn(keys, t, y, x)
    return jnp.mean(error)


# %%

from neural_diffusion_processes.bidimensional_attention_model import BiDimensionalAttentionScoreModel

key, nkey = jax.random.split(key)
network = BiDimensionalAttentionScoreModel(
    num_bidim_attention_blocks=2,
    hidden_dim=16,
    num_heads=4,
    key=nkey,
)
# %%

learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4, peak_value=1e-3, warmup_steps=1000, decay_steps=5_000, end_value=1e-4
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

opt_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))


@eqx.filter_jit
def step(
    network: eqx.Module,
    batch: DataBatch,
    key: RNGKey,
    opt_state: optax.OptState,
    opt_update,
) -> Tuple[Array, eqx.Module, optax.OptState]:
    loss_fn = eqx.filter_value_and_grad(loss)
    loss_value, grads = loss_fn(network, batch, key)
    updates, opt_state = opt_update(grads, opt_state)
    network = eqx.apply_updates(network, updates)
    return loss_value, network, opt_state


# %%

NUM_STEPS = 5_000
metrices = []
progress_bar = tqdm.tqdm(list(range(1, NUM_STEPS + 1)), miniters=1)

for i, batch in zip(progress_bar, dataset_iter):
    key, subkey = jax.random.split(key)
    value, network, opt_state = step(network, batch, subkey, opt_state, optimizer.update)
    metrices.append({'train_loss': value})

    if i % 100 == 0:
        progress_bar.set_description(f"loss {value:.2f}")

metrices = pd.DataFrame(metrices)
# %%
plt.plot(metrices['train_loss'].values[::100])
plt.plot(metrices['train_loss'].rolling(10).mean().values[::100])
# %%
# %%

# We now have a `trained` score network that we can use to reverse the SDE, and practically initialise the process to N(0,1) and generate function draws from our data distribution.


@eqx.filter_jit
def reverse_process(scoremodel, yT, x, key):

    def reverse_drift(t, yt, x):
        score = scoremodel(t[None], yt[None], x[None], key=key)[0]
        _, std = marginal_distribution(t, yt * jnp.nan)
        score = - score / std
        return drift(t, yt) - 0.5 * beta_schedule(t) * score
        # return drift(t, yt) - 0.5 * (diffusion(t, yt) ** 2) * score

    term = dfx.ODETerm(reverse_drift)
    solver = dfx.Euler()
    # reverse time, solve from t1 to t0
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 10)[::-1])
    sol = dfx.diffeqsolve(term, solver, t1, t0, dt0=-dt/2, y0=yT, args=x, adjoint=dfx.NoAdjoint(), saveat=saveat)
    return sol.ys


num_samples = 64
x_test = jnp.linspace(0, 1, 50)[:, None]
key, yTkey = jax.random.split(key)
yT = jax.random.normal(yTkey, (num_samples, len(x_test), 1))

rkey = jax.random.split(key, num=num_samples)
sol = jax.vmap(lambda yT, key: reverse_process(network, yT, x_test, key))(
    yT, rkey
)

# Conditional samples
# %%
# plt.plot(x_test, sol.ys[-1, ...,0].T)
# plt.plot(x_test, sol.ys[0], '.')
plt.plot(x_test, sol[:, -1, :, 0].T, '.')
plt.ylim(-4, 4)

# %%