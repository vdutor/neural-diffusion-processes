from __future__ import annotations
from typing import Tuple, List, Callable, Mapping, Optional

import abc
from dataclasses import dataclass
import jaxkern
import jax
import gpjax
import jax.numpy as jnp
from jaxtyping import Float, Array
import distrax
from gpjax import Prior

import jax

from neural_diffusion_processes.types import Batch


@dataclass
class UniformDiscrete:
    lower: int
    upper: int

    def sample(self, key, shape):
        if self.lower == self.upper:
            return jnp.ones(shape, dtype=jnp.int32) * self.lower
        return jax.random.randint(key, shape, minval=self.lower, maxval=self.upper + 1)


DATASETS = [
    "se",
    "matern",
    "sawtooth",
    "step",
]

TASKS = [
    "training",
    "interpolation",
]


@dataclass
class TaskConfig:
    x_context_dist: distrax.Distribution
    x_target_dist: distrax.Distribution


@dataclass
class DatasetConfig:
    max_input_dim: int  # (incl.)
    is_gp: bool
    train_num_target: UniformDiscrete = UniformDiscrete(1, 60)
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)


_NOISE_VAR = 0.05**2
_KERNEL_VAR = 1.0
_LENGTHSCALE = .25

_DATASET_CONFIGS = {
    "se": DatasetConfig(max_input_dim=3, is_gp=True),
    "matern": DatasetConfig(max_input_dim=3, is_gp=True),
    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),
    "step": DatasetConfig(max_input_dim=1, is_gp=False),
}

_TASK_CONFIGS = {
    "training": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
    "interpolation": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
}


@dataclass
class FuntionalDistribution(abc.ABC):
    # All GP datasets are naturally normalized so do not need additional normalization.
    # Sawtooth is not normalized so we need to normalize it in the Mixture but not when used
    # in isolation.
    is_data_naturally_normalized: bool = True
    normalize: bool = False

    @abc.abstractmethod
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        raise NotImplementedError()
        

class GPFunctionalDistribution(FuntionalDistribution):

    def __init__(self, kernel: jaxkern.base.AbstractKernel, params: Mapping):
        self.kernel = kernel
        self.params = params
        self.mean = gpjax.mean_functions.Zero()
        self.prior = Prior(self.kernel, self.mean)
    
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        f = self.prior.predict(self.params)(x).sample(seed=key, sample_shape=()).reshape(x.shape[0], 1)
        sigma2 = self.params["noise_variance"]
        y = f + (jax.random.normal(key, shape=f.shape) * jnp.sqrt(sigma2))
        return y


DatasetFactory: Callable[[List[int]], FuntionalDistribution]

_DATASET_FACTORIES: Mapping[str, DatasetFactory] = {}

def register_dataset_factory(name: str):

    def wrap(f: DatasetFactory):
        _DATASET_FACTORIES[name] = f
    
    return wrap

    
@register_dataset_factory("se")
def _se_dataset_factory(active_dim: List[int]):
    k = jaxkern.stationary.RBF(active_dims=active_dim)
    # white = jaxkern.White(active_dims=active_dim)
    # kernel = jaxkern.SumKernel([rbf, white])
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
            # {"variance": 0,},
        # ],
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(k, params)


@register_dataset_factory("matern")
def _matern_dataset_factory(active_dim: List[int]):
    k = jaxkern.stationary.Matern52(active_dims=active_dim)
    # white = jaxkern.White(active_dims=active_dim)
    # kernel = jaxkern.SumKernel([mat, white])
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
            # {"variance": _NOISE_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(k, params)


class Sawtooth(FuntionalDistribution):

    A = 1.
    K_max = 20
    mean = 0.5
    variance = 0.07965

    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        fkey, skey, kkey = jax.random.split(key, 3)
        f = jax.random.uniform(fkey, (), minval=3., maxval=5.)
        s = jax.random.uniform(skey, (), minval=-5., maxval=5.)
        ks = jnp.arange(1, self.K_max + 1, dtype=x.dtype)[None, :]
        vals = (-1.) ** ks * jnp.sin(2. * jnp.pi * ks * f * (x - s)) / ks
        k = jax.random.randint(kkey, (), minval=10, maxval=self.K_max + 1)
        mask = jnp.where(ks < k, jnp.ones_like(ks), jnp.zeros_like(ks))
        # we substract the mean A/2
        fs = self.A/2 + self.A/jnp.pi * jnp.sum(vals * mask, axis=1, keepdims=True)
        fs = fs - self.mean
        if self.normalize:
            fs = fs / jnp.sqrt(self.variance)
        return fs


@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory(*_):
    return Sawtooth(is_data_naturally_normalized=False, normalize=False)


class Step(FuntionalDistribution):

    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        s = jax.random.uniform(key, (), minval=-2., maxval=2.)
        fs = jnp.where(x < s, jnp.zeros_like(x), jnp.ones_like(x))
        return fs


@register_dataset_factory("step")
def _sawtooth_dataset_factory(*_):
    return Step()
    

def get_batch(key, batch_size: int, name: str, task: str, input_dim: int):
    if name not in DATASETS:
        raise NotImplementedError("Unknown dataset: %s." % name)
    if task not in TASKS:
        raise NotImplementedError("Unknown task: %s." % task)
    
    
    if input_dim > _DATASET_CONFIGS[name].max_input_dim:
        raise NotImplementedError(
            "Too many active dims for dataset %s. Max: %d, got: %d." % (
                name, _DATASET_CONFIGS[name].max_input_dim, len(active_dims)
            )
        )

    if task == "training":
        max_n_target = _DATASET_CONFIGS[name].train_num_target.upper
        max_n_context = 0
    else:
        max_n_target = _DATASET_CONFIGS[name].eval_num_target.upper
        max_n_context = _DATASET_CONFIGS[name].eval_num_context.upper * input_dim

    key, ckey, tkey = jax.random.split(key, 3)
    task = _TASK_CONFIGS[task]
    x_context = task.x_context_dist.sample(seed=ckey, sample_shape=(batch_size, max_n_context, input_dim))

    # if task is "training":
    #     mask_context = None
    # else:
    #     mask_context = 

    x_target = task.x_target_dist.sample(seed=tkey, sample_shape=(batch_size, max_n_target, input_dim))
    x = jnp.concatenate([x_context, x_target], axis=1)

    keys = jax.random.split(key, batch_size)
    active_dims = list(range(input_dim))
    sample_func = _DATASET_FACTORIES[name](active_dims).sample
    y = jax.vmap(sample_func)(keys, x)
    return Batch(
        x_target=x_target,
        y_target=y[:, max_n_context:, :],
        x_context=x_context,
        y_context=y[:, :max_n_context, :],
        mask_context=None,
        mask_target=None,
    )


# class DatasetFromGenerator:
#     def __init__(self, generator, key):
#         self._key = key
#         self._generator  = generator
#         self._preprocess = []
    
#     def map(self, function):
#         self._preprocess.append(function)
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         batch = next(self._generator)
#         for func in self._preprocess:
#             self._key, key = jax.random.split(self._key)
#             batch = func(batch, key=key)
#         return batch


# def data_generator(key, dataset, task, total_num_samples, batch_size, num_epochs: Optional[int] = None):
#     """
#     :param num_epochs: if `None` generator runs forever
#     """
#     assert total_num_samples % batch_size == 0

#     @jax.jit
#     def batch(key) -> DataBatch:
#         return get_batch(key, batch_size, dataset, task)

#     _ = batch(key)

#     if num_epochs is None:
#         num_epochs = jnp.inf
    
#     count_epochs = 0
#     while count_epochs < num_epochs:
#         count_epochs += 1
#         for _ in range(total_num_samples // batch_size):
#             key, bkey = jax.random.split(key)
#             yield batch(bkey)



# def get_padding_function(dataset: str, task: str):
#     if task == "training":
#         target_num_data_sampler = _DATASET_CONFIGS[dataset].train_num_target
#         context_num_data_sampler = None
#     else:
#         target_num_data_sampler = _DATASET_CONFIGS[dataset].eval_num_target
#         context_num_data_sampler = _DATASET_CONFIGS[dataset].eval_num_context

#     @jax.jit
#     def padding(batch: DataBatch, key):
#         num_data_total = batch.xs.shape[1]
#         num_data = target_num_data_sampler.sample(key, shape=())
#         mask = jnp.where(
#             jnp.arange(num_data_total)[None, :, None] < num_data,
#             jnp.zeros_like(batch.xs),  # keep
#             jnp.ones_like(batch.xs)  # ignore
#         )[..., 0]

#         # repeat for context
#         if context_num_data_sampler is not None:
#             num_data_total = batch.xc.shape[1]
#             num_data = context_num_data_sampler.sample(key, shape=())
#             mask_context = jnp.where(
#                 jnp.arange(num_data_total)[None, :, None] < num_data,
#                 jnp.zeros_like(batch.xc),  # keep
#                 jnp.ones_like(batch.xc),  # ignore
#             )[..., 0]
#         else:
#             mask_context = None

#         return DataBatch(
#             xs=batch.xs,
#             ys=batch.ys,
#             mask=mask,
#             xc=batch.xc,
#             yc=batch.yc,
#             mask_context=mask_context
#         )

#     return padding


# def get_dataset(dataset: str, task: str, *, key, batch_size: int, samples_per_epoch: int, num_epochs: Optional[int] = None) -> DatasetFromGenerator:
#     gkey, dskey = jax.random.split(key)
#     gen = data_generator(gkey, dataset, task, samples_per_epoch, batch_size, num_epochs)
#     ds = DatasetFromGenerator(gen, dskey)
#     ds.map(get_padding_function(dataset, task))
#     return ds


#%%
if __name__ == "__main__":
    import matplotlib
    jax.config.update("jax_enable_x64", True)


    def plot_data():
        import numpy
        import matplotlib.pyplot as plt
        import itertools

        def info(a, name):
            print(name)
            print(a.shape)
            print("="*10)

        def plot_data(xc, yc, xt, yt, ax, legend=True, ns=1):
            info(xc, "context")
            info(xt, "target")
            ax.plot(xt[:ns, :, 0].T, yt[:ns, :, 0].T, "C1.", label="target")
            ax.plot(xc[:ns, :, 0].T, yc[:ns, :, 0].T, "C0.", label="context")
            handles, labels = ax.get_legend_handles_labels()
            labels, ids = numpy.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            if legend:
                ax.legend(handles, labels, loc='best')


        key = jax.random.PRNGKey(0)

        fig, axes = plt.subplots(len(_DATASET_FACTORIES), len(_TASK_CONFIGS), figsize=(15, 5), sharex=True, tight_layout=True)

        for (i, dataset), (j, task) in itertools.product(enumerate(_DATASET_FACTORIES.keys()), enumerate(_TASK_CONFIGS.keys())):
            print(dataset, task)
            ax = axes[i,j]
            ax.set_xlim(-4, 6)
            data = get_batch(key, 16, dataset, task)
            plot_data(data.xc, data.yc, data.xs, data.ys, ax, legend=(i==0) and (j==0))
            if i == 0:
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(dataset)
        
        plt.savefig("fig1.png")


        nrows = len(_DATASET_FACTORIES)
        fig, axes = plt.subplots(nrows, 1, figsize=(15, 3 * nrows), sharex=True)
        for i, name in enumerate(_DATASET_FACTORIES.keys()):
            ax = axes[i]
            keys = jax.random.split(key, 16)
            x = jnp.linspace(-2, 3, 500)[:, None]
            y = jax.vmap(_DATASET_FACTORIES[name].sample, in_axes=[0, None])(keys, x)
            ax.set_title(name)
            ax.plot(x, y[:3, :, 0].T)

        plt.savefig("fig2.png")
    
    plot_data()