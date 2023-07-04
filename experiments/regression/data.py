from __future__ import annotations
from typing import Tuple, List, Callable, Mapping, Optional

import abc
from dataclasses import dataclass
import gpjax.kernels as jaxkern
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

    def __init__(self, kernel: jaxkern.AbstractKernel, params: Mapping):
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
    k = jaxkern.RBF(active_dims=active_dim)
    input_dim = len(active_dim)
    factor = jnp.sqrt(input_dim)
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE * factor, "variance": _KERNEL_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(k, params)


@register_dataset_factory("matern")
def _matern_dataset_factory(active_dim: List[int]):
    k = jaxkern.Matern52(active_dims=active_dim)
    input_dim = len(active_dim)
    factor = jnp.sqrt(input_dim)
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE * factor, "variance": _KERNEL_VAR,},
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
        min_n_target = _DATASET_CONFIGS[name].eval_num_target.lower
        max_n_target = (
            _DATASET_CONFIGS[name].eval_num_target.upper
            + _DATASET_CONFIGS[name].eval_num_context.upper * input_dim
        )  # input_dim * num_context + num_target
        max_n_context = 0
    else:
        max_n_target = _DATASET_CONFIGS[name].eval_num_target.upper
        max_n_context = _DATASET_CONFIGS[name].eval_num_context.upper * input_dim

    key, ckey, tkey, mkey = jax.random.split(key, 4)
    x_context = _TASK_CONFIGS[task].x_context_dist.sample(seed=ckey, sample_shape=(batch_size, max_n_context, input_dim))

    x_target = _TASK_CONFIGS[task].x_target_dist.sample(seed=tkey, sample_shape=(batch_size, max_n_target, input_dim))
    x = jnp.concatenate([x_context, x_target], axis=1)

    if task == "training":
        num_keep_target = jax.random.randint(mkey, (), minval=min_n_target, maxval=max_n_target)
        mask_target = jnp.where(
            jnp.arange(max_n_target)[None, :] < num_keep_target,
            jnp.zeros_like(x_target)[..., 0],  # keep
            jnp.ones_like(x_target)[..., 0]  # ignore
        )
        mask_context = jnp.zeros_like(x_context[..., 0])
    elif task == "interpolation":
        num_keep_context = jax.random.randint(mkey, (), minval=1, maxval=max_n_context)
        mask_context = jnp.where(
            jnp.arange(max_n_context)[None, :] < num_keep_context,
            jnp.zeros_like(x_context)[..., 0],  # keep
            jnp.ones_like(x_context)[..., 0]  # ignore
        )
        mask_target = jnp.zeros_like(x_target[..., 0])

    keys = jax.random.split(key, batch_size)
    active_dims = list(range(input_dim))
    sample_func = _DATASET_FACTORIES[name](active_dims).sample
    y = jax.vmap(sample_func)(keys, x)
    return Batch(
        x_target=x_target,
        y_target=y[:, max_n_context:, :],
        x_context=x_context,
        y_context=y[:, :max_n_context, :],
        mask_target=mask_target,
        mask_context=mask_context,
    )
