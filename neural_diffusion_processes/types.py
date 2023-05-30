from __future__ import annotations

from typing import Any, Generator, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from simple_pytree import dataclass, Pytree

ndarray = Union[jnp.ndarray, np.ndarray]
Dtype = Any
Rng = jax.random.KeyArray
Params = optax.Params
Config = Any


@dataclass
class Batch(Pytree):
    x_target: ndarray
    y_target: ndarray
    x_context: ndarray | None = None
    y_context: ndarray | None = None
    mask_target: ndarray | None = None
    mask_context: ndarray | None = None


Dataset = Generator[Batch, None, None]