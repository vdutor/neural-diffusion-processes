from typing import Optional

import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from jaxtyping import Array

JITTER = 1e-12


@check_shapes(
    "mean: [num_points, 1]",
    "cov: [num_points, num_points]",
    "return: [num_samples, num_points, 1] if num_samples",
    "return: [num_points, 1] if not num_samples",
)
def sample_mvn(key, mean: Array, cov: Array, num_samples: Optional[int] = None):
    """Returns samples from a GP(mean, kernel) at x."""
    num_samples = num_samples or 1
    L = jnp.linalg.cholesky(cov + JITTER * jnp.eye(len(mean)))
    eps = jax.random.normal(key, (len(mean), num_samples), dtype=mean.dtype)
    s = mean + L @ eps
    s = jnp.transpose(s)[..., None]
    if num_samples == 1:
        return s[0]
    else:
        return s
