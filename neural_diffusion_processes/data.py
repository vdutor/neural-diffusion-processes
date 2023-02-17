from typing import Iterator, Mapping, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxkern
from check_shapes import check_shape, check_shapes
from jaxtyping import Array

from .misc import sample_mvn
from .types import RNGKey


def get_gp_data(
    key: RNGKey,
    kernel: jaxkern.base.AbstractKernel,
    num_samples: int,
    *,
    x_range=(-1.0, 1.0),
    num_points: int = 100,
    input_dim: int = 1,
    output_dim: int = 1,
    params: Optional[Mapping[str, float]] = None
):
    """
    Returns tuple of inputs and outputs. The outputs are drawn from a GP prior with a fixed kernel.
    """
    assert input_dim == 1
    assert output_dim == 1

    if params is None:
        params = {
            "lengthscale": 0.25,
            "variance": 1.0,
        }

    def sample_single(key):
        input_key, output_key = jax.random.split(key, 2)
        x = jax.random.uniform(
            input_key, [num_points, 1], minval=x_range[0], maxval=x_range[1], dtype=jnp.float64
        )
        x = x.sort(axis=0)
        gram = kernel.gram(params, x).to_dense()
        y = sample_mvn(output_key, jnp.zeros_like(x), gram)
        return x, y

    x, y = jax.vmap(sample_single)(jax.random.split(key, num_samples))
    return x, y


class DataBatch(eqx.Module):
    function_inputs: Array
    function_outputs: Array

    def __len__(self) -> int:
        return len(self.function_inputs)

    @check_shapes()
    def __post_init__(self) -> None:
        check_shape(self.function_inputs, "[batch, num_points, input_dim]")
        check_shape(self.function_outputs, "[batch, num_points, output_dim]")


@check_shapes(
    "data[0]: [len_data, num_points, input_dim]",
    "data[1]: [len_data, num_points, output_dim]",
)
def dataloader(data: Tuple[Array, Array], batch_size: int, *, key) -> Iterator[DataBatch]:
    """Yields minibatches of size `batch_size` from the data."""
    x, y = data
    dataset_size = len(x)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield DataBatch(function_inputs=x[batch_perm], function_outputs=y[batch_perm])
            start = end
            end = start + batch_size
