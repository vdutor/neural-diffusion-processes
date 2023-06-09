from __future__ import annotations

from typing import Callable, Iterator, Tuple

import haiku as hk
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from check_shapes import check_shapes

from neural_diffusion_processes.model import (
    BiDimensionalAttentionBlock,
    BiDimensionalAttentionModel,
)


class Consts:
    batch_size = 7
    num_points = 31
    hidden_dim = 8
    num_heads = 4
    num_layers = 2


def _get_key_iter(seed: int) -> Iterator[jr.PRNGKey]:
    key = jr.PRNGKey(seed)
    while True:
        key, ykey = jr.split(key)
        yield ykey


_KEY_ITER = _get_key_iter(seed=0)


@pytest.fixture(name="inputs", params=[3, 5])
def _inputs_fixuture(request):
    input_dim = request.param
    x = jr.normal(next(_KEY_ITER), (Consts.batch_size, Consts.num_points, input_dim))
    y = jr.normal(next(_KEY_ITER), (Consts.batch_size, Consts.num_points, 1))
    t = jr.uniform(next(_KEY_ITER), (Consts.batch_size,), minval=0, maxval=1.0)
    mask = jnp.zeros((Consts.batch_size, Consts.num_points))
    return x, y, t, mask


@pytest.fixture(name="hidden_inputs", params=[3, 5])
def _hidden_inputs_fixuture(request):
    input_dim = request.param
    shape = (Consts.batch_size, Consts.num_points, input_dim, Consts.hidden_dim)
    x_embedded = jr.normal(next(_KEY_ITER), shape)
    t_embedded = jr.normal(
        next(_KEY_ITER),
        (Consts.batch_size, Consts.hidden_dim),
    )
    mask = jnp.zeros((Consts.batch_size, Consts.num_points))
    return x_embedded, t_embedded, mask


@pytest.fixture(name="bidimensional_attention_model")
def _bidim_attn_model_fixuture():
    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = BiDimensionalAttentionModel(
            n_layers=Consts.num_layers,
            hidden_dim=Consts.hidden_dim,
            num_heads=Consts.num_heads,
            init_zero=False,
        )
        return model(x, y, t, mask)

    t = jnp.zeros((Consts.batch_size,))
    x = jnp.zeros((Consts.batch_size, Consts.num_points, 1))
    y = jnp.zeros((Consts.batch_size, Consts.num_points, 1))
    mask = jnp.ones((Consts.batch_size, Consts.num_points))

    params = network.init(next(_KEY_ITER), t=t, y=y, x=x, mask=mask)
    return lambda *args: network.apply(params, *args)


@pytest.fixture(name="bidimensional_attention_block")
def _bidim_attn_block_fixuture():
    @hk.without_apply_rng
    @hk.transform
    def network(t, s, mask):
        model = BiDimensionalAttentionBlock(
            hidden_dim=Consts.hidden_dim, num_heads=Consts.num_heads
        )
        return model(s, t, mask)

    t_embedded = jnp.zeros((Consts.batch_size, Consts.hidden_dim))
    s_embedded = jnp.zeros((Consts.batch_size, Consts.num_points, 1, Consts.hidden_dim))
    mask = jnp.ones((Consts.batch_size, Consts.num_points))
    params = network.init(next(_KEY_ITER), t=t_embedded, s=s_embedded, mask=mask)
    return lambda *args: network.apply(params, *args)


def permute(
    key: jr.PRNGKey | None, a: jnp.ndarray, axis: int, shuffled_inds: jnp.ndarray | None = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Permutes a tensor `a` along dimension `axis`. Returns the permuted tensor as well
    as the indices. When `shuffled_inds` is None, generates a new permutation,
    otherwise reuses the indices.
    """
    if shuffled_inds is None:
        len_axis = jnp.shape(a)[axis]
        inds = jnp.arange(len_axis, dtype=int)
        shuffled_inds = jr.shuffle(key, inds)  # Random shuffle

    return jnp.take(a, indices=shuffled_inds, axis=axis), shuffled_inds


def _check_invariance(key: jr.PRNGKey, f: Callable, axis: int, *args: jnp.ndarray):
    """
    Tests that shuffling the input `args` along `axis` does not change the output of `f`.
    """
    permuted_args = []
    indices = None
    for arg in args:
        arg_p, indices = permute(key, arg, axis=axis, shuffled_inds=indices)
        permuted_args.append(arg_p)

    outputs_original = f(*args)
    outputs_permuted_inputs = f(*permuted_args)

    if not isinstance(outputs_original, tuple):
        outputs_original = (outputs_original,)
        outputs_permuted_inputs = (outputs_permuted_inputs,)

    for output_orginal, output_permuted_inputs in zip(outputs_original, outputs_permuted_inputs):
        # we assume invariance: the output should not be affected by shuffled inputs
        np.testing.assert_array_almost_equal(output_orginal, output_permuted_inputs)


def _check_equivariance(
    key: jr.PRNGKey, f: Callable, axis: int, output_axis: int, *args: jnp.ndarray
):
    """
    Tests that shuffling the input `args` along `axis` changes the output of `f`. The output
    of `f` will be shuffled along `output_axis` in the same way as the shuffled inputs.
    """

    key1, key2 = jr.split(key)
    permuted_args = []
    indices = None
    for arg in args:
        arg_p, indices = permute(key1, arg, axis=axis, shuffled_inds=indices)
        permuted_args.append(arg_p)

    outputs_original = f(*args)
    outputs_permuted_inputs = f(*permuted_args)

    if not isinstance(outputs_original, tuple):
        outputs_original = (outputs_original,)
        outputs_permuted_inputs = (outputs_permuted_inputs,)

    for output_orginal, output_permuted_inputs in zip(outputs_original, outputs_permuted_inputs):
        output_orginal_permuted, _ = permute(
            key2, output_orginal, axis=output_axis, shuffled_inds=indices
        )
        # we assume equivariance: the output should affected by shuffled inputs in the same
        # way as shuffling the output itself.
        np.testing.assert_allclose(
            actual=output_permuted_inputs,
            desired=output_orginal_permuted,
            atol=3e-5,
            rtol=1e-2,
        )


@check_shapes(
    "hidden_inputs[0]: [batch_size, seq_len, input_dim, hidden_dim]",
    "hidden_inputs[1]: [batch_size, hidden_dim]",
    "hidden_inputs[2]: [batch_size, seq_len]",
)
def test_attention_block_equivariance_for_input_dimensionality(
    hidden_inputs, bidimensional_attention_block
):
    x_emb, t_emb, mask = hidden_inputs

    def f(x):
        return bidimensional_attention_block(t_emb, x, mask)

    _check_equivariance(next(_KEY_ITER), f, 2, 2, x_emb)


@check_shapes(
    "hidden_inputs[0]: [batch_size, seq_len, input_dim, hidden_dim]",
    "hidden_inputs[1]: [batch_size, hidden_dim]",
    "hidden_inputs[2]: [batch_size, seq_len]",
)
def test_attention_block_equivariance_for_data_sequence(
    hidden_inputs, bidimensional_attention_block
):
    x_emb, t_emb, mask = hidden_inputs

    def f(x_):
        return bidimensional_attention_block(t_emb, x_, mask)

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x_emb)


@check_shapes(
    "inputs[0]: [batch_size, seq_len, input_dim]",
    "inputs[1]: [batch_size, seq_len, 1]",
    "inputs[2]: [batch_size,]",
    "inputs[3]: [batch_size, seq_len]",
)
def test_scoremodel_model_equivariance_for_data_sequence(inputs, bidimensional_attention_model):
    x, y, t, mask = inputs

    def f(x_, y_):
        return bidimensional_attention_model(t, y_, x_, mask)

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x, y)


@check_shapes(
    "inputs[0]: [batch_size, seq_len, input_dim]",
    "inputs[1]: [batch_size, seq_len, 1]",
    "inputs[2]: [batch_size,]",
    "inputs[3]: [batch_size, seq_len]",
)
def test_scoremodel_model_invariance_for_input_dimensionality(
    inputs, bidimensional_attention_model
):
    x, y, t, mask = inputs

    def f(x_):
        return bidimensional_attention_model(t, y, x_, mask)

    _check_invariance(next(_KEY_ITER), f, 2, x)


def test_full_scoremodel_model_equivariance_for_data_sequence(
    inputs, bidimensional_attention_model
):
    x, y, t, mask = inputs

    def f(x_, y_):
        return bidimensional_attention_model(t, y_, x_, mask)

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x, y)
