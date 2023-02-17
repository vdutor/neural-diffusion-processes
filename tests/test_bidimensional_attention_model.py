from __future__ import annotations

from typing import Callable, Iterator, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from check_shapes import check_shapes

from neural_diffusion_processes.bidimensional_attention_model import (
    BiDimensionalAttentionBlock,
    BiDimensionalAttentionScoreModel,
)


class Consts:
    batch_size = 32
    num_points = 101
    hidden_dim = 16
    num_heads = 8
    num_layers = 3


def _get_key_iter(seed: int) -> Iterator[jr.PRNGKey]:
    key = jr.PRNGKey(seed)
    while True:
        key, ykey = jr.split(key)
        yield ykey


_SEED = 0
_KEY_ITER = _get_key_iter(seed=_SEED)


@pytest.fixture(name="inputs", params=[3, 5])
def _inputs_fixuture(request):
    input_dim = request.param
    x = jr.normal(next(_KEY_ITER), (Consts.batch_size, Consts.num_points, input_dim))
    y = jr.normal(next(_KEY_ITER), (Consts.batch_size, Consts.num_points, 1))
    t = jr.uniform(next(_KEY_ITER), (Consts.batch_size,), minval=0, maxval=1.0)
    return x, y, t


@pytest.fixture(name="hidden_inputs", params=[3, 5])
def _hidden_inputs_fixuture(request):
    input_dim = request.param
    shape = (Consts.batch_size, Consts.num_points, input_dim, Consts.hidden_dim)
    x_embedded = jr.normal(next(_KEY_ITER), shape)
    t_embedded = jr.normal(
        next(_KEY_ITER),
        (Consts.batch_size, Consts.hidden_dim),
    )
    return x_embedded, t_embedded


@pytest.fixture(name="bidimensional_attention_block")
def _bidim_attn_block_fixuture():
    return BiDimensionalAttentionBlock(Consts.hidden_dim, Consts.num_heads, key=next(_KEY_ITER))


@pytest.fixture(name="scoremodel")
def _bidim_attn_scoremodel_fixuture():
    return BiDimensionalAttentionScoreModel(
        Consts.num_layers, Consts.hidden_dim, Consts.num_heads, key=next(_KEY_ITER)
    )


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
)
def test_attention_block_equivariance_for_input_dimensionality(
    hidden_inputs, bidimensional_attention_block
):
    x_emb, t_emb = hidden_inputs

    def f(x):
        return bidimensional_attention_block(x, t_emb, key=next(_KEY_ITER))

    _check_equivariance(next(_KEY_ITER), f, 2, 2, x_emb)


@check_shapes(
    "hidden_inputs[0]: [batch_size, seq_len, input_dim, hidden_dim]",
    "hidden_inputs[1]: [batch_size, hidden_dim]",
)
def test_attention_block_equivariance_for_data_sequence(
    hidden_inputs, bidimensional_attention_block
):
    x_emb, t_emb = hidden_inputs

    def f(x_):
        return bidimensional_attention_block(x_, t_emb, key=next(_KEY_ITER))

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x_emb)


@check_shapes(
    "inputs[0]: [batch_size, seq_len, input_dim]",
    "inputs[1]: [batch_size, seq_len, 1]",
    "inputs[2]: [batch_size,]",
)
def test_scoremodel_model_equivariance_for_data_sequence(inputs, scoremodel):
    x, y, t = inputs

    def f(x_, y_):
        return scoremodel(t, y_, x_, key=next(_KEY_ITER))

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x, y)


@check_shapes(
    "inputs[0]: [batch_size, seq_len, input_dim]",
    "inputs[1]: [batch_size, seq_len, 1]",
    "inputs[2]: [batch_size,]",
)
def test_scoremodel_model_invariance_for_input_dimensionality(inputs, scoremodel):
    x, y, t = inputs

    def f(x_):
        return scoremodel(t, y, x_, key=next(_KEY_ITER))

    _check_invariance(next(_KEY_ITER), f, 2, x)


def test_full_scoremodel_model_equivariance_for_data_sequence(inputs, scoremodel):
    x, y, t = inputs

    def f(x_, y_):
        return scoremodel(t, y_, x_, key=next(_KEY_ITER))

    _check_equivariance(next(_KEY_ITER), f, 1, 1, x, y)
