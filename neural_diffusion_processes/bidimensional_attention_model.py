from __future__ import annotations

import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from check_shapes import check_shape as cs
from check_shapes import check_shapes
from einops import reduce

from .types import RNGKey


@check_shapes(
    "t: [batch_size]",
    "return: [batch_size, embedding_dim]",
)
def timestep_embedding(t: jnp.ndarray, embedding_dim: int, max_positions: int = 10_000):
    """Sinusoidal embedding"""
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (t.shape[0], embedding_dim)
    return emb


class BiDimensionalAttentionBlock(eqx.Module):
    """
    Neural network block which:
     - transforms time linearly,
     - applies attention across datapoints,
     - applies attention across input dimensions.
    """

    linear_time: eqx.nn.Linear
    attention_n: eqx.nn.MultiheadAttention
    attention_d: eqx.nn.MultiheadAttention
    hidden_dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()

    def __init__(self, hidden_dim: int, num_heads: int, *, key) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        keys = jax.random.split(key, num=3)
        self.linear_time = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[0])
        self.attention_n = eqx.nn.MultiheadAttention(
            num_heads, query_size=hidden_dim, output_size=2 * hidden_dim, key=keys[1]
        )
        self.attention_d = eqx.nn.MultiheadAttention(
            num_heads, query_size=hidden_dim, output_size=2 * hidden_dim, key=keys[2]
        )

    @check_shapes(
        "s: [batch_size, num_points, input_dim, hidden_dim]",
        "t: [batch_size, hidden_dim]",
        "return[0]: [batch_size, num_points, input_dim, hidden_dim]",
        "return[1]: [batch_size, num_points, input_dim, hidden_dim]",
    )
    def __call__(self, s: jnp.ndarray, t: jnp.ndarray, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bi-dimensional attention block. Main computation block in the NDP noise model.
        """
        t = cs(jax.vmap(self.linear_time)(t)[:, None, None, :], "[batch_size, 1, 1, hidden_dim]")
        y = cs(s + t, "[batch_size, num_points, input_dim, hidden_dim]")

        y_att_d = jax.vmap(jax.vmap(self.attention_d))(y, y, y)
        y_att_d = cs(y_att_d, "[batch_size, num_points, input_dim, hidden_dim_x2]")

        y_r = cs(jnp.swapaxes(y, 1, 2), "[batch_size, input_dim, num_points, hidden_dim]")
        y_att_n = jax.vmap(jax.vmap(self.attention_n))(y_r, y_r, y_r)
        y_att_n = cs(y_att_n, "[batch_size, input_dim, num_points, hidden_dim_x2]")
        y_att_n = cs(
            jnp.swapaxes(y_att_n, 1, 2), "[batch_size, num_points, input_dim, hidden_dim_x2]"
        )

        y = y_att_n + y_att_d

        residual, skip = jnp.split(y, 2, axis=-1)
        residual = jax.nn.gelu(residual)
        skip = jax.nn.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


class BiDimensionalAttentionScoreModel(eqx.Module):
    linear_embedding: eqx.nn.Linear
    linear_hidden: eqx.nn.Linear
    linear_output: eqx.nn.Linear
    bidim_attention_blocks: list  # List[BiDimensionalAttentionBlock]

    num_bidim_attention_blocks: int = eqx.static_field()
    hidden_dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()
    output_dim: int = eqx.static_field()

    def __init__(
        self,
        num_bidim_attention_blocks: int,
        hidden_dim: int,
        num_heads: int,
        output_dim: int = 1,
        *,
        key: RNGKey,
    ) -> None:
        super().__init__()
        self.num_bidim_attention_blocks = num_bidim_attention_blocks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        key, ekey = jax.random.split(key, 2)
        self.linear_embedding = eqx.nn.Linear(in_features=2, out_features=self.hidden_dim, key=ekey)

        self.bidim_attention_blocks = []
        for _ in range(self.num_bidim_attention_blocks):
            key, subkey = jax.random.split(key)
            self.bidim_attention_blocks.append(
                BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads, key=subkey)
            )

        key, o1key, o2key = jax.random.split(key, 3)
        self.linear_hidden = eqx.nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim, key=o1key
        )
        self.linear_output = eqx.nn.Linear(
            in_features=self.hidden_dim, out_features=self.output_dim, use_bias=False, key=o2key
        )
        # Init weights to zero
        # self.linear_output = eqx.tree_at(
        #     lambda layer: layer.weight,
        #     linear_output,
        #     replace_fn=lambda w: jnp.zeros_like(w)
        # )

    @check_shapes(
        "x: [batch_size, num_points, input_dim]",
        "y: [batch_size, num_points, 1]",
        "return: [batch_size, num_points, input_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x dimensions for dimension-agnostic processing.
        """
        num_x_dims = jnp.shape(x)[-1]
        x = jnp.expand_dims(x, axis=-1)
        y = jnp.repeat(jnp.expand_dims(y, axis=-1), num_x_dims, axis=2)
        return jnp.concatenate([x, y], axis=-1)

    @check_shapes(
        "t: [batch_size]",
        "yt: [batch_size, num_points, output_dim]",
        "x: [batch_size, num_points, input_dim]",
        "return: [batch_size, num_points, output_dim]",
    )
    def __call__(
        self,
        t,
        yt: jnp.ndarray,
        x: jnp.ndarray,
        *,
        key: RNGKey,
    ) -> jnp.ndarray:
        """Network to estimate score."""
        x = cs(self.process_inputs(x, yt), "[batch_size, num_points, input_dim, 2]")

        x = cs(
            jax.vmap(jax.vmap(jax.vmap(self.linear_embedding)))(x),
            "[batch_size, num_points, input_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for layer in self.bidim_attention_blocks:
            key, subkey = jax.random.split(key)
            x, skip_connection = layer(x, t_embedding, key=subkey)
            skip = skip_connection if skip is None else skip_connection + skip

        x = cs(x, "[batch_size, num_points, input_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, input_dim, hidden_dim]")

        skip = cs(reduce(skip, "b n d h -> b n h", "mean"), "[batch, num_points, hidden_dim]")

        eps = skip / math.sqrt(self.num_bidim_attention_blocks * 1.0)
        eps = jax.nn.gelu(jax.vmap(jax.vmap(self.linear_hidden))(eps))
        eps = jax.vmap(jax.vmap(self.linear_output))(eps)
        return eps
