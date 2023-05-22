from typing import Tuple, Protocol
import jax.numpy as jnp
import jax
from einops import repeat
from check_shapes import check_shapes

from .types import Rng, ndarray


class EpsModel(Protocol):

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "mask: [N,]", "return: [N, y_dim]")
    def __call__(self, t: ndarray, yt: ndarray, x: ndarray, mask: ndarray, *, key: Rng) -> ndarray:
        ...


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def cosine_schedule(beta_start, beta_end, timesteps, s=0.008, **kwargs):
    x = jnp.linspace(0, timesteps, timesteps + 1)
    ft = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0.0001, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class GaussianDiffusion:
    betas: ndarray
    alphas: ndarray
    alpha_bars: ndarray

    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars=jnp.cumprod(1.0 - betas)

    @check_shapes("y0: [N, y_dim]", "t: []", "return[0]: [N, y_dim]", "return[1]: [N, y_dim]")
    def forward(self, key: Rng, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        alpha_bars = expand_to(self.alpha_bars[t], y0)
        noise = jax.random.normal(key, y0.shape)
        yt = jnp.sqrt(alpha_bars) * y0 + jnp.sqrt(1.0 - alpha_bars) * noise
        return yt, noise

    @check_shapes("yt: [N, y_dim]", "t: []", "noise: [N, y_dim]", "return: [N, y_dim]")
    def ddpm_backward_step(
        self,
        key: Rng,
        noise: ndarray,
        yt: ndarray,
        t: ndarray
    ) -> ndarray:
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        # z = jnp.where(
        #     expand_to(t, yt) > 0, jax.random.normal(key, yt.shape), jnp.zeros_like(yt)
        # )
        z = (t > 0) * jax.random.normal(key, shape=yt.shape, dtype=yt.dtype)

        a = 1.0 / jnp.sqrt(alpha_t)
        b = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        yt_minus_one = a * (yt - b * noise) + jnp.sqrt(beta_t) * z
        return yt_minus_one

    @check_shapes("yT: [N, y_dim]", "x: [N, x_dim]", "mask: [N,]", "return: [N, y_dim]")
    def ddpm_backward(self, key, yT, x, mask, *, model_fn: EpsModel):

        if mask is None:
            mask = jnp.zeros_like(x[:, 0])

        yt = yT

        for t in range(self.steps - 1, -1, -1):
            key, ekey, skey = jax.random.split(key, num=3)
            noise_hat = model_fn(t, yt, x, mask, key=ekey)
            yt = self.ddpm_backward_step(skey, noise_hat, yt, t)

        return yt    


    # def sample(self, *, key, model_fn: EpsModel, y, t, x, return_all: bool = False):
    #     keys = jax.random.split(key, len(t))
    #     t = repeat(t, "t -> t b", b=y.shape[0])

    #     def scan_fn(y, inputs):
    #         t, key = inputs
    #         mkey, rkey = jax.random.split(key)
    #         noise_hat = model_fn(t, y, x, key=mkey)
    #         y = self.ddpm_backward_step(key=rkey, noise=noise_hat, yt=y, t=t)
    #         out = y if return_all else None
    #         return y, out

    #     yf, yt = jax.lax.scan(scan_fn, y, (t, keys))
    #     return yt if yt is not None else yf


# GaussianDiffusion.forward = jax.jit(GaussianDiffusion.forward)
# GaussianDiffusion.reverse = jax.jit(GaussianDiffusion.reverse)
# GaussianDiffusion.sample = jax.jit(GaussianDiffusion.sample, static_argnames=("model_fn", "return_all"))


from neural_diffusion_processes.types import Batch



def loss(process: GaussianDiffusion, network: EpsModel, batch: Batch, key: Rng, *, num_timesteps: int, loss_type: str = "l1"):

    if loss_type == "l1":
        loss_metric = lambda a, b: jnp.abs(a - b)
    elif loss_type == "l2":
        loss_metric = lambda a, b: (a - b) ** 2
    else:
        raise ValueError(f"Unknown loss type {loss_type}")

    @check_shapes("t: []", "y: [N, y_dim]", "x: [N, x_dim]", "mask: [N,] if mask", "return: []")
    def loss_fn(key, t, y, x, mask):
        yt, noise = process.forward(key, y, t)
        noise_hat = network(t, yt, x, mask, key=key)
        l = jnp.sum(loss_metric(noise, noise_hat, axis=1))  # [N,]
        l = l * (1. - mask[:, None])
        num_points = len(mask) - jnp.count_nonzero(mask)
        return jnp.sum(l) / num_points

    batch_size = len(batch.x_target)

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=0, maxval=num_timesteps / batch_size)
    t = t + (num_timesteps / batch_size) * jnp.arange(batch_size)

    keys = jax.random.split(key, batch_size)

    if batch.mask_target is None:
        # consider all points
        mask_target = jnp.zeros_like(batch.x_target[..., 0])
    else:
        mask_target = batch.mask_target

    losses = jax.vmap(loss_fn)(keys, t, batch.y_target, batch.x_target, mask_target)
    return jnp.mean(losses)