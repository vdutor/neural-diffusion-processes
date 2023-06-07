# Based on GPJAX but with some modifications
from typing import Callable, Dict, Optional, Tuple, Union
from gpjax.gaussian_distribution import GaussianDistribution
from jaxutils import config
import jax.numpy as jnp
from jaxlinop import identity, DiagonalLinearOperator

from gpjax.gps import Prior


from .types import ndarray


def predict(
    prior: Prior,
    params: Dict,
    train_data: Tuple[ndarray, ndarray],
    diag: bool = False,
) -> Callable[[ndarray], GaussianDistribution]:
    """Conditional on a training data set, compute the GP's posterior
    predictive distribution for a given set of parameters. The returned
    function can be evaluated at a set of test inputs to compute the
    corresponding predictive density.

    The predictive distribution of a conjugate GP is given by
    .. math::

        p(\\mathbf{f}^{\\star}\mid \mathbf{y}) & = \\int p(\\mathbf{f}^{\\star} \\mathbf{f} \\mid \\mathbf{y})\\\\
        & =\\mathcal{N}(\\mathbf{f}^{\\star} \\boldsymbol{\mu}_{\mid \mathbf{y}}, \\boldsymbol{\Sigma}_{\mid \mathbf{y}}
    where

    .. math::

        \\boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left(k(\\mathbf{x}, \\mathbf{x}')+\\sigma^2\\mathbf{I}_n\\right)^{-1}\\mathbf{y}  \\\\
        \\boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\\mathbf{x}^{\\star}, \\mathbf{x}^{\\star\\prime}) -k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left( k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_n \\right)^{-1}k(\\mathbf{x}, \\mathbf{x}^{\\star}).

    The conditioning set is a GPJax ``Dataset`` object, whilst predictions
    are made on a regular Jax array.

    £xample:
        For a ``posterior`` distribution, the following code snippet will
        evaluate the predictive distribution.

        >>> import gpjax as gpx
        >>>
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
        >>>
        >>> params = gpx.initialise(posterior)
        >>> predictive_dist = posterior.predict(params, gpx.Dataset(X=xtrain, y=ytrain))
        >>> predictive_dist(xtest)

    Args:
        params (Dict): A dictionary of parameters that should be used to
            compute the posterior.
        train_data (Dataset): A `gpx.Dataset` object that contains the
            input and output data used for training dataset.

    Returns:
        Callable[[Float[Array, "N D"]], GaussianDistribution]: A
            function that accepts an input array and returns the predictive
            distribution as a ``GaussianDistribution``.
    """
    jitter = config.get_global_config()["jitter"]

    # Unpack training data
    x, y = train_data
    n = x.shape[0]

    # Unpack mean function and kernel
    mean_function = prior.mean_function
    kernel = prior.kernel

    # Observation noise σ²
    obs_noise = params["noise_variance"]
    μx = mean_function(params["mean_function"], x)

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = kernel.gram(params["kernel"], x)
    Kxx += identity(n) * jitter

    # Σ = Kxx + Iσ²
    Sigma = Kxx + identity(n) * obs_noise

    def predict(test_inputs: ndarray) -> GaussianDistribution:
        """Compute the predictive distribution at a set of test inputs.

        Args:
            test_inputs (Float[Array, "N D"]): A Jax array of test inputs.

        Returns:
            GaussianDistribution: A ``GaussianDistribution``
            object that represents the predictive distribution.
        """

        # Unpack test inputs
        t = test_inputs
        n_test = test_inputs.shape[0]

        μt = mean_function(params["mean_function"], t)
        Ktt = kernel.gram(params["kernel"], t)
        Kxt = kernel.cross_covariance(params["kernel"], x, t)

        # Σ⁻¹ Kxt
        Sigma_inv_Kxt = Sigma.solve(Kxt)

        # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
        mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - μx)

        # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += identity(n_test) * (jitter + obs_noise)

        if diag:
            # Obviously not the most efficient way to do this
            covariance = DiagonalLinearOperator(covariance.diagonal())

        return GaussianDistribution(
            jnp.atleast_1d(mean.squeeze()), covariance
        )

    return predict