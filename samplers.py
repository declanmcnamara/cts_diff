import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Key


class BetaModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def beta(self, t):
        pass

    @abstractmethod
    def beta_int(self):
        pass


def get_beta_fn(beta_integral_fn: Union[Callable, eqx.Module]) -> Callable:
    """Obtain beta function from a beta beta integral."""

    def _beta_fn(t):
        _, beta = jax.jvp(
            beta_integral_fn, primals=(t,), tangents=(jnp.ones_like(t),), has_aux=False
        )
        return beta

    return _beta_fn


class BetaIntegralDefinedScheduler(eqx.Module, BetaModule):
    beta_int_fn: Callable
    beta_fn: Callable

    def __init__(self, integral_fcn):
        self.beta_int_fn = integral_fcn
        self.beta_fn = get_beta_fn(self.beta_int_fn)

    def beta(self, t):
        return self.beta_fn(t)

    def beta_int(self, t):
        return self.beta_int_fn(t)


class ForwardSDE(eqx.Module):
    beta_int_fcn: Callable
    beta_module: eqx.Module
    dt: float

    def __init__(self, beta_int_fcn: Callable, dt: float = 0.01):
        """
        Construct an SDE.
        """
        super().__init__()
        self.dt = dt
        self.beta_int_fcn = beta_int_fcn
        self.beta_module = BetaIntegralDefinedScheduler(beta_int_fcn)

    def forward_dist(self, t, x0):
        """Return the distribution of x_t conditional on x_0.

        Form of Eq. (29) of SBGM.
        """
        beta_int = self.beta_module.beta_int(t)
        mu = x0 * jnp.exp((-1 / 2) * beta_int)
        sig_sq = 1.0 - jnp.exp((-1 * beta_int))
        scale = jnp.sqrt(sig_sq)
        xt_dist = dist.Normal(mu, scale)
        return xt_dist

    def forward_sample(self, t, x0, key):
        xt_dist = self.forward_dist(t, x0)
        return xt_dist.sample(key)

    def forward_sample_rparam(self, t, x0, key):
        beta_int = self.beta_module.beta_int(t)
        mu = x0 * jnp.exp((-1 / 2) * beta_int)
        sig_sq = 1.0 - jnp.exp((-1 * beta_int))
        scale = jnp.sqrt(sig_sq)
        epsilon = jr.normal(key, x0.shape)
        return mu, scale, epsilon

    def marginal_log_prob(self, xt, t, x0):
        xt_dist = self.forward_dist(t, x0)
        return xt_dist.log_prob(xt).sum()

    def f(self, x, t):
        """Define quantity $f(x, t)$ in Eq. (15) of SBGM."""
        return (-1 / 2) * self.beta_module.beta(t) * x

    def G(self, x, t):
        """Define quantity $G(x,t)$ in Eq. (15) of SBGM.

        NOTE: We do not explicitly return G(x,t) as a matrix.
        This allows us to simply keep images in their original shape for
        ease rather than dealing with reshaping, matrix multiplication, etc.
        """
        value = math.sqrt(self.beta_module.beta(t))
        return value


class ReverseSDE(eqx.Module):
    dt: float
    score_model: eqx.Module
    forward_sde: eqx.Module
    base_dist: numpyro.distributions.Distribution
    shape: tuple
    epsilon: float

    def __init__(self, dt, score_model, forward_sde):
        """
        Construct an SDE.
        """
        super().__init__()
        self.dt = dt
        self.score_model = score_model
        self.forward_sde = forward_sde
        self.shape = (1, 28, 28)
        self.base_dist = dist.Normal(loc=jnp.zeros(self.shape))
        self.epsilon = 1e-5

    def tilde_f(self, x, t):
        """The reverse SDE drift coefficient.

        Defined in terms of x, t as well as
        f, G from the forward SDE, as well as the
        the score function (replaced by a trained model
        of the score function here.)
        """
        fxt = self.forward_sde.f(x, t)
        gxt = self.forward_sde.G(x, t)  # scalar
        score = self.score_model(t, x)
        return fxt - (gxt**2) * score

    def tilde_G(self, x, t):
        """Reverse time scale coefficient; same as forward."""
        return self.forward_sde.G(x, t)

    def sample(self, key):
        key = jr.split(key)[0]
        x1 = self.base_dist.sample(key=key)
        time_grid = jnp.arange(
            start=10.0, stop=self.epsilon - self.dt, step=-1 * self.dt
        )

        curr_x = x1
        for time in time_grid:
            """
            Euler Maryuma iteration:
            dx <- [f(x, t) - g^2(t) * score(x, t, q)] * dt + g(t) * sqrt(dt) * eps_t
            x <- x + dx
            t <- t + dt
            """

            key = jr.split(key)[0]
            time = jnp.array(time)

            eps_t = jr.normal(key, self.shape)
            drift = self.tilde_f(curr_x, time)
            diffusion = self.tilde_G(curr_x, time)
            next_x_mean = curr_x - drift * self.dt  # mu_x = x + drift * -step

            curr_x = next_x_mean + diffusion * jnp.sqrt(self.dt) * eps_t

        return next_x_mean

    def sample_K(self, K, key):
        key = jr.split(key)[0]
        x1s = self.base_dist.sample(key, sample_shape=(K,))
        time_grid = jnp.arange(start=10.0, stop=self.epsilon, step=-1 * self.dt)
        curr_xs = x1s
        for time in time_grid:
            key = jr.split(key)[0]
            time = jnp.array(time)

            drift = partial(self.tilde_f, t=jnp.array(time))
            noise = partial(self.tilde_G, t=jnp.array(time))
            batch_drift = jax.vmap(drift)
            batch_noise = jax.vmap(noise)

            eps_t = jr.normal(key, x1s.shape)
            drifts = batch_drift(curr_xs)
            diffusions = batch_noise(curr_xs)
            next_xs_means = curr_xs - drifts * self.dt  # mu_x = x + drift * -step

            curr_xs = next_xs_means + diffusions[0] * jnp.sqrt(self.dt) * eps_t  # HACK

        return next_xs_means

    def plot_grid(self, nrow, ncol, key):
        K = nrow * ncol
        out = self.sample_K(K=K, key=key)

        fig, ax = plt.subplots(nrow, ncol)
        for i in range(nrow):
            for j in range(ncol):
                entry = ncol * i + j
                this_image = out[entry][0]
                ax[i, j].imshow(this_image)

        for axi in ax.flat:
            axi.set_xticks([])
            axi.set_yticks([])
            axi.set_xticklabels([])
            axi.set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("example_fig.png", bbox_inches="tight")
        plt.clf()
