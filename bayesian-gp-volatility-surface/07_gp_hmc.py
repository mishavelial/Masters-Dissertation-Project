"""
In this module we implement the Bayesian inference via NUTS. First, we obtain the residuals w(k,T) = w_observed(k,T) -
w_prior(k,T). Then we standardise the inputs and residuals. Later compute the anisotropic Matérn 5/2 covariance matrix/kernel.
"""

import importlib
import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

import jax
# Enforce double-precision float64 arithmetic. Gaussian process covariance matrices are notoriously
# ill-conditioned (κ(K) >> 10^12); float32 causes catastrophic precision loss during Cholesky decomposition in JAX autodiff
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

#parallelise chains across available CPU cores for efficient MCMC sampling
numpyro.set_host_device_count(2)

#import parametric prior model (e.g., SVI or deterministic variance model)
TotalVariancePrior = importlib.import_module("03_prior_model").TotalVariancePrior


def matern52_kernel(X, Z, lengthscales, variance):
    """
    Computes the anisotropic Matérn 5/2 covariance matrix.
    """
    X_scaled = X / lengthscales
    Z_scaled = Z / lengthscales

    #gram matrix distance computation via vectorization
    dist_sq = jnp.sum(X_scaled ** 2, axis=-1)[:, None] + jnp.sum(Z_scaled ** 2, axis=-1) - 2 * jnp.dot(X_scaled,
                                                                                                       Z_scaled.T)

    # clip negative values arising from floating-point roundoff in inner product
    dist_sq = jnp.maximum(dist_sq, 0.0)

    # 1e-12 jitter under the square root prevents undefined gradients (d/dx sqrt(x) -> inf at x=0) which would otherwise
    # crash NUTS reverse-mode automatic differentiation leapfrog steps.
    d = jnp.sqrt(dist_sq + 1e-12)

    return variance * (1.0 + jnp.sqrt(5.0) * d + (5.0 / 3.0) * dist_sq) * jnp.exp(-jnp.sqrt(5.0) * d)


def exact_gp_model(X, y=None, jitter=1e-6):
    """
    Defines the GP log-marginal likelihood for Bayesian inference via NUTS.
    """
    N = X.shape[0]

    #we set hard-bounded Uniform Priors so we have a structural constraint ("electric fence") on the hyperparameter space.
    # Restricting ell_k > 1.0 explicitly prevents the optimizer from descending into the overfitted short-lengthscale regime
    # where high local curvature violates static no-arbitrage bounds and ell_t < 3.0 to not allow the optimiser overgeneralise the surface.
    # This is a conscious choice made to avoid other suitable distributions (i. e., Gamma or Inverse Gamma which
    # were tested but did not prove themselves to have better fitting.
    ell_k = numpyro.sample("ell_k", dist.Uniform(low=1.0, high=3.0))
    ell_t = numpyro.sample("ell_t", dist.Uniform(low=0.2, high=1.5))

    #signal and noise variance priors in log-space where the Prior mean of log_sigma_n = -5.0 corresponds to an expected
    # observation noise variance sigma_n^2 ~ 10^-4. This prevents the GP from over-smoothing genuine market structure while
    # absorbing residual quote noise (e.g., bid-ask micro-structure gaps).
    log_sigma_f = numpyro.sample("log_sigma_f", dist.Normal(0.0, 1.0))
    log_sigma_n = numpyro.sample("log_sigma_n", dist.Normal(-5.0, 0.5))

    ell = jnp.array([ell_k, ell_t])
    variance_f = jnp.exp(log_sigma_f) ** 2
    variance_n = jnp.exp(log_sigma_n) ** 2

    K = matern52_kernel(X, X, ell, variance_f)

    #tikhonov regularization (jitter * I) guarantees positive-definiteness for Cholesky factorization of the covariance matrix Ky.
    K_y = K + (variance_n + jitter) * jnp.eye(N)

    #marginal likelihood condition y ~ N(0, K_y)
    numpyro.sample("y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=K_y), obs=y)


def run_hmc(date_str, num_warmup=1000, num_samples=1000, proc_dir="data/processed", prior_dir="results/surfaces",
            out_dir="results/diagnostics"):
    """
    Orchestrates the HMC/NUTS sampling over total variance residuals.
    """
    print(f"Running HMC Inference for {date_str}...")

    data = np.load(os.path.join(proc_dir, f"design_matrix_{date_str}.npz"))
    X_train, w_train = data['X_raw'][data['train_idx']], data['w_raw'][data['train_idx']]
    prior_data = np.load(os.path.join(prior_dir, f"prior_fit_{date_str}.npz"))
    prior = TotalVariancePrior()
    prior.params = prior_data['params']

    #residuals
    res_train = w_train - prior.predict(X_train)

    #isotropic standardization: Transforms input coordinates and residuals to zero mean and unit variance, allowing uniform
    # isotropic priors on lengthscales to map evenly across both log-moneyness and maturity dimensions.
    scaler_X, scaler_y = StandardScaler().fit(X_train), StandardScaler().fit(res_train.reshape(-1, 1))
    X_train_s = jnp.array(scaler_X.transform(X_train))
    y_train_s = jnp.array(scaler_y.transform(res_train.reshape(-1, 1)).ravel())

    rng_key = jax.random.PRNGKey(42)

    # NUTS Configuration:
    # 1. dense_mass=True: a dense mass matrix estimates off-diagonal momentum correlations between lengthscales and signal variance.
    # 2. target_accept_prob=0.95: force smaller leapfrog integrator steps to safely navigate high-curvature boundaries and prevent divergence
    nuts_kernel = NUTS(exact_gp_model, target_accept_prob=0.95, dense_mass=True)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=2, progress_bar=True)

    mcmc.run(rng_key, X=X_train_s, y=y_train_s)
    samples = mcmc.get_samples()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hmc_samples_{date_str}.npz")
    np.savez(out_path, **{k: np.array(v) for k, v in samples.items()})
    print(f"Saved {num_samples} posterior samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    run_hmc(args.date)
