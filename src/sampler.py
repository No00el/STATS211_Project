"""Student-t Metropolis-within-Gibbs sampler implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln


@dataclass
class PriorConfig:
    m0: float = 0.0
    kappa0: float = 0.01
    alpha0: float = 2.0
    beta0: float = 1.0
    a0: float = 2.0
    b0: float = 0.1


@dataclass
class SamplerConfig:
    n_iter: int = 200_000
    burn_in: int = 5_000
    seed: int = 123
    prop_sd: float = 0.3
    target_accept: float = 0.30
    adapt: bool = True
    adapt_start: int = 200
    adapt_end: Optional[int] = None
    adapt_interval: int = 50
    enforce_nu_gt2: bool = False
    priors: PriorConfig = field(default_factory=PriorConfig)

    def effective_adapt_end(self) -> int:
        return self.adapt_end if self.adapt_end is not None else self.burn_in


@dataclass
class SamplerResult:
    mu: np.ndarray
    sigma2: np.ndarray
    nu: np.ndarray
    accept_rate: float
    final_prop_sd: float
    enforce_nu_gt2: bool

    def burn_samples(self, burn: int) -> "SamplerResult":
        return SamplerResult(
            mu=self.mu[burn:],
            sigma2=self.sigma2[burn:],
            nu=self.nu[burn:],
            accept_rate=self.accept_rate,
            final_prop_sd=self.final_prop_sd,
            enforce_nu_gt2=self.enforce_nu_gt2,
        )


# ================================
#   Log posterior for nu
# ================================
def log_p_nu(nu: float, lambdas: np.ndarray, priors: PriorConfig) -> float:
    if nu <= 0:
        return -np.inf
    n = lambdas.size
    sum_log_lambda = np.sum(np.log(lambdas))
    sum_lambda = np.sum(lambdas)
    log_prior = (priors.a0 - 1.0) * np.log(nu) - priors.b0 * nu
    half_nu = nu / 2.0
    log_lik = (
        n * (half_nu * np.log(half_nu) - gammaln(half_nu))
        + (half_nu - 1.0) * sum_log_lambda
        - half_nu * sum_lambda
    )
    return log_prior + log_lik


# ================================
#   Sampler driver
# ================================
def run_student_t_sampler(y: ArrayLike, config: SamplerConfig) -> SamplerResult:
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        raise ValueError("Need at least two log-return observations for sampling.")

    priors = config.priors
    n = y.size
    n_iter = config.n_iter
    rng = np.random.default_rng(config.seed)

    mu_chain = np.zeros(n_iter)
    sigma2_chain = np.zeros(n_iter)
    nu_chain = np.zeros(n_iter)
    mu = float(np.mean(y))
    sigma2 = float(np.var(y))
    nu = 5.0
    lambdas = np.ones(n)

    accept_count = 0
    curr_prop_sd = float(config.prop_sd)
    log_prop_sd = np.log(curr_prop_sd)
    adapt_end = config.effective_adapt_end()

    for it in range(n_iter):
        shape_lam = (nu + 1.0) / 2.0
        rate_lam = (nu + (y - mu) ** 2 / sigma2) / 2.0
        lambdas = rng.gamma(shape_lam, 1.0 / rate_lam)

        W = np.sum(lambdas)
        sum_wy = np.sum(lambdas * y)
        kappa_n = priors.kappa0 + W
        m_n = (priors.kappa0 * priors.m0 + sum_wy) / kappa_n
        mu_var = sigma2 / kappa_n
        mu = rng.normal(m_n, np.sqrt(mu_var))

        alpha_n = priors.alpha0 + n / 2.0
        beta_n = priors.beta0 + 0.5 * np.sum(lambdas * (y - mu) ** 2)
        gamma_sample = rng.gamma(alpha_n, 1.0 / beta_n)
        sigma2 = 1.0 / gamma_sample

        if config.enforce_nu_gt2:
            xi_curr = np.log(max(nu - 2.0, 1e-12))
            xi_prop = xi_curr + rng.normal(0.0, curr_prop_sd)
            nu_prop = 2.0 + np.exp(xi_prop)
            logpost_curr = log_p_nu(nu, lambdas, priors) + np.log(max(nu - 2.0, 1e-12))
            logpost_prop = log_p_nu(nu_prop, lambdas, priors) + np.log(nu_prop - 2.0)
        else:
            eta_curr = np.log(nu)
            eta_prop = eta_curr + rng.normal(0.0, curr_prop_sd)
            nu_prop = np.exp(eta_prop)
            logpost_curr = log_p_nu(nu, lambdas, priors) + np.log(nu)
            logpost_prop = log_p_nu(nu_prop, lambdas, priors) + np.log(nu_prop)

        log_acc_ratio = logpost_prop - logpost_curr
        if np.log(rng.uniform()) < log_acc_ratio:
            nu = nu_prop
            accept_count += 1

        if config.adapt and (config.adapt_start <= it + 1 <= adapt_end) and ((it + 1) % config.adapt_interval == 0):
            acc_rate = accept_count / float(it + 1)
            log_prop_sd += 0.05 * (acc_rate - config.target_accept)
            log_prop_sd = np.clip(log_prop_sd, np.log(1e-3), np.log(5.0))
            curr_prop_sd = float(np.exp(log_prop_sd))

        mu_chain[it] = mu
        sigma2_chain[it] = sigma2
        nu_chain[it] = nu

    accept_rate = accept_count / float(n_iter)
    return SamplerResult(
        mu=mu_chain,
        sigma2=sigma2_chain,
        nu=nu_chain,
        accept_rate=accept_rate,
        final_prop_sd=curr_prop_sd,
        enforce_nu_gt2=config.enforce_nu_gt2,
    )
