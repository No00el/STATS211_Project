# Methodology: Student-t via Metropolis-within-Gibbs

This document explains the hierarchical model, full conditionals, the MH-within-Gibbs step for the degrees-of-freedom parameter `nu`, and practical tuning guidance used in `main.py`.

## Model

We model log-returns with a Student-t distribution using a normal–gamma mixture representation.

- Observation layer:  
  $y_i \mid \mu, \sigma^2, \lambda_i \sim \mathcal{N}(\mu, \sigma^2/\lambda_i)$
- Mixing weights:  
  $\lambda_i \mid \nu \sim \text{Gamma}(\tfrac{\nu}{2},\, \text{rate} = \tfrac{\nu}{2})$
- Priors:  
  $\mu\mid\sigma^2 \sim \mathcal{N}(m_0, \sigma^2/\kappa_0)$,  
  $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0,\,\beta_0)$,  
  $\nu \sim \text{Gamma}(a_0,\,\text{rate}=b_0)$.

Marginally, $y_i \sim t_\nu(\mu, \sigma^2)$ with heavy tails controlled by `nu`.

## Full Conditionals (Gibbs parts)

1) $\lambda_i \mid \mu,\sigma^2,\nu, y_i \sim \text{Gamma}\Big(\tfrac{\nu+1}{2},\, \text{rate}=\tfrac{\nu + (y_i-\mu)^2/\sigma^2}{2}\Big)$

2) $\mu \mid \sigma^2, \{\lambda_i\}, \{y_i\} \sim \mathcal{N}(m_n, \sigma^2/\kappa_n)$  
$\kappa_n = \kappa_0 + \sum_i \lambda_i$,  
$m_n = (\kappa_0 m_0 + \sum_i \lambda_i y_i)/\kappa_n$.

3) $\sigma^2 \mid \mu, \{\lambda_i\}, \{y_i\} \sim \text{Inv-Gamma}\Big(\alpha_0 + \tfrac{n}{2},\; \beta_0 + \tfrac{1}{2}\sum_i \lambda_i(y_i-\mu)^2\Big)$

These are conjugate and sampled via Gibbs.

## MH for `nu` (Metropolis-within-Gibbs)

The complete conditional for `nu` is not of a standard form due to the $\log\Gamma(\nu/2)$ term. We therefore use a random-walk Metropolis step inside the Gibbs sampler.

Up to an additive constant, the log-posterior is:

$$
\log p(\nu\mid\lambda) \propto (a_0-1)\log\nu - b_0\nu
+ n\Big(\tfrac{\nu}{2}\log\tfrac{\nu}{2} - \log\Gamma(\tfrac{\nu}{2})\Big)
+ (\tfrac{\nu}{2}-1)\sum_i\log\lambda_i - \tfrac{\nu}{2}\sum_i\lambda_i.
$$

### Transformations and Jacobian

We do random-walk proposals in a transformed space to enforce positivity and optionally finite variance:

- Default: $\eta = \log \nu$ (ensures $\nu>0$). Proposal: $\eta' = \eta + \mathcal{N}(0, s^2)$, with $\nu' = e^{\eta'}$.  
  Acceptance compares $\log p(\nu'\mid\lambda) + \log\nu'$ to $\log p(\nu\mid\lambda) + \log\nu$ (Jacobian term $\log|d\nu/d\eta|=\log\nu$).

- Enforce finite variance: $\xi = \log(\nu - 2)$ (ensures $\nu>2$). Proposal: $\xi' = \xi + \mathcal{N}(0, s^2)$, with $\nu' = 2 + e^{\xi'}$.  
  Acceptance compares $\log p(\nu'\mid\lambda) + \log(\nu' - 2)$ to $\log p(\nu\mid\lambda) + \log(\nu - 2)$ (Jacobian $\log(\nu-2)$).

### Adaptation (Burn-in only)

We (optionally) adapt the proposal std `s` during burn-in to target a given acceptance rate (default 0.30). A small Robbins–Monro style update is applied to $\log s$ every `adapt_interval` iterations and then frozen after burn-in to preserve correct stationary behavior.

## Practical Tuning

- Target MH acceptance: 20%–50% is reasonable; 30% is a common target.
- If acceptance is too low: reduce proposal std (`--prop-sd`), or increase `--target-accept` slightly during adaptation.
- If mixing is slow (high ACF): increase total iterations, adjust proposal std, or enforce `nu>2` to stabilize variance.
- Priors: Smaller `a_0` or larger `b_0` favors heavier tails (smaller `nu`). Use weakly-informative priors unless data are scarce.

## Outputs & Diagnostics

- Posterior means for `mu`, `sigma^2`, `nu`.
- `accept_rate` and `final_prop_sd` for the `nu` MH step.
- Trace and ACF plots to assess mixing and autocorrelation (burn-in removed).
- Optional CSV of posterior samples, and JSON summary of run configuration.

## References
- Geweke, J. (1993). Bayesian treatment of the independent Student-t linear model.
- Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC.
