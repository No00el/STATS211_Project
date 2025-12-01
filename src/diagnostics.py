"""Plotting helpers for trace and ACF diagnostics."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .sampler import SamplerResult


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.array([1.0])
    var = np.dot(x, x) / n
    if var == 0:
        return np.ones(min(max_lag, max(n - 1, 1)) + 1)
    K = min(max_lag, n - 1)
    acf = np.empty(K + 1)
    acf[0] = 1.0
    for k in range(1, K + 1):
        acf[k] = np.dot(x[: n - k], x[k:]) / (n * var)
    return acf


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_traces(result: SamplerResult, burn: int, out_path: Path) -> None:
    ensure_parent(out_path)
    mu = result.mu[burn:]
    sigma = np.sqrt(result.sigma2[burn:])
    nu = result.nu[burn:]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(mu, lw=0.8)
    axes[0].set_title("Trace: mu")
    axes[1].plot(sigma, lw=0.8)
    axes[1].set_title("Trace: sigma")
    axes[2].plot(nu, lw=0.8)
    axes[2].set_title("Trace: nu")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("MCMC Trace Plots (burn-in removed)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acf(
    result: SamplerResult,
    burn: int,
    max_lag: int,
    out_path: Path,
    thin: int = 50,
    only_nu: bool = False,
) -> None:
    ensure_parent(out_path)
    mu = result.mu[burn:]
    sigma = np.sqrt(result.sigma2[burn:])
    nu = result.nu[burn:]

    thin = max(1, int(thin))
    mu_t = mu[::thin]
    sigma_t = sigma[::thin]
    nu_t = nu[::thin]

    if only_nu:
        acf_nu = _autocorr(nu_t, max_lag=max_lag)
        lags = np.arange(acf_nu.size)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
        ax.stem(lags, acf_nu)
        ax.set_xlim(0, lags.max() if lags.size else 0)
        ax.set_title(f"ACF: nu (thinning={thin})")
        ax.grid(True, alpha=0.3)
        fig.suptitle("Sample Autocorrelation (burn-in removed)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    acf_mu = _autocorr(mu_t, max_lag=max_lag)
    acf_sigma = _autocorr(sigma_t, max_lag=max_lag)
    acf_nu = _autocorr(nu_t, max_lag=max_lag)
    lags_mu = np.arange(acf_mu.size)
    lags_sigma = np.arange(acf_sigma.size)
    lags_nu = np.arange(acf_nu.size)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7), constrained_layout=True)
    axes[0].stem(lags_mu, acf_mu)
    axes[0].set_title(f"ACF: mu (thinning={thin})")
    axes[1].stem(lags_sigma, acf_sigma)
    axes[1].set_title(f"ACF: sigma (thinning={thin})")
    axes[2].stem(lags_nu, acf_nu)
    axes[2].set_title(f"ACF: nu (thinning={thin})")
    axes[0].set_xlim(0, lags_mu.max() if lags_mu.size else 0)
    axes[1].set_xlim(0, lags_sigma.max() if lags_sigma.size else 0)
    axes[2].set_xlim(0, lags_nu.max() if lags_nu.size else 0)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sample Autocorrelation (burn-in removed)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
