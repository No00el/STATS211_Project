import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt

# ================================
# 1. nu 的对数后验（只差一个常数）
# ================================
def log_p_nu(nu, lambdas, a0=2.0, b0=0.1):
    """log p(nu | lambda) up to constant, with Gamma(a0,b0) prior on nu (shape-rate)."""
    if nu <= 0:
        return -np.inf
    n = len(lambdas)
    sum_log_lambda = np.sum(np.log(lambdas))
    sum_lambda = np.sum(lambdas)
    log_prior = (a0 - 1.0) * np.log(nu) - b0 * nu
    half_nu = nu / 2.0
    log_lik = (
        n * (half_nu * np.log(half_nu) - gammaln(half_nu))
        + (half_nu - 1.0) * sum_log_lambda
        - half_nu * sum_lambda
    )
    return log_prior + log_lik


def metropolis_within_gibbs_t(
    y,
    n_iter=200000,
    m0=0.0,
    kappa0=0.01,
    alpha0=2.0,
    beta0=1.0,
    a0=2.0,
    b0=0.1,
    prop_sd=0.1,
    seed=123,
    enforce_nu_gt_2=False,
    adapt=True,
    adapt_start=200,
    adapt_end=10000,
    adapt_interval=50,
    target_accept=0.30,
):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    mu_chain = np.zeros(n_iter)
    sigma2_chain = np.zeros(n_iter)
    nu_chain = np.zeros(n_iter)
    mu = np.mean(y)
    sigma2 = np.var(y)
    nu = 5.0
    lambdas = np.ones(n)
    accept_count = 0
    accept_count_window = 0
    curr_prop_sd = float(prop_sd)
    log_prop_sd = np.log(curr_prop_sd)
    for it in range(n_iter):
        shape_lam = (nu + 1.0) / 2.0
        rate_lam = (nu + (y - mu) ** 2 / sigma2) / 2.0
        lambdas = rng.gamma(shape_lam, 1.0 / rate_lam)
        W = np.sum(lambdas)
        sum_wy = np.sum(lambdas * y)
        kappa_n = kappa0 + W
        m_n = (kappa0 * m0 + sum_wy) / kappa_n
        mu_var = sigma2 / kappa_n
        mu = rng.normal(m_n, np.sqrt(mu_var))
        alpha_n = alpha0 + n / 2.0
        beta_n = beta0 + 0.5 * np.sum(lambdas * (y - mu) ** 2)
        gamma_sample = rng.gamma(alpha_n, 1.0 / beta_n)
        sigma2 = 1.0 / gamma_sample
        if enforce_nu_gt_2:
            xi_curr = np.log(nu - 2.0) if nu > 2.0 else -5.0
            xi_prop = xi_curr + rng.normal(0.0, curr_prop_sd)
            nu_prop = 2.0 + np.exp(xi_prop)
            logpost_curr = log_p_nu(nu, lambdas, a0=a0, b0=b0) + np.log(max(nu - 2.0, 1e-12))
            logpost_prop = log_p_nu(nu_prop, lambdas, a0=a0, b0=b0) + np.log(nu_prop - 2.0)
        else:
            eta_curr = np.log(nu)
            eta_prop = eta_curr + rng.normal(0.0, curr_prop_sd)
            nu_prop = np.exp(eta_prop)
            logpost_curr = log_p_nu(nu, lambdas, a0=a0, b0=b0) + np.log(nu)
            logpost_prop = log_p_nu(nu_prop, lambdas, a0=a0, b0=b0) + np.log(nu_prop)
        log_acc_ratio = logpost_prop - logpost_curr
        if np.log(rng.uniform()) < log_acc_ratio:
            nu = nu_prop
            accept_count += 1
            accept_count_window += 1
        if adapt and (adapt_start <= it + 1 <= adapt_end) and ((it + 1) % adapt_interval == 0):
            # Use recent-window acceptance (over the last `adapt_interval` proposals)
            # This is more responsive than the cumulative acceptance rate.
            acc_rate = accept_count_window / float(adapt_interval)
            # A slightly larger step (0.10) helps the scale adapt faster in short runs.
            log_prop_sd += 0.1 * (acc_rate - target_accept)
            log_prop_sd = np.clip(log_prop_sd, np.log(1e-3), np.log(5.0))
            curr_prop_sd = float(np.exp(log_prop_sd))
            accept_count_window = 0
        mu_chain[it] = mu
        sigma2_chain[it] = sigma2
        nu_chain[it] = nu
    accept_rate = accept_count / float(n_iter)
    return {
        "mu": mu_chain,
        "sigma2": sigma2_chain,
        "nu": nu_chain,
        "accept_rate": accept_rate,
        "final_prop_sd": curr_prop_sd,
        "enforce_nu_gt_2": enforce_nu_gt_2,
    }


# ==========================================
# 辅助：绘图与简单 ACF
# ==========================================
def _autocorr(x, max_lag=40):
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])
    var = np.dot(x, x) / n
    if var == 0:
        return np.ones(min(max_lag, n - 1) + 1)
    K = min(max_lag, n - 1)
    acf = np.empty(K + 1)
    acf[0] = 1.0
    for k in range(1, K + 1):
        acf[k] = np.dot(x[: n - k], x[k:]) / (n * var)
    return acf


def save_trace_plots(samples, burn=0, out_path="traceplots.png"):
    mu = samples["mu"][burn:]
    sigma = np.sqrt(samples["sigma2"][burn:])
    nu = samples["nu"][burn:]

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


def save_acf_plots(samples, burn=0, max_lag=40, out_path="acfplots.png", thin=50, only_nu=False):
    mu = samples["mu"][burn:]
    sigma = np.sqrt(samples["sigma2"][burn:])
    nu = samples["nu"][burn:]

    # Thinning
    thin = max(1, int(thin))
    mu_t = mu[::thin]
    sigma_t = sigma[::thin]
    nu_t = nu[::thin]

    if only_nu:
        acf_nu = _autocorr(nu_t, max_lag=max_lag)
        lags = np.arange(len(acf_nu))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
        ax.stem(lags, acf_nu)
        ax.set_xlim(0, max(lags) if len(lags) else 0)
        ax.set_title(f"ACF: nu (thinning={thin})")
        ax.grid(True, alpha=0.3)
        fig.suptitle("Sample Autocorrelation (burn-in removed)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    acf_mu = _autocorr(mu_t, max_lag=max_lag)
    acf_sigma = _autocorr(sigma_t, max_lag=max_lag)
    acf_nu = _autocorr(nu_t, max_lag=max_lag)
    lags_mu = np.arange(len(acf_mu))
    lags_sigma = np.arange(len(acf_sigma))
    lags_nu = np.arange(len(acf_nu))

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7), constrained_layout=True)
    axes[0].stem(lags_mu, acf_mu)
    axes[0].set_title(f"ACF: mu (thinning={thin})")
    axes[1].stem(lags_sigma, acf_sigma)
    axes[1].set_title(f"ACF: sigma (thinning={thin})")
    axes[2].stem(lags_nu, acf_nu)
    axes[2].set_title(f"ACF: nu (thinning={thin})")
    for ax in axes:
        # Set xlim to the length available for each subplot
        x_max = 0
        if ax is axes[0]:
            x_max = max(lags_mu) if len(lags_mu) else 0
        elif ax is axes[1]:
            x_max = max(lags_sigma) if len(lags_sigma) else 0
        else:
            x_max = max(lags_nu) if len(lags_nu) else 0
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sample Autocorrelation (burn-in removed)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_hist_plots(samples, burn=0, out_path="histplots.png"):
    mu = samples["mu"][burn:]
    sigma2 = samples["sigma2"][burn:]
    nu = samples["nu"][burn:]
    params = [
        (mu, "mu", "#a7c8e5"),
        (sigma2, r"sigma^2", "#f7b7a3"),
        (nu, "nu", "#b5d99c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (arr, label, color) in zip(axes, params):
        mean = float(np.mean(arr))
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        ax.hist(arr, bins=40, density=True, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(mean, color="#1f77b4", lw=1.8, label="Mean")
        ax.axvline(ci_low, color="#c80000", ls="--", lw=1.4, label="95% CI")
        ax.axvline(ci_high, color="#c80000", ls="--", lw=1.4)
        ax.set_title(f"Posterior: {label}")
        ax.text(
            0.5,
            0.95,
            f"Mean: {mean:.6f}\n95% CI: [{ci_low:.6f}, {ci_high:.6f}]",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle("Posterior Histograms with 95% Credible Intervals (burn-in removed)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================
# 3. 主程序：读取 SP500.csv，计算 log return 并跑 MCMC
# ==========================================
if __name__ == "__main__":
    import argparse, json, time, csv
    parser = argparse.ArgumentParser(description="Metropolis-within-Gibbs sampler for Student-t model of log returns.")
    parser.add_argument("--csv", default="SP500.csv")
    parser.add_argument("--price-col", default="SP500")
    parser.add_argument("--date-col", default="observation_date")
    parser.add_argument("--n-iter", type=int, default=200000)
    parser.add_argument("--burn", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prop-sd", type=float, default=0.1)
    parser.add_argument("--target-accept", type=float, default=0.30)
    parser.add_argument("--adapt", action="store_true")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--adapt-start", type=int, default=200)
    parser.add_argument("--adapt-end", type=int, default=None)
    parser.add_argument("--adapt-interval", type=int, default=50)
    parser.add_argument("--nu-gt2", action="store_true")
    parser.add_argument("--no-nu-gt2", action="store_true")
    parser.add_argument("--m0", type=float, default=0.0)
    parser.add_argument("--kappa0", type=float, default=0.01)
    parser.add_argument("--alpha0", type=float, default=2.0)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--a0", type=float, default=2.0)
    parser.add_argument("--b0", type=float, default=0.1)
    parser.add_argument("--out-prefix", default="results")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--acf-lag", type=int, default=40)
    parser.add_argument("--acf-thin", type=int, default=100, help="Thinning factor for ACF computation only (default 50).")
    parser.add_argument("--acf-only-nu", action="store_true", help="Plot only nu's ACF using optional thinning.")
    args = parser.parse_args()
    adapt_flag = args.adapt and not args.no_adapt
    nu_gt2_flag = args.nu_gt2 and not args.no_nu_gt2
    burn_in = args.burn
    adapt_end_iter = args.adapt_end if args.adapt_end is not None else burn_in
    t0 = time.time()
    df = pd.read_csv(args.csv)
    df[args.price_col] = pd.to_numeric(df[args.price_col], errors="coerce")
    df = df.dropna(subset=[args.price_col]).sort_values(args.date_col)
    prices = df[args.price_col].values
    log_returns = np.log(prices[1:] / prices[:-1])
    print(f"共有 {len(log_returns)} 条日对数收益率数据。")
    out = metropolis_within_gibbs_t(
        log_returns,
        n_iter=args.n_iter,
        m0=args.m0,
        kappa0=args.kappa0,
        alpha0=args.alpha0,
        beta0=args.beta0,
        a0=args.a0,
        b0=args.b0,
        prop_sd=args.prop_sd,
        seed=args.seed,
        enforce_nu_gt_2=nu_gt2_flag,
        adapt=adapt_flag,
        adapt_start=args.adapt_start,
        adapt_end=adapt_end_iter,
        adapt_interval=args.adapt_interval,
        target_accept=args.target_accept,
    )
    mu_est = out["mu"][burn_in:].mean()
    sigma2_est = out["sigma2"][burn_in:].mean()
    nu_est = out["nu"][burn_in:].mean()
    mu_ci = np.percentile(out["mu"][burn_in:], [2.5, 97.5])
    sigma2_ci = np.percentile(out["sigma2"][burn_in:], [2.5, 97.5])
    nu_ci = np.percentile(out["nu"][burn_in:], [2.5, 97.5])
    elapsed = time.time() - t0
    print("\n后验均值：")
    print(f"mu      ≈ {mu_est:.6f}")
    print(f"sigma^2 ≈ {sigma2_est:.8f}")
    print(f"sigma   ≈ {np.sqrt(sigma2_est):.6f}")
    print(f"nu      ≈ {nu_est:.3f}")
    print("95% 置信区间(credible intervals)：")
    print(f"mu:      ({mu_ci[0]:.6f}, {mu_ci[1]:.6f})")
    print(f"sigma^2: ({sigma2_ci[0]:.8f}, {sigma2_ci[1]:.8f})")
    print(f"nu:      ({nu_ci[0]:.3f}, {nu_ci[1]:.3f})")
    print(f"MH 接受率 ≈ {out['accept_rate']*100:.1f}% | 最终步长 ≈ {out['final_prop_sd']:.3f} | enforce nu>2: {out['enforce_nu_gt_2']}")
    print(f"耗时 ≈ {elapsed:.2f} 秒")
    trace_path = f"{args.out_prefix}_trace.png"
    acf_path = f"{args.out_prefix}_acf.png"
    hist_path = f"{args.out_prefix}_hist.png"
    save_trace_plots(out, burn=burn_in, out_path=trace_path)
    save_acf_plots(out, burn=burn_in, max_lag=args.acf_lag, out_path=acf_path, thin=args.acf_thin, only_nu=args.acf_only_nu)
    save_hist_plots(out, burn=burn_in, out_path=hist_path)
    print(f"保存: {trace_path}, {acf_path}, {hist_path}")
    summary = {
        "n_iter": args.n_iter,
        "burn_in": burn_in,
        "mu_mean": float(mu_est),
        "sigma2_mean": float(sigma2_est),
        "sigma_mean": float(np.sqrt(sigma2_est)),
        "nu_mean": float(nu_est),
        "mu_ci": [float(mu_ci[0]), float(mu_ci[1])],
        "sigma2_ci": [float(sigma2_ci[0]), float(sigma2_ci[1])],
        "nu_ci": [float(nu_ci[0]), float(nu_ci[1])],
        "accept_rate": float(out["accept_rate"]),
        "final_prop_sd": float(out["final_prop_sd"]),
        "enforce_nu_gt_2": out["enforce_nu_gt_2"],
        "adapt": adapt_flag,
        "target_accept": args.target_accept,
        "runtime_seconds": elapsed,
    }
    with open(f"{args.out_prefix}_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"保存: {args.out_prefix}_summary.json")
    if args.save_samples or args.save_full:
        post_mu = out["mu"][burn_in:] if args.save_samples else out["mu"]
        post_sigma2 = out["sigma2"][burn_in:] if args.save_samples else out["sigma2"]
        post_nu = out["nu"][burn_in:] if args.save_samples else out["nu"]
        with open(f"{args.out_prefix}_samples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mu", "sigma2", "nu"])
            for a, b, c in zip(post_mu, post_sigma2, post_nu):
                w.writerow([f"{a:.8f}", f"{b:.8f}", f"{c:.5f}"])
        print(f"保存: {args.out_prefix}_samples.csv (rows={len(post_mu)})")