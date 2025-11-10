import numpy as np
import pandas as pd
from scipy.special import gammaln

# ================================
# 1. nu 的对数后验（只差一个常数）
# ================================
def log_p_nu(nu, lambdas, a0=2.0, b0=0.1):
    """
    计算 log p(nu | lambda)（只差常数），用于 Metropolis 步骤。
    先验: nu ~ Gamma(a0, b0)（shape=a0, rate=b0）
    lambdas: 一维数组 lambda_i
    """
    if nu <= 0:
        return -np.inf

    n = len(lambdas)
    sum_log_lambda = np.sum(np.log(lambdas))
    sum_lambda = np.sum(lambdas)

    # Gamma(a0, b0) 先验 (shape, rate)：log p(nu) ∝ (a0-1)*log(nu) - b0*nu
    log_prior = (a0 - 1.0) * np.log(nu) - b0 * nu

    # lambda_i ~ Gamma(nu/2, nu/2) (shape, rate)
    half_nu = nu / 2.0
    log_lik = (
        n * (half_nu * np.log(half_nu) - gammaln(half_nu))
        + (half_nu - 1.0) * sum_log_lambda
        - half_nu * sum_lambda
    )

    return log_prior + log_lik


# ==========================================
# 2. Metropolis within Gibbs 主函数
# ==========================================
def metropolis_within_gibbs_t(
    y,
    n_iter=20000,
    m0=0.0, kappa0=0.01,      # mu | sigma^2 的正态先验超参数
    alpha0=2.0, beta0=1.0,    # sigma^2 的逆 Gamma 先验超参数
    a0=2.0, b0=0.1,           # nu 的 Gamma 先验超参数
    prop_sd=0.3,              # log(nu) 随机游走提议的标准差
    seed=123
):
    """
    Metropolis-within-Gibbs 采样器:
        y_i ~ t_nu(mu, sigma^2)

    先验:
        mu | sigma^2 ~ N(m0, sigma^2 / kappa0)
        sigma^2 ~ Inv-Gamma(alpha0, beta0)
        nu ~ Gamma(a0, b0) (shape, rate)

    返回值:
        dict: 包含 'mu', 'sigma2', 'nu' 三条链
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)

    # 存储链
    mu_chain = np.zeros(n_iter)
    sigma2_chain = np.zeros(n_iter)
    nu_chain = np.zeros(n_iter)

    # 初始值（可以比较随意给一个合理的起点）
    mu = np.mean(y)
    sigma2 = np.var(y)
    nu = 5.0
    lambdas = np.ones(n)

    for it in range(n_iter):
        # ----------------------------- #
        # 1. 更新 lambda_i | mu, sigma2, nu, y （Gibbs）
        #    lambda_i ~ Gamma((nu+1)/2, (nu + (y_i-mu)^2/sigma2)/2) (shape, rate)
        shape_lam = (nu + 1.0) / 2.0
        rate_lam = (nu + (y - mu) ** 2 / sigma2) / 2.0
        # numpy 的 Gamma(k, theta) 用的是 scale=theta，因此要用 1/rate
        lambdas = rng.gamma(shape_lam, 1.0 / rate_lam)

        # ----------------------------- #
        # 2. 更新 mu | sigma2, lambdas, y （Gibbs）
        #    是一个带权重的正态模型
        W = np.sum(lambdas)
        sum_wy = np.sum(lambdas * y)

        kappa_n = kappa0 + W
        m_n = (kappa0 * m0 + sum_wy) / kappa_n

        mu_var = sigma2 / kappa_n
        mu = rng.normal(m_n, np.sqrt(mu_var))

        # ----------------------------- #
        # 3. 更新 sigma^2 | mu, lambdas, y （Gibbs）
        #    sigma^2 ~ Inv-Gamma(alpha_n, beta_n)
        alpha_n = alpha0 + n / 2.0
        beta_n = beta0 + 0.5 * np.sum(lambdas * (y - mu) ** 2)

        # 如果 X ~ Gamma(alpha_n, scale=1/beta_n)，则 1/X ~ Inv-Gamma(alpha_n, beta_n)
        gamma_sample = rng.gamma(alpha_n, 1.0 / beta_n)
        sigma2 = 1.0 / gamma_sample

        # ----------------------------- #
        # 4. 更新 nu | lambdas （Metropolis 在 log(nu) 空间）
        eta_curr = np.log(nu)
        eta_prop = eta_curr + rng.normal(0.0, prop_sd)
        nu_prop = np.exp(eta_prop)

        # 在 eta 空间的 log 后验 = log p(nu | lambda) + log |d nu / d eta| = log p(nu|lambda) + log(nu)
        logpost_curr = log_p_nu(nu, lambdas, a0=a0, b0=b0) + np.log(nu)
        logpost_prop = log_p_nu(nu_prop, lambdas, a0=a0, b0=b0) + np.log(nu_prop)

        log_acc_ratio = logpost_prop - logpost_curr
        if np.log(rng.uniform()) < log_acc_ratio:
            nu = nu_prop  # 接受新值

        # ----------------------------- #
        # 保存样本
        mu_chain[it] = mu
        sigma2_chain[it] = sigma2
        nu_chain[it] = nu

    return {
        "mu": mu_chain,
        "sigma2": sigma2_chain,
        "nu": nu_chain,
    }


# ==========================================
# 3. 主程序：读取 SP500.csv，计算 log return 并跑 MCMC
# ==========================================
if __name__ == "__main__":
    # 1）读取数据
    df = pd.read_csv("SP500.csv")

    # 确保价格列是数值型
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
    df = df.dropna(subset=["SP500"])

    # 按日期排序（保险起见）
    df = df.sort_values("observation_date")

    # 2）计算每日对数收益率: r_t = log(P_t / P_{t-1})
    prices = df["SP500"].values
    log_returns = np.log(prices[1:] / prices[:-1])

    print(f"共有 {len(log_returns)} 条日对数收益率数据。")

    # 3）运行 Metropolis within Gibbs 采样
    out = metropolis_within_gibbs_t(
        log_returns,
        n_iter=20000,
        m0=0.0, kappa0=0.01,
        alpha0=2.0, beta0=1.0,
        a0=2.0, b0=0.1,
        prop_sd=0.3,
        seed=123
    )

    # 4）去掉 burn-in，计算后验均值作为参数估计
    burn_in = 5000
    mu_est = out["mu"][burn_in:].mean()
    sigma2_est = out["sigma2"][burn_in:].mean()
    nu_est = out["nu"][burn_in:].mean()

    print("\n基于 t 分布的参数后验均值估计：")
    print(f"mu（位置参数）       ≈ {mu_est:.6f}")
    print(f"sigma^2（方差参数） ≈ {sigma2_est:.8f}")
    print(f"sigma（标准差）     ≈ {np.sqrt(sigma2_est):.6f}")
    print(f"nu（自由度）        ≈ {nu_est:.3f}")