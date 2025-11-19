import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 1. 加载数据
# ==========================================
file_path = 'SP500_Log.csv'
df = pd.read_csv(file_path)
# 确保按日期排序
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.sort_values('observation_date')
data = df['log_return'].values

print(f"数据加载完成，样本量: {len(data)}")

# ==========================================
# 2. Metropolis-within-Gibbs 采样器
# ==========================================
def run_mwg_sampler(data, n_iter=10000, burn_in=2000):
    """
    输入: 对数收益率数据
    输出: 参数的后验样本 (mu, sigma2, nu)
    """
    n = len(data)
    
    # --- 初始化参数 (Initialization) ---
    mu = np.mean(data)
    sigma2 = np.var(data)
    nu = 5.0  # 自由度初始猜测
    lambdas = np.ones(n) # 潜变量 lambda_t，每个数据点一个
    
    # 存储采样结果
    samples = {
        'mu': np.zeros(n_iter),
        'sigma2': np.zeros(n_iter),
        'nu': np.zeros(n_iter)
    }
    
    # --- 先验分布超参数 (Hyperparameters) ---
    # Mu ~ N(0, 100)
    mu0, tau2 = 0, 100  
    # Sigma2 ~ InverseGamma(2, 0.0001) (弱信息先验)
    alpha_prior, beta_prior = 2.0, 0.0001 
    
    print(f"开始采样 (总迭代: {n_iter}, Burn-in: {burn_in})...")
    
    for i in range(n_iter):
        # ---------------------------------------
        # Step 1: 更新 Mu (Gibbs)
        # ---------------------------------------
        # 后验为正态分布
        # Precision = 1/tau2 + sum(lambda)/sigma2
        inv_tau2_post = 1/tau2 + np.sum(lambdas)/sigma2
        tau2_post = 1/inv_tau2_post
        mu_post_mean = tau2_post * (mu0/tau2 + np.sum(lambdas * data)/sigma2)
        mu = np.random.normal(mu_post_mean, np.sqrt(tau2_post))
        
        # ---------------------------------------
        # Step 2: 更新 Sigma2 (Gibbs)
        # ---------------------------------------
        # 后验为逆伽马分布
        alpha_post = alpha_prior + n / 2
        # 核心项: sum(lambda * (y - mu)^2)
        beta_post = beta_prior + 0.5 * np.sum(lambdas * (data - mu)**2)
        sigma2 = stats.invgamma.rvs(alpha_post, scale=beta_post)
        
        # ---------------------------------------
        # Step 3: 更新 Lambdas (Gibbs)
        # ---------------------------------------
        # 后验为 Gamma 分布
        # Shape = (nu + 1) / 2
        # Rate = (nu + (y - mu)^2 / sigma2) / 2
        shape_post = (nu + 1) / 2
        rate_post = (nu + (data - mu)**2 / sigma2) / 2
        # 注意：Numpy/Scipy 的 Gamma scale 参数是 1/rate
        lambdas = np.random.gamma(shape_post, 1.0/rate_post)
        
        # ---------------------------------------
        # Step 4: 更新 Nu (Metropolis-Hastings)
        # ---------------------------------------
        # Nu 的后验没有标准形式，使用 MH 算法
        nu_current = nu
        nu_proposal = np.random.normal(nu_current, 0.5) # 随机游走建议
        
        # 自由度必须 > 2 (或 > 0)，金融数据通常约束 > 2 保证方差存在
        if nu_proposal > 2.0: 
            # 计算接受率
            # 似然函数源自 lambda ~ Gamma(nu/2, nu/2)
            # log p(lambda | nu)
            log_lik_curr = np.sum(stats.gamma.logpdf(lambdas, a=nu_current/2, scale=2/nu_current))
            log_lik_prop = np.sum(stats.gamma.logpdf(lambdas, a=nu_proposal/2, scale=2/nu_proposal))
            
            # 假设 Nu 先验为均匀分布 (Flat Prior)，则只需比较似然
            log_r = log_lik_prop - log_lik_curr
            
            if np.log(np.random.rand()) < log_r:
                nu = nu_proposal # 接受新值
        
        # 存储
        samples['mu'][i] = mu
        samples['sigma2'][i] = sigma2
        samples['nu'][i] = nu
        
        if (i+1) % 1000 == 0:
            print(f"Iteration {i+1}/{n_iter} completed.")

    # 去除 Burn-in 阶段
    return {k: v[burn_in:] for k, v in samples.items()}

# ==========================================
# 3. 运行与诊断
# ==========================================
# 运行采样
posterior_samples = run_mwg_sampler(data, n_iter=10000, burn_in=2000)

# 打印估计值
print("\n" + "="*30)
print("参数估计结果 (后验均值)")
print("="*30)
print(f"Mu (均值):     {np.mean(posterior_samples['mu']):.8f}")
print(f"Sigma2 (方差): {np.mean(posterior_samples['sigma2']):.8f}")
print(f"Nu (自由度):   {np.mean(posterior_samples['nu']):.4f}")

# 绘制 Trace Plots (收敛性诊断)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(posterior_samples['mu'], alpha=0.7)
axes[0].set_ylabel(r'$\mu$')
axes[0].set_title('Trace Plots: Parameter Convergence')

axes[1].plot(posterior_samples['sigma2'], alpha=0.7, color='orange')
axes[1].set_ylabel(r'$\sigma^2$')

axes[2].plot(posterior_samples['nu'], alpha=0.7, color='green')
axes[2].set_ylabel(r'$\nu$')
axes[2].set_xlabel('Iterations (after burn-in)')

plt.tight_layout()
plt.show()