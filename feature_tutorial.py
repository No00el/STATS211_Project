"""
feature_tutorial.py

说明：
- 这个脚本演示如何修改/增加 `main.py` 中的要素（带代码片段与原因说明），
  并提供可运行的对比实验（短链 smoke tests）来观察修改效果。

用法示例：
    python feature_tutorial.py --smoke
    python feature_tutorial.py --compare --out-prefix compare1

"""
import argparse
import json
import time
import os
import sys
from typing import List, Dict

# 导入主采样器函数
try:
    from main import metropolis_within_gibbs_t
except Exception as e:
    print("无法导入 main.metropolis_within_gibbs_t，请先确保 main.py 在相同目录且可导入。", e)
    raise


def run_experiment(y, name: str, n_iter=1200, burn=300, prop_sd=0.3, enforce_nu_gt_2=True, adapt=True, seed=123,
                   a0=2.0, b0=0.1):
    """运行一次短链实验并返回摘要字典（适合课堂演示）"""
    print(f"运行实验 {name}: n_iter={n_iter}, burn={burn}, prop_sd={prop_sd}, enforce_nu_gt_2={enforce_nu_gt_2}, adapt={adapt}")
    t0 = time.time()
    out = metropolis_within_gibbs_t(
        y,
        n_iter=n_iter,
        m0=0.0, kappa0=0.01,
        alpha0=2.0, beta0=1.0,
        a0=a0, b0=b0,
        prop_sd=prop_sd,
        seed=seed,
        enforce_nu_gt_2=enforce_nu_gt_2,
        adapt=adapt,
        adapt_start=50,
        adapt_end=burn,
        adapt_interval=25,
        target_accept=0.3,
    )
    elapsed = time.time() - t0
    mu_mean = float(out['mu'][burn:].mean())
    sigma2_mean = float(out['sigma2'][burn:].mean())
    nu_mean = float(out['nu'][burn:].mean())
    return {
        'name': name,
        'n_iter': n_iter,
        'burn': burn,
        'mu_mean': mu_mean,
        'sigma2_mean': sigma2_mean,
        'sigma_mean': sigma2_mean ** 0.5,
        'nu_mean': nu_mean,
        'accept_rate': float(out['accept_rate']),
        'final_prop_sd': float(out['final_prop_sd']),
        'enforce_nu_gt_2': out['enforce_nu_gt_2'],
        'runtime_sec': elapsed,
    }


def load_data(csv_path='SP500.csv', price_col='SP500', date_col='observation_date'):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.dropna(subset=[price_col]).sort_values(date_col)
    prices = df[price_col].values
    log_returns = (np.log(prices[1:] / prices[:-1]))
    return log_returns


# 下面是针对常见修改的“代码片段 + 说明”（中文）
MOD_SNIPPETS = [
    {
        'title': '1) 记录并打印 nu 接受率的滑动窗口（更细粒度的诊断）',
        'why': '能观察接受率在采样过程中的动态，帮助判断自适应是否稳定或是否需要提前停止/调整。',
        'snippet': """
# 在 metropolis_within_gibbs_t 的循环中，维护 recent_accepts 窗口：
recent_accepts = collections.deque(maxlen=100)
...
if np.log(rng.uniform()) < log_acc_ratio:
    nu = nu_prop
    accept_count += 1
    recent_accepts.append(1)
else:
    recent_accepts.append(0)
...
if (it+1) % 100 == 0:
    print(f"iter {it+1}: recent accept rate (last 100) = {sum(recent_accepts)/len(recent_accepts):.3f}")
"""
    },
    {
        'title': '2) 使用对数变换 xi=log(nu-2) 强制 nu>2（已实现）',
        'why': '确保 t 分布存在有限方差，常用于金融建模中避免估计不稳定。',
        'snippet': """
# 见 main.py 中的 enforce_nu_gt_2 实现：
# xi_curr = np.log(nu - 2.0)
# xi_prop = xi_curr + rng.normal(0, prop_sd)
# nu_prop = 2.0 + np.exp(xi_prop)
# 比较 log p(nu') + log(nu'-2) 与 log p(nu) + log(nu-2)
"""
    },
    {
        'title': '3) 为 nu 使用不同的先验（例如弱信息）',
        'why': '先验对 nu 的后验有影响，尤其当数据量有限时。较弱先验能让数据主导；更强先验可以稳定估计。',
        'snippet': """
# 修改 main.py 中的默认 a0,b0，例如：
# a0 = 1.1; b0 = 0.01  # 更宽的 Gamma 先验（更弱信息）
# 在 CLI 中添加参数 --a0 --b0（已实现），并通过命令行传入。
"""
    },
    {
        'title': '4) 多链并行（MPI/ multiprocessing）',
        'why': '多链可以诊断收敛并提高样本效率（可并行运行不同随机种子与初值）。',
        'snippet': """
# 使用 multiprocessing 运行多条链：
from multiprocessing import Pool
configs = [(y, 1000, 200, 0.3, True, True, s) for s in seeds]
with Pool(4) as p:
    results = p.starmap(run_experiment, configs)
# 合并 results，检查 Gelman-Rubin 等诊断
"""
    },
    {
        'title': '5) 使用 slice sampler 或 MH 的自适应变体替代 RW 提议',
        'why': 'RW-MH 有时收敛慢；slice sampler 对尺度不敏感，能在某些情形下改进混合。',
        'snippet': """
# 替换 nu 的更新块为 slice sampling（需要实现或使用 PyMC/其他库）
# 简单示例可参照 Neal 的 slice sampling 算法实现。
"""
    },
]


def show_mod_snippets():
    for s in MOD_SNIPPETS:
        print('\n' + '='*60)
        print(s['title'])
        print('为什么要这么做：', s['why'])
        print('\n示例代码片段：')
        print(s['snippet'])


# 比较实验：变更几个要素并保存对比表
def compare_variants(y, out_prefix='compare', variants: List[Dict]=None, n_iter=1200, burn=300):
    if variants is None:
        variants = [
            {'name': 'base', 'prop_sd': 0.3, 'enforce_nu_gt_2': True, 'adapt': True},
            {'name': 'wide-prop', 'prop_sd': 0.6, 'enforce_nu_gt_2': True, 'adapt': True},
            {'name': 'no-enforce', 'prop_sd': 0.3, 'enforce_nu_gt_2': False, 'adapt': True},
        ]
    results = []
    for v in variants:
        r = run_experiment(y, v['name'], n_iter=n_iter, burn=burn, prop_sd=v['prop_sd'], enforce_nu_gt_2=v['enforce_nu_gt_2'], adapt=v['adapt'])
        results.append(r)
    # 保存为 JSON 与 CSV
    with open(f"{out_prefix}_variants.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    try:
        import csv
        keys = ['name', 'n_iter', 'burn', 'mu_mean', 'sigma2_mean', 'sigma_mean', 'nu_mean', 'accept_rate', 'final_prop_sd', 'runtime_sec']
        with open(f"{out_prefix}_variants.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in results:
                w.writerow([r[k] for k in keys])
    except Exception:
        pass
    print(f"保存对比结果: {out_prefix}_variants.json / .csv")
    return results


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Run a short smoke test to demonstrate.')
    parser.add_argument('--compare', action='store_true', help='Run small comparison experiments.')
    parser.add_argument('--out-prefix', default='tutorial', help='Output prefix for comparison files.')
    args = parser.parse_args()

    import numpy as np

    if args.smoke:
        # 载入数据并运行一个短链
        try:
            from main import metropolis_within_gibbs_t
            import pandas as pd
            df = pd.read_csv('SP500.csv')
            df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')
            df = df.dropna(subset=['SP500']).sort_values('observation_date')
            prices = df['SP500'].values
            log_returns = np.log(prices[1:] / prices[:-1])
        except Exception as e:
            print('载入数据失败，请确保 SP500.csv 存在并包含 observation_date 和 SP500 列。', e)
            sys.exit(1)
        res = run_experiment(log_returns, 'smoke', n_iter=800, burn=200, prop_sd=0.3, enforce_nu_gt_2=True, adapt=True, seed=42)
        print('\nSummary:')
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print('\n--- 修改建议及代码片段（节选）---')
        show_mod_snippets()

    if args.compare:
        try:
            import pandas as pd
            df = pd.read_csv('SP500.csv')
            df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')
            df = df.dropna(subset=['SP500']).sort_values('observation_date')
            prices = df['SP500'].values
            log_returns = np.log(prices[1:] / prices[:-1])
        except Exception as e:
            print('载入数据失败，请确保 SP500.csv 存在并包含 observation_date 和 SP500 列。', e)
            sys.exit(1)
        variants = [
            {'name': 'base', 'prop_sd': 0.3, 'enforce_nu_gt_2': True, 'adapt': True},
            {'name': 'wide-prop', 'prop_sd': 0.7, 'enforce_nu_gt_2': True, 'adapt': True},
            {'name': 'no-enforce', 'prop_sd': 0.3, 'enforce_nu_gt_2': False, 'adapt': True},
        ]
        compare_variants(log_returns, out_prefix=args.out_prefix, variants=variants, n_iter=1000, burn=250)
    
    if not (args.smoke or args.compare):
        print('运行本脚本可以：')
        print('- python feature_tutorial.py --smoke   (短链演示 + 修改片段)')
        print('- python feature_tutorial.py --compare (对比多种设置并保存 CSV/JSON)')
