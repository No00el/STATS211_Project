"""Small experiments that demonstrate sampler configuration tweaks."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .cli import DATA_DIR, PROJECT_ROOT
from .data_prep import DataConfig, prepare_log_returns
from .sampler import PriorConfig, SamplerConfig, run_student_t_sampler


def _load_returns(csv_path: Path | None = None) -> np.ndarray:
    cfg = DataConfig(csv_path=csv_path or DATA_DIR / "SP500.csv", price_col="SP500", date_col="observation_date")
    cfg = cfg.resolve_paths(project_root=PROJECT_ROOT)
    _, log_returns = prepare_log_returns(cfg)
    return log_returns


def run_experiment(
    y: np.ndarray,
    name: str,
    n_iter: int = 1_200,
    burn: int = 300,
    prop_sd: float = 0.3,
    enforce_nu_gt_2: bool = True,
    adapt: bool = True,
    seed: int = 123,
    a0: float = 2.0,
    b0: float = 0.1,
) -> Dict[str, float | str]:
    priors = PriorConfig(a0=a0, b0=b0)
    cfg = SamplerConfig(
        n_iter=n_iter,
        burn_in=burn,
        seed=seed,
        prop_sd=prop_sd,
        adapt=adapt,
        adapt_start=50,
        adapt_end=burn,
        adapt_interval=25,
        enforce_nu_gt2=enforce_nu_gt_2,
        priors=priors,
    )
    print(
        f"Running {name}: n_iter={n_iter}, burn={burn}, prop_sd={prop_sd}, "
        f"enforce_nu_gt2={enforce_nu_gt_2}, adapt={adapt}"
    )
    t0 = time.time()
    result = run_student_t_sampler(y, cfg)
    elapsed = time.time() - t0
    mu_mean = float(result.mu[burn:].mean())
    sigma2_mean = float(result.sigma2[burn:].mean())
    nu_mean = float(result.nu[burn:].mean())
    return {
        "name": name,
        "n_iter": n_iter,
        "burn": burn,
        "mu_mean": mu_mean,
        "sigma2_mean": sigma2_mean,
        "sigma_mean": sigma2_mean ** 0.5,
        "nu_mean": nu_mean,
        "accept_rate": float(result.accept_rate),
        "final_prop_sd": float(result.final_prop_sd),
        "enforce_nu_gt_2": result.enforce_nu_gt2,
        "runtime_sec": elapsed,
    }


MOD_SNIPPETS = [
    {
        "title": "1) Sliding window acceptance diagnostics",
        "why": "Track MH acceptance locally to see if adaptation is stable.",
        "snippet": """
# In sampler loop, maintain a deque of recent accept indicators and print stats periodically.
recent_accepts = collections.deque(maxlen=100)
...
if np.log(rng.uniform()) < log_acc_ratio:
    nu = nu_prop
    accept_count += 1
    recent_accepts.append(1)
else:
    recent_accepts.append(0)
...
if (it + 1) % 100 == 0:
    print(f"iter {it+1}: recent accept rate={(sum(recent_accepts)/len(recent_accepts)):.3f}")
""",
    },
    {
        "title": "2) Enforce nu>2 via log-shift",
        "why": "Guarantees finite variance; matches the final project assumption.",
        "snippet": """
xi_curr = np.log(nu - 2.0)
xi_prop = xi_curr + rng.normal(0, prop_sd)
nu_prop = 2.0 + np.exp(xi_prop)
logpost_curr = log_p_nu(nu, lambdas, priors) + np.log(nu - 2.0)
logpost_prop = log_p_nu(nu_prop, lambdas, priors) + np.log(nu_prop - 2.0)
""",
    },
    {
        "title": "3) Try weaker priors on nu",
        "why": "When sample size is moderate, nu's prior can dominate; experiment with smaller a0/b0.",
        "snippet": """
priors = PriorConfig(a0=1.1, b0=0.01)
result = run_student_t_sampler(log_returns, sampler_cfg)
""",
    },
    {
        "title": "4) Parallel chains",
        "why": "Multiple seeds help diagnose convergence (Gelman-Rubin, etc.).",
        "snippet": """
from multiprocessing import Pool
seeds = [11, 22, 33, 44]
with Pool(len(seeds)) as pool:
    configs = [cfg.replace(seed=s) for s in seeds]
    outputs = pool.starmap(run_student_t_sampler, [(log_returns, c) for c in configs])
""",
    },
    {
        "title": "5) Alternative proposal mechanisms",
        "why": "Slice sampling or adaptive MH can improve nu mixing when tails are tricky.",
        "snippet": """
# Swap RW-MH for slice sampling in the nu block using Neal's stepping-out procedure.
# Accept/reject step disappears because slice sampling is rejection-free.
""",
    },
]


def show_mod_snippets() -> None:
    for snippet in MOD_SNIPPETS:
        print("\n" + "=" * 60)
        print(snippet["title"])
        print("Why:", snippet["why"])
        print("\nSnippet:")
        print(snippet["snippet"])


def compare_variants(
    y: np.ndarray,
    out_prefix: str = "compare",
    variants: List[Dict[str, object]] | None = None,
    n_iter: int = 1_200,
    burn: int = 300,
) -> List[Dict[str, object]]:
    if variants is None:
        variants = [
            {"name": "base", "prop_sd": 0.3, "enforce_nu_gt_2": True, "adapt": True},
            {"name": "wide-prop", "prop_sd": 0.6, "enforce_nu_gt_2": True, "adapt": True},
            {"name": "no-enforce", "prop_sd": 0.3, "enforce_nu_gt_2": False, "adapt": True},
        ]
    results: List[Dict[str, object]] = []
    for variant in variants:
        res = run_experiment(
            y,
            variant["name"],
            n_iter=n_iter,
            burn=burn,
            prop_sd=variant["prop_sd"],
            enforce_nu_gt_2=variant["enforce_nu_gt_2"],
            adapt=variant["adapt"],
        )
        results.append(res)

    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"{out_prefix}_variants.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved comparison JSON: {json_path.relative_to(PROJECT_ROOT)}")
    csv_path = RESULTS_DIR / f"{out_prefix}_variants.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "name",
            "n_iter",
            "burn",
            "mu_mean",
            "sigma2_mean",
            "sigma_mean",
            "nu_mean",
            "accept_rate",
            "final_prop_sd",
            "runtime_sec",
        ]
        writer.writerow(header)
        for row in results:
            writer.writerow([row[key] for key in header])
    print(f"Saved comparison CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classroom tutorial utilities for the Student-t sampler")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test")
    parser.add_argument("--compare", action="store_true", help="Run comparison experiments")
    parser.add_argument("--out-prefix", default="tutorial", help="Prefix for comparison outputs")
    args = parser.parse_args()

    if args.smoke:
        y = _load_returns()
        res = run_experiment(y, "smoke", n_iter=800, burn=200, prop_sd=0.3, enforce_nu_gt_2=True, adapt=True, seed=42)
        print("\nSummary")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("\n--- Suggested modifications ---")
        show_mod_snippets()

    if args.compare:
        y = _load_returns()
        variants = [
            {"name": "base", "prop_sd": 0.3, "enforce_nu_gt_2": True, "adapt": True},
            {"name": "wide-prop", "prop_sd": 0.7, "enforce_nu_gt_2": True, "adapt": True},
            {"name": "no-enforce", "prop_sd": 0.3, "enforce_nu_gt_2": False, "adapt": True},
        ]
        compare_variants(y, out_prefix=args.out_prefix, variants=variants, n_iter=1_000, burn=250)

    if not (args.smoke or args.compare):
        print("Usage examples:")
        print("  python -m src.tutorial --smoke")
        print("  python -m src.tutorial --compare --out-prefix compare1")


if __name__ == "__main__":
    main()
