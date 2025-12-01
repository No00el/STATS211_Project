"""Command line interface for the STATS211 Student-t project."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .data_prep import DataConfig, prepare_log_returns
from .diagnostics import plot_acf, plot_traces
from .sampler import PriorConfig, SamplerConfig, run_student_t_sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figure"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metropolis-within-Gibbs sampler for Student-t log-return model")
    parser.add_argument("--csv", default=str(DATA_DIR / "SP500.csv"))
    parser.add_argument("--price-col", default="SP500")
    parser.add_argument("--date-col", default="observation_date")
    parser.add_argument("--clean-csv", default=None, help="Optional path to save the cleaned log-return CSV")
    parser.add_argument("--n-iter", type=int, default=200_000)
    parser.add_argument("--burn", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prop-sd", type=float, default=0.3)
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
    parser.add_argument("--acf-thin", type=int, default=50)
    parser.add_argument("--acf-only-nu", action="store_true")
    return parser.parse_args()


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    adapt_flag = args.adapt and not args.no_adapt
    nu_gt2_flag = args.nu_gt2 and not args.no_nu_gt2

    clean_csv_path = _resolve_path(args.clean_csv)
    data_cfg = DataConfig(
        csv_path=_resolve_path(args.csv) or DATA_DIR / "SP500.csv",
        price_col=args.price_col,
        date_col=args.date_col,
        output_clean_path=clean_csv_path,
    )
    data_cfg = data_cfg.resolve_paths(project_root=PROJECT_ROOT)

    priors = PriorConfig(
        m0=args.m0,
        kappa0=args.kappa0,
        alpha0=args.alpha0,
        beta0=args.beta0,
        a0=args.a0,
        b0=args.b0,
    )

    sampler_cfg = SamplerConfig(
        n_iter=args.n_iter,
        burn_in=args.burn,
        seed=args.seed,
        prop_sd=args.prop_sd,
        target_accept=args.target_accept,
        adapt=adapt_flag,
        adapt_start=args.adapt_start,
        adapt_end=args.adapt_end,
        adapt_interval=args.adapt_interval,
        enforce_nu_gt2=nu_gt2_flag,
        priors=priors,
    )

    t0 = time.time()
    df, log_returns = prepare_log_returns(data_cfg)
    print(f"Loaded {log_returns.size} daily log returns from {data_cfg.csv_path}.")

    result = run_student_t_sampler(log_returns, sampler_cfg)
    burn = args.burn
    mu_est = float(result.mu[burn:].mean())
    sigma2_est = float(result.sigma2[burn:].mean())
    nu_est = float(result.nu[burn:].mean())
    elapsed = time.time() - t0

    print("\nPosterior means (burn-in removed):")
    print(f"mu      ≈ {mu_est:.6f}")
    print(f"sigma^2 ≈ {sigma2_est:.8f}")
    print(f"sigma   ≈ {np.sqrt(sigma2_est):.6f}")
    print(f"nu      ≈ {nu_est:.3f}")
    print(
        f"MH acceptance ≈ {result.accept_rate*100:.1f}% | final step size ≈ {result.final_prop_sd:.3f} | enforce nu>2: {result.enforce_nu_gt2}"
    )
    print(f"Runtime ≈ {elapsed:.2f} sec")

    FIGURE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    trace_path = FIGURE_DIR / f"{args.out_prefix}_trace.png"
    acf_path = FIGURE_DIR / f"{args.out_prefix}_acf.png"
    plot_traces(result, burn=burn, out_path=trace_path)
    plot_acf(result, burn=burn, max_lag=args.acf_lag, out_path=acf_path, thin=args.acf_thin, only_nu=args.acf_only_nu)
    print(f"Saved figures: {trace_path.relative_to(PROJECT_ROOT)}, {acf_path.relative_to(PROJECT_ROOT)}")

    summary = {
        "n_iter": args.n_iter,
        "burn_in": burn,
        "mu_mean": mu_est,
        "sigma2_mean": sigma2_est,
        "sigma_mean": float(np.sqrt(sigma2_est)),
        "nu_mean": nu_est,
        "accept_rate": float(result.accept_rate),
        "final_prop_sd": float(result.final_prop_sd),
        "enforce_nu_gt_2": result.enforce_nu_gt2,
        "adapt": adapt_flag,
        "target_accept": args.target_accept,
        "runtime_seconds": elapsed,
        "csv_path": str(data_cfg.csv_path),
    }
    summary_path = RESULTS_DIR / f"{args.out_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path.relative_to(PROJECT_ROOT)}")

    if args.save_samples or args.save_full:
        post_mu = result.mu[burn:] if args.save_samples else result.mu
        post_sigma2 = result.sigma2[burn:] if args.save_samples else result.sigma2
        post_nu = result.nu[burn:] if args.save_samples else result.nu
        samples_path = RESULTS_DIR / f"{args.out_prefix}_samples.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mu", "sigma2", "nu"])
            for a, b, c in zip(post_mu, post_sigma2, post_nu):
                writer.writerow([f"{a:.8f}", f"{b:.8f}", f"{c:.5f}"])
        print(f"Saved samples: {samples_path.relative_to(PROJECT_ROOT)} (rows={post_mu.size})")


if __name__ == "__main__":
    main()
