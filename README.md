# STATS211_Project

Bayesian Student-t model for S&P 500 log returns. The sampler uses Metropolis-within-Gibbs and mirrors the class methodology notes.

## Layout

- `data/`: raw and cleaned price CSV files.
- `figure/`: trace, ACF, and notebook visuals.
- `results/`: JSON or CSV summaries from runs.
- `src/`: reusable modules (data prep, sampler, CLI, tutorial).
- `STATS211_Project.ipynb`: final narrative notebook.
- `MCMC_METHOD.md` and `Project+Guidance.pdf`: reference documents.

## Quick Start

```bash
python -m pip install numpy pandas scipy matplotlib seaborn
python -m src.cli --n-iter 20000 --burn 5000 --adapt --nu-gt2 --out-prefix run1
```

Outputs:
- `figure/run1_trace.png` and `figure/run1_acf.png`
- `results/run1_summary.json`
- optional `results/run1_samples.csv` when `--save-samples` or `--save-full`

## CLI Flags

```
python -m src.cli [--csv data/SP500.csv] [--price-col SP500] [--date-col observation_date]
    [--clean-csv data/SP500_clean.csv]
    [--n-iter 20000] [--burn 5000] [--seed 123]
    [--prop-sd 0.3] [--target-accept 0.3]
    [--adapt | --no-adapt] [--adapt-start 200] [--adapt-end BURN] [--adapt-interval 50]
    [--nu-gt2 | --no-nu-gt2]
    [--m0 0.0] [--kappa0 0.01] [--alpha0 2.0] [--beta0 1.0] [--a0 2.0] [--b0 0.1]
    [--out-prefix results] [--save-samples | --save-full]
    [--acf-lag 40] [--acf-thin 50] [--acf-only-nu]
```

Example runs:
- `python -m src.cli --n-iter 1500 --burn 300 --adapt --nu-gt2 --out-prefix smoketest`
- `python -m src.cli --n-iter 20000 --burn 5000 --no-nu-gt2 --no-adapt --out-prefix free_nu`
- `python -m src.cli --n-iter 100000 --burn 20000 --acf-lag 500 --acf-thin 50 --acf-only-nu --out-prefix nu_focus`

## Tips

- Heavy tails: smaller $\nu$ means fatter tails; large $\nu$ looks Gaussian.
- Target MH acceptance near 0.3 by tuning `--prop-sd`.
- Inspect figures under `figure/` to confirm mixing.
- Priors: `kappa0` controls the mean prior, `alpha0/beta0` shape the variance prior, `a0/b0` affect tail flexibility.

## Notebook and Tutorial

- `STATS211_Project.ipynb` documents the full workflow in concise English Markdown plus code cells.
- `python -m src.tutorial --smoke` shows a short chain and simple upgrade ideas.
- `python -m src.tutorial --compare --out-prefix compare1` saves small comparison tables under `results/`.
