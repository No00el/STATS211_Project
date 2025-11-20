# STATS211_Project

Student-t model via Metropolis-within-Gibbs on S&P 500 log-returns.

## Quick Start

Requirements: Python 3.11+, `numpy`, `pandas`, `scipy`, `matplotlib`.

```powershell
C:/Users/angel/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m pip install numpy pandas scipy matplotlib
C:/Users/angel/AppData/Local/Microsoft/WindowsApps/python3.11.exe .\main.py --n-iter 20000 --burn 5000 --adapt --nu-gt2 --out-prefix run1
```

Outputs:
- `run1_trace.png`, `run1_acf.png`: Trace and ACF (burn-in removed)
- `run1_summary.json`: Posterior means, MH accept rate, final step size, config
- `run1_samples.csv`: Optional posterior samples (enable with `--save-samples`)

## CLI Usage

```
main.py [--csv SP500.csv] [--price-col SP500] [--date-col observation_date]
	[--n-iter 20000] [--burn 5000] [--seed 123]
	[--prop-sd 0.3] [--target-accept 0.3]
	[--adapt|--no-adapt] [--adapt-start 200] [--adapt-end BURN] [--adapt-interval 50]
	[--nu-gt2|--no-nu-gt2]
	[--m0 0.0] [--kappa0 0.01] [--alpha0 2.0] [--beta0 1.0] [--a0 2.0] [--b0 0.1]
	[--out-prefix results] [--save-samples|--save-full] [--acf-lag 40]
	[--acf-thin 1] [--acf-only-nu]
```

Common examples:

```powershell
# Short smoke test
C:/Users/angel/AppData/Local/Microsoft/WindowsApps/python3.11.exe .\main.py --n-iter 1500 --burn 300 --adapt --nu-gt2 --out-prefix smoketest --save-samples

# Disable enforce nu>2 and disable adaptation
C:/Users/angel/AppData/Local/Microsoft/WindowsApps/python3.11.exe .\main.py --n-iter 20000 --burn 5000 --no-nu-gt2 --no-adapt --out-prefix no_enforce

# Teacher's suggestion: ACF thinning (only nu)
C:/Users/angel/AppData/Local/Microsoft/WindowsApps/python3.11.exe .\main.py ^
	--n-iter 100000 ^
	--burn 20000 ^
	--adapt --nu-gt2 ^
	--acf-lag 500 ^
	--acf-thin 50 ^
	--acf-only-nu ^
	--out-prefix nu_acf_thin50
```

Key flags:
- `--nu-gt2`: Enforce `nu > 2` via reparameterization `xi = log(nu-2)` (ensures finite variance).
- `--adapt`: Adapt proposal std during burn-in toward `--target-accept` (default 0.3).
- `--save-samples`: Save posterior draws after burn-in to CSV.

## Interpretation Cheatsheet
- Heavy tails: Smaller `nu` → heavier tails; larger `nu` → closer to Normal.
- Tuning: Target MH accept rate around 20%–50% (default target 30%).
- Diagnostics: Check trace and ACF. If mixing is slow, increase `n-iter`, tune `--prop-sd`, or use `--nu-gt2`.

## Methodology
See `MCMC_METHOD.md` for the hierarchical model, full conditionals, MH details, and tuning advice.
