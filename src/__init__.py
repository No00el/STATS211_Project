"""STATS211 S&P 500 Student-t project package."""
from .data_prep import DataConfig, prepare_log_returns, load_price_history, attach_log_returns, save_clean_data
from .diagnostics import plot_acf, plot_traces
from .sampler import PriorConfig, SamplerConfig, SamplerResult, run_student_t_sampler

__all__ = [
    "DataConfig",
    "PriorConfig",
    "SamplerConfig",
    "SamplerResult",
    "attach_log_returns",
    "prepare_log_returns",
    "load_price_history",
    "plot_acf",
    "plot_traces",
    "run_student_t_sampler",
    "save_clean_data",
]
