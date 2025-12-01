"""Data preparation utilities for the S&P 500 Student-t project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    """Configuration describing how to load and clean the price data."""

    csv_path: Path
    price_col: str = "SP500"
    date_col: str = "observation_date"
    output_clean_path: Optional[Path] = None

    def resolve_paths(self, project_root: Optional[Path] = None) -> "DataConfig":
        cfg = DataConfig(
            csv_path=self._resolve(self.csv_path, project_root),
            price_col=self.price_col,
            date_col=self.date_col,
            output_clean_path=self._resolve(self.output_clean_path, project_root),
        )
        return cfg

    @staticmethod
    def _resolve(value: Optional[Path], project_root: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        value = Path(value)
        if not value.is_absolute() and project_root is not None:
            return project_root / value
        return value


def load_price_history(config: DataConfig) -> pd.DataFrame:
    """Read, clean, and sort the historical S&P 500 price data."""

    df = pd.read_csv(config.csv_path)
    df[config.price_col] = pd.to_numeric(df[config.price_col], errors="coerce")
    df = df.dropna(subset=[config.price_col]).copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col)
    return df[[config.date_col, config.price_col]].reset_index(drop=True)


def attach_log_returns(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute log returns and append the column to the DataFrame."""

    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df


def prepare_log_returns(config: DataConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return a cleaned DataFrame and the numpy array of log returns."""

    df = load_price_history(config)
    df = attach_log_returns(df, config.price_col)
    if config.output_clean_path is not None:
        output_path = config.output_clean_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df, df["log_return"].to_numpy()


def save_clean_data(csv_path: Path, output_path: Path, price_col: str = "SP500", date_col: str = "observation_date") -> None:
    """Helper that mimics the old Data_Preprocess script behavior."""

    cfg = DataConfig(csv_path=csv_path, price_col=price_col, date_col=date_col, output_clean_path=output_path)
    prepare_log_returns(cfg)
