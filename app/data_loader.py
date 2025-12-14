"""Data loading helpers for the panel analytics interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

CATEGORY_COLUMNS = [
    "uni_type",
    "profile",
    "federal_district",
    "region",
    "agency",
]


def load_panel_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load panel data from CSV or Parquet.

    Parameters
    ----------
    path:
        Optional path to a CSV or Parquet file. When ``None``, a bundled sample
        dataset is used to let the interface start instantly.
    """
    if path is None:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_panel.csv"
        return pd.read_csv(sample_path)

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    raise ValueError("Unsupported file format. Use CSV or Parquet.")


def available_categories(df: pd.DataFrame) -> list[str]:
    """Return a list of category columns present in the dataframe."""
    return [col for col in CATEGORY_COLUMNS if col in df.columns]
