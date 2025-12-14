"""Data filtering and aggregation utilities."""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def filter_by_categories(
    df: pd.DataFrame, filters: dict[str, Iterable[str]]
) -> pd.DataFrame:
    """Filter dataframe by multiple categorical selections."""
    filtered = df.copy()
    for column, values in filters.items():
        if values:
            filtered = filtered[filtered[column].isin(values)]
    return filtered


def filter_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Keep rows where ``year`` is inside the inclusive range."""
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)]


def aggregate_values(
    df: pd.DataFrame,
    group_by: Optional[str],
    how: str = "mean",
) -> pd.DataFrame:
    """Aggregate values by year and optional category.

    Parameters
    ----------
    df:
        Filtered dataframe containing at least ``value`` and ``year``.
    group_by:
        Optional category column to aggregate by. When ``None``, aggregation is
        performed by year only.
    how:
        Aggregation function name supported by :meth:`pandas.core.groupby.GroupBy.agg`.
    """
    if group_by:
        grouped = df.groupby([group_by, "year"], as_index=False)["value"].agg(how)
    else:
        grouped = df.groupby("year", as_index=False)["value"].agg(how)
    return grouped


def slice_static_view(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return data for the selected year for static charts."""
    return df[df["year"] == year]
