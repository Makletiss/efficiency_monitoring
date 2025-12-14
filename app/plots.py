"""Plotting helpers using Plotly Express."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px


COLORS = px.colors.sequential.Plasma


def line_trend(
    df: pd.DataFrame,
    group_by: Optional[str],
    variable_code: str,
    variable_description: str,
) -> px.line:
    """Build a line chart showing dynamics across years."""
    if group_by and group_by in df.columns:
        fig = px.line(
            df,
            x="year",
            y="value",
            color=group_by,
            markers=True,
            color_discrete_sequence=COLORS,
        )
    else:
        fig = px.line(
            df,
            x="year",
            y="value",
            markers=True,
            color_discrete_sequence=COLORS,
        )

    fig.update_layout(
        title=f"Динамика показателя {variable_code}",
        xaxis_title="Год",
        yaxis_title=variable_description,
        legend_title=group_by if group_by else None,
        hovermode="x unified",
        template="simple_white",
    )
    return fig


def static_chart(
    df: pd.DataFrame,
    chart_type: str,
    group_by: Optional[str],
    variable_code: str,
    variable_description: str,
) -> px.scatter:
    """Build a static chart for a single year or aggregated view."""
    chart_type = chart_type.lower()
    if group_by and group_by in df.columns:
        x_axis = group_by
    else:
        x_axis = "org_id"

    if chart_type == "bar":
        fig = px.bar(df, x=x_axis, y="value", color=group_by, color_discrete_sequence=COLORS)
    elif chart_type == "box":
        fig = px.box(df, x=x_axis, y="value", color=group_by, color_discrete_sequence=COLORS)
    else:
        fig = px.scatter(
            df,
            x=x_axis,
            y="value",
            color=group_by,
            size_max=12,
            color_discrete_sequence=COLORS,
        )

    fig.update_layout(
        title=f"Статическое распределение {variable_code}",
        xaxis_title=x_axis,
        yaxis_title=variable_description,
        template="simple_white",
    )
    return fig
