"""Streamlit interface for exploratory analysis of panel data across organizations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data_loader import available_categories, load_panel_data
from app.plots import line_trend, static_chart
from app.processing import aggregate_values, filter_by_categories, filter_year_range, slice_static_view

st.set_page_config(page_title="Панельные данные вузов", layout="wide")


def read_uploaded(file: st.runtime.uploaded_file_manager.UploadedFile | None) -> pd.DataFrame:
    if file is None:
        return load_panel_data()
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    st.warning("Используем встроенный пример: поддерживаются только CSV или Parquet")
    return load_panel_data()


def sidebar_controls(df: pd.DataFrame) -> dict:
    variable_codes = sorted(df["variable_code"].unique())
    variable_map = (
        df[["variable_code", "variable_description"]]
        .drop_duplicates()
        .set_index("variable_code")["variable_description"]
        .to_dict()
    )

    start_year, end_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider(
        "Период",
        min_value=start_year,
        max_value=end_year,
        value=(start_year, end_year),
        step=1,
    )

    selected_code = st.sidebar.selectbox(
        "Показатель",
        variable_codes,
        format_func=lambda v: f"{v} — {variable_map.get(v, '')}",
    )
    group_by = st.sidebar.selectbox("Агрегировать по", options=[None] + available_categories(df))
    agg_method = st.sidebar.selectbox("Метод агрегации", options=["mean", "median", "sum"])

    category_filters = {}
    st.sidebar.subheader("Фильтры")
    for column in available_categories(df):
        options = sorted(df[column].dropna().unique())
        category_filters[column] = st.sidebar.multiselect(column, options)

    static_year = st.sidebar.selectbox("Год для статической диаграммы", options=sorted(df["year"].unique()))
    chart_type = st.sidebar.selectbox("Тип статической диаграммы", options=["bar", "box", "scatter"])

    return {
        "year_range": year_range,
        "selected_code": selected_code,
        "group_by": group_by,
        "agg_method": agg_method,
        "category_filters": category_filters,
        "static_year": static_year,
        "chart_type": chart_type,
    }


def main():
    st.title("Интерактивный анализ панельных данных по вузам")
    st.markdown(
        """
        Выберите период, показатель и фильтры для построения динамики и статических распределений.
        Интерфейс работает локально и поддерживает любые данные формата CSV/Parquet
        со столбцами ``org_id, year, variable_code, value`` и категориальными атрибутами.
        """
    )

    st.sidebar.header("Исходные данные")
    uploaded = st.sidebar.file_uploader("Загрузить CSV/Parquet", type=["csv", "parquet"])
    st.sidebar.caption("Если ничего не загружать — используется демонстрационный датасет.")

    df = read_uploaded(uploaded)
    controls = sidebar_controls(df)

    variable_df = df[df["variable_code"] == controls["selected_code"]].copy()
    variable_df.sort_values("year", inplace=True)
    variable_description = variable_df["variable_description"].iloc[0]

    filtered = filter_by_categories(variable_df, controls["category_filters"])
    filtered = filter_year_range(filtered, *controls["year_range"])

    aggregated = aggregate_values(
        filtered, group_by=controls["group_by"], how=controls["agg_method"]
    )

    trend_fig = line_trend(
        aggregated,
        group_by=controls["group_by"],
        variable_code=controls["selected_code"],
        variable_description=variable_description,
    )

    static_df = slice_static_view(filtered, controls["static_year"])
    static_fig = static_chart(
        static_df,
        chart_type=controls["chart_type"],
        group_by=controls["group_by"],
        variable_code=controls["selected_code"],
        variable_description=variable_description,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Динамика")
        st.plotly_chart(trend_fig, use_container_width=True)
    with col2:
        st.subheader("Статика")
        st.plotly_chart(static_fig, use_container_width=True)

    st.markdown("### Текущее количество наблюдений")
    st.write(filtered.shape[0])

    st.markdown("### Таблица (отфильтрованные данные)")
    st.dataframe(filtered.reset_index(drop=True))


if __name__ == "__main__":
    main()
