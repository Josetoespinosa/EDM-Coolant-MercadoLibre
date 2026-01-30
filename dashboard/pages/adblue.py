from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st


def _parse_sold_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    raw = _normalize_text(text)
    raw = raw.replace("vendidos", "").replace("vendido", "").replace("+", "").strip()
    raw = raw.replace("mas de", "").replace(" ", "").strip()
    if not raw:
        return None
    if "mil" in raw:
        num_part = raw.replace("mil", "")
        try:
            value = float(num_part) if num_part else 1.0
            return int(value * 1000)
        except ValueError:
            return None
    try:
        return int(raw)
    except ValueError:
        return None


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    cleaned = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return cleaned.lower()


def _infer_volume_liters(title: Optional[str]) -> Optional[float]:
    if not title:
        return None
    t = _normalize_text(title).replace(",", ".")
    candidates: list[float] = []

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(litros|litro|ltros|ltro|lts|lt|l)\b", t):
        try:
            candidates.append(float(match.group(1)))
        except ValueError:
            continue

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*ml\b", t):
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        liters = val if val < 20 else val / 1000.0
        candidates.append(liters)

    if candidates:
        return max(candidates)
    return None


def _compute_quality(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    if total == 0:
        return {
            "missing_price_pct": 0.0,
            "missing_seller_pct": 0.0,
            "missing_rating_pct": 0.0,
            "missing_image_pct": 0.0,
        }
    return {
        "missing_price_pct": df["price_value"].isna().mean() * 100,
        "missing_seller_pct": df["seller"].isna().mean() * 100,
        "missing_rating_pct": df["rating"].isna().mean() * 100,
        "missing_image_pct": df["thumbnail_url"].isna().mean() * 100,
    }


def _bin_prices_fixed(df: pd.DataFrame, step: int = 5000) -> pd.DataFrame:
    prices = df["price_value"].dropna()
    if prices.empty or step <= 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "count", "label"])
    min_val = int(prices.min())
    max_val = int(prices.max())
    min_edge = (min_val // step) * step
    max_edge = ((max_val + step - 1) // step) * step
    if max_edge == min_edge:
        max_edge += step
    edges = list(range(min_edge, max_edge + step, step))
    bins = pd.IntervalIndex.from_breaks(edges, closed="left")
    binned = pd.cut(prices, bins=bins)
    counts = binned.value_counts().reindex(bins, fill_value=0)
    df_bins = pd.DataFrame(
        {
            "bin_start": [int(interval.left) for interval in bins],
            "bin_end": [int(interval.right) for interval in bins],
            "count": counts.values,
        }
    )
    df_bins["label"] = df_bins["bin_start"].astype(str) + "-" + df_bins["bin_end"].astype(str)
    return df_bins


def _aggregate_sold_by_price_bins(df: pd.DataFrame, step: int = 5000) -> pd.DataFrame:
    data = df.dropna(subset=["price_value", "sold_estimate"]).copy()
    if data.empty or step <= 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "label", "sold_median", "count"])

    max_val = int(data["price_value"].max())
    max_edge = ((max_val + step - 1) // step) * step
    edges = list(range(0, max_edge + step, step))
    bins = pd.IntervalIndex.from_breaks(edges, closed="right")
    binned = pd.cut(data["price_value"], bins=bins, include_lowest=True)

    grouped = data.groupby(binned, observed=True)["sold_estimate"].agg(["median", "count"]).reindex(bins)
    df_bins = pd.DataFrame(
        {
            "bin_start": [0 if interval.left < 0 else int(interval.left) for interval in bins],
            "bin_end": [int(interval.right) for interval in bins],
            "sold_median": grouped["median"].values,
            "count": grouped["count"].fillna(0).astype(int).values,
        }
    )
    labels = []
    for start, end in zip(df_bins["bin_start"], df_bins["bin_end"]):
        label_start = start if start == 0 else start + 1
        labels.append(f"{label_start}-{end}")
    df_bins["label"] = labels
    df_bins = df_bins[df_bins["count"] > 0]
    return df_bins


def _safe_currency(currency: Optional[str]) -> str:
    return currency if currency else "CLP"


def _ensure_adblue_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adblue_runs (
            run_id TEXT PRIMARY KEY,
            source TEXT,
            scraped_at_utc TEXT,
            status TEXT,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adblue_items (
            run_id TEXT,
            rank INTEGER,
            title TEXT,
            seller TEXT,
            seller_is_official INTEGER,
            rating REAL,
            sold_text TEXT,
            price_value INTEGER,
            currency TEXT,
            volume_liters REAL,
            is_ad INTEGER,
            permalink TEXT,
            thumbnail_url TEXT,
            raw_item_html TEXT
        )
        """
    )


def _load_adblue_items(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    select run_id, rank, title, seller, seller_is_official, rating, sold_text,
           price_value, currency, volume_liters, is_ad, permalink, thumbnail_url, raw_item_html
    from adblue_items
    """
    return pd.read_sql_query(query, conn)


def main() -> None:
    st.set_page_config(page_title="AdBlue - Estudio de Mercado", layout="wide")
    st.title("AdBlue - Estudio de Mercado")

    with st.sidebar:
        st.header("Fuente de datos")
        db_path = st.text_input("Ruta DB", value="out/ml_prices.db")
        st.caption("Ej: out/ml_prices.db")

    db_file = Path(db_path)
    if not db_file.exists():
        st.error("No se encuentra la base de datos. Revisa la ruta.")
        st.stop()

    try:
        conn = sqlite3.connect(str(db_file))
    except sqlite3.Error as exc:
        st.error(f"No se pudo abrir la base de datos: {exc}")
        st.stop()

    _ensure_adblue_tables(conn)
    df = _load_adblue_items(conn)
    conn.close()

    if df.empty:
        st.warning("No hay items de AdBlue en la base de datos.")
        st.stop()

    df["sold_estimate"] = df["sold_text"].apply(_parse_sold_text)
    df["currency"] = df["currency"].apply(_safe_currency)
    df["volume_liters_inferred"] = df["title"].apply(_infer_volume_liters)
    df["volume_liters_norm"] = df["volume_liters"].fillna(df["volume_liters_inferred"])

    sellers = sorted([s for s in df["seller"].dropna().unique().tolist()])
    liters_available = df["volume_liters_norm"].notna().any()

    with st.sidebar:
        st.header("Filtros")
        st.subheader("Formato")
        if liters_available:
            liters_options = sorted(df["volume_liters_norm"].dropna().unique().tolist())
            liters_filter = st.multiselect("Litros", liters_options, default=[])
            include_missing_liters = st.checkbox("Incluir sin litros", value=True)
        else:
            st.caption("Sin datos de litros")
            liters_filter = []
            include_missing_liters = True

        st.subheader("Vendedor")
        seller_filter = st.multiselect("Vendedor", sellers, default=[])

    filtered = df.copy()

    if liters_available:
        if liters_filter:
            liters_mask = filtered["volume_liters_norm"].isin(liters_filter)
        else:
            liters_mask = True
        if include_missing_liters:
            liters_mask = liters_mask | filtered["volume_liters_norm"].isna()
        filtered = filtered[liters_mask]

    if seller_filter:
        filtered = filtered[filtered["seller"].isin(seller_filter)]

    st.subheader("KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Items", len(filtered))
    k2.metric("Vendedores unicos", filtered["seller"].nunique())
    if filtered["price_value"].notna().any():
        k3.metric("Precio mediana", int(filtered["price_value"].median()))
        k4.metric("Precio promedio", int(filtered["price_value"].mean()))
        k5.metric("Precio minimo", int(filtered["price_value"].min()))
    else:
        k3.metric("Precio mediana", "N/A")
        k4.metric("Precio promedio", "N/A")
        k5.metric("Precio minimo", "N/A")

    st.subheader("Indicadores adicionales")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("% Ads", round(filtered["is_ad"].mean() * 100, 1))
    a2.metric("% Oficial", round(filtered["seller_is_official"].mean() * 100, 1))
    a3.metric("Rating medio", round(filtered["rating"].mean(), 2) if filtered["rating"].notna().any() else "N/A")
    a4.metric("Vendidos estimados (mediana)", int(filtered["sold_estimate"].median()) if filtered["sold_estimate"].notna().any() else "N/A")

    st.subheader("Calidad de datos")
    quality = _compute_quality(filtered)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("% sin precio", f"{quality['missing_price_pct']:.1f}%")
    q2.metric("% sin seller", f"{quality['missing_seller_pct']:.1f}%")
    q3.metric("% sin rating", f"{quality['missing_rating_pct']:.1f}%")
    q4.metric("% sin imagen", f"{quality['missing_image_pct']:.1f}%")
    q5.empty()

    st.subheader("Distribucion de precios")
    price_bins = _bin_prices_fixed(filtered, step=5000)
    if not price_bins.empty:
        price_chart = (
            alt.Chart(price_bins)
            .mark_bar()
            .encode(
                x=alt.X("bin_start:Q", title="Precio (CLP)", axis=alt.Axis(format=",.0f")),
                x2="bin_end:Q",
                y=alt.Y("count:Q", title="Cantidad"),
                tooltip=[
                    alt.Tooltip("label:N", title="Rango"),
                    alt.Tooltip("count:Q", title="Items"),
                ],
            )
        )
        st.altair_chart(price_chart, width="stretch")
    else:
        st.info("No hay precios para graficar.")

    st.subheader("Vendidos estimados")
    sold_bins = _aggregate_sold_by_price_bins(filtered, step=5000)
    if not sold_bins.empty:
        sold_chart = (
            alt.Chart(sold_bins)
            .mark_bar()
            .encode(
                x=alt.X("bin_start:Q", title="Precio (CLP)", axis=alt.Axis(format=",.0f")),
                x2="bin_end:Q",
                y=alt.Y("sold_median:Q", title="Vendidos estimados (mediana)"),
                tooltip=[
                    alt.Tooltip("label:N", title="Rango"),
                    alt.Tooltip("sold_median:Q", title="Vendidos (mediana)"),
                    alt.Tooltip("count:Q", title="Items"),
                ],
            )
        )
        st.altair_chart(sold_chart, width="stretch")
    else:
        st.info("No hay datos de vendidos y precio para graficar.")

    st.subheader("Top vendedores")
    seller_stats = (
        filtered.groupby("seller")
        .agg(
            items=("permalink", "count"),
            price_median=("price_value", "median"),
            rating_mean=("rating", "mean"),
        )
        .sort_values("items", ascending=False)
        .head(15)
    )
    st.dataframe(seller_stats, width="stretch")

    st.subheader("Items filtrados")
    filtered_display = filtered.copy()
    filtered_display["volume_liters"] = filtered_display["volume_liters_norm"]

    st.dataframe(
        filtered_display[[
            "rank",
            "title",
            "seller",
            "price_value",
            "currency",
            "volume_liters",
            "rating",
            "sold_text",
            "is_ad",
            "seller_is_official",
            "permalink",
        ]],
        width="stretch",
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV filtrado", data=csv_data, file_name="adblue_filtrados.csv", mime="text/csv")


if __name__ == "__main__":
    main()
