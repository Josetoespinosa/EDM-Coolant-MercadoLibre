from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st
import re


def _parse_sold_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    raw = text.strip().lower()
    raw = raw.replace("vendidos", "").replace("vendido", "").replace("+", "").strip()
    raw = raw.replace("más de", "").replace("mas de", "").strip()
    raw = raw.replace(" ", "")

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
    replacements = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
        }
    )
    return text.lower().translate(replacements)


def _infer_product_type(title: Optional[str]) -> str:
    if not title:
        return "Sin título"
    t = _normalize_text(title)

    if "tester" in t or "test" in t:
        return "Tester/Accesorio"
    if "motocool" in t or "motul" in t:
        return "Moto"
    if "concentrado" in t or "97%" in t:
        return "Concentrado"
    if "ready mix" in t or "readymix" in t:
        return "Ready Mix"
    if "50/50" in t or "50-50" in t or "5050" in t:
        return "Mezcla 50/50"
    if "33%" in t or "33 %" in t:
        return "Mezcla 33%"
    if "35%" in t or "35 %" in t:
        return "Mezcla 35%"

    if "anticongelante" in t and ("refrigerante" in t or "coolant" in t or "antifreeze" in t):
        return "Refrigerante + Anticongelante"
    if "refrigerante" in t or "coolant" in t or "antifreeze" in t:
        return "Refrigerante/Coolant"
    if "anticongelante" in t:
        return "Anticongelante"

    return "Otros"


def _infer_color(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = _normalize_text(title)
    tokens = set(re.findall(r"[a-z]+", t))

    color_map = [
        ("ROSADO", {"rosado", "rosa", "pink"}),
        ("ROJO", {"rojo", "roja", "red"}),
        ("VERDE", {"verde", "green"}),
        ("AZUL", {"azul", "blue"}),
        ("NARANJA", {"naranja", "orange"}),
        ("AMARILLO", {"amarillo", "amarilla", "yellow"}),
        ("GRIS", {"gris", "gray", "grey"}),
        ("BLANCO", {"blanco", "blanca", "white"}),
        ("NEGRO", {"negro", "negra", "black"}),
        ("TURQUESA", {"turquesa", "teal", "cyan", "aqua"}),
        ("MORADO", {"morado", "violeta", "purple"}),
        ("DORADO", {"dorado", "gold"}),
        ("BEIGE", {"beige"}),
    ]

    for canon, variants in color_map:
        if tokens.intersection(variants):
            return canon
    return None


def _infer_volume_liters(title: Optional[str]) -> float:
    if not title:
        return 3.78
    t = _normalize_text(title).replace(",", ".")

    candidates: list[float] = []

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(galones|galon|gal|gallon|gallons|gl)\b", t):
        try:
            val = float(match.group(1))
            candidates.append(val * 3.78)
        except ValueError:
            continue

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(litros|litro|ltros|ltro|lts|lt|l)\b", t):
        try:
            val = float(match.group(1))
            candidates.append(val)
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
    return 3.78


def _load_runs_df(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    select run_id, url, scraped_at_utc, status, notes
    from runs
    order by scraped_at_utc desc
    """
    return pd.read_sql_query(query, conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    additions = [
        ("volume_liters", "REAL"),
        ("viscosity", "REAL"),
        ("color", "TEXT"),
        ("temp_c", "REAL"),
    ]
    for column, col_type in additions:
        if column not in existing:
            conn.execute(f"ALTER TABLE items ADD COLUMN {column} {col_type}")
    if "temp_c" in existing and "viscosity" in existing:
        conn.execute("UPDATE items SET viscosity = temp_c WHERE viscosity IS NULL AND temp_c IS NOT NULL")


def _load_items_all(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    select run_id, rank, title, seller, seller_is_official, rating, sold_text,
           price_value, currency, volume_liters, viscosity, color, temp_c, installments_text, is_ad,
           permalink, thumbnail_url, raw_item_html
    from items
    """
    return pd.read_sql_query(query, conn)


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


def _bin_numeric_series(values: pd.Series, bins: int = 8) -> pd.DataFrame:
    data = values.dropna()
    if data.empty or bins <= 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "count", "label"])
    min_val = float(data.min())
    max_val = float(data.max())
    if min_val == max_val:
        edges = [min_val, min_val + 1.0]
    else:
        _, edges = pd.cut(data, bins=bins, retbins=True, right=False, include_lowest=True)
        edges = [float(edge) for edge in edges]
    if len(edges) < 2:
        return pd.DataFrame(columns=["bin_start", "bin_end", "count", "label"])
    bins_index = pd.IntervalIndex.from_breaks(edges, closed="left")
    binned = pd.cut(data, bins=bins_index)
    counts = binned.value_counts().reindex(bins_index, fill_value=0)
    df_bins = pd.DataFrame(
        {
            "bin_start": [interval.left for interval in bins_index],
            "bin_end": [interval.right for interval in bins_index],
            "count": counts.values,
        }
    )
    df_bins["label"] = df_bins["bin_start"].round(0).astype(int).astype(str) + "-" + df_bins["bin_end"].round(0).astype(int).astype(str)
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


def main() -> None:
    st.set_page_config(page_title="Coolant - Estudio de Mercado", layout="wide")
    st.title("Coolant - Estudio de Mercado")

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

    _ensure_columns(conn)
    df = _load_items_all(conn)
    conn.close()

    if df.empty:
        st.warning("No hay items en la base de datos.")
        st.stop()

    df["sold_estimate"] = df["sold_text"].apply(_parse_sold_text)
    df["currency"] = df["currency"].apply(_safe_currency)
    df["product_type"] = df["title"].apply(_infer_product_type)
    df["color_inferred"] = df["title"].apply(_infer_color)
    df["color_norm"] = df["color"].fillna(df["color_inferred"]).astype(str).str.strip()
    df.loc[df["color_norm"].isin(["None", "nan", "NaN", ""]), "color_norm"] = None
    df["volume_liters_inferred"] = df["title"].apply(_infer_volume_liters)
    df["volume_liters_norm"] = df["volume_liters"].fillna(df["volume_liters_inferred"])

    sellers = sorted([s for s in df["seller"].dropna().unique().tolist()])
    product_types = sorted([s for s in df["product_type"].dropna().unique().tolist()])
    color_values = sorted([s for s in df["color_norm"].dropna().unique().tolist()])

    liters_available = df["volume_liters_norm"].notna().any()
    viscosity_available = df["viscosity"].notna().any()
    temp_available = df["temp_c"].notna().any()

    with st.sidebar:
        st.header("Filtros")
        st.subheader("Formato: viscosidad, temperatura y litros")
        if liters_available:
            liters_options = sorted(df["volume_liters_norm"].dropna().unique().tolist())
            liters_filter = st.multiselect("Litros", liters_options, default=[])
            include_missing_liters = st.checkbox("Incluir sin litros", value=True)
        else:
            st.caption("Sin datos de litros")
            liters_filter = []
            include_missing_liters = True

        if viscosity_available:
            allowed_viscosities = [33.0, 35.0, 50.0]
            viscosity_options = [v for v in allowed_viscosities if v in set(df["viscosity"].dropna().unique().tolist())]
            viscosity_filter = st.multiselect("Viscosidad", viscosity_options, default=[])
            include_missing_visc = st.checkbox("Incluir sin viscosidad", value=True)
        else:
            st.caption("Sin datos de viscosidad")
            viscosity_filter = []
            include_missing_visc = True

        if temp_available:
            temp_options = sorted(df["temp_c"].dropna().unique().tolist())
            temp_filter = st.multiselect("Temperatura (°C)", temp_options, default=[])
            include_missing_temp = st.checkbox("Incluir sin temperatura", value=True)
        else:
            st.caption("Sin datos de temperatura")
            temp_filter = []
            include_missing_temp = True

        st.subheader("Producto y color")
        product_filter = st.multiselect("Tipo de producto (semántico)", product_types, default=[])
        color_filter = st.multiselect("Color", color_values, default=[])

        st.subheader("Vendedor y Ads")
        seller_filter = st.multiselect("Vendedor", sellers, default=[])
        ad_only = st.selectbox("Ads", ["Todos", "Solo Ads", "Sin Ads"])
        official_only = st.selectbox("Tienda oficial", ["Todos", "Solo oficial", "No oficial"])

        st.subheader("Precio y rating")
        price_min = int(df["price_value"].min()) if df["price_value"].notna().any() else 0
        price_max = int(df["price_value"].max()) if df["price_value"].notna().any() else 0
        if price_min >= price_max:
            st.caption("Rango de precio no disponible (valor único o sin datos).")
            price_range = (price_min, price_max)
        else:
            price_range = st.slider(
                "Rango de precio (CLP)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max),
            )

        rating_min = float(df["rating"].min()) if df["rating"].notna().any() else 0.0
        rating_max = float(df["rating"].max()) if df["rating"].notna().any() else 5.0
        rating_range = st.slider("Rango de rating", min_value=0.0, max_value=5.0, value=(rating_min, rating_max))

        st.subheader("Búsqueda")
        search_text = st.text_input("Buscar en título")

        st.subheader("Opciones")
        dedupe = st.checkbox("Deduplicar por permalink (opcional)", value=False)
        if dedupe:
            df = df.drop_duplicates(subset=["permalink"])

    filtered = df.copy()

    if seller_filter:
        filtered = filtered[filtered["seller"].isin(seller_filter)]
    if product_filter:
        filtered = filtered[filtered["product_type"].isin(product_filter)]
    if color_filter:
        filtered = filtered[filtered["color_norm"].isin(color_filter)]
    if ad_only == "Solo Ads":
        filtered = filtered[filtered["is_ad"] == 1]
    elif ad_only == "Sin Ads":
        filtered = filtered[filtered["is_ad"] == 0]

    if official_only == "Solo oficial":
        filtered = filtered[filtered["seller_is_official"] == 1]
    elif official_only == "No oficial":
        filtered = filtered[filtered["seller_is_official"] == 0]

    if liters_available:
        if liters_filter:
            liters_mask = filtered["volume_liters_norm"].isin(liters_filter)
        else:
            liters_mask = True
        if include_missing_liters:
            liters_mask = liters_mask | filtered["volume_liters_norm"].isna()
        filtered = filtered[liters_mask]

    if viscosity_available:
        if viscosity_filter:
            visc_mask = filtered["viscosity"].isin(viscosity_filter)
        else:
            visc_mask = True
        if include_missing_visc:
            visc_mask = visc_mask | filtered["viscosity"].isna()
        filtered = filtered[visc_mask]

    if temp_available:
        if temp_filter:
            temp_mask = filtered["temp_c"].isin(temp_filter)
        else:
            temp_mask = True
        if include_missing_temp:
            temp_mask = temp_mask | filtered["temp_c"].isna()
        filtered = filtered[temp_mask]

    filtered = filtered[
        (filtered["price_value"].isna())
        | ((filtered["price_value"] >= price_range[0]) & (filtered["price_value"] <= price_range[1]))
    ]
    filtered = filtered[
        (filtered["rating"].isna())
        | ((filtered["rating"] >= rating_range[0]) & (filtered["rating"] <= rating_range[1]))
    ]

    if search_text:
        filtered = filtered[filtered["title"].str.contains(search_text, case=False, na=False)]

    st.subheader("KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Items", len(filtered))
    k2.metric("Vendedores únicos", filtered["seller"].nunique())
    if filtered["price_value"].notna().any():
        k3.metric("Precio mediana", int(filtered["price_value"].median()))
        k4.metric("Precio promedio", int(filtered["price_value"].mean()))
        k5.metric("Precio mínimo", int(filtered["price_value"].min()))
    else:
        k3.metric("Precio mediana", "N/A")
        k4.metric("Precio promedio", "N/A")
        k5.metric("Precio mínimo", "N/A")

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

    st.subheader("Distribución de precios")
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
            "viscosity",
            "color_norm",
            "temp_c",
            "rating",
            "sold_text",
            "is_ad",
            "seller_is_official",
            "permalink",
        ]],
        width="stretch",
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV filtrado", data=csv_data, file_name="items_filtrados.csv", mime="text/csv")


if __name__ == "__main__":
    main()
