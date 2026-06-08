"""
FireGuard Dashboard — Forest Fire Risk Monitor.

Visualisasi peta hotspot historis + prediksi besok per provinsi.
Connect ke MLflow native serving via REST API /invocations (LK10).

Run lokal:
    streamlit run src/dashboard/app.py

Run via docker compose:
    docker compose up -d streamlit-dashboard
    # Buka http://localhost:8501
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.dashboard.api_client import (
    API_BASE,
    build_features_from_today,
    derive_risk,
    get_health,
    predict_one,
)
from src.dashboard.data_loader import (
    aggregate_daily,
    filter_hotspots,
    load_all_hotspots,
)
from src.dashboard.map_renderer import (
    PROVINCE_DISPLAY,
    render_forecast_map,
    render_hotspot_map,
)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — dark theme menyerupai NASA FIRMS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FireGuard Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS untuk look dark/satellite
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #ff5722 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔥 FireGuard Dashboard")
st.caption("Forest Fire Risk Monitor — Indonesia Hotspot Detection + ML Forecast")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading hotspot data..."):
    all_hotspots = load_all_hotspots()

if all_hotspots.empty:
    st.error(
        "❌ Tidak ada data hotspot. Pastikan `data/raw/firms/` punya CSV "
        "(jalankan `python -m src.data.bulk_fetch` atau "
        "`python -m src.data.split_archive_by_province` dulu)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — filter & controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Filter")

    data_max = all_hotspots["acq_date"].max()
    data_min = all_hotspots["acq_date"].min()
    default_start = max(data_min, data_max - pd.Timedelta(days=30))

    date_range = st.date_input(
        "Rentang tanggal",
        value=(default_start.date(), data_max.date()),
        min_value=data_min.date(),
        max_value=data_max.date(),
    )
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = pd.Timestamp(default_start)
        end_date = data_max

    all_provs = sorted(all_hotspots["province_id"].dropna().unique().tolist())
    selected_provs = st.multiselect(
        "Provinsi",
        options=all_provs,
        default=all_provs,
        format_func=lambda p: PROVINCE_DISPLAY.get(p, p),
    )

    min_conf = st.selectbox(
        "Min confidence",
        options=["low", "nominal", "high"],
        index=0,
        help="FIRMS VIIRS confidence (l=low, n=nominal, h=high)",
    )

    st.markdown("---")

    if st.button("🔄 Reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    do_predict = st.button(
        "🔮 Predict besok (5 provinsi)",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")

    st.subheader("🤖 MLflow Model Server")
    health = get_health()
    if health:
        st.success(f"✓ Connected\n\n`{API_BASE}`\n\nEndpoint: `/invocations`")
    else:
        st.error(f"❌ Unreachable\n\n`{API_BASE}`")


# ---------------------------------------------------------------------------
# Filter data
# ---------------------------------------------------------------------------
filtered = filter_hotspots(
    all_hotspots,
    start_date=start_date,
    end_date=end_date,
    provinces=selected_provs,
    min_confidence=min_conf,
)


# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔥 Hotspot", f"{len(filtered):,}", help="Setelah filter")
col2.metric("📅 Hari", f"{(end_date - start_date).days + 1}")
col3.metric("🌏 Provinsi", len(selected_provs))
col4.metric("⚡ Mean FRP",
            f"{filtered['frp'].mean():.1f} MW" if not filtered.empty else "—")


# ---------------------------------------------------------------------------
# Predict per province
# ---------------------------------------------------------------------------
if "risk_forecast" not in st.session_state or do_predict:
    with st.spinner("Calling MLflow /invocations untuk 5 provinsi..."):
        recent_date = filtered["acq_date"].max() if not filtered.empty else end_date
        rolling_all = filter_hotspots(
            all_hotspots,
            start_date=recent_date - pd.Timedelta(days=8),
            end_date=recent_date,
            provinces=selected_provs,
            min_confidence="low",
        )

        forecasts: dict[str, dict] = {}
        for prov_id in selected_provs:
            df_today = rolling_all[
                (rolling_all["province_id"] == prov_id)
                & (rolling_all["acq_date"].dt.date == recent_date.date())
            ]
            if df_today.empty:
                forecasts[prov_id] = None
                continue
            df_hist = rolling_all[rolling_all["province_id"] == prov_id]
            features = build_features_from_today(df_today, df_hist)
            if not features:
                forecasts[prov_id] = None
                continue
            pred = predict_one(features)
            if pred is None:
                forecasts[prov_id] = None
                continue
            risk_lvl, risk_label = derive_risk(pred)
            forecasts[prov_id] = {
                "hotspot_count_tomorrow": pred,
                "risk_level": risk_lvl,
                "risk_label": risk_label,
            }

        st.session_state["risk_forecast"] = forecasts

risk_forecast = st.session_state.get("risk_forecast", {})


# ---------------------------------------------------------------------------
# MAIN — Dua peta terpisah (tab) supaya tidak saling rusak saat rerun
# ---------------------------------------------------------------------------
tab_forecast, tab_hotspot = st.tabs(
    ["🔮 Prediksi Esok Hari", "🛰️ Range Hotspot"]
)

with tab_forecast:
    st.caption("Kotak provinsi diwarnai sesuai prediksi risk besok; titik hotspot bisa di-toggle.")
    fmap_forecast = render_forecast_map(
        risk_forecast=risk_forecast,
        hotspots_df=filtered,
    )
    # key stabil -> komponen tidak di-remount saat rerun (peta tidak blank/rusak)
    st_folium(
        fmap_forecast,
        height=600,
        use_container_width=True,
        returned_objects=[],
        key="map_forecast",
    )

with tab_hotspot:
    st.caption("Titik merah = deteksi hotspot historis (hasil filter sidebar).")
    if filtered.empty:
        st.info("Tidak ada hotspot di filter ini. Coba perlebar rentang tanggal.")
    else:
        fmap_hotspot = render_hotspot_map(filtered)
        st_folium(
            fmap_hotspot,
            height=600,
            use_container_width=True,
            returned_objects=[],
            key="map_hotspot",
        )


# ---------------------------------------------------------------------------
# Bottom — time-series + risk table
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Hotspot Count Harian")
    if not filtered.empty:
        daily = aggregate_daily(filtered)
        if not daily.empty:
            pivot = daily.pivot(
                index="acq_date", columns="province_id", values="hotspot_count"
            ).fillna(0)
            st.line_chart(pivot, height=300)

with col_right:
    st.subheader("📋 Forecast Besok")
    if risk_forecast:
        rows = []
        for prov_id, fc in risk_forecast.items():
            prov_name = PROVINCE_DISPLAY.get(prov_id, prov_id)
            if fc is None:
                rows.append({
                    "Provinsi": prov_name,
                    "Today": "—",
                    "Besok": "—",
                    "Risk": "❌",
                })
                continue
            today_count = filtered[
                (filtered["province_id"] == prov_id)
                & (filtered["acq_date"].dt.date == filtered["acq_date"].max().date())
            ].shape[0]
            risk_lvl = fc["risk_level"]
            emoji = {0: "🟢", 1: "🟡", 2: "🔴"}[risk_lvl]
            rows.append({
                "Provinsi": prov_name,
                "Today": today_count,
                "Besok": f"{fc['hotspot_count_tomorrow']:.0f}",
                "Risk": f"{emoji} {fc['risk_label']}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Klik '🔮 Predict besok' di sidebar untuk forecast.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"🔥 FireGuard MLOps | "
    f"Data: NASA FIRMS Archive {data_min.date()} → {data_max.date()} | "
    f"Total: {len(all_hotspots):,} hotspots | "
    f"Model: MLflow native via `{API_BASE}/invocations` (LK10)"
)
