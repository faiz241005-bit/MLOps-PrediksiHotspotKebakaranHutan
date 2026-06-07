"""
Folium map renderer — style menyerupai NASA FIRMS fire map.

Visual:
    - Satellite imagery basemap (Esri.WorldImagery) — gelap, realistik
    - Red CircleMarker untuk hotspot (no border, semi-transparent, FRP-scaled)
    - Opsional: province bbox rectangle dengan warna risk forecast

Returns folium.Map object untuk st_folium().
"""
from __future__ import annotations

import logging
from typing import Optional

import folium
import pandas as pd

LOG = logging.getLogger(__name__)

# Bbox provinsi Indonesia (5 fire-prone)
PROVINCE_BBOXES = {
    "riau":    [0.0, 100.0, 4.5, 106.5],
    "kalteng": [-3.5, 110.5, 1.5, 116.5],
    "kalbar":  [-3.0, 108.0, 2.5, 118.0],
    "sumsel":  [-5.5, 102.0, -1.0, 108.5],
    "jambi":   [-3.0, 101.0, -0.5, 105.0],
}

PROVINCE_DISPLAY = {
    "riau":    "Riau",
    "kalteng": "Kalimantan Tengah",
    "kalbar":  "Kalimantan Barat",
    "sumsel":  "Sumatera Selatan",
    "jambi":   "Jambi",
}

# Risk colors — bright untuk over satellite dark
RISK_COLORS = {
    0: "#4caf50",  # Aman — hijau
    1: "#ffeb3b",  # Waspada — kuning bright
    2: "#ff1744",  # Bahaya — merah bright
}

RISK_LABELS = {0: "Aman", 1: "Waspada", 2: "Bahaya"}

# Map default: center di tengah Indonesia (Pulau Sulawesi)
_DEFAULT_CENTER = [-1.0, 117.0]
_DEFAULT_ZOOM = 5


def render_map(
    hotspots_df: pd.DataFrame,
    risk_forecast: Optional[dict[str, dict]] = None,
    max_hotspots_to_render: int = 5000,
    show_provinces: bool = True,
) -> folium.Map:
    """
    Bangun folium.Map dengan satellite imagery + red hotspot markers.

    Style menyerupai NASA FIRMS fire map:
        - Dark satellite basemap (Esri.WorldImagery)
        - Red dots untuk tiap hotspot detection
        - Province bbox overlay (opsional) dengan warna risk_level
    """
    # Satellite imagery basemap — match NASA FIRMS look
    m = folium.Map(
        location=_DEFAULT_CENTER,
        zoom_start=_DEFAULT_ZOOM,
        tiles=None,           # tidak pakai default OSM
        control_scale=True,
    )

    # Esri WorldImagery — satellite imagery dark + detail tinggi
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri",
        name="Satellite (Esri WorldImagery)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Alternative tile (toggle-able): CartoDB Dark
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark (CartoDB)",
        overlay=False,
        control=True,
    ).add_to(m)

    # ---- Layer 1: Province bbox overlay (forecast risk) -------------------
    if show_provinces:
        province_group = folium.FeatureGroup(name="🌏 Provinsi (forecast)", show=True)
        for prov_id, bbox in PROVINCE_BBOXES.items():
            lat_min, lon_min, lat_max, lon_max = bbox
            bounds = [[lat_min, lon_min], [lat_max, lon_max]]

            color = "#9e9e9e"
            risk_label = "—"
            forecast_count = None
            if risk_forecast and prov_id in risk_forecast and risk_forecast[prov_id]:
                fc = risk_forecast[prov_id]
                color = RISK_COLORS.get(fc.get("risk_level", 0), "#9e9e9e")
                risk_label = fc.get("risk_label", "—")
                forecast_count = fc.get("hotspot_count_tomorrow")

            popup_html = (
                f"<b>{PROVINCE_DISPLAY.get(prov_id, prov_id)}</b><br>"
                f"Risk besok: <span style='color:{color};font-weight:bold'>{risk_label}</span><br>"
                + (f"Prediksi hotspot: <b>{forecast_count:.0f}</b><br>"
                   if forecast_count is not None else "")
            )

            folium.Rectangle(
                bounds=bounds,
                color=color,
                weight=2,
                fill=False,
                dash_array="5, 5",
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{PROVINCE_DISPLAY.get(prov_id, prov_id)} — {risk_label}",
            ).add_to(province_group)
        province_group.add_to(m)

    # ---- Layer 2: Hotspot scatter (red dots) ------------------------------
    if not hotspots_df.empty:
        df = hotspots_df.copy()
        if len(df) > max_hotspots_to_render:
            LOG.info("Subsample hotspots %d → %d untuk performa",
                     len(df), max_hotspots_to_render)
            df = df.sample(n=max_hotspots_to_render, random_state=42)

        hotspot_group = folium.FeatureGroup(name="🔥 Hotspot", show=True)

        # Age-based opacity (recent = bright, old = faded)
        max_date = df["acq_date"].max()
        for _, row in df.iterrows():
            age_days = (max_date - row["acq_date"]).days
            opacity = max(0.4, 1.0 - age_days / 30)

            # FRP-scaled radius (3-12 px)
            frp_val = row.get("frp") or 5
            radius = max(3, min(12, frp_val / 8))

            popup_html = (
                f"<div style='font-family: monospace; font-size: 11px;'>"
                f"<b>{row.get('province_id', '?').upper()}</b><br>"
                f"Date: {row['acq_date'].strftime('%Y-%m-%d')}<br>"
                f"FRP: {row.get('frp', 0):.1f} MW<br>"
                f"Confidence: {row.get('confidence', '?')}<br>"
                f"Day/Night: {row.get('daynight', '?')}<br>"
                f"Coord: ({row['latitude']:.3f}, {row['longitude']:.3f})"
                f"</div>"
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color="#ff1744",        # bright red border
                fill=True,
                fill_color="#ff5722",   # orange-red fill
                fill_opacity=opacity,
                weight=1,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(hotspot_group)
        hotspot_group.add_to(m)

    # ---- Layer control --------------------------------------------------
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    return m
