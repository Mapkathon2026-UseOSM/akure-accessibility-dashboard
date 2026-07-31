"""
dashboard/app.py

Streamlit dashboard presenting the health/education accessibility
analysis for Akure North and Akure South LGAs.

Expects pre-computed outputs from notebooks 01-04, specifically:
    data/processed/akure_north/grid_access_scored.geojson
    data/processed/akure_south/grid_access_scored.geojson

Optionally uses, if present (notebook 03, Section 5.2):
    data/processed/{lga}/isochrones_health_walk.geojson
This powers the "walking catchments" overlay toggle. It is genuinely
optional -- the dashboard works normally without it, since not every
deployment will necessarily have re-run notebook 03 since this overlay
was added.

Run with (from repo root):
    streamlit run dashboard/app.py
"""

import geopandas as gpd
import pandas as pd
import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(page_title="Akure Access Dashboard", layout="wide")

DATA_PATHS = {
    "Akure North": "data/processed/akure_north/grid_access_scored.geojson",
    "Akure South": "data/processed/akure_south/grid_access_scored.geojson",
}

# Precomputed health-facility walking catchments (see
# notebooks/03_accessibility_analysis.ipynb, Section 5.2). This is an
# OPTIONAL overlay -- loaded only if the file exists, so the dashboard
# still works normally for an LGA where this notebook hasn't been
# re-run since this feature was added.
ISOCHRONE_PATHS = {
    "Akure North": "data/processed/akure_north/isochrones_health_walk.geojson",
    "Akure South": "data/processed/akure_south/isochrones_health_walk.geojson",
}


@st.cache_data
def load_data():
    frames = {}
    for lga, path in DATA_PATHS.items():
        try:
            gdf = gpd.read_file(path)
            gdf["lga"] = lga
            frames[lga] = gdf
        except Exception:
            frames[lga] = None
    return frames


@st.cache_data
def load_isochrones():
    """
    Loads precomputed health-facility walking catchments per LGA, if
    the file exists. Returns None (not an error) for any LGA where it
    doesn't -- this is an optional overlay, not a required input, so
    its absence should never block the rest of the dashboard from
    working normally.
    """
    frames = {}
    for lga, path in ISOCHRONE_PATHS.items():
        try:
            frames[lga] = gpd.read_file(path)
        except Exception:
            frames[lga] = None
    return frames


st.title("Mapping the Gap: Health & Education Accessibility in Akure")
st.write(
    "An OSM-driven analysis of physical access to healthcare and education "
    "facilities across Akure North and Akure South LGAs, Ondo State, with an "
    "integrated check on where OSM's own data coverage may be shaping the "
    "picture."
)

data = load_data()
missing = [lga for lga, gdf in data.items() if gdf is None]
if missing:
    st.warning(
        f"No processed data found yet for: {', '.join(missing)}. "
        "Run notebooks 01-03 for that LGA first."
    )

available_lgas = [lga for lga, gdf in data.items() if gdf is not None]
if not available_lgas:
    st.stop()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    lga_choice = st.selectbox("Study area", available_lgas + (["Both (compare)"] if len(available_lgas) == 2 else []))
with col2:
    view_choice = st.radio("Access view", ["Combined", "Health only", "Education only"], horizontal=True)
with col3:
    mode_choice = st.selectbox(
        "Transport mode",
        ["walk", "okada", "drive"],
        format_func=lambda m: {"walk": "Walking", "okada": "Okada (motorcycle)", "drive": "Private/shared vehicle"}[m],
    )

threshold_note = st.caption(
    "Underserved thresholds differ by mode: a 30-minute walk covers far less ground "
    "than 30 minutes by okada or car, so each mode uses its own distance/time budget. "
    "Cells with no visible buildings are excluded from scoring."
)

isochrone_data = load_isochrones()
show_isochrones = st.checkbox(
    "Show 15/30/45-min walking catchments around health facilities",
    value=False,
    help=(
        "An illustrative overlay showing roughly how far someone can walk from "
        "each health facility within 15, 30, or 45 minutes — a convex-hull "
        "approximation, not the project's actual access-deficit scoring (which "
        "uses exact network routing; see the methodology notes for detail)."
    ),
)


def score_column(view, mode):
    suffix = f"_{mode}"
    if view == "Health only":
        col = f"health_time_min{suffix}"
    elif view == "Education only":
        col = f"education_time_min{suffix}"
    else:
        col = f"{mode}_access_deficit_score"
    return col


def render_map(gdf, view, mode, isochrones_gdf=None, show_isochrones=False):
    m = leafmap.Map()
    col = score_column(view, mode)
    settled = gdf[gdf["building_count"] > 0]
    if not settled.empty and col in settled.columns:
        m.add_data(
            settled,
            column=col,
            cmap="RdYlGn_r",
            legend_title=col,
            layer_name=f"{view} access ({mode})",
        )
    elif col not in settled.columns:
        st.info(f"Column '{col}' not found — re-run notebook 03 with modes including '{mode}'.")

    if show_isochrones and isochrones_gdf is not None and not isochrones_gdf.empty:
        # Reproject to WGS84 for web-map display, matching the same
        # one-way CRS conversion done for the kepler.gl exports in
        # notebook 05 -- this overlay is for visualization only, no
        # further analysis happens on it here.
        isochrones_wgs84 = isochrones_gdf.to_crs("EPSG:4326")
        m.add_data(
            isochrones_wgs84,
            column="trip_time_min",
            cmap="Blues",
            legend_title="Walking catchment (min)",
            layer_name="Health facility walking catchments",
        )
    elif show_isochrones and (isochrones_gdf is None or isochrones_gdf.empty):
        st.info(
            "No precomputed walking catchments found for this LGA — "
            "re-run notebook 03 (Section 5.2) to generate them."
        )

    m.to_streamlit(height=600)


st.subheader("Access map")
if lga_choice == "Both (compare)":
    tab1, tab2 = st.tabs(available_lgas)
    with tab1:
        render_map(
            data[available_lgas[0]], view_choice, mode_choice,
            isochrones_gdf=isochrone_data.get(available_lgas[0]),
            show_isochrones=show_isochrones,
        )
    with tab2:
        render_map(
            data[available_lgas[1]], view_choice, mode_choice,
            isochrones_gdf=isochrone_data.get(available_lgas[1]),
            show_isochrones=show_isochrones,
        )
else:
    render_map(
        data[lga_choice], view_choice, mode_choice,
        isochrones_gdf=isochrone_data.get(lga_choice),
        show_isochrones=show_isochrones,
    )

st.subheader("Most underserved settlements")
frames_to_rank = (
    [data[l] for l in available_lgas] if lga_choice == "Both (compare)" else [data[lga_choice]]
)
combined = pd.concat(frames_to_rank, ignore_index=True)
deficit_col = f"{mode_choice}_access_deficit_score"
if deficit_col in combined.columns:
    ranked = combined[combined[deficit_col] > 0].sort_values(deficit_col, ascending=False)
    display_cols = ["lga", "cell_id"] + [
        c for c in [
            f"health_time_min_{mode_choice}", f"health_distance_km_{mode_choice}",
            f"education_time_min_{mode_choice}", f"education_distance_km_{mode_choice}",
            deficit_col,
        ] if c in ranked.columns
    ]
    st.dataframe(ranked[display_cols].head(15), use_container_width=True)
else:
    st.info(f"No scored data found for mode '{mode_choice}' yet — re-run notebook 03 with this mode included.")

st.subheader("Findings summary")

all_settled = pd.concat(
    [data[l][data[l]["building_count"] > 0] for l in available_lgas],
    ignore_index=True,
)

if all_settled.empty:
    st.info("No settled grid cells found in the loaded data.")
else:
    summary_lines = []

    # Cross-mode comparison, computed live from whatever modes are present
    mode_stats = []
    for m in ["walk", "okada", "drive"]:
        col = f"{m}_access_deficit_score"
        if col in all_settled.columns:
            pct_any = 100 * (all_settled[col] > 0).mean()
            pct_both = 100 * (all_settled[col] == 2).mean()
            mode_stats.append((m, pct_any, pct_both))

    if mode_stats:
        summary_lines.append("**Underserved rate by transport mode** (across all loaded study areas):")
        for m, pct_any, pct_both in mode_stats:
            summary_lines.append(
                f"- **{m.capitalize()}**: {pct_any:.1f}% underserved for at least one service, "
                f"{pct_both:.1f}% underserved for both"
            )
        if len(mode_stats) > 1:
            walk_pct = next((p for m, p, _ in mode_stats if m == "walk"), None)
            fastest_pct = min(p for m, p, _ in mode_stats if m != "walk") if len(mode_stats) > 1 else None
            if walk_pct is not None and fastest_pct is not None:
                gap = walk_pct - fastest_pct
                summary_lines.append(
                    f"\nWalking-only analysis would overstate underserved communities by roughly "
                    f"**{gap:.0f} percentage points** compared to okada/driving access — a key reason "
                    f"this project models all three modes rather than walking distance alone."
                )

    # Completeness cross-check, computed live
    if "health_completeness_flag" in all_settled.columns and "walk_access_deficit_score" in all_settled.columns:
        walk_underserved = all_settled[all_settled["walk_access_deficit_score"] > 0]
        if len(walk_underserved) > 0:
            pct_health_gap = 100 * walk_underserved["health_completeness_flag"].mean()
            pct_edu_gap = 100 * walk_underserved["education_completeness_flag"].mean()
            summary_lines.append(
                f"\n**Completeness caveat:** among walking-underserved cells, "
                f"{pct_health_gap:.1f}% also carry a possible health-facility OSM data gap, and "
                f"{pct_edu_gap:.1f}% carry a possible education-facility data gap. Some portion of "
                f"the underserved findings above may reflect incomplete OSM tagging rather than a "
                f"confirmed absence of nearby facilities — see the Methodology tab / README for detail."
            )

    st.markdown("\n".join(summary_lines))

st.caption(
    "See the accompanying ArcGIS StoryMap and written project report for the full narrative "
    "writeup and methodology detail."
)
