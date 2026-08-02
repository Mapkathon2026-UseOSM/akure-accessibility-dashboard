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
optional, since the dashboard works normally without it, and not every
deployment will necessarily have re-run notebook 03 since this overlay
was added.

Visual design: see the CSS block below for the full token system
(palette, type, the concentric-ring signature motif used as section
dividers). Grounded in the subject rather than a generic dashboard
theme: the accent colors reference Akure's laterite-red roads and
soil, and Southwest Nigeria's adire indigo-dyeing tradition.

Run with (from repo root):
    streamlit run dashboard/app.py
"""

import json
import os
import sys

import geopandas as gpd
import pandas as pd
import streamlit as st
import leafmap.foliumap as leafmap

# Streamlit Cloud installs from dashboard/requirements.txt, not
# pyproject.toml, and that file has never actually pip-installed the
# local akure_access package (confirmed: it's absent from the full
# list of installed packages in the deploy logs). akure_access has
# only ever been importable there by accident of working directory,
# not because it was properly installed. Rather than depend on that
# continuing to work, explicitly add the repo root (this file's
# grandparent directory: dashboard/app.py -> dashboard/ -> repo root)
# to sys.path before importing anything from akure_access, so this
# works regardless of whether pip install ever ran for the local
# package, on Streamlit Cloud or anywhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from akure_access.insights import describe_interactive_view

st.set_page_config(page_title="Akure Access Dashboard", page_icon="◎", layout="wide")

# ============================================================
# Visual design system
# ------------------------------------------------------------
# Palette:
#   #141625  background      (adire indigo-black)
#   #1c1f33  panel/card       (a step lighter, for contrast)
#   #C4622D  primary accent   (laterite road/soil red-orange)
#   #4C9A8C  secondary accent (vegetation teal)
#   #E8B84B  highlight        (soft gold, for standout figures)
#   #F2EFE9  body text        (warm off-white)
# Type:
#   Space Grotesk  - headings, a technical/civic display face
#   IBM Plex Sans  - body copy, legible and unshowy
#   IBM Plex Mono  - data figures, ties to the coordinate/data nature
#                    of a GIS tool
# Signature motif:
#   Concentric rings, echoing the isochrone catchment rings that are
#   the actual visual/conceptual core of an accessibility study, used
#   as section dividers and the page icon.
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* Global font-size bump. Since Streamlit's own built-in widgets,
       labels, radio buttons, and dataframes all size themselves in rem
       (relative to this root value), increasing it here is what makes
       "everything else" bigger app-wide, not just our own custom CSS
       classes below. The hero title/subtitle are deliberately set in
       fixed px further down (not rem), specifically so they DON'T also
       get multiplied by this root increase, keeping their bump smaller
       and independent, as requested. */
    html {
        font-size: 118%;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #141625;
        color: #F2EFE9;
    }
    h1, h2, h3, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }
    code, .stDataFrame, [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Hero header */
    .hero-band {
        padding: 1.75rem 2rem 1.5rem 2rem;
        margin-bottom: 1.25rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #1c1f33 0%, #191c2d 100%);
        border: 1px solid rgba(196, 98, 45, 0.25);
    }
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        color: #F2EFE9;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-ring {
        color: #C4622D;
        font-size: 1.6rem;
    }
    .hero-sub {
        font-size: 18px;
        color: #b9b6ad;
        max-width: 62rem;
        line-height: 1.5;
    }

    /* Section divider: a thin gradient rule with a ring mark,
       standing in for st.subheader's default styling */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.6rem 0 1rem 0;
    }
    .section-divider .ring-mark {
        color: #C4622D;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .section-divider .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #F2EFE9;
        white-space: nowrap;
    }
    .section-divider .rule {
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(196,98,45,0.6), rgba(196,98,45,0));
    }

    /* Metric cards for the findings summary */
    .metric-card {
        background: #1c1f33;
        border: 1px solid rgba(76, 154, 140, 0.25);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }
    .metric-card .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #4C9A8C;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-figure {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 500;
        color: #E8B84B;
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-note {
        font-size: 0.88rem;
        color: #b9b6ad;
        line-height: 1.4;
    }
    .callout {
        background: rgba(76, 154, 140, 0.08);
        border-left: 3px solid #4C9A8C;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        font-size: 0.95rem;
        color: #d9d6cd;
        line-height: 1.5;
        margin-top: 0.75rem;
    }

    /* Checkbox / radio visibility fix. Streamlit's default BaseWeb
       widget styling uses a subtle, low-contrast border color for the
       unchecked box/circle, which was nearly invisible against this
       page's dark custom background, making both "Use colorblind-safe
       palette" and "Show 15/30/45-min walking catchments" look like
       plain unclickable text rather than actual checkboxes. This gives
       every checkbox/radio control an explicit, clearly visible
       accent-colored outline in both the checked and unchecked state. */
    [data-baseweb="checkbox"] div[role="checkbox"],
    [data-baseweb="radio"] div[role="radio"] {
        border: 2px solid #C4622D !important;
        background-color: transparent !important;
    }
    [data-baseweb="checkbox"] div[role="checkbox"][aria-checked="true"],
    [data-baseweb="radio"] div[role="radio"][aria-checked="true"] {
        background-color: #C4622D !important;
        border-color: #C4622D !important;
    }
    [data-baseweb="checkbox"] div[role="checkbox"] svg,
    [data-baseweb="radio"] div[role="radio"] svg {
        fill: #141625 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_divider(title: str):
    """Renders the concentric-ring section divider in place of st.subheader,
    keeping the signature motif consistent across all section breaks."""
    st.markdown(
        f"""
        <div class="section-divider">
            <span class="ring-mark">◎</span>
            <span class="section-title">{title}</span>
            <span class="rule"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Data loading
# ============================================================
DATA_PATHS = {
    "Akure North": "data/processed/akure_north/grid_access_scored.geojson",
    "Akure South": "data/processed/akure_south/grid_access_scored.geojson",
}

# Precomputed health-facility walking catchments (see
# notebooks/03_accessibility_analysis.ipynb, Section 5.2). This is an
# optional overlay, loaded only if the file exists, so the dashboard
# still works normally for an LGA where this notebook hasn't been
# re-run since this feature was added.
ISOCHRONE_PATHS = {
    "Akure North": "data/processed/akure_north/isochrones_health_walk.geojson",
    "Akure South": "data/processed/akure_south/isochrones_health_walk.geojson",
}

# Basemap options exposed in the UI, mapped to the provider names
# leafmap/xyzservices expects. All three are free, token-free tile
# sources (folium's Leaflet base, not Mapbox), so this toggle carries
# none of the credential concerns that applied to the kepler.gl exports
# in notebook 05.
BASEMAP_OPTIONS = {
    "Light (CartoDB Positron)": "CartoDB.Positron",
    "Streets (OpenStreetMap)": "OpenStreetMap",
    "Satellite (Esri)": "Esri.WorldImagery",
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
    doesn't, since this is an optional overlay, not a required input,
    so its absence should never block the rest of the dashboard from
    working normally.
    """
    frames = {}
    for lga, path in ISOCHRONE_PATHS.items():
        try:
            frames[lga] = gpd.read_file(path)
        except Exception:
            frames[lga] = None
    return frames


# ============================================================
# Hero header
# ============================================================
st.markdown(
    """
    <div class="hero-band">
        <div class="hero-title"><span class="hero-ring">◎</span>Mapping the Gap: Health &amp; Education Accessibility in Akure</div>
        <div class="hero-sub">An OSM-driven analysis of physical access to healthcare and education
        facilities across Akure North and Akure South LGAs, Ondo State, with an
        integrated check on where OSM's own data coverage may be shaping the
        picture.</div>
    </div>
    """,
    unsafe_allow_html=True,
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

col4, col5 = st.columns([1, 2])
with col4:
    basemap_choice = st.selectbox("Basemap", list(BASEMAP_OPTIONS.keys()))
with col5:
    colorblind_safe = st.checkbox(
        "Use colorblind-safe palette",
        value=False,
        help=(
            "Switches the map colors from the default red/yellow/green "
            "(hard to distinguish under red-green color blindness) to the "
            "Okabe-Ito palette for the deficit score, and viridis for the "
            "continuous time-based views."
        ),
    )

threshold_note = st.caption(
    "Underserved thresholds differ by mode, since a 30-minute walk covers far less ground "
    "than 30 minutes by okada or car, so each mode uses its own distance/time budget. "
    "Cells with no visible buildings are excluded from scoring."
)

isochrone_data = load_isochrones()
show_isochrones = st.checkbox(
    "Show 15/30/45-min walking catchments around health facilities",
    value=False,
    help=(
        "An illustrative overlay showing roughly how far someone can walk from "
        "each health facility within 15, 30, or 45 minutes: a convex-hull "
        "approximation, not the project's actual access-deficit scoring (which "
        "uses exact network routing; see the methodology notes for detail)."
    ),
)

if show_isochrones:
    # Check availability for whichever LGA(s) are actually in view right
    # now, and say so immediately, next to the checkbox itself, rather
    # than only via a small caption buried under a 600px map further
    # down the page, which is easy to miss entirely, this file is an
    # OPTIONAL output of notebook 03 Section 5.2, so it's expected to be
    # absent unless that section has specifically been run.
    lgas_to_check = available_lgas if lga_choice == "Both (compare)" else [lga_choice]
    missing_isochrones = [
        lga for lga in lgas_to_check
        if isochrone_data.get(lga) is None or isochrone_data.get(lga).empty
    ]
    if missing_isochrones:
        st.warning(
            f"No precomputed walking catchments found for: {', '.join(missing_isochrones)}. "
            "This overlay needs `isochrones_health_walk.geojson`, an optional output of "
            "Notebook 03, Section 5.2, run that section for the LGA(s) above to enable it."
        )


MODE_LABELS = {"walk": "Walking", "okada": "Okada", "drive": "Driving"}

# Standard (traffic-light) vs colorblind-safe palette for the discrete
# 0/1/2 access-deficit score. The colorblind-safe option uses the
# Okabe-Ito palette, specifically designed to remain distinguishable
# under the common forms of red-green color blindness (deuteranopia,
# protanopia), which the default traffic-light red/yellow/green is not.
DEFICIT_COLORS = {
    "standard": ["#2ECC71", "#F1C40F", "#C0392B"],
    "colorblind_safe": ["#0072B2", "#E69F00", "#D55E00"],
}
DEFICIT_LABELS = ["Well served", "Underserved (1 service)", "Underserved (both services)"]

# Continuous (minutes-to-nearest-facility) legend: a sequential warm
# ramp by default, or viridis (perceptually uniform, colorblind-safe)
# as the alternative.
CONTINUOUS_CMAP = {"standard": "YlOrRd", "colorblind_safe": "viridis"}


def score_column(view, mode):
    suffix = f"_{mode}"
    if view == "Health only":
        col = f"health_time_min{suffix}"
    elif view == "Education only":
        col = f"education_time_min{suffix}"
    else:
        col = f"{mode}_access_deficit_score"
    return col


def render_map(gdf, view, mode, isochrones_gdf=None, show_isochrones=False,
                basemap="CartoDB.Positron", colorblind_safe=False):
    m = leafmap.Map()
    m.add_basemap(basemap)

    col = score_column(view, mode)
    settled = gdf[gdf["building_count"] > 0]
    palette_key = "colorblind_safe" if colorblind_safe else "standard"

    if not settled.empty and col in settled.columns:
        if view == "Combined":
            # Discrete 0/1/2 score: explicit categorical colors/labels,
            # rather than a numeric quantile legend that would show raw
            # score values with no explanation of what they mean.
            m.add_data(
                settled,
                column=col,
                scheme="UserDefined",
                classification_kwds={"bins": [0, 1, 2]},
                colors=DEFICIT_COLORS[palette_key],
                labels=DEFICIT_LABELS,
                legend_title=f"Access Deficit ({MODE_LABELS[mode]})",
                layer_name=f"{view} access ({mode})",
            )
        else:
            service = "Health" if view == "Health only" else "Education"
            m.add_data(
                settled,
                column=col,
                cmap=CONTINUOUS_CMAP[palette_key],
                legend_title=f"{service} Access Time, {MODE_LABELS[mode]} (min)",
                layer_name=f"{view} access ({mode})",
            )
    elif col not in settled.columns:
        st.info(f"Column '{col}' not found. Re-run notebook 03 with modes including '{mode}'.")

    if show_isochrones and isochrones_gdf is not None and not isochrones_gdf.empty:
        # Reproject to WGS84 for web-map display, matching the same
        # one-way CRS conversion done for the kepler.gl exports in
        # notebook 05. This overlay is for visualization only; no
        # further analysis happens on it here.
        isochrones_wgs84 = isochrones_gdf.to_crs("EPSG:4326")
        m.add_data(
            isochrones_wgs84,
            column="trip_time_min",
            scheme="UserDefined",
            classification_kwds={"bins": [15, 30, 45]},
            colors=["#B3D9FF", "#5B9BD5", "#1F4E79"] if not colorblind_safe else ["#9AD1D4", "#3D8B95", "#0B4F55"],
            labels=["Within 15 min", "Within 30 min", "Within 45 min"],
            legend_title="Health Facility Walking Catchment",
            layer_name="Health facility walking catchments",
        )
    elif show_isochrones and (isochrones_gdf is None or isochrones_gdf.empty):
        st.info(
            "No precomputed walking catchments found for this LGA. "
            "Re-run notebook 03 (Section 5.2) to generate them."
        )

    m.to_streamlit(height=600)


section_divider("Access map")
st.caption(
    "Combined view colors cells by deficit score (green = well served, "
    "amber/red = underserved). Health/Education-only views show a continuous "
    "gradient of travel time in minutes. Legend appears bottom-right of the map."
)
if lga_choice == "Both (compare)":
    # Side-by-side columns, not tabs: "compare" implies seeing both
    # LGAs at once, tabs would only let you switch between them one at
    # a time, which isn't actually a comparison view.
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown(f"**{available_lgas[0]}**")
        render_map(
            data[available_lgas[0]], view_choice, mode_choice,
            isochrones_gdf=isochrone_data.get(available_lgas[0]),
            show_isochrones=show_isochrones,
            basemap=BASEMAP_OPTIONS[basemap_choice],
            colorblind_safe=colorblind_safe,
        )
        st.markdown(
            f'<div class="callout">{describe_interactive_view(data[available_lgas[0]], available_lgas[0], mode_choice, view_choice)}</div>',
            unsafe_allow_html=True,
        )
    with map_col2:
        st.markdown(f"**{available_lgas[1]}**")
        render_map(
            data[available_lgas[1]], view_choice, mode_choice,
            isochrones_gdf=isochrone_data.get(available_lgas[1]),
            show_isochrones=show_isochrones,
            basemap=BASEMAP_OPTIONS[basemap_choice],
            colorblind_safe=colorblind_safe,
        )
        st.markdown(
            f'<div class="callout">{describe_interactive_view(data[available_lgas[1]], available_lgas[1], mode_choice, view_choice)}</div>',
            unsafe_allow_html=True,
        )
else:
    render_map(
        data[lga_choice], view_choice, mode_choice,
        isochrones_gdf=isochrone_data.get(lga_choice),
        show_isochrones=show_isochrones,
        basemap=BASEMAP_OPTIONS[basemap_choice],
        colorblind_safe=colorblind_safe,
    )
    # Dynamic map description and inferred result: recomputed straight
    # from the currently selected LGA/mode/view every time any of those
    # change (Streamlit reruns this script on every widget interaction),
    # so the caption shown always matches exactly what the map above is
    # currently displaying, rather than a fixed caption written once
    # that could silently drift out of sync with the actual selection.
    st.markdown(
        f'<div class="callout">{describe_interactive_view(data[lga_choice], lga_choice, mode_choice, view_choice)}</div>',
        unsafe_allow_html=True,
    )

section_divider("Most underserved settlements")
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
    # Human-readable column headers and rounded figures, rather than raw
    # column names (e.g. "health_time_min_walk") and five-decimal floats.
    friendly_names = {
        "lga": "LGA",
        "cell_id": "Cell ID",
        f"health_time_min_{mode_choice}": "Health time (min)",
        f"health_distance_km_{mode_choice}": "Health distance (km)",
        f"education_time_min_{mode_choice}": "Education time (min)",
        f"education_distance_km_{mode_choice}": "Education distance (km)",
        deficit_col: "Deficit score (0-2)",
    }
    display_df = ranked[display_cols].head(15).round(1).rename(columns=friendly_names)
    st.dataframe(display_df, use_container_width=True)
    st.caption(
        "Deficit score: **0** = well served, **1** = underserved for one service "
        "(health or education), **2** = underserved for both."
    )
else:
    st.info(f"No scored data found for mode '{mode_choice}' yet. Re-run notebook 03 with this mode included.")

section_divider("Findings summary")

# Match the same LGA-selection behavior as "Most underserved settlements"
# above: respect lga_choice rather than always combining every LGA. Before
# this fix, this section silently ignored lga_choice and always showed a
# combined North+South statistic, so switching the LGA tab visibly updated
# the map and table above but left these percentage cards unchanged, a
# confusing inconsistency for anyone comparing the two sections side by side.
settled_frames = (
    [data[l][data[l]["building_count"] > 0] for l in available_lgas]
    if lga_choice == "Both (compare)"
    else [data[lga_choice][data[lga_choice]["building_count"] > 0]]
)
all_settled = pd.concat(settled_frames, ignore_index=True)

summary_scope_label = "Combined (Akure North + Akure South)" if lga_choice == "Both (compare)" else lga_choice
st.caption(f"Scope: {summary_scope_label}")

if all_settled.empty:
    st.info("No settled grid cells found in the loaded data.")
else:
    # Cross-mode comparison, computed live from whatever modes are present
    mode_stats = []
    for m in ["walk", "okada", "drive"]:
        col = f"{m}_access_deficit_score"
        if col in all_settled.columns:
            pct_any = 100 * (all_settled[col] > 0).mean()
            pct_both = 100 * (all_settled[col] == 2).mean()
            mode_stats.append((m, pct_any, pct_both))

    if mode_stats:
        mode_labels = {"walk": "Walking", "okada": "Okada", "drive": "Driving"}
        cards = st.columns(len(mode_stats))
        for card_col, (m, pct_any, pct_both) in zip(cards, mode_stats):
            with card_col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{mode_labels[m]}</div>
                        <div class="metric-figure">{pct_any:.1f}%</div>
                        <div class="metric-note">underserved for at least one service<br>
                        {pct_both:.1f}% underserved for both</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if len(mode_stats) > 1:
            walk_pct = next((p for m, p, _ in mode_stats if m == "walk"), None)
            fastest_pct = min(p for m, p, _ in mode_stats if m != "walk") if len(mode_stats) > 1 else None
            if walk_pct is not None and fastest_pct is not None:
                gap = walk_pct - fastest_pct
                st.markdown(
                    f"""
                    <div class="callout">
                    Walking-only analysis would overstate underserved communities by roughly
                    <strong>{gap:.0f} percentage points</strong> compared to okada/driving access,
                    a key reason this project models all three modes rather than walking distance alone.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Completeness cross-check, computed live
    if "health_completeness_flag" in all_settled.columns and "walk_access_deficit_score" in all_settled.columns:
        walk_underserved = all_settled[all_settled["walk_access_deficit_score"] > 0]
        if len(walk_underserved) > 0:
            pct_health_gap = 100 * walk_underserved["health_completeness_flag"].mean()
            pct_edu_gap = 100 * walk_underserved["education_completeness_flag"].mean()
            st.markdown(
                f"""
                <div class="callout">
                <strong>Completeness caveat:</strong> among walking-underserved cells,
                {pct_health_gap:.1f}% also carry a possible health-facility OSM data gap, and
                {pct_edu_gap:.1f}% carry a possible education-facility data gap. Some portion of
                the underserved findings above may reflect incomplete OSM tagging rather than a
                confirmed absence of nearby facilities. See the Methodology tab / README for detail.
                </div>
                """,
                unsafe_allow_html=True,
            )

section_divider("Static / publication maps")
st.markdown(
    "Downloadable, report-ready versions of these maps and charts, with an OSM basemap, "
    "gridlines, north arrow, scale bar, and legend/colorbar, styled to match this project's "
    "cartographic reference standard."
)

VISUALS_DIR = "visuals"
static_scope_lgas = available_lgas if lga_choice == "Both (compare)" else [lga_choice]

# Each LGA folder produced by Notebook 03 Section 6.2 contains print-
# quality (300dpi) figures at the top level, plus a lighter web/
# subfolder (150dpi by default) meant specifically for in-app display,
# generated from the SAME rendered figure so the two tiers always
# show identical content, just different resolution. Preferring web/
# here means this page loads faster; falling back to the top-level
# files keeps this working against older runs generated before the
# web/ subfolder existed.
static_files_found = {}  # {lga: (display_dir, [filenames])}
static_captions = {}  # {lga: {filename: caption_text}}
for lga in static_scope_lgas:
    lga_dir = os.path.join(VISUALS_DIR, lga.replace(" ", "_"))
    web_dir = os.path.join(lga_dir, "web")
    display_dir = web_dir if os.path.isdir(web_dir) else lga_dir
    if os.path.isdir(display_dir):
        files = sorted(
            f for f in os.listdir(display_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if files:
            static_files_found[lga] = (display_dir, files)

    # captions.json is always written to the top-level LGA folder (not
    # the web/ subfolder) by generate_all_static_outputs, but its keys
    # are plain filenames shared by both the print and web tiers (same
    # figure, same filename, different folder/resolution), so this
    # lookup works correctly regardless of which tier is being displayed.
    captions_path = os.path.join(lga_dir, "captions.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                static_captions[lga] = json.load(f)
        except (json.JSONDecodeError, OSError):
            static_captions[lga] = {}

if not static_files_found:
    st.info(
        "No static maps found yet. Run Notebook 03, Section 6.2 "
        "(`generate_all_static_outputs`) to produce them, they'll be picked up here "
        f"automatically from `{VISUALS_DIR}/{{lga_name}}/` once generated."
    )
else:
    # Group filenames by figure type (deficit / health time / education time /
    # completeness / mode comparison) so related figures across LGAs and modes
    # sit together, rather than one long alphabetical list.
    def _categorize(filename: str) -> str:
        if "mode_comparison" in filename:
            return "Mode comparison"
        if "completeness" in filename:
            return "Data completeness"
        if "health_time" in filename:
            return "Health access time"
        if "education_time" in filename:
            return "Education access time"
        if "deficit" in filename:
            return "Access deficit"
        return "Other"

    category_order = [
        "Access deficit", "Health access time", "Education access time",
        "Data completeness", "Mode comparison", "Other",
    ]

    for lga in static_scope_lgas:
        if lga not in static_files_found:
            continue
        display_dir, files = static_files_found[lga]
        st.markdown(f"**{lga}**")

        by_category = {}
        for fname in files:
            by_category.setdefault(_categorize(fname), []).append(fname)

        for category in category_order:
            cat_files = by_category.get(category)
            if not cat_files:
                continue
            with st.expander(f"{category} ({len(cat_files)})", expanded=(category == "Access deficit")):
                # Always allocate a fixed 3-column grid, regardless of how
                # many images are in this category. Using
                # min(3, len(cat_files)) here previously meant a category
                # with only ONE image (e.g. "Mode comparison") got a
                # single full-width column, stretching that one chart to
                # fill the whole page while every other category's images
                # stayed a normal, consistent size.
                cols = st.columns(3)
                lga_captions = static_captions.get(lga, {})
                for idx, fname in enumerate(cat_files):
                    with cols[idx % 3]:
                        st.image(os.path.join(display_dir, fname), use_container_width=True)
                        caption_text = lga_captions.get(fname)
                        if not caption_text:
                            # Backward compatibility: older runs generated
                            # before captions.json existed won't have a
                            # real caption available, fall back to a
                            # readable label rather than showing nothing.
                            caption_text = os.path.splitext(fname)[0].replace("_", " ")
                        st.caption(caption_text)

    zip_path = os.path.join(VISUALS_DIR, "akure_access_static_maps.zip")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button(
                "Download all static maps (ZIP)", data=f, file_name="akure_access_static_maps.zip",
                mime="application/zip",
            )

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
        <a href="https://github.com/Mapkathon2026-UseOSM/akure-accessibility-dashboard"
           target="_blank" rel="noopener noreferrer"
           style="display: flex; align-items: center; gap: 0.5rem; color: #b9b6ad; text-decoration: none;">
            <svg height="20" width="20" viewBox="0 0 16 16" fill="#b9b6ad" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
                1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            <span>View the full source code and project history on GitHub</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
