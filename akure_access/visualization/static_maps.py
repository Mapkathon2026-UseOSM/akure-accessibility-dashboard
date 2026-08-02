"""
akure_access.visualization.static_maps

Publication-styled static maps and charts for the Akure accessibility
analysis: OSM basemap, lat/long gridlines, north arrow, scale bar, and
legend/colorbar.

WHY THIS MODULE EXISTS, AND WHY IT'S VECTOR-BASED, NOT RASTER
----------------------------------------------------------------------
Standard reference-map styling (OSM basemap, gridlines, north arrow,
scale bar, legend) is normally built for single/multi-band *rasters*
(e.g. satellite imagery, rasterio + a GeoTIFF per layer). This
project's accessibility outputs are *vector* data instead: a polygon
grid (`grid_access_scored.geojson`) with per-cell scores, plus point/
line facility and road layers. This module ports the cartographic
STYLING conventions (gridlines, north arrow, scale bar, OSM basemap,
legend/colorbar placement, dpi/format choices) to vector plotting, via
`gdf.plot()` (GeoPandas' matplotlib-based vector plotting) rather than
`ax.imshow()`.

Every plotting function in this module accepts a GeoDataFrame in
EPSG:4326 (lat/lon) and internally reprojects to Web Mercator
(EPSG:3857) only for the OSM basemap layer via contextily, using
`contextily.add_basemap(..., crs='EPSG:4326')` to reproject OSM tiles
on the fly rather than reprojecting the data. This keeps every
function's public interface in the same CRS the rest of
`akure_access` already standardizes on for grid/scoring work.

REQUIRES LIVE INTERNET ACCESS (OSM tile servers) TO SHOW A BASEMAP.
In an offline/sandboxed environment, `add_osm_basemap()` degrades
gracefully to a plain light-gray background with a small on-figure
note, rather than raising, so the rest of the map (data, gridlines,
legend, scale bar) still renders and the function is still safe to
call from automated/CI contexts.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    import contextily as ctx
    _HAS_CONTEXTILY = True
except ImportError:  # pragma: no cover - contextily is an optional/heavy dependency
    _HAS_CONTEXTILY = False


# Reuses the exact same palette as the Streamlit dashboard's discrete
# access-deficit legend (dashboard/app.py: DEFICIT_PALETTES / DEFICIT_LABELS),
# so a static JPEG and the interactive map always agree on what "amber"
# or "red" means, rather than each maintaining its own independent
# color scheme that could silently drift apart.
DEFICIT_PALETTES = {
    "standard": ["#2ECC71", "#F1C40F", "#C0392B"],
    "colorblind_safe": ["#0072B2", "#E69F00", "#D55E00"],
}
DEFICIT_LABELS = ["Well served", "Underserved (1 service)", "Underserved (both services)"]

# Brand accent colors from the dashboard's own theme (dashboard/app.py),
# reused here so charts/maps produced by the notebook visually match the
# Streamlit app they'll sit alongside, rather than looking like a
# separate, disconnected artifact.
ACCENT_PRIMARY = "#C4622D"    # laterite road/soil red-orange
ACCENT_SECONDARY = "#4C9A8C"  # vegetation teal
ACCENT_HIGHLIGHT = "#E8B84B"  # soft gold


# ---------------------------------------------------------------------------
# Cartographic building blocks: add_north_arrow, add_scale_bar, add_gridlines,
# add_osm_basemap
# ---------------------------------------------------------------------------


def _ensure_lonlat(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reprojects to EPSG:4326 (lon/lat) if not already there.

    The rest of this module (gridline spacing, scale-bar km-per-degree
    math, the OSM basemap's crs= argument) all assume geographic
    coordinates, matching standard reference-map convention. But
    akure_access.accessibility deliberately reprojects everything to a
    metric UTM CRS for correct distance/area math (see clean.py's and
    scoring.py's own CRS-handling docstrings), so the real
    grid_access_scored.geojson this module actually receives is in
    EPSG:32631 (or similar), not EPSG:4326. Without this conversion,
    gridline/scale-bar spacing computed as if UTM meters were degrees
    would try to draw tens of thousands of gridlines across a ~40,000
    (meters) span and hang, exactly the failure caught while testing
    this module against real Akure North data.
    """
    if gdf.crs is None:
        warnings.warn("Input GeoDataFrame has no CRS set; assuming EPSG:4326.")
        return gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        return gdf.to_crs("EPSG:4326")
    return gdf


def add_north_arrow(ax, xy=(0.94, 0.94)):
    """Simple 'N' arrow in the upper-right of the axes, same placement
    convention used throughout this module."""
    ax.annotate(
        "N", xy=xy, xytext=(xy[0], xy[1] - 0.07),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(facecolor="black", width=4, headwidth=12),
        ha="center", fontsize=13, fontweight="bold", zorder=15,
    )


def add_scale_bar(ax, bounds_lonlat, length_km=None, xy=(0.05, 0.05)):
    """
    Draws a ground-truth scale bar in the lower-left of the map.

    Since the axes are in
    geographic degrees (lon/lat), a scale bar can't just be "N pixels
    long" the way it could in a projected CRS, the physical length of
    one degree of longitude varies with latitude (it shrinks toward the
    poles). So this converts a target km length into the correct number
    of *degrees* at the map's actual mid-latitude, using the standard
    111.32 km/degree-of-latitude constant scaled by cos(latitude) for
    longitude. This is an approximation (true for a sphere, not an
    ellipsoid), fine at map-reading scale, but not survey-grade.

    If `length_km` isn't given, picks a "nice" round number sized to
    roughly a fifth of the map's width, similar to how QGIS/ArcGIS
    auto-scale bars behave, so grid cells at very different physical
    scales (a single LGA vs. a wide comparison view) each get a
    sensibly-sized bar rather than one hardcoded length.
    """
    west, south, east, north = bounds_lonlat
    lat_mid = (south + north) / 2
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mid))
    width_km = (east - west) * km_per_deg_lon

    if length_km is None:
        target = width_km / 5
        # Round to a "nice" number: 1, 2, 5, 10, 20, 50, 100...
        magnitude = 10 ** np.floor(np.log10(max(target, 0.1)))
        for m in (1, 2, 5, 10):
            candidate = m * magnitude
            if candidate >= target:
                length_km = candidate
                break
        else:
            length_km = 10 * magnitude

    bar_deg = length_km / km_per_deg_lon
    x0 = west + (east - west) * xy[0]
    y0 = south + (north - south) * xy[1]
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="black", linewidth=3,
             transform=ax.transData, zorder=15, solid_capstyle="butt")
    ax.text(x0 + bar_deg / 2, y0 + (north - south) * 0.015,
             f"{length_km:g} km", ha="center", fontsize=9, zorder=15)


def add_gridlines(ax, bounds_lonlat, interval=None, color="gray", alpha=0.6, fontsize=9,
                   label_sides=("left", "bottom")):
    """
    Lat/long graticule with edge labels, styled after standard printed
    reference maps: coordinate labels appear on the LEFT (latitude) and
    BOTTOM (longitude) only, not mirrored on all four sides. Mirrored
    labels (top+bottom, left+right showing the same values twice) are a
    plotting-library default, not a cartographic convention, standard
    topographic and reference maps label each axis once. `interval` is
    auto-picked from the map's extent if not given, since an LGA-scale
    map (a few km wide) needs a much finer graticule (e.g. 0.02 degrees)
    than a continental-scale map's typical 1.0-degree spacing would give,
    a fixed 1.0 degree interval would draw zero or one gridline across
    an entire LGA and be useless.
    """
    west, south, east, north = bounds_lonlat
    if interval is None:
        span = max(east - west, north - south)
        # Pick a "nice" interval so grid density scales sensibly across
        # very different map sizes (single ward vs. full LGA vs. both
        # LGAs side by side).
        for candidate in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            if span / candidate <= 8:
                interval = candidate
                break
        else:
            interval = 1.0

    lons = np.arange(np.floor(west / interval) * interval, east + interval, interval)
    lats = np.arange(np.floor(south / interval) * interval, north + interval, interval)

    for lon in lons:
        ax.axvline(lon, color=color, linewidth=0.5, linestyle=":", alpha=alpha, zorder=5)
    for lat in lats:
        ax.axhline(lat, color=color, linewidth=0.5, linestyle=":", alpha=alpha, zorder=5)

    def _fmt_lon(l):
        return f"{abs(l):.3f}\u00b0{'E' if l >= 0 else 'W'}"

    def _fmt_lat(l):
        return f"{abs(l):.3f}\u00b0{'N' if l >= 0 else 'S'}"

    ax.set_xticks(lons)
    ax.set_yticks(lats)
    ax.set_xticklabels([_fmt_lon(l) for l in lons], fontsize=fontsize, rotation=30)
    ax.set_yticklabels([_fmt_lat(l) for l in lats], fontsize=fontsize)

    # Standard single-side labeling: ticks (short marks) can still show
    # on all four sides for a finished, "boxed" map frame, but the text
    # labels themselves are drawn only once per axis (left + bottom),
    # not duplicated on the opposite side.
    ax.tick_params(
        direction="out", length=4,
        top="top" in label_sides, labeltop="top" in label_sides,
        bottom=True, labelbottom="bottom" in label_sides,
        right="right" in label_sides, labelright="right" in label_sides,
        left=True, labelleft="left" in label_sides,
    )


def add_osm_basemap(ax, crs="EPSG:4326", timeout=8):
    """
    Adds OpenStreetMap tiles behind the data (zorder=0), reprojecting
    tiles on the fly to match `crs` rather than reprojecting the data
    into Web Mercator first.

    Degrades gracefully (light-gray background + small note) rather
    than raising if contextily isn't installed or tiles can't be
    fetched (e.g. no internet access), so callers in automated/CI
    contexts still get a usable figure with everything except the
    basemap itself. `timeout` (seconds) keeps a blocked/unreachable
    tile server from hanging the whole map-generation run, rather than
    contextily's default of retrying indefinitely.
    """
    if not _HAS_CONTEXTILY:
        ax.set_facecolor("#e8e8e8")
        ax.text(0.5, 0.5, "OSM basemap unavailable\n(contextily not installed)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888", zorder=1)
        return
    try:
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik,
                         zorder=0, attribution_size=6, timeout=timeout)
    except Exception as exc:  # pragma: no cover - network-dependent
        warnings.warn(
            f"OSM basemap could not be fetched ({exc!r}); falling back to a plain "
            f"background. This is expected in offline/sandboxed environments, in "
            f"Colab or Streamlit Cloud (both have internet access) this will "
            f"normally succeed."
        )
        ax.set_facecolor("#e8e8e8")


def _figure_bounds(gdf: gpd.GeoDataFrame, pad_frac: float = 0.04):
    """Bounding box in lon/lat with a small padding margin so the data
    doesn't touch the frame edge, matching standard reference-map use
    of the raster's own bounds as the map extent."""
    west, south, east, north = gdf.total_bounds
    dx, dy = (east - west) * pad_frac, (north - south) * pad_frac
    return (west - dx, south - dy, east + dx, north + dy)


def _finalize_and_save(fig, ax, bounds, title, out_path, scale_bar_km, dpi, web_path=None, web_dpi=None):
    """
    Finalizes a map figure (gridlines, north arrow, scale bar, title,
    extent) and saves it, optionally saving a SECOND, lower-resolution
    copy from the exact same already-rendered figure.

    The second save is nearly free: the expensive parts (fetching OSM
    basemap tiles, plotting the data layer, computing gridlines) have
    already happened once by the time this runs, a second `savefig()`
    call at a different dpi just re-rasterizes the same in-memory
    figure, it does NOT re-fetch tiles or re-run the plot. This is
    what makes a "print-quality download" + "fast web display" pair
    practical to generate together in one pass, rather than needing to
    build the whole figure twice (which would also double the load on
    OSM's tile servers for no benefit).
    """
    add_gridlines(ax, bounds)
    add_north_arrow(ax)
    add_scale_bar(ax, bounds, length_km=scale_bar_km)
    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if web_path is not None:
        os.makedirs(os.path.dirname(web_path) or ".", exist_ok=True)
        fig.savefig(web_path, dpi=web_dpi or dpi, bbox_inches="tight")

    plt.close(fig)
    return out_path


def _map_title(lga_name: str, metric_label: str) -> str:
    """
    Standard title format for every static map/chart: "{LGA}: {metric}",
    e.g. "Akure South: Access Deficit (Walk)". Kept as one shared
    function so every figure type (deficit, continuous, completeness,
    mode-comparison) uses the exact same title convention, rather than
    each caller formatting it slightly differently.
    """
    return f"{lga_name}: {metric_label}"


# ---------------------------------------------------------------------------
# Map builders: categorical and continuous vector choropleths, the vector
# equivalent of a raster make_map(mode='categorical' | 'continuous') helper
# ---------------------------------------------------------------------------


def plot_deficit_map(
    grid_gdf: gpd.GeoDataFrame,
    mode: str,
    title: str,
    out_path: str,
    settled_only: bool = True,
    palette: str = "standard",
    scale_bar_km: Optional[float] = None,
    dpi: int = 300,
    web_path: Optional[str] = None,
    web_dpi: Optional[int] = None,
    figsize=(11, 12),
):
    """
    Categorical map of the 0/1/2 access-deficit score for one travel
    mode (walk/okada/drive), styled to match a standard cartographic reference's
    categorical make_map() output (OSM basemap, gridlines, north arrow,
    scale bar, legend) and using the SAME palette as the interactive
    Streamlit map, so this static export and the dashboard never
    disagree on what a given color means.

    `settled_only=True` (default) excludes unsettled cells (no
    buildings, and therefore no deficit score) from the plotted layer
    entirely, rather than coloring them, since "no people, no score" is
    a different concept from "well served" and conflating the two would
    misrepresent unsettled land as a positive finding.
    """
    col = f"{mode}_access_deficit_score"
    if col not in grid_gdf.columns:
        raise KeyError(
            f"'{col}' not found. Expected columns like 'walk_access_deficit_score' "
            f"produced by akure_access.accessibility.add_access_deficit_score()."
        )

    grid_gdf = _ensure_lonlat(grid_gdf)
    plot_gdf = grid_gdf[grid_gdf["building_count"] > 0] if settled_only else grid_gdf
    if plot_gdf.empty:
        raise ValueError("No settled cells to plot for this LGA/mode.")

    colors = DEFICIT_PALETTES[palette]
    bounds = _figure_bounds(grid_gdf)  # full grid extent, not just settled cells, for context

    fig, ax = plt.subplots(figsize=figsize)
    add_osm_basemap(ax)

    for score, color, label in zip((0, 1, 2), colors, DEFICIT_LABELS):
        subset = plot_gdf[plot_gdf[col] == score]
        if not subset.empty:
            subset.plot(ax=ax, facecolor=color, edgecolor=color, linewidth=0.1,
                        alpha=0.85, zorder=2, label=label)

    legend_patches = [mpatches.Patch(facecolor=c, label=l) for c, l in zip(colors, DEFICIT_LABELS)]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, frameon=True,
              title=f"Access Deficit ({mode.capitalize()})", title_fontsize=10)

    return _finalize_and_save(fig, ax, bounds, title, out_path, scale_bar_km, dpi, web_path=web_path, web_dpi=web_dpi)


def plot_continuous_map(
    grid_gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_path: str,
    colorbar_label: str,
    settled_only: bool = True,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.9,
    scale_bar_km: Optional[float] = None,
    dpi: int = 300,
    web_path: Optional[str] = None,
    web_dpi: Optional[int] = None,
    figsize=(11, 12),
):
    """
    Continuous choropleth (e.g. health_time_min_walk) with a colorbar,
    styled to match a standard cartographic reference's continuous choropleth output.
    NaN cells (unsettled, or unreachable-then-sanitized) are left
    uncolored/transparent rather than plotted as a value, the same
    "don't visually claim a score exists where it doesn't" principle
    used throughout the scoring pipeline itself.
    """
    if value_col not in grid_gdf.columns:
        raise KeyError(f"'{value_col}' not found in grid_gdf columns: {list(grid_gdf.columns)}")

    grid_gdf = _ensure_lonlat(grid_gdf)
    plot_gdf = grid_gdf[grid_gdf["building_count"] > 0] if settled_only else grid_gdf
    plot_gdf = plot_gdf[plot_gdf[value_col].notna()]
    if plot_gdf.empty:
        raise ValueError(f"No non-null '{value_col}' values to plot for this LGA/mode.")

    bounds = _figure_bounds(grid_gdf)
    fig, ax = plt.subplots(figsize=figsize)
    add_osm_basemap(ax)

    vmin = plot_gdf[value_col].min() if vmin is None else vmin
    vmax = plot_gdf[value_col].max() if vmax is None else vmax

    plot_gdf.plot(
        ax=ax, column=value_col, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolor="none", alpha=alpha, zorder=2,
        legend=True,
        legend_kwds={"label": colorbar_label, "shrink": 0.6, "pad": 0.03},
    )

    return _finalize_and_save(fig, ax, bounds, title, out_path, scale_bar_km, dpi, web_path=web_path, web_dpi=web_dpi)


def plot_completeness_map(
    grid_gdf: gpd.GeoDataFrame,
    service: str,
    title: str,
    out_path: str,
    scale_bar_km: Optional[float] = None,
    dpi: int = 300,
    web_path: Optional[str] = None,
    web_dpi: Optional[int] = None,
    figsize=(11, 12),
):
    """
    Categorical map distinguishing three states for settled cells:
    confirmed nearby facility, possible data gap (flagged by
    grid_check.flag_completeness), and unsettled/not scored. Kept as
    its own function (rather than reusing plot_deficit_map) because the
    underlying concept is different, this is about OSM COVERAGE
    confidence, not about access itself, and conflating the two palettes
    would blur exactly the distinction the completeness module exists to
    preserve.
    """
    flag_col = f"{service}_completeness_flag"
    if flag_col not in grid_gdf.columns:
        raise KeyError(f"'{flag_col}' not found. Expected output of grid_check.flag_completeness().")

    grid_gdf = _ensure_lonlat(grid_gdf)
    settled = grid_gdf[grid_gdf["building_count"] > 0]
    if settled.empty:
        raise ValueError("No settled cells to plot.")

    bounds = _figure_bounds(grid_gdf)
    fig, ax = plt.subplots(figsize=figsize)
    add_osm_basemap(ax)

    confirmed = settled[settled[flag_col] == False]  # noqa: E712 (explicit bool compare for clarity against nullable dtype)
    possible_gap = settled[settled[flag_col] == True]  # noqa: E712

    if not confirmed.empty:
        confirmed.plot(ax=ax, facecolor=ACCENT_SECONDARY, edgecolor="none", alpha=0.85,
                        zorder=2, label="Facility confirmed nearby")
    if not possible_gap.empty:
        possible_gap.plot(ax=ax, facecolor=ACCENT_HIGHLIGHT, edgecolor="none", alpha=0.85,
                           zorder=2, label="Possible OSM data gap")

    legend_patches = [
        mpatches.Patch(facecolor=ACCENT_SECONDARY, label="Facility confirmed nearby"),
        mpatches.Patch(facecolor=ACCENT_HIGHLIGHT, label="Possible OSM data gap"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10, frameon=True,
              title=f"{service.capitalize()} data completeness", title_fontsize=10)

    return _finalize_and_save(fig, ax, bounds, title, out_path, scale_bar_km, dpi, web_path=web_path, web_dpi=web_dpi)


# ---------------------------------------------------------------------------
# Charts (not maps): mode-comparison bar chart, matching dashboard branding
# ---------------------------------------------------------------------------


def plot_mode_comparison_chart(
    mode_stats: Sequence[tuple],
    title: str,
    out_path: str,
    dpi: int = 300,
    web_path: Optional[str] = None,
    web_dpi: Optional[int] = None,
    figsize=(8, 5.5),
):
    """
    Bar chart of % underserved (any service) vs % underserved (both
    services) per mode, the static-export equivalent of the Streamlit
    dashboard's "Findings summary" metric cards, so the same headline
    numbers a judge sees on the live dashboard are also available as a
    standalone, downloadable figure for a report/slide deck.

    `mode_stats`: sequence of (mode_name, pct_any, pct_both) tuples,
    same shape produced inline in dashboard/app.py's Findings Summary
    section, kept as a plain tuple sequence (not a bespoke class) so
    both the notebook and the Streamlit app can build this input the
    same simple way.
    """
    if not mode_stats:
        raise ValueError("mode_stats is empty; nothing to chart.")

    labels = [m.capitalize() for m, _, _ in mode_stats]
    pct_any = [p for _, p, _ in mode_stats]
    pct_both = [p for _, _, p in mode_stats]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, pct_any, width, label="Underserved (\u22651 service)", color=ACCENT_PRIMARY)
    ax.bar(x + width / 2, pct_both, width, label="Underserved (both services)", color=ACCENT_SECONDARY)

    for xi, v in zip(x - width / 2, pct_any):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, pct_both):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% of settled cells")
    ax.set_ylim(0, max(pct_any + pct_both) * 1.2)
    ax.set_title(title, fontsize=14, pad=12)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    if web_path is not None:
        os.makedirs(os.path.dirname(web_path) or ".", exist_ok=True)
        fig.savefig(web_path, dpi=web_dpi or dpi, bbox_inches="tight")

    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator: generate every map/chart for one LGA in one call
# ---------------------------------------------------------------------------


def generate_all_static_outputs(
    lga_name: str,
    grid_gdf: gpd.GeoDataFrame,
    out_dir: str,
    modes: Iterable[str] = ("walk", "okada", "drive"),
    palette: str = "standard",
    dpi: int = 300,
    web_dpi: Optional[int] = 150,
) -> dict:
    """
    Produces the full standard set of publication-styled outputs for one
    LGA: a deficit map + health-time map + education-time map per mode,
    plus one completeness map per service, plus one mode-comparison
    chart, mirroring the file-per-layer, saved-to-disk pattern of the
    standard cartographic reference described above.

    Every title uses the same "{LGA name}: {metric}" convention (e.g.
    "Akure South: Access Deficit (Walk)"), via _map_title(), so every
    figure this function produces reads consistently, rather than each
    figure type phrasing its title slightly differently.

    Each figure is rendered ONCE and saved TWICE, at `dpi` (default
    300, print/download quality) into `out_dir`, and at `web_dpi`
    (default 150, if not None) into `out_dir/web/`. Rendering once and
    saving twice, rather than calling each plot function twice, avoids
    fetching the OSM basemap tiles a second time for the same figure,
    which would double both the runtime and the load on OSM's tile
    servers for no visual benefit. Pass `web_dpi=None` to skip
    generating the web tier entirely.

    Returns a dict: {"print": [...], "web": [...]}, so the caller
    (notebook or Streamlit app) can zip the print-quality set for
    download while pointing the in-app display at the lighter web set,
    without needing to know this function's internal naming scheme.
    """
    os.makedirs(out_dir, exist_ok=True)
    web_dir = os.path.join(out_dir, "web")
    if web_dpi is not None:
        os.makedirs(web_dir, exist_ok=True)

    produced = {"print": [], "web": []}
    safe_lga = lga_name.replace(" ", "_")

    def _web_kwargs(fname):
        if web_dpi is None:
            return {}
        web_path = os.path.join(web_dir, fname)
        produced["web"].append(web_path)
        return {"web_path": web_path, "web_dpi": web_dpi}

    for mode in modes:
        fname = f"{safe_lga}_deficit_{mode}.jpg"
        deficit_path = os.path.join(out_dir, fname)
        produced["print"].append(plot_deficit_map(
            grid_gdf, mode, _map_title(lga_name, f"Access Deficit ({mode.capitalize()})"),
            deficit_path, palette=palette, dpi=dpi, **_web_kwargs(fname),
        ))

        for service, label in (("health", "Health"), ("education", "Education")):
            col = f"{service}_time_min_{mode}"
            if col in grid_gdf.columns and grid_gdf[col].notna().any():
                fname = f"{safe_lga}_{service}_time_{mode}.jpg"
                cont_path = os.path.join(out_dir, fname)
                produced["print"].append(plot_continuous_map(
                    grid_gdf, col,
                    _map_title(lga_name, f"{label} Access Time ({mode.capitalize()}, minutes)"),
                    cont_path, colorbar_label="Minutes", dpi=dpi, **_web_kwargs(fname),
                ))

    for service in ("health", "education"):
        flag_col = f"{service}_completeness_flag"
        if flag_col in grid_gdf.columns:
            fname = f"{safe_lga}_completeness_{service}.jpg"
            comp_path = os.path.join(out_dir, fname)
            produced["print"].append(plot_completeness_map(
                grid_gdf, service,
                _map_title(lga_name, f"{service.capitalize()} Facility Data Completeness"),
                comp_path, dpi=dpi, **_web_kwargs(fname),
            ))

    settled = grid_gdf[grid_gdf["building_count"] > 0]
    mode_stats = []
    for m in modes:
        col = f"{m}_access_deficit_score"
        if col in settled.columns:
            pct_any = 100 * (settled[col] > 0).mean()
            pct_both = 100 * (settled[col] == 2).mean()
            mode_stats.append((m, pct_any, pct_both))
    if mode_stats:
        fname = f"{safe_lga}_mode_comparison.jpg"
        chart_path = os.path.join(out_dir, fname)
        produced["print"].append(plot_mode_comparison_chart(
            mode_stats, _map_title(lga_name, "Underserved by Travel Mode"),
            chart_path, dpi=dpi, **_web_kwargs(fname),
        ))

    return produced
