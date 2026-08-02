"""
akure_access.insights

Generates short, data-driven captions ("what this map shows, what the
legend colors mean, and what the numbers actually say") for both the
interactive Streamlit map and the static exported maps/charts.

Every caption is computed directly from the same scored GeoDataFrame the
map itself is drawn from, never hand-written once and left sitting next
to a map. This means re-running the analysis notebook with updated data
automatically produces updated, still-accurate captions everywhere they
appear, in the live dashboard and in every downloaded static figure,
without anyone needing to remember to update wording by hand.

This module intentionally has no dependency on matplotlib or Streamlit,
only geopandas/numpy, so it's safe to import from dashboard/app.py
without pulling in the heavier static-map plotting stack, and safe to
import from the static map generator without pulling in Streamlit.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

MODE_LABELS = {"walk": "walking", "okada": "okada", "drive": "driving"}
SERVICE_LABELS = {"health": "health", "education": "education"}

# Default mode-specific "underserved" thresholds, in minutes. Used only
# to produce threshold-aware phrasing (e.g. "above the 30-minute walking
# threshold used for this analysis"), the actual underserved/well-served
# classification always comes straight from the grid's own
# *_access_deficit_score columns, which were computed by
# akure_access.accessibility.scoring using whatever threshold was
# actually passed in at scoring time. Pass thresholds=... to any
# function below to override this default with the authoritative values
# from a specific analysis run, kept here in sync with
# ACCESS_THRESHOLDS_MIN in Notebook 03's configuration cell.
DEFAULT_THRESHOLDS_MIN = {"walk": 30, "okada": 20, "drive": 15}


def _article(word: str) -> str:
    """Returns 'an' if `word` starts with a vowel sound, else 'a'. Used
    so generated captions read as "a health facility" but "an education
    facility", rather than always defaulting to "a" regardless of the
    following word."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _settled(grid_gdf):
    return grid_gdf[grid_gdf["building_count"] > 0]


def _compass_skew(target_gdf, reference_gdf) -> Optional[str]:
    """
    Compares the centroid of `target_gdf` (e.g. underserved cells)
    against the centroid of `reference_gdf` (e.g. all settled cells) and
    returns a short compass phrase like "the southeast" if the
    difference is large enough to be worth mentioning, or None if the
    two centroids are close enough that no direction clearly stands out.

    Works whether the input is in a projected (meters) or geographic
    (degrees) CRS: both preserve the convention that increasing X is
    east and increasing Y is north, so only the SIGN of the offset
    matters here, and that sign is the same in either CRS.
    """
    if target_gdf.empty or reference_gdf.empty:
        return None

    # geopandas warns that .centroid on a geographic (lat/lon) CRS is
    # imprecise, true for accurate area/distance work, but irrelevant
    # here: this function only needs the SIGN of the offset between two
    # centroids (is it further north/south/east/west), not a precise
    # distance, and that sign is unaffected by the same approximation
    # that would distort an actual area or distance calculation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tx, ty = target_gdf.geometry.centroid.x.mean(), target_gdf.geometry.centroid.y.mean()
        rx, ry = reference_gdf.geometry.centroid.x.mean(), reference_gdf.geometry.centroid.y.mean()

    bounds = reference_gdf.total_bounds  # west, south, east, north
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if width == 0 or height == 0:
        return None

    # Normalize by the study area's own extent, so "large enough to
    # mention" scales with the size of the area rather than using a
    # fixed distance that would behave inconsistently between a
    # projected CRS (meters) and a geographic one (degrees).
    dx_frac = (tx - rx) / width
    dy_frac = (ty - ry) / height

    skew_threshold = 0.08
    ns = "north" if dy_frac > skew_threshold else ("south" if dy_frac < -skew_threshold else "")
    ew = "east" if dx_frac > skew_threshold else ("west" if dx_frac < -skew_threshold else "")

    if not ns and not ew:
        return None
    return f"the {ns}{ew}" if ns and ew else f"the {ns or ew}"


def _mode_ranking_phrase(grid_gdf, mode: str) -> Optional[str]:
    """
    Compares the current mode's "underserved for at least one service"
    rate against whichever other modes are also present as columns in
    this grid, and returns a short phrase describing where the current
    mode ranks, or None if fewer than two modes are available to
    compare.
    """
    settled = _settled(grid_gdf)
    if settled.empty:
        return None

    rates = {}
    for m in ("walk", "okada", "drive"):
        col = f"{m}_access_deficit_score"
        if col in settled.columns:
            rates[m] = (settled[col] > 0).mean() * 100

    if mode not in rates or len(rates) < 2:
        return None

    ranked = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    position = [m for m, _ in ranked].index(mode)

    if position == 0:
        return "the most restrictive of the modes shown here"
    if position == len(ranked) - 1:
        return "the least restrictive of the modes shown here"
    return "in between the other modes shown here"


def describe_deficit_map(grid_gdf, lga_name: str, mode: str, thresholds: Optional[dict] = None) -> str:
    """
    Caption for the categorical access-deficit map (green/yellow/red):
    what the colors mean, and the actual underserved percentages for
    this LGA and mode, plus a note on how this mode compares to the
    others and a rough sense of where the underserved areas cluster.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS_MIN
    mode_label = MODE_LABELS.get(mode, mode)
    threshold = thresholds.get(mode)
    col = f"{mode}_access_deficit_score"

    settled = _settled(grid_gdf)
    if settled.empty or col not in settled.columns:
        return (
            f"No scored data available for {mode_label} access in {lga_name} yet. "
            f"Re-run the analysis to generate it."
        )

    pct_any = (settled[col] > 0).mean() * 100
    pct_both = (settled[col] == 2).mean() * 100
    ranking = _mode_ranking_phrase(grid_gdf, mode)
    underserved = settled[settled[col] > 0]
    direction = _compass_skew(underserved, settled)

    threshold_phrase = f" (within {threshold} minutes by {mode_label} of both)" if threshold else ""
    legend = (
        f"Green cells are well served{threshold_phrase}, yellow cells lack one "
        f"of the two services, and red cells lack both."
    )

    result = (
        f"{pct_any:.1f}% of settled cells in {lga_name} are underserved for at "
        f"least one service by {mode_label}, and {pct_both:.1f}% are underserved "
        f"for both."
    )
    if ranking:
        result += f" {mode_label.capitalize()} is {ranking}."
    if direction:
        result += f" Underserved cells are concentrated toward {direction} of the study area."

    return f"{legend} {result}"


def describe_continuous_map(
    grid_gdf, lga_name: str, mode: str, service: str, thresholds: Optional[dict] = None
) -> str:
    """
    Caption for the continuous travel-time map (colorbar in minutes):
    what the gradient means, plus real summary statistics (mean,
    median, share exceeding the mode's threshold) and where the
    longest travel times cluster.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS_MIN
    mode_label = MODE_LABELS.get(mode, mode)
    service_label = SERVICE_LABELS.get(service, service)
    threshold = thresholds.get(mode)
    col = f"{service}_time_min_{mode}"

    settled = _settled(grid_gdf)
    if settled.empty or col not in settled.columns or settled[col].notna().sum() == 0:
        return (
            f"No {service_label} travel-time data available for {mode_label} in "
            f"{lga_name} yet. Re-run the analysis to generate it."
        )

    valid = settled[settled[col].notna()]
    times = valid[col]
    mean_t, median_t = times.mean(), times.median()

    legend = (
        f"Color shows travel time in minutes to the nearest {service_label} "
        f"facility by {mode_label}: cooler/darker colors are shorter times, "
        f"warmer colors are longer ones."
    )

    result = (
        f"Across settled cells in {lga_name}, the average {service_label} travel "
        f"time by {mode_label} is {mean_t:.0f} minutes (median {median_t:.0f} minutes)."
    )
    if threshold:
        pct_over = (times > threshold).mean() * 100
        result += (
            f" {pct_over:.1f}% of cells exceed the {threshold}-minute threshold "
            f"used to flag underserved areas for this mode."
        )
        worst = valid[times > threshold]
    else:
        worst = valid[times > times.quantile(0.75)]

    direction = _compass_skew(worst, settled)
    if direction:
        result += f" The longest travel times are concentrated toward {direction} of the study area."

    return f"{legend} {result}"


def describe_completeness_map(grid_gdf, lga_name: str, service: str) -> str:
    """
    Caption for the data-completeness map: what "confirmed nearby" vs
    "possible data gap" means, and the real share of settled cells that
    fall into each category, so the reader is warned when a large
    portion of an "underserved" reading might reflect thin OSM coverage
    rather than a confirmed absence of service.
    """
    service_label = SERVICE_LABELS.get(service, service)
    flag_col = f"{service}_completeness_flag"

    settled = _settled(grid_gdf)
    if settled.empty or flag_col not in settled.columns:
        return f"No {service_label} completeness data available for {lga_name} yet."

    pct_gap = (settled[flag_col] == True).mean() * 100  # noqa: E712
    pct_confirmed = 100 - pct_gap

    legend = (
        f"Teal cells have {_article(service_label)} {service_label} facility confirmed "
        f"nearby in OpenStreetMap; gold cells are flagged as a possible data gap, "
        f"meaning no nearby facility was found in OSM, which may reflect a real "
        f"service gap or simply that OSM's coverage is still incomplete there."
    )
    result = (
        f"{pct_confirmed:.1f}% of settled cells in {lga_name} have a confirmed "
        f"nearby {service_label} facility, while {pct_gap:.1f}% are flagged as a "
        f"possible data gap rather than a confirmed absence of service."
    )
    return f"{legend} {result}"


def describe_mode_comparison_chart(mode_stats, lga_name: str) -> str:
    """
    Caption for the mode-comparison bar chart. `mode_stats` is the same
    sequence of (mode, pct_any, pct_both) tuples used to draw the chart
    itself (see visualization.static_maps.plot_mode_comparison_chart and
    dashboard/app.py's own Findings Summary section), so the caption and
    the bars it describes can never disagree.
    """
    if not mode_stats:
        return f"No mode-comparison data available for {lga_name} yet."

    ranked = sorted(mode_stats, key=lambda t: t[1], reverse=True)
    worst_mode, worst_pct, _ = ranked[0]
    best_mode, best_pct, _ = ranked[-1]
    worst_label, best_label = MODE_LABELS.get(worst_mode, worst_mode), MODE_LABELS.get(best_mode, best_mode)

    # If the highest and lowest rates are close enough to be
    # indistinguishable at this precision, "highest at X%, lowest at X%"
    # reads as a contradiction rather than useful information, describe
    # the modes as comparably restrictive instead of forcing a ranking
    # sentence onto numbers that don't actually differ meaningfully.
    if abs(worst_pct - best_pct) < 0.5:
        mode_labels_all = ", ".join(MODE_LABELS.get(m, m) for m, _, _ in mode_stats)
        result = (
            f"Underserved rates in {lga_name} are similarly high across every mode "
            f"analyzed ({mode_labels_all}), all around {worst_pct:.1f}% of settled "
            f"cells. This suggests the access gap here is severe enough that "
            f"switching travel mode alone would not meaningfully change who counts "
            f"as underserved."
        )
        return result

    result = (
        f"Across the travel modes analyzed for {lga_name}, {worst_label} shows "
        f"the highest underserved rate at {worst_pct:.1f}% of settled cells, "
        f"while {best_label} shows the lowest at {best_pct:.1f}%. This gap shows "
        f"how much the choice of travel mode alone changes who counts as "
        f"underserved, not just the underlying facility locations."
    )
    return result


def describe_interactive_view(
    grid_gdf, lga_name: str, mode: str, view_choice: str, thresholds: Optional[dict] = None
) -> str:
    """
    Single entry point for the interactive Streamlit map: dispatches to
    the right caption generator based on the current "Access view"
    selection ("Combined", "Health only", "Education only"), so the
    caption shown in the dashboard always matches whatever the map
    itself is currently displaying.
    """
    if view_choice == "Health only":
        return describe_continuous_map(grid_gdf, lga_name, mode, "health", thresholds=thresholds)
    if view_choice == "Education only":
        return describe_continuous_map(grid_gdf, lga_name, mode, "education", thresholds=thresholds)
    return describe_deficit_map(grid_gdf, lga_name, mode, thresholds=thresholds)
