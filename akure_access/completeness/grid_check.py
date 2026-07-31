"""
grid_check.py

Lightweight OSM completeness assessment: flags grid cells that show
visible building density (a settlement proxy) but have no nearby
OSM-tagged health facility or school, suggesting the area may be
under-mapped rather than genuinely unserved.

This intentionally avoids a full land-cover classification pipeline
(no satellite imagery classification required) to keep the check fast
to build; the building layer already extracted by lga_extractor is
used as the settlement-presence signal.
"""

import geopandas as gpd
import pandas as pd


DEFAULT_BUILDING_PRESENCE_THRESHOLD = 3  # min buildings in a cell to call it "settled"
DEFAULT_FACILITY_SEARCH_RADIUS_M = 1000  # radius to check for a nearby facility


def flag_completeness(
    grid_gdf: gpd.GeoDataFrame,
    health_gdf: gpd.GeoDataFrame,
    schools_gdf: gpd.GeoDataFrame,
    building_threshold: int = DEFAULT_BUILDING_PRESENCE_THRESHOLD,
    search_radius_m: float = DEFAULT_FACILITY_SEARCH_RADIUS_M,
) -> gpd.GeoDataFrame:
    """
    Flag grid cells as potentially under-mapped in OSM.

    A cell is flagged as under-mapped for a given service (health or
    education) if:
      - it has building_count >= building_threshold (visibly settled), AND
      - no facility of that type exists within search_radius_m of the
        cell centroid.

    This distinguishes "genuinely underserved" (settled, no nearby
    facility, and OSM completeness looks fine elsewhere in the area)
    from "possibly a data gap" (settled, no nearby OSM facility tag at
    all, and this pattern repeats suspiciously).

    Implementation note: nearest-facility distance for every cell is
    computed via geopandas.sjoin_nearest(), which uses a spatial index
    (an STRtree, built once over the facility points) internally. This
    is O(n log m) for n cells and m facilities, rather than the O(n x m)
    cost of comparing every cell's centroid against every facility with
    a plain per-cell distance scan. At this project's actual scale (a
    few thousand grid cells, a few dozen facilities per LGA) the
    difference is not the dominant runtime cost -- unlike the network
    routing in accessibility/isochrones.py, which was the real
    bottleneck -- but this scales correctly if ever applied to a much
    larger area or a much denser facility dataset.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Grid with a 'building_count' column, as produced by
        scoring.add_building_density(), in EPSG:32631.
    health_gdf, schools_gdf : geopandas.GeoDataFrame
        Cleaned facility point layers, in EPSG:32631.
    building_threshold : int
        Minimum building count for a cell to be considered "settled".
    search_radius_m : float
        Search radius (metres) used to check for a nearby facility.

    Returns
    -------
    geopandas.GeoDataFrame
        grid_gdf with added boolean columns 'health_completeness_flag'
        and 'education_completeness_flag' (True = settled but no
        nearby OSM facility of that type -- a likely OSM data gap
        rather than confirmed non-access).
    """
    grid = grid_gdf.copy()
    centroids_gdf = gpd.GeoDataFrame(geometry=grid.geometry.centroid.values, crs=grid.crs)

    grid["health_completeness_flag"] = _flag_via_spatial_index(
        centroids_gdf, grid["building_count"], health_gdf, building_threshold, search_radius_m
    )
    grid["education_completeness_flag"] = _flag_via_spatial_index(
        centroids_gdf, grid["building_count"], schools_gdf, building_threshold, search_radius_m
    )

    return grid


def _flag_via_spatial_index(
    centroids_gdf: gpd.GeoDataFrame,
    building_counts: pd.Series,
    facilities_gdf: gpd.GeoDataFrame,
    building_threshold: int,
    search_radius_m: float,
) -> list:
    """
    Vectorized completeness flagging for one facility type, using a
    spatial-indexed nearest-neighbor join instead of a per-cell loop.
    """
    building_counts = pd.Series(building_counts).reset_index(drop=True)
    settled_mask = building_counts >= building_threshold

    if facilities_gdf.empty:
        # No facilities of this type exist anywhere in the LGA: every
        # settled cell is, by definition, unmapped for this service.
        return settled_mask.tolist()

    # sjoin_nearest builds a spatial index over the right-hand
    # GeoDataFrame (facilities) once, then finds each left-hand row's
    # (cell centroid's) nearest match in a single indexed query --
    # rather than a linear distance() comparison against every facility
    # per cell.
    joined = gpd.sjoin_nearest(
        centroids_gdf.reset_index(drop=True),
        facilities_gdf[["geometry"]].reset_index(drop=True),
        distance_col="_nearest_dist",
        how="left",
    )
    # Exact ties (multiple facilities at identical nearest distance) can
    # produce duplicate rows for the same cell; keep just one match per
    # cell since we only need the nearest distance value itself.
    joined = joined[~joined.index.duplicated(keep="first")]
    nearest_dist = joined["_nearest_dist"].reindex(range(len(centroids_gdf)))

    flags = settled_mask & (nearest_dist > search_radius_m)
    return flags.tolist()


def summarize_completeness(grid_gdf: gpd.GeoDataFrame) -> dict:
    """
    Produce a simple summary of completeness flags for reporting.

    Returns
    -------
    dict
        Counts and percentages of settled cells flagged as potentially
        under-mapped for health and education, respectively.
    """
    settled = grid_gdf[grid_gdf["building_count"] > 0]
    n_settled = len(settled)

    if n_settled == 0:
        return {"settled_cells": 0, "health_flagged": 0, "education_flagged": 0}

    return {
        "settled_cells": n_settled,
        "health_flagged": int(settled["health_completeness_flag"].sum()),
        "health_flagged_pct": round(100 * settled["health_completeness_flag"].mean(), 1),
        "education_flagged": int(settled["education_completeness_flag"].sum()),
        "education_flagged_pct": round(100 * settled["education_completeness_flag"].mean(), 1),
    }
