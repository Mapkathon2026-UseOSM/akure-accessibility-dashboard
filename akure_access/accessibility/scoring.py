"""
scoring.py

Generates a settlement/analysis grid over an LGA, uses OSM building
density as a population proxy, computes nearest-facility travel times
per cell (health + education), and derives a composite access-deficit
score.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from .network_graph import graph_from_roads
from .isochrones import batch_nearest_facility_distances, lookup_nearest_distance_time

DEFAULT_ACCESS_THRESHOLD_MIN = 30
DEFAULT_GRID_CELL_SIZE_M = 500


def build_grid(boundary_gdf: gpd.GeoDataFrame, cell_size_m: float = DEFAULT_GRID_CELL_SIZE_M) -> gpd.GeoDataFrame:
    """
    Build a regular square grid covering the LGA boundary.

    Parameters
    ----------
    boundary_gdf : geopandas.GeoDataFrame
        LGA boundary, will be reprojected to EPSG:32631 if needed.
    cell_size_m : float
        Grid cell size in metres.

    Returns
    -------
    geopandas.GeoDataFrame
        Grid cells (squares) clipped to the boundary, in EPSG:32631,
        with a 'cell_id' column.
    """
    boundary_m = boundary_gdf.to_crs("EPSG:32631")
    minx, miny, maxx, maxy = boundary_m.total_bounds

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    cells = [box(x, y, x + cell_size_m, y + cell_size_m) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:32631")

    boundary_union = boundary_m.union_all()
    grid = grid[grid.intersects(boundary_union)].reset_index(drop=True)
    grid["cell_id"] = grid.index

    return grid


def add_building_density(grid_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a building_count column to each grid cell, used as a population
    proxy in the absence of fine-grained population data.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Output of build_grid(), in EPSG:32631.
    buildings_gdf : geopandas.GeoDataFrame
        Cleaned buildings layer, in EPSG:32631.

    Returns
    -------
    geopandas.GeoDataFrame
        grid_gdf with an added 'building_count' column.
    """
    grid = grid_gdf.copy()

    if buildings_gdf.empty:
        grid["building_count"] = 0
        return grid

    joined = gpd.sjoin(buildings_gdf, grid[["cell_id", "geometry"]], how="inner", predicate="intersects")
    counts = joined.groupby("cell_id").size().rename("building_count")

    grid = grid.merge(counts, on="cell_id", how="left")
    grid["building_count"] = grid["building_count"].fillna(0).astype(int)

    return grid


def add_access_times(
    grid_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    health_gdf: gpd.GeoDataFrame,
    schools_gdf: gpd.GeoDataFrame,
    boundary_polygon_wgs84=None,
    modes=("walk",),
) -> gpd.GeoDataFrame:
    """
    Compute nearest-facility distance and travel time for health and
    education access, for every populated grid cell, across one or
    more transport modes.

    Only cells with building_count > 0 are routed (empty/unsettled
    cells are left as NaN), to keep runtime reasonable. A separate
    network graph is built per mode (see network_graph.MODE_CONFIG),
    since "okada"/"drive" use the OSM drive network while "walk" uses
    the OSM walk network.

    Performance note: for each mode/service combination, this computes
    nearest-facility distance to every graph node ONCE via multi-source
    Dijkstra (batch_nearest_facility_distances()), then looks up each
    grid cell's result in O(1), rather than running a fresh
    shortest-path search per grid cell per facility. This is
    substantially faster on real road networks with thousands of nodes
    and hundreds of settled cells (the previous per-cell approach could
    take over an hour for a full LGA across all three modes; this
    approach reduces the routing cost to roughly one Dijkstra run per
    mode per service, regardless of how many grid cells there are).

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Output of add_building_density(), in EPSG:32631.
    roads_gdf : geopandas.GeoDataFrame
        Cleaned roads layer, in EPSG:32631 (used only as a fallback if
        boundary_polygon_wgs84 is not provided).
    health_gdf, schools_gdf : geopandas.GeoDataFrame
        Cleaned facility point layers, in EPSG:32631.
    boundary_polygon_wgs84 : shapely.geometry.Polygon, optional
        LGA boundary in EPSG:4326, passed through to graph_from_roads()
        for the recommended OSMnx-based graph construction path.
    modes : tuple of str
        Transport modes to compute, from {"walk", "okada", "drive"}.
        Defaults to walking only for backward compatibility; pass
        modes=("walk", "okada", "drive") for the full comparison.

    Returns
    -------
    geopandas.GeoDataFrame
        grid_gdf with added columns per mode and service, e.g.:
        'health_time_min_walk', 'health_distance_km_walk',
        'education_time_min_walk', 'education_distance_km_walk',
        and equivalents for 'okada' / 'drive' if requested.
        For backward compatibility, when modes == ("walk",) the plain
        'health_time_min' / 'education_time_min' columns (no suffix)
        are also populated.
    """
    grid = grid_gdf.copy()

    for mode in modes:
        G = graph_from_roads(roads_gdf, boundary_polygon=boundary_polygon_wgs84, mode=mode)

        if boundary_polygon_wgs84 is not None:
            health_pts = health_gdf.to_crs("EPSG:4326")
            school_pts = schools_gdf.to_crs("EPSG:4326")
            # Compute centroids in the grid's own projected CRS (EPSG:32631,
            # metres) first, then reproject the resulting points to
            # EPSG:4326, NOT the other way around. Taking a centroid
            # directly in a geographic (lat/lon) CRS distorts the result,
            # since degrees aren't equal-area/equal-distance units; this
            # also silences GeoPandas' "Geometry is in a geographic CRS"
            # warning, which was flagging exactly this issue.
            grid_centroids = gpd.GeoSeries(grid.geometry.centroid, crs=grid.crs).to_crs("EPSG:4326")
        else:
            health_pts = health_gdf
            school_pts = schools_gdf
            grid_centroids = grid.geometry.centroid

        # Multi-source Dijkstra: compute nearest-facility distance to
        # EVERY node in the graph in one pass per facility type, rather
        # than running a fresh shortest-path search per grid cell per
        # facility. This is the key fix for real-LGA runtime, the
        # previous per-cell approach did `settled_cells x facilities`
        # separate shortest-path searches per mode, which is what made a
        # full run take over an hour in practice on real road networks.
        health_distances_by_node = batch_nearest_facility_distances(G, health_pts)
        school_distances_by_node = batch_nearest_facility_distances(G, school_pts)

        health_times, health_dists = [], []
        edu_times, edu_dists = [], []

        for centroid, count in zip(grid_centroids, grid["building_count"]):
            if count == 0:
                health_times.append(np.nan); health_dists.append(np.nan)
                edu_times.append(np.nan); edu_dists.append(np.nan)
                continue

            h_dist, h_time = lookup_nearest_distance_time(G, centroid, health_distances_by_node)
            e_dist, e_time = lookup_nearest_distance_time(G, centroid, school_distances_by_node)

            health_times.append(h_time); health_dists.append(h_dist)
            edu_times.append(e_time); edu_dists.append(e_dist)

        grid[f"health_time_min_{mode}"] = health_times
        grid[f"health_distance_km_{mode}"] = health_dists
        grid[f"education_time_min_{mode}"] = edu_times
        grid[f"education_distance_km_{mode}"] = edu_dists

        # NOTE: inf values are kept as-is here (not converted to NaN),
        # because add_access_deficit_score() below relies on detecting
        # inf specifically to correctly treat unreachable facilities as
        # underserved. Converting inf -> NaN here would cause NaN's
        # "unknown, treat as adequately served" fillna(0) handling in
        # add_access_deficit_score() to silently misclassify genuinely
        # unreachable cells as served. inf is only sanitized to NaN for
        # GeoJSON export, see sanitize_for_export(), which must be
        # called AFTER add_access_deficit_score(), never before.

        # Backward-compatible unsuffixed columns when only walking is requested
        if modes == ("walk",):
            grid["health_time_min"] = grid[f"health_time_min_{mode}"]
            grid["education_time_min"] = grid[f"education_time_min_{mode}"]

    return grid


def add_access_deficit_score(
    grid_gdf: gpd.GeoDataFrame, threshold_min: float = DEFAULT_ACCESS_THRESHOLD_MIN, mode: str = "walk"
) -> gpd.GeoDataFrame:
    """
    Derive a composite access-deficit score per cell, for a given mode:
        0 = adequately served for both health and education
        1 = underserved for one service
        2 = underserved for both services
    Unsettled cells (no building_count) are left unscored (NaN).

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        Output of add_access_times(). If add_access_times() was called
        with multiple modes, this reads the '{service}_time_min_{mode}'
        columns; if called with the default single "walk" mode, it
        falls back to the unsuffixed 'health_time_min' /
        'education_time_min' columns for backward compatibility.
    threshold_min : float
        Travel-time threshold (minutes) beyond which a cell is
        considered underserved for a given service, for this mode.
        Note that a sensible threshold typically differs by mode, 30 minutes walking covers far less ground than 30 minutes by
        okada or car, so consider passing a mode-appropriate threshold
        (e.g. a larger distance/time budget for faster modes) rather
        than reusing the walking default unchanged.
    mode : str
        Which mode's columns to score ("walk", "okada", or "drive").

    Returns
    -------
    geopandas.GeoDataFrame
        grid_gdf with added '{mode}_health_underserved',
        '{mode}_education_underserved', and
        '{mode}_access_deficit_score' columns (also mirrored to the
        unsuffixed legacy names when mode == "walk" and suffixed
        columns aren't present).
    """
    grid = grid_gdf.copy()

    health_col = f"health_time_min_{mode}" if f"health_time_min_{mode}" in grid.columns else "health_time_min"
    edu_col = f"education_time_min_{mode}" if f"education_time_min_{mode}" in grid.columns else "education_time_min"

    def _underserved(time_val):
        if pd_isna(time_val):
            return np.nan
        return int(time_val > threshold_min or time_val == float("inf"))

    health_flag = grid[health_col].apply(_underserved)
    edu_flag = grid[edu_col].apply(_underserved)
    deficit_score = health_flag.fillna(0) + edu_flag.fillna(0)
    deficit_score[grid["building_count"] == 0] = np.nan

    grid[f"{mode}_health_underserved"] = health_flag
    grid[f"{mode}_education_underserved"] = edu_flag
    grid[f"{mode}_access_deficit_score"] = deficit_score

    if mode == "walk":
        grid["health_underserved"] = health_flag
        grid["education_underserved"] = edu_flag
        grid["access_deficit_score"] = deficit_score

    return grid


def pd_isna(val) -> bool:
    """Small helper to avoid importing pandas just for isna in this module."""
    try:
        return val != val  # NaN != NaN is True
    except Exception:
        return val is None


def sanitize_for_export(grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Replace inf/-inf with NaN across all numeric columns, in preparation
    for GeoJSON export.

    This must only be called AFTER add_access_deficit_score() has already
    run for every mode of interest. inf values are the correct, meaningful
    representation of "facility unreachable" while scoring is happening, add_access_deficit_score() specifically checks for inf to correctly
    treat unreachable cells as underserved, and calling this function
    beforehand would silently break that (NaN's "unknown" handling treats
    a cell as adequately served instead of underserved, which is wrong).

    Once scoring is done and inf's job is finished, inf becomes purely a
    liability: it isn't valid JSON, and GeoJSON writers either error or
    silently null it out with a warning. This function makes that
    conversion explicit and intentional instead, right before export.

    Parameters
    ----------
    grid_gdf : geopandas.GeoDataFrame
        A grid that has already been fully scored (add_access_times() +
        add_access_deficit_score() for every mode you plan to export).

    Returns
    -------
    geopandas.GeoDataFrame
        A copy with inf/-inf replaced by NaN in all numeric columns.
        Geometry and non-numeric columns are left untouched.
    """
    grid = grid_gdf.copy()
    numeric_cols = grid.select_dtypes(include=["number"]).columns
    grid[numeric_cols] = grid[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return grid
