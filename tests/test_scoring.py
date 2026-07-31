"""
tests/test_scoring.py

Offline tests for akure_access.accessibility.scoring: grid generation, building
density joins, and access-deficit scoring logic. These use synthetic
geometries and do not require network access or a live OSM graph.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from akure_access.accessibility.scoring import (
    build_grid,
    add_building_density,
    add_access_deficit_score,
    sanitize_for_export,
    DEFAULT_GRID_CELL_SIZE_M,
)


def _synthetic_boundary(size_m=2000):
    return gpd.GeoDataFrame(geometry=[box(0, 0, size_m, size_m)], crs="EPSG:32631")


def test_build_grid_covers_boundary():
    boundary = _synthetic_boundary(2000)
    grid = build_grid(boundary, cell_size_m=500)

    # A 2000x2000m area with 500m cells should produce a 4x4 grid (16 cells)
    assert len(grid) == 16
    assert "cell_id" in grid.columns
    assert grid.crs.to_string() == "EPSG:32631"


def test_build_grid_respects_custom_cell_size():
    boundary = _synthetic_boundary(1000)
    grid_coarse = build_grid(boundary, cell_size_m=1000)
    grid_fine = build_grid(boundary, cell_size_m=250)

    assert len(grid_fine) > len(grid_coarse)


def test_add_building_density_counts_correctly():
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)

    # 3 buildings in the first cell (0,0)-(500,500), 1 in a different cell
    buildings = gpd.GeoDataFrame(
        geometry=[Point(100, 100), Point(200, 200), Point(400, 400), Point(900, 900)],
        crs="EPSG:32631",
    )
    grid = add_building_density(grid, buildings)

    assert "building_count" in grid.columns
    assert grid["building_count"].sum() == 4
    assert grid["building_count"].max() == 3


def test_add_building_density_handles_empty_buildings():
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    empty_buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    grid = add_building_density(grid, empty_buildings)

    assert (grid["building_count"] == 0).all()


def test_access_deficit_score_composite_logic():
    """
    Directly tests the 0/1/2 composite scoring logic without needing a
    real network graph: builds a grid with pre-set time columns and
    checks the deficit score matches expectations.
    """
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    grid["building_count"] = [5, 5, 5, 0]  # last cell unsettled

    # health_time_min / education_time_min are the unsuffixed columns
    # add_access_deficit_score falls back to when mode="walk" and no
    # suffixed columns are present.
    grid["health_time_min"] = [10, 40, 40, np.nan]     # served, over, over, n/a
    grid["education_time_min"] = [10, 10, 40, np.nan]  # served, served, over, n/a

    scored = add_access_deficit_score(grid, threshold_min=30, mode="walk")

    # Cell 0: both served -> score 0
    assert scored.loc[0, "access_deficit_score"] == 0
    # Cell 1: health underserved only -> score 1
    assert scored.loc[1, "access_deficit_score"] == 1
    # Cell 2: both underserved -> score 2
    assert scored.loc[2, "access_deficit_score"] == 2
    # Cell 3: unsettled -> score should be NaN, not 0
    assert pd.isna(scored.loc[3, "access_deficit_score"])


def test_access_deficit_score_mode_suffix_columns():
    """
    When mode-specific suffixed columns are present (as produced by
    add_access_times(modes=(...))), add_access_deficit_score should read
    those instead of the unsuffixed legacy columns.
    """
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    grid["building_count"] = [5, 5, 5, 5]

    grid["health_time_min_okada"] = [5, 25, 5, 25]
    grid["education_time_min_okada"] = [5, 5, 25, 25]

    scored = add_access_deficit_score(grid, threshold_min=20, mode="okada")

    assert scored.loc[0, "okada_access_deficit_score"] == 0
    assert scored.loc[1, "okada_access_deficit_score"] == 1
    assert scored.loc[2, "okada_access_deficit_score"] == 1
    assert scored.loc[3, "okada_access_deficit_score"] == 2


def test_access_deficit_score_treats_inf_as_underserved():
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    grid["building_count"] = [5, 5, 5, 5]
    grid["health_time_min"] = [10, float("inf"), 10, 10]
    grid["education_time_min"] = [10, 10, 10, 10]

    scored = add_access_deficit_score(grid, threshold_min=30, mode="walk")

    # An unreachable facility (inf travel time) must count as underserved,
    # not silently pass the threshold check.
    assert scored.loc[1, "access_deficit_score"] == 1


def test_sanitize_for_export_does_not_break_deficit_scoring():
    """
    Regression test: sanitize_for_export() must only ever be called AFTER
    add_access_deficit_score(), never before. This test locks in that
    ordering by checking that a genuinely unreachable, settled cell is
    still correctly scored as underserved (not silently "adequately
    served") even once the resulting grid has been through export
    sanitization -- i.e. the scoring decision, once made, survives the
    inf -> NaN conversion; only the raw time/distance values change.
    """
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    grid["building_count"] = [5, 5, 5, 5]
    grid["health_time_min_walk"] = [10, float("inf"), 10, 10]
    grid["education_time_min_walk"] = [10, 10, 10, 10]

    scored = add_access_deficit_score(grid, threshold_min=30, mode="walk")
    assert scored.loc[1, "walk_access_deficit_score"] == 1  # correctly underserved

    exported = sanitize_for_export(scored)

    # The deficit score itself (already computed) must be unaffected...
    assert exported.loc[1, "walk_access_deficit_score"] == 1
    # ...while the raw inf time value is now a clean NaN, safe for GeoJSON.
    assert pd.isna(exported.loc[1, "health_time_min_walk"])


def test_sanitize_for_export_called_before_scoring_gives_wrong_result():
    """
    Negative test demonstrating WHY the ordering matters: calling
    sanitize_for_export() before add_access_deficit_score() converts inf
    to NaN first, which add_access_deficit_score() then treats as
    "unknown" (fillna(0) = adequately served) instead of correctly
    detecting it as unreachable. This is the exact regression caught
    during development -- kept here as a permanent guard against
    reintroducing it.
    """
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    grid["building_count"] = [5, 5, 5, 5]
    grid["health_time_min_walk"] = [10, float("inf"), 10, 10]
    grid["education_time_min_walk"] = [10, 10, 10, 10]

    # WRONG ORDER: sanitize before scoring
    sanitized_early = sanitize_for_export(grid)
    scored_wrong = add_access_deficit_score(sanitized_early, threshold_min=30, mode="walk")

    # This demonstrates the bug: the unreachable cell is now
    # (incorrectly) treated as adequately served, because inf was
    # already gone before scoring ever saw it.
    assert scored_wrong.loc[1, "walk_access_deficit_score"] == 0
