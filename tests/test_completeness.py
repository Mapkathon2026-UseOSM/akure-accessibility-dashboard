"""
tests/test_completeness.py

Offline tests for akure_access.completeness.grid_check: flagging settled grid
cells that lack a nearby OSM health/education facility tag. Uses
synthetic geometries; no network access required.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import geopandas as gpd
from shapely.geometry import Point, box

from akure_access.accessibility.scoring import build_grid, add_building_density
from akure_access.completeness.grid_check import flag_completeness, summarize_completeness


def _synthetic_boundary(size_m=2000):
    return gpd.GeoDataFrame(geometry=[box(0, 0, size_m, size_m)], crs="EPSG:32631")


def test_flag_completeness_flags_settled_unmapped_cell():
    boundary = _synthetic_boundary(2000)
    grid = build_grid(boundary, cell_size_m=500)

    # Settle cell (0,0)-(500,500) with 5 buildings; leave others empty
    buildings = gpd.GeoDataFrame(
        geometry=[Point(100, 100), Point(150, 150), Point(200, 200), Point(250, 250), Point(300, 300)],
        crs="EPSG:32631",
    )
    grid = add_building_density(grid, buildings)

    # No health facilities anywhere -> the settled cell should be flagged
    health = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    grid = flag_completeness(grid, health, schools, building_threshold=3, search_radius_m=1000)

    settled_cell = grid[grid["building_count"] >= 3].iloc[0]
    assert settled_cell["health_completeness_flag"] == True
    assert settled_cell["education_completeness_flag"] == True


def test_flag_completeness_does_not_flag_when_facility_nearby():
    boundary = _synthetic_boundary(2000)
    grid = build_grid(boundary, cell_size_m=500)

    buildings = gpd.GeoDataFrame(
        geometry=[Point(100, 100), Point(150, 150), Point(200, 200), Point(250, 250)],
        crs="EPSG:32631",
    )
    grid = add_building_density(grid, buildings)

    # A health facility right next to the settled cell -> should NOT be flagged
    health = gpd.GeoDataFrame(geometry=[Point(260, 260)], crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    grid = flag_completeness(grid, health, schools, building_threshold=3, search_radius_m=1000)

    settled_cell = grid[grid["building_count"] >= 3].iloc[0]
    assert settled_cell["health_completeness_flag"] == False
    assert settled_cell["education_completeness_flag"] == True  # still no school nearby


def test_flag_completeness_ignores_unsettled_cells():
    boundary = _synthetic_boundary(2000)
    grid = build_grid(boundary, cell_size_m=500)
    # No buildings at all -> every cell has building_count == 0
    empty_buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    grid = add_building_density(grid, empty_buildings)

    health = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    grid = flag_completeness(grid, health, schools, building_threshold=3, search_radius_m=1000)

    # Unsettled cells should never be flagged, regardless of facility absence
    assert not grid["health_completeness_flag"].any()
    assert not grid["education_completeness_flag"].any()


def test_summarize_completeness_reports_correct_percentages():
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)  # 4 cells

    # Settle all 4 cells
    buildings = gpd.GeoDataFrame(
        geometry=[
            Point(100, 100), Point(150, 150), Point(200, 200),
            Point(600, 100), Point(650, 150), Point(700, 200),
            Point(100, 600), Point(150, 650), Point(200, 700),
            Point(600, 600), Point(650, 650), Point(700, 700),
        ],
        crs="EPSG:32631",
    )
    grid = add_building_density(grid, buildings)

    # Completeness is checked from each cell's CENTROID, not its nearest
    # building, so place the health facility exactly at one cell's
    # centroid (250, 250) to guarantee that cell is not flagged, while
    # the other three cells' centroids remain far outside the radius.
    health = gpd.GeoDataFrame(geometry=[Point(250, 250)], crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    grid = flag_completeness(grid, health, schools, building_threshold=3, search_radius_m=100)
    summary = summarize_completeness(grid)

    assert summary["settled_cells"] == 4
    # Only the cell whose centroid coincides with the facility should be
    # considered covered; the other 3 settled cells' centroids are ~500m+
    # away, well outside the 100m search radius.
    assert summary["health_flagged"] == 3
    # No schools at all -> all 4 settled cells flagged for education
    assert summary["education_flagged"] == 4
    assert summary["education_flagged_pct"] == 100.0


def test_summarize_completeness_handles_no_settled_cells():
    boundary = _synthetic_boundary(1000)
    grid = build_grid(boundary, cell_size_m=500)
    empty_buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    grid = add_building_density(grid, empty_buildings)

    health = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    grid = flag_completeness(grid, health, schools)

    summary = summarize_completeness(grid)
    assert summary["settled_cells"] == 0


def _naive_linear_scan_flag(centroid, building_count, facilities_gdf, building_threshold, search_radius_m):
    """Reference implementation (the original pre-spatial-index approach), kept
    here only as a correctness oracle for the equivalence test below."""
    if building_count < building_threshold:
        return False
    if facilities_gdf.empty:
        return True
    nearby = facilities_gdf[facilities_gdf.distance(centroid) <= search_radius_m]
    return nearby.empty


def test_spatial_index_flagging_matches_naive_linear_scan():
    """
    flag_completeness() now uses gpd.sjoin_nearest() (spatial-indexed)
    instead of a per-cell linear distance scan against every facility.
    This test verifies the two approaches give IDENTICAL results across
    a realistic mix of settled/unsettled cells and near/far facilities,
    to confirm the performance rewrite didn't change behavior.
    """
    boundary = _synthetic_boundary(3000)
    grid = build_grid(boundary, cell_size_m=500)  # 36 cells

    import numpy as np
    rng = np.random.default_rng(42)
    building_points = [
        Point(rng.uniform(0, 3000), rng.uniform(0, 3000)) for _ in range(200)
    ]
    buildings = gpd.GeoDataFrame(geometry=building_points, crs="EPSG:32631")
    grid = add_building_density(grid, buildings)

    facility_points = [Point(rng.uniform(0, 3000), rng.uniform(0, 3000)) for _ in range(8)]
    health = gpd.GeoDataFrame(geometry=facility_points, crs="EPSG:32631")
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")  # also test the empty-facilities path

    scored = flag_completeness(grid, health, schools, building_threshold=3, search_radius_m=800)

    centroids = grid.geometry.centroid
    expected_health_flags = [
        _naive_linear_scan_flag(c, count, health, 3, 800)
        for c, count in zip(centroids, grid["building_count"])
    ]
    expected_edu_flags = [
        _naive_linear_scan_flag(c, count, schools, 3, 800)
        for c, count in zip(centroids, grid["building_count"])
    ]

    assert scored["health_completeness_flag"].tolist() == expected_health_flags
    assert scored["education_completeness_flag"].tolist() == expected_edu_flags
