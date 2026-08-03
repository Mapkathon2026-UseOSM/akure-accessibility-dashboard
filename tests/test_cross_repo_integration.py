"""
tests/test_cross_repo_integration.py

Integration tests verifying that lga-osm-extractor's actual output
schema is what akure-accessibility-dashboard's analysis functions expect, i.e. that the two sibling repositories genuinely work together, not
just that each one's own unit tests pass in isolation.

These tests require lga_extractor to be installed alongside this
repo's own package (akure_access). They are automatically skipped
(not failed) if lga_extractor is not importable, so the regular
single-repo test suite / CI job is unaffected. A dedicated CI workflow
(.github/workflows/cross-repo-integration.yml) checks out both
repositories and installs both packages specifically to run this file.

No live OSM/Overpass calls are made here, lga_extractor's cleaning
and export functions are exercised directly against small synthetic
GeoDataFrames shaped like real OSM output, keeping this fast and
network-independent while still testing the real schema contract
between the two packages.
"""

import os
import sys
import tempfile
import shutil

import pytest
import geopandas as gpd
from shapely.geometry import Point, LineString

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

lga_extractor = pytest.importorskip(
    "lga_extractor",
    reason="lga_extractor not installed, cross-repo integration tests skipped. "
    "Install it with 'pip install -e ../lga-osm-extractor' to run these.",
)

from lga_extractor.clean import clean_layers
from lga_extractor.export import export_layers

from akure_access.accessibility import (
    build_grid,
    add_building_density,
    add_access_times,
    add_access_deficit_score,
    sanitize_for_export,
)
from akure_access.completeness import flag_completeness


def _synthetic_raw_extraction():
    """
    Mimics the shape of raw layers.extract_layers() output for one small
    synthetic LGA: a few buildings, a couple of facilities, and a tiny
    connected road grid, enough to exercise the full pipeline without
    needing a real OSM/Overpass query.

    Coordinates are small offsets (thousandths of a degree, i.e. roughly
    ~100m scale) around a real-world reference point near Akure, Nigeria
    (7.25 N, 5.20 E), rather than arbitrary large numbers. This matters:
    labeling large plane-scale coordinates (e.g. x=650, y=650) as
    EPSG:4326 lat/lon would put them far outside valid longitude/
    latitude range, and reprojecting such a "coordinate" to a UTM CRS
    sends it to infinity, exactly the kind of bug this synthetic
    fixture is supposed to help catch in the REAL pipeline, not
    introduce via unrealistic test data of its own.
    """
    lon0, lat0 = 5.200, 7.250  # a real-world reference point near Akure
    d = 0.001  # roughly 100m per unit at this latitude

    buildings = gpd.GeoDataFrame(
        {"building": ["yes"] * 6},
        geometry=[
            Point(lon0 + 1 * d, lat0 + 1 * d), Point(lon0 + 1.5 * d, lat0 + 1.2 * d), Point(lon0 + 1.8 * d, lat0 + 0.9 * d),
            Point(lon0 + 6.5 * d, lat0 + 6.5 * d), Point(lon0 + 7.0 * d, lat0 + 6.8 * d), Point(lon0 + 7.2 * d, lat0 + 6.2 * d),
        ],
        crs="EPSG:4326",
    )
    health = gpd.GeoDataFrame(
        {"amenity": ["clinic"]},
        geometry=[Point(lon0 + 1.4 * d, lat0 + 1.1 * d)],
        crs="EPSG:4326",
    )
    schools = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")  # deliberately empty

    # A small connected grid of road segments covering both building clusters
    coords = [c * d for c in [0, 2, 4, 6, 8]]
    lines = []
    for y in coords:
        for i in range(len(coords) - 1):
            lines.append(LineString([(lon0 + coords[i], lat0 + y), (lon0 + coords[i + 1], lat0 + y)]))
    for x in coords:
        for i in range(len(coords) - 1):
            lines.append(LineString([(lon0 + x, lat0 + coords[i]), (lon0 + x, lat0 + coords[i + 1])]))
    roads = gpd.GeoDataFrame({"highway": ["residential"] * len(lines)}, geometry=lines, crs="EPSG:4326")

    return {
        "roads": roads,
        "buildings": buildings,
        "health_facilities": health,
        "schools": schools,
        "_warnings": [],
    }


def test_extractor_output_schema_matches_dashboard_expectations():
    """
    Runs lga_extractor's real cleaning + export functions on synthetic
    raw data, then feeds the exported files straight into
    akure_access's scoring pipeline, verifying the column names and
    types lga_extractor actually produces (osmid, name, geometry) are
    exactly what akure_access's functions expect to consume, with no
    schema mismatch requiring manual adjustment.
    """
    raw = _synthetic_raw_extraction()

    # Pass a synthetic boundary near Akure so clean_layers() exercises its
    # auto-UTM-zone-selection path (resolve_target_crs()) exactly the way
    # pipeline.extract_lga() calls it in production, rather than only
    # testing the no-boundary fallback path.
    lon0, lat0 = 5.200, 7.250
    d = 0.001
    synthetic_boundary = gpd.GeoDataFrame(
        geometry=[gpd.GeoSeries(raw["roads"].geometry).union_all().convex_hull.buffer(d)],
        crs="EPSG:4326",
    )
    cleaned = clean_layers(raw, boundary_gdf=synthetic_boundary)
    cleaned.pop("_warnings", None)

    # Akure is in UTM Zone 31N, confirms the auto-selected zone matches
    # the original hardcoded default for this project's actual study area.
    assert cleaned["roads"].crs.to_string() == "EPSG:32631"

    tmp_dir = tempfile.mkdtemp()
    try:
        exported = export_layers(cleaned, tmp_dir)

        # Confirm the extractor actually produced the files the dashboard expects
        for layer in ["roads", "buildings", "health_facilities"]:
            assert layer in exported, f"Expected '{layer}' in extractor output"
            assert os.path.exists(exported[layer]["geojson"])

        roads_gdf = gpd.read_file(exported["roads"]["geojson"])
        buildings_gdf = gpd.read_file(exported["buildings"]["geojson"])
        health_gdf = gpd.read_file(exported["health_facilities"]["geojson"])
        schools_gdf = gpd.GeoDataFrame(geometry=[], crs=roads_gdf.crs)  # was empty, correctly skipped

        # Feed extractor output directly into the dashboard's grid pipeline
        boundary = gpd.GeoDataFrame(
            geometry=[roads_gdf.union_all().convex_hull.buffer(50)], crs=roads_gdf.crs
        )
        grid = build_grid(boundary, cell_size_m=200)
        grid = add_building_density(grid, buildings_gdf)
        assert grid["building_count"].sum() == len(buildings_gdf), (
            "Building count from extractor output doesn't match what the dashboard "
            "grid join found, possible schema mismatch between the two repos."
        )

        grid = flag_completeness(grid, health_gdf, schools_gdf, building_threshold=1, search_radius_m=300)
        assert "health_completeness_flag" in grid.columns
        assert "education_completeness_flag" in grid.columns

        grid = add_access_times(
            grid, roads_gdf, health_gdf, schools_gdf,
            boundary_polygon_wgs84=None,  # use geometry-fallback graph (no live OSM needed)
            modes=("walk",),
        )
        assert "health_time_min_walk" in grid.columns

        grid = add_access_deficit_score(grid, threshold_min=30, mode="walk")
        assert "walk_access_deficit_score" in grid.columns

        final = sanitize_for_export(grid)
        assert not (final["health_time_min_walk"] == float("inf")).any(), (
            "sanitize_for_export should have converted any inf values to NaN"
        )

    finally:
        shutil.rmtree(tmp_dir)


def test_extractor_run_log_captures_environment_for_reproducibility():
    """
    Confirms lga_extractor's run log (used for reproducibility, per the
    Map<>kathon judging criteria) actually captures package versions, this is what a reviewer checking reproducibility across the two
    repos would rely on.
    """
    from lga_extractor.logging_utils import log_run

    tmp_dir = tempfile.mkdtemp()
    try:
        log_path = log_run(
            lga_name="Test LGA",
            state_name="Test State",
            tag_config={"roads": {"highway": True}},
            output_dir=tmp_dir,
            boundary_source="test:synthetic",
        )
        import json
        with open(log_path) as f:
            log = json.load(f)

        assert "environment" in log
        assert "package_versions" in log["environment"]
        assert "osmnx" in log["environment"]["package_versions"]
    finally:
        shutil.rmtree(tmp_dir)
