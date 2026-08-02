"""
tests/test_insights.py

Regression tests for akure_access.insights, the module that generates
data-driven captions for both the interactive dashboard and static
maps. These tests use synthetic but realistic scored grids to verify
every caption function produces accurate, non-crashing output across
the range of situations the real data actually hits: missing columns,
all-NaN service layers (the exact shape of the real Akure North bug
found earlier in this project), tied mode comparisons, and directional
skew detection.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from akure_access.insights import (
    describe_deficit_map,
    describe_continuous_map,
    describe_completeness_map,
    describe_mode_comparison_chart,
    describe_interactive_view,
    DEFAULT_THRESHOLDS_MIN,
)


def _grid(n_side=8, crs="EPSG:4326"):
    """A synthetic settled grid with real-shaped scored columns,
    deliberately skewed so the eastern half is worse-served, to give the
    directional-skew logic something genuine to detect."""
    cells, rows = [], []
    rng = np.random.default_rng(0)
    for i in range(n_side):
        for j in range(n_side):
            x0, y0 = 5.0 + i * 0.01, 7.0 + j * 0.01
            cells.append(Polygon([(x0, y0), (x0 + 0.01, y0), (x0 + 0.01, y0 + 0.01), (x0, y0 + 0.01)]))
            is_east = i >= n_side // 2
            walk_score = 2 if is_east else rng.integers(0, 2)
            rows.append({
                "building_count": rng.integers(1, 5),
                "walk_access_deficit_score": walk_score,
                "okada_access_deficit_score": rng.integers(0, 3),
                "drive_access_deficit_score": rng.integers(0, 2),
                "health_time_min_walk": rng.uniform(40, 60) if is_east else rng.uniform(5, 25),
                "education_time_min_walk": rng.uniform(10, 90),
                "health_completeness_flag": bool(rng.choice([True, False])),
                "education_completeness_flag": bool(rng.choice([True, False])),
            })
    return gpd.GeoDataFrame(rows, geometry=cells, crs=crs)


def test_describe_deficit_map_returns_real_numbers():
    grid = _grid()
    text = describe_deficit_map(grid, "Test LGA", "walk")
    assert "Test LGA" in text
    assert "%" in text
    assert "Green cells are well served" in text


def test_describe_deficit_map_missing_mode_column_does_not_crash():
    grid = _grid()
    text = describe_deficit_map(grid, "Test LGA", "not_a_real_mode")
    assert "No scored data available" in text


def test_describe_deficit_map_reports_directional_skew():
    """The synthetic fixture deliberately makes the eastern half
    worse-served; the walk deficit map should detect and mention this."""
    grid = _grid()
    text = describe_deficit_map(grid, "Test LGA", "walk")
    assert "east" in text.lower()


def test_describe_continuous_map_returns_real_numbers():
    grid = _grid()
    text = describe_continuous_map(grid, "Test LGA", "walk", "health")
    assert "average health travel time" in text
    assert "%" in text


def test_describe_continuous_map_all_nan_column_does_not_crash():
    grid = _grid()
    grid["health_time_min_walk"] = float("nan")
    text = describe_continuous_map(grid, "Test LGA", "walk", "health")
    assert "No health travel-time data available" in text


def test_describe_continuous_map_missing_column_does_not_crash():
    grid = _grid()
    text = describe_continuous_map(grid, "Test LGA", "okada", "health")  # no *_okada column in fixture
    assert "No health travel-time data available" in text


def test_describe_completeness_map_returns_real_numbers():
    grid = _grid()
    text = describe_completeness_map(grid, "Test LGA", "health")
    assert "%" in text
    assert "possible data gap" in text


def test_describe_completeness_map_missing_column_does_not_crash():
    grid = _grid()
    text = describe_completeness_map(grid, "Test LGA", "not_a_real_service")
    assert "No not_a_real_service completeness data available" in text


def test_describe_mode_comparison_chart_normal_case():
    text = describe_mode_comparison_chart(
        [("walk", 88.6, 40.0), ("okada", 44.8, 20.0), ("drive", 42.2, 18.0)], "Test LGA"
    )
    assert "walking shows the highest" in text
    assert "driving shows the lowest" in text


def test_describe_mode_comparison_chart_handles_ties_gracefully():
    """Regression test for the real Akure North (pre-fix) data shape:
    walk and drive both showed exactly 100.0% underserved, which without
    this handling produced the nonsensical 'highest at 100.0%, lowest at
    100.0%' phrasing."""
    text = describe_mode_comparison_chart(
        [("walk", 100.0, 92.9), ("drive", 100.0, 60.4)], "Test LGA"
    )
    assert "highest at 100.0%" not in text
    assert "similarly high" in text


def test_describe_mode_comparison_chart_empty_input_does_not_crash():
    text = describe_mode_comparison_chart([], "Test LGA")
    assert "No mode-comparison data available" in text


def test_describe_interactive_view_dispatches_correctly():
    grid = _grid()
    combined = describe_interactive_view(grid, "Test LGA", "walk", "Combined")
    health = describe_interactive_view(grid, "Test LGA", "walk", "Health only")
    education = describe_interactive_view(grid, "Test LGA", "walk", "Education only")

    assert "Green cells are well served" in combined
    assert "average health travel time" in health
    assert "average education travel time" in education


def test_no_em_dash_in_any_generated_caption():
    """Every generated caption must be free of em-dashes, per the
    project's own style requirement."""
    grid = _grid()
    texts = [
        describe_deficit_map(grid, "Test LGA", "walk"),
        describe_continuous_map(grid, "Test LGA", "walk", "health"),
        describe_completeness_map(grid, "Test LGA", "health"),
        describe_mode_comparison_chart([("walk", 80.0, 40.0), ("drive", 30.0, 10.0)], "Test LGA"),
        describe_interactive_view(grid, "Test LGA", "walk", "Combined"),
    ]
    for text in texts:
        assert "\u2014" not in text, f"em-dash found in: {text}"


def test_describe_completeness_map_uses_correct_article():
    """'an education facility', not 'a education facility'; 'a health
    facility', not 'an health facility'."""
    grid = _grid()
    education_text = describe_completeness_map(grid, "Test LGA", "education")
    health_text = describe_completeness_map(grid, "Test LGA", "health")
    assert "an education facility" in education_text
    assert "a education facility" not in education_text
    assert "a health facility" in health_text
    assert "an health facility" not in health_text


def test_default_thresholds_match_notebook_config():
    """Locks in that the module's default thresholds stay in sync with
    ACCESS_THRESHOLDS_MIN in Notebook 03's configuration cell."""
    assert DEFAULT_THRESHOLDS_MIN == {"walk": 30, "okada": 20, "drive": 15}
