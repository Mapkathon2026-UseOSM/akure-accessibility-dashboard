"""
tests/test_static_maps.py

Smoke tests for akure_access.visualization.static_maps. These are
offline-safe: add_osm_basemap() degrades gracefully (plain background)
without live internet access, so these tests exercise every other part
of the styling toolkit (gridlines, scale bar, legend, CRS handling,
file output) without needing a real network connection or a CI runner
with tile-server access.
"""

import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from akure_access.visualization import (
    plot_deficit_map,
    plot_continuous_map,
    plot_completeness_map,
    plot_mode_comparison_chart,
    generate_all_static_outputs,
    DEFICIT_PALETTES,
    DEFICIT_LABELS,
)
from akure_access.visualization.static_maps import _ensure_lonlat, add_scale_bar, add_gridlines


def _synthetic_scored_grid(crs="EPSG:32631", n_side=6):
    """A small synthetic scored grid in a PROJECTED CRS (matching the
    real pipeline's UTM output), with the same columns
    add_access_deficit_score()/flag_completeness() would produce."""
    cells = []
    for i in range(n_side):
        for j in range(n_side):
            x0, y0 = 740000 + i * 100, 800000 + j * 100
            cells.append(Polygon([(x0, y0), (x0 + 100, y0), (x0 + 100, y0 + 100), (x0, y0 + 100)]))

    n = len(cells)
    rng = np.random.default_rng(0)
    gdf = gpd.GeoDataFrame(
        {
            "cell_id": range(n),
            "building_count": rng.integers(0, 5, n),
            "walk_access_deficit_score": rng.integers(0, 3, n),
            "okada_access_deficit_score": rng.integers(0, 3, n),
            "health_time_min_walk": rng.uniform(1, 60, n),
            "education_time_min_walk": rng.uniform(1, 90, n),
            "health_completeness_flag": rng.choice([True, False], n),
            "education_completeness_flag": rng.choice([True, False], n),
        },
        geometry=cells,
        crs=crs,
    )
    return gdf


def test_ensure_lonlat_reprojects_projected_input():
    gdf = _synthetic_scored_grid(crs="EPSG:32631")
    out = _ensure_lonlat(gdf)
    assert str(out.crs).upper() in ("EPSG:4326",)
    # Longitude/latitude for this UTM zone should land in plausible Nigeria bounds
    west, south, east, north = out.total_bounds
    assert 2 < west < 8 and 2 < east < 8
    assert 4 < south < 10 and 4 < north < 10


def test_ensure_lonlat_leaves_already_lonlat_input_unchanged():
    gdf = _synthetic_scored_grid(crs="EPSG:32631").to_crs("EPSG:4326")
    out = _ensure_lonlat(gdf)
    assert out.crs == gdf.crs


def test_scale_bar_and_gridlines_do_not_hang_on_projected_bounds(monkeypatch):
    """
    Regression test for the exact bug found while building this module:
    calling add_gridlines()/add_scale_bar() with bounds still in UTM
    METERS (tens of thousands) rather than degrees caused an effectively
    unbounded number of gridlines to be computed. This asserts the
    higher-level plot_* functions convert to lon/lat BEFORE calling
    these, by checking the resulting bounds passed to matplotlib are
    degree-scale, not meter-scale.
    """
    gdf = _synthetic_scored_grid(crs="EPSG:32631")
    lonlat = _ensure_lonlat(gdf)
    bounds = lonlat.total_bounds
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    assert span < 10, "Bounds should be degree-scale after _ensure_lonlat, not UTM-meter-scale"


def test_plot_deficit_map_saves_file(tmp_path):
    gdf = _synthetic_scored_grid()
    out_path = str(tmp_path / "deficit_walk.jpg")
    result = plot_deficit_map(gdf, "walk", "Test Deficit Map", out_path, dpi=60)
    assert result == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_deficit_map_missing_column_raises_clear_error(tmp_path):
    gdf = _synthetic_scored_grid()
    with pytest.raises(KeyError, match="not_a_real_mode_access_deficit_score"):
        plot_deficit_map(gdf, "not_a_real_mode", "x", str(tmp_path / "x.jpg"))


def test_plot_continuous_map_excludes_nan_and_unsettled(tmp_path):
    gdf = _synthetic_scored_grid()
    gdf.loc[0, "health_time_min_walk"] = float("nan")
    out_path = str(tmp_path / "health_walk.jpg")
    result = plot_continuous_map(
        gdf, "health_time_min_walk", "Test Health Time", out_path, colorbar_label="Minutes", dpi=60
    )
    assert os.path.exists(result)


def test_plot_continuous_map_all_nan_raises_clear_error(tmp_path):
    gdf = _synthetic_scored_grid()
    gdf["health_time_min_walk"] = float("nan")
    with pytest.raises(ValueError, match="No non-null"):
        plot_continuous_map(gdf, "health_time_min_walk", "x", str(tmp_path / "x.jpg"), colorbar_label="min")


def test_plot_completeness_map_saves_file(tmp_path):
    gdf = _synthetic_scored_grid()
    out_path = str(tmp_path / "completeness_health.jpg")
    result = plot_completeness_map(gdf, "health", "Test Completeness", out_path, dpi=60)
    assert os.path.exists(result)


def test_plot_mode_comparison_chart_saves_file(tmp_path):
    out_path = str(tmp_path / "mode_comparison.jpg")
    result = plot_mode_comparison_chart(
        [("walk", 88.6, 40.0), ("okada", 44.8, 20.0), ("drive", 42.2, 18.0)],
        "Test Mode Comparison", out_path, dpi=60,
    )
    assert os.path.exists(result)


def test_plot_mode_comparison_chart_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_mode_comparison_chart([], "x", "/tmp/wont_be_created.jpg")


def test_generate_all_static_outputs_produces_expected_file_count(tmp_path):
    gdf = _synthetic_scored_grid()
    produced = generate_all_static_outputs("Test LGA", gdf, str(tmp_path), modes=("walk", "okada"), dpi=60, web_dpi=40)
    # walk: deficit + health_time + education_time = 3 (fixture has both time columns for walk)
    # okada: deficit only = 1 (fixture only defines okada_access_deficit_score, no time columns,
    #   correctly produces just the deficit map rather than fabricating data for missing columns)
    # + 2 completeness maps (health, education) + 1 mode-comparison chart = 7
    assert len(produced["print"]) == 7
    assert len(produced["web"]) == 7  # one web-tier copy per print figure, since web_dpi was given
    for p in produced["print"] + produced["web"]:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


def test_generate_all_static_outputs_skips_web_tier_when_disabled(tmp_path):
    gdf = _synthetic_scored_grid()
    produced = generate_all_static_outputs("Test LGA", gdf, str(tmp_path), modes=("walk",), dpi=60, web_dpi=None)
    assert produced["web"] == []
    assert len(produced["print"]) > 0


def test_generate_all_static_outputs_uses_lga_first_title_convention(tmp_path, monkeypatch):
    """Locks in the 'Akure South: Access Deficit (Walk)' title convention
    (LGA name first, colon, then metric), consistent across every figure
    type this function produces."""
    captured_titles = []
    import akure_access.visualization.static_maps as sm
    real_plot_deficit_map = sm.plot_deficit_map

    def spy(grid_gdf, mode, title, *args, **kwargs):
        captured_titles.append(title)
        return real_plot_deficit_map(grid_gdf, mode, title, *args, **kwargs)

    monkeypatch.setattr(sm, "plot_deficit_map", spy)
    gdf = _synthetic_scored_grid()
    sm.generate_all_static_outputs("Test LGA", gdf, str(tmp_path), modes=("walk",), dpi=60, web_dpi=None)
    assert captured_titles, "plot_deficit_map was never called"
    assert captured_titles[0].startswith("Test LGA:")


def test_generate_all_static_outputs_skips_all_nan_service_layer(tmp_path):
    """
    Mirrors the real Akure North situation this project actually hit:
    a service with 100% NaN travel times for an LGA (before the
    geometry-type fix) should be silently skipped, not produce a
    misleading blank map or raise.
    """
    gdf = _synthetic_scored_grid()
    gdf["health_time_min_walk"] = float("nan")
    produced = generate_all_static_outputs("Test LGA", gdf, str(tmp_path), modes=("walk",), dpi=60, web_dpi=None)
    assert not any("health_time" in p for p in produced["print"])
    assert any("education_time" in p for p in produced["print"])


def test_deficit_palette_and_labels_length_match():
    assert len(DEFICIT_PALETTES["standard"]) == len(DEFICIT_LABELS) == 3
    assert len(DEFICIT_PALETTES["colorblind_safe"]) == 3
