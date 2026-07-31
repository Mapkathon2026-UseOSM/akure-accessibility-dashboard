"""
tests/test_network_graph.py

Offline tests for akure_access.accessibility.network_graph, using the
geometry-fallback graph construction path (no live OSM/network calls),
plus a live-OSM integration test marked separately.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from akure_access.accessibility.network_graph import graph_from_roads, MODE_CONFIG
from akure_access.accessibility.isochrones import (
    nearest_facility_distance_and_time,
    batch_nearest_facility_distances,
    lookup_nearest_distance_time,
    compute_isochrone_polygon,
    build_isochrones_for_facilities,
)


def _simple_road_grid(size_m=1000, step=500):
    """
    A simple connected grid of roads for fallback graph construction.

    Built as individual segments between adjacent grid points (rather
    than one long LineString per row/column) so that intersections
    share an actual coordinate/node -- a single 2-point LineString
    spanning an entire row would not create a shared node at interior
    crossing points, leaving the graph disconnected in a way that's easy
    to miss.
    """
    coords = list(range(0, size_m + 1, step))
    lines = []

    for y in coords:
        for i in range(len(coords) - 1):
            lines.append(LineString([(coords[i], y), (coords[i + 1], y)]))

    for x in coords:
        for i in range(len(coords) - 1):
            lines.append(LineString([(x, coords[i]), (x, coords[i + 1])]))

    return gpd.GeoDataFrame({"osmid": range(len(lines))}, geometry=lines, crs="EPSG:32631")


def test_mode_config_has_expected_modes():
    assert set(MODE_CONFIG.keys()) == {"walk", "okada", "drive"}
    assert MODE_CONFIG["walk"]["speed_kph"] < MODE_CONFIG["okada"]["speed_kph"]
    assert MODE_CONFIG["okada"]["speed_kph"] < MODE_CONFIG["drive"]["speed_kph"]


def test_graph_from_roads_geometry_fallback_assigns_travel_times():
    roads = _simple_road_grid()
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    assert G.graph["mode"] == "walk"
    assert G.number_of_edges() > 0
    for _, _, data in G.edges(data=True):
        assert "travel_time_min" in data
        assert data["travel_time_min"] >= 0


def test_graph_from_roads_speed_override():
    roads = _simple_road_grid()
    G_default = graph_from_roads(roads, boundary_polygon=None, mode="walk")
    G_override = graph_from_roads(roads, boundary_polygon=None, mode="walk", speed_kph=10.0)

    # Doubling speed should roughly halve travel time for the same edges
    default_times = [d["travel_time_min"] for _, _, d in G_default.edges(data=True)]
    override_times = [d["travel_time_min"] for _, _, d in G_override.edges(data=True)]
    assert sum(override_times) < sum(default_times)


def test_graph_from_roads_invalid_mode_raises():
    roads = _simple_road_grid()
    with pytest.raises(ValueError):
        graph_from_roads(roads, boundary_polygon=None, mode="teleport")


def test_nearest_facility_distance_and_time_finds_closer_facility():
    roads = _simple_road_grid()
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    origin = Point(0, 0)
    facilities = gpd.GeoDataFrame(
        geometry=[Point(500, 0), Point(1000, 1000)],  # one near, one far
        crs="EPSG:32631",
    )

    distance_km, time_min = nearest_facility_distance_and_time(G, origin, facilities)

    assert distance_km != float("inf")
    assert time_min != float("inf")
    # Should route to the nearer facility (500m away along the grid), not the far one
    assert distance_km < 1.0


def test_nearest_facility_distance_and_time_handles_empty_facilities():
    roads = _simple_road_grid()
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    origin = Point(0, 0)
    empty_facilities = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")

    distance_km, time_min = nearest_facility_distance_and_time(G, origin, empty_facilities)
    assert distance_km == float("inf")
    assert time_min == float("inf")


def test_batch_nearest_facility_distances_matches_naive_per_pair_approach():
    """
    The whole point of batch_nearest_facility_distances() +
    lookup_nearest_distance_time() is to give the SAME result as the
    naive per-pair nearest_facility_distance_and_time(), just computed
    once for the whole graph instead of once per origin. This test
    verifies that equivalence directly, for several origins against the
    same facility set, on the geometry-fallback graph.
    """
    roads = _simple_road_grid()
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    facilities = gpd.GeoDataFrame(
        geometry=[Point(500, 0), Point(1000, 1000)],
        crs="EPSG:32631",
    )

    distances_by_node = batch_nearest_facility_distances(G, facilities)
    assert len(distances_by_node) > 0

    test_origins = [Point(0, 0), Point(1000, 0), Point(0, 1000), Point(500, 500)]

    for origin in test_origins:
        naive_dist, naive_time = nearest_facility_distance_and_time(G, origin, facilities)
        batch_dist, batch_time = lookup_nearest_distance_time(G, origin, distances_by_node)

        assert batch_dist == pytest.approx(naive_dist, rel=1e-6), (
            f"Distance mismatch at {origin}: naive={naive_dist}, batch={batch_dist}"
        )
        assert batch_time == pytest.approx(naive_time, rel=1e-6), (
            f"Time mismatch at {origin}: naive={naive_time}, batch={batch_time}"
        )


def test_batch_nearest_facility_distances_handles_empty_facilities():
    roads = _simple_road_grid()
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    empty_facilities = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    distances_by_node = batch_nearest_facility_distances(G, empty_facilities)
    assert distances_by_node == {}

    origin = Point(0, 0)
    dist, time = lookup_nearest_distance_time(G, origin, distances_by_node)
    assert dist == float("inf")
    assert time == float("inf")


def test_batch_nearest_facility_distances_much_faster_than_naive_at_scale():
    """
    Performance regression guard: confirms the batch approach is
    genuinely faster than the naive per-pair approach when there are
    many origins to score against the same facility set -- this is the
    actual scenario (many grid cells, few facilities) that caused the
    real-world hour-long runtime this fix addresses. Uses a generous
    margin (batch should be at least 2x faster) rather than a tight
    threshold, to avoid flakiness on slower CI runners.
    """
    import time as time_module

    roads = _simple_road_grid(size_m=2000, step=100)  # a denser grid, more nodes
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    facilities = gpd.GeoDataFrame(
        geometry=[Point(1000, 1000), Point(0, 0), Point(2000, 2000)],
        crs="EPSG:32631",
    )
    origins = [Point(x, y) for x in range(0, 2001, 200) for y in range(0, 2001, 200)]

    start = time_module.perf_counter()
    for origin in origins:
        nearest_facility_distance_and_time(G, origin, facilities)
    naive_elapsed = time_module.perf_counter() - start

    start = time_module.perf_counter()
    distances_by_node = batch_nearest_facility_distances(G, facilities)
    for origin in origins:
        lookup_nearest_distance_time(G, origin, distances_by_node)
    batch_elapsed = time_module.perf_counter() - start

    assert batch_elapsed < naive_elapsed / 2, (
        f"Expected batch approach to be at least 2x faster: "
        f"naive={naive_elapsed:.4f}s, batch={batch_elapsed:.4f}s"
    )


def test_compute_isochrone_polygon_returns_larger_area_for_longer_trip_time():
    """
    Core sanity check for the isochrone feature now exposed in the
    dashboard: a longer trip-time budget must never produce a SMALLER
    reachable-area polygon than a shorter one, on the same graph from
    the same origin. (They may be equal if the graph is small enough
    that the whole reachable area is already covered at the shorter
    time -- this is not a failure, just means the test graph is small
    relative to the trip times used.)
    """
    roads = _simple_road_grid(size_m=2000, step=200)
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")
    origin = Point(1000, 1000)

    poly_15 = compute_isochrone_polygon(G, origin, trip_time_min=15)
    poly_30 = compute_isochrone_polygon(G, origin, trip_time_min=30)

    assert poly_15 is not None
    assert poly_30 is not None
    assert poly_30.area >= poly_15.area


def test_compute_isochrone_polygon_returns_none_for_unmatchable_origin():
    """
    An origin point cannot always be matched to a graph node (e.g. an
    empty graph, or a malformed point) -- this must return None rather
    than raising, since build_isochrones_for_facilities() relies on
    this to skip unmatchable facilities gracefully rather than aborting
    the whole batch.
    """
    import networkx as nx
    empty_graph = nx.MultiDiGraph()
    result = compute_isochrone_polygon(empty_graph, Point(0, 0), trip_time_min=15)
    assert result is None


def test_build_isochrones_for_facilities_one_row_per_facility_per_trip_time():
    """
    Verifies the actual output contract dashboard/app.py depends on:
    one row per (facility, trip_time) combination, with the expected
    columns, for a small set of facilities and trip times.
    """
    roads = _simple_road_grid(size_m=2000, step=200)
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    facilities = gpd.GeoDataFrame(
        {"name": ["Clinic A", "Clinic B"], "osmid": [101, 102]},
        geometry=[Point(500, 500), Point(1500, 1500)],
        crs="EPSG:32631",
    )

    isochrones = build_isochrones_for_facilities(G, facilities, trip_times_min=(15, 30))

    assert len(isochrones) == 4  # 2 facilities x 2 trip times
    assert set(isochrones.columns) >= {"facility_name", "osmid", "trip_time_min", "geometry"}
    assert set(isochrones["trip_time_min"].unique()) == {15, 30}
    assert set(isochrones["facility_name"].unique()) == {"Clinic A", "Clinic B"}
    assert isochrones.crs.to_string() == "EPSG:32631"


def test_build_isochrones_for_facilities_handles_empty_facilities():
    roads = _simple_road_grid(size_m=2000, step=200)
    G = graph_from_roads(roads, boundary_polygon=None, mode="walk")

    empty_facilities = gpd.GeoDataFrame(geometry=[], crs="EPSG:32631")
    isochrones = build_isochrones_for_facilities(G, empty_facilities, trip_times_min=(15, 30))

    assert isochrones.empty


@pytest.mark.integration
def test_graph_from_roads_live_osm_polygon():
    """
    Integration test building a real graph from a live OSM boundary
    polygon. Requires network access. Run explicitly with:
        pytest -m integration
    """
    from lga_extractor import resolve_boundary

    boundary = resolve_boundary(lga_name="Akure North", state_name="Ondo")
    polygon = boundary.geometry.iloc[0]

    G = graph_from_roads(gpd.GeoDataFrame(), boundary_polygon=polygon, mode="walk")
    assert G.number_of_nodes() > 0
    assert G.graph["mode"] == "walk"
