"""
akure_access.accessibility

Network-based accessibility analysis: routable graph construction,
isochrone computation, settlement grid generation, and access-deficit
scoring for health and education facilities.
"""

from .network_graph import graph_from_roads, WALKING_SPEED_KPH, MODE_CONFIG
from .isochrones import (
    build_isochrones_for_facilities,
    compute_isochrone_polygon,
    nearest_facility_travel_time,
    nearest_facility_distance_and_time,
    batch_nearest_facility_distances,
    lookup_nearest_distance_time,
)
from .scoring import (
    build_grid,
    add_building_density,
    add_access_times,
    add_access_deficit_score,
    sanitize_for_export,
    DEFAULT_ACCESS_THRESHOLD_MIN,
    DEFAULT_GRID_CELL_SIZE_M,
)

__all__ = [
    "graph_from_roads",
    "WALKING_SPEED_KPH",
    "MODE_CONFIG",
    "build_isochrones_for_facilities",
    "compute_isochrone_polygon",
    "nearest_facility_travel_time",
    "nearest_facility_distance_and_time",
    "batch_nearest_facility_distances",
    "lookup_nearest_distance_time",
    "build_grid",
    "add_building_density",
    "add_access_times",
    "add_access_deficit_score",
    "sanitize_for_export",
    "DEFAULT_ACCESS_THRESHOLD_MIN",
    "DEFAULT_GRID_CELL_SIZE_M",
]
