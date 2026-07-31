"""
akure_access.completeness

Lightweight OSM data-completeness assessment for health and education
facilities, based on building density as a settlement-presence proxy.
"""

from .grid_check import (
    flag_completeness,
    summarize_completeness,
    DEFAULT_BUILDING_PRESENCE_THRESHOLD,
    DEFAULT_FACILITY_SEARCH_RADIUS_M,
)

__all__ = [
    "flag_completeness",
    "summarize_completeness",
    "DEFAULT_BUILDING_PRESENCE_THRESHOLD",
    "DEFAULT_FACILITY_SEARCH_RADIUS_M",
]
