"""
akure_access

Health and education accessibility analysis for Nigerian LGAs, built
for Akure North and Akure South (Ondo State) as part of Map<>kathon
2026.

This package answers one question per settlement: "how long would it
take to reach the nearest health facility or school, on foot, by
okada (motorcycle taxi), or by vehicle?" -- combined with an OSM
completeness check, so that "no facility found nearby" can be
distinguished from "OSM simply hasn't mapped it yet."

Two subpackages:
    akure_access.accessibility
        Builds the analysis grid, estimates population via building
        density, constructs routable road networks per transport
        mode, and computes travel-time-based access-deficit scores.
    akure_access.completeness
        Flags grid cells that look settled (dense buildings) but have
        no nearby OSM-tagged facility -- a likely data gap rather
        than a confirmed lack of access.

This package expects its input data (roads, buildings, health
facilities, schools) to come from the companion `lga_extractor`
package (see the sibling `lga-osm-extractor` repository), which
handles OSM querying, cleaning, and export. akure_access itself never
queries OSM directly.

Quick start
-----------
    from akure_access.accessibility import build_grid, add_building_density
    from akure_access.completeness import flag_completeness

    grid = build_grid(boundary_gdf, cell_size_m=500)
    grid = add_building_density(grid, buildings_gdf)
    grid = flag_completeness(grid, health_gdf, schools_gdf)

See notebooks/01-05 for the full, worked analysis pipeline, and
dashboard/app.py for the Streamlit presentation layer built on top of
this package's output.
"""

from .accessibility import (
    build_grid,
    add_building_density,
    add_access_times,
    add_access_deficit_score,
    sanitize_for_export,
)
from .completeness import flag_completeness, summarize_completeness

__all__ = [
    "build_grid",
    "add_building_density",
    "add_access_times",
    "add_access_deficit_score",
    "sanitize_for_export",
    "flag_completeness",
    "summarize_completeness",
]

__version__ = "0.1.0"
