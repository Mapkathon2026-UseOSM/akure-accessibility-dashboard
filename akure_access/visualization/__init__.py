"""
akure_access.visualization

Publication-styled static maps and charts for the accessibility
analysis (OSM basemap, gridlines, north arrow, scale bar, legend).
See static_maps.py for the full toolkit and design notes.
"""

from .static_maps import (
    add_north_arrow,
    add_scale_bar,
    add_gridlines,
    add_osm_basemap,
    plot_deficit_map,
    plot_continuous_map,
    plot_completeness_map,
    plot_mode_comparison_chart,
    generate_all_static_outputs,
    DEFICIT_PALETTES,
    DEFICIT_LABELS,
)

__all__ = [
    "add_north_arrow",
    "add_scale_bar",
    "add_gridlines",
    "add_osm_basemap",
    "plot_deficit_map",
    "plot_continuous_map",
    "plot_completeness_map",
    "plot_mode_comparison_chart",
    "generate_all_static_outputs",
    "DEFICIT_PALETTES",
    "DEFICIT_LABELS",
]
