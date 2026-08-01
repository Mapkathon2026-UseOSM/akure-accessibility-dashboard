# akure_access/

## Purpose

The core, tested analysis package behind this project. Everything in
here is pure Python (no notebook-only logic), given the right input
GeoDataFrames, these functions build the analysis grid, score
accessibility, and flag likely OSM data gaps, independent of whether
they're called from a notebook, `dashboard/app.py`, or a future script.

## Architecture

```
akure_access/
├── accessibility/
│   ├── network_graph.py   # builds a routable graph from road geometries, per transport mode
│   ├── isochrones.py      # nearest-facility routing (naive + fast batch versions); isochrone-polygon helpers for the dashboard's optional catchment overlay
│   └── scoring.py         # grid generation, building density, orchestrates routing into a final access-deficit score
└── completeness/
    └── grid_check.py      # flags settled-but-unmapped cells (spatial-indexed nearest-facility check)
```

## Workflow (how these modules fit together)

1. `scoring.build_grid()` creates a uniform 500m grid over the LGA boundary.
2. `scoring.add_building_density()` counts OSM buildings per cell, as a
   population proxy (see `../docs/methodology.md` for why building
   density rather than actual population data).
3. `completeness.flag_completeness()` checks each settled cell against
   the health/school facility layers using a spatial index
   (`geopandas.sjoin_nearest`), flagging cells that look inhabited but
   have no nearby OSM-tagged facility.
4. `scoring.add_access_times()` builds a routable graph per mode
   (`network_graph.graph_from_roads()`) and computes nearest-facility
   travel time per cell, using multi-source Dijkstra
   (`isochrones.batch_nearest_facility_distances()`) rather than one
   shortest-path search per cell, see that function's docstring for
   why this matters at scale.
5. `scoring.add_access_deficit_score()` combines health + education
   travel times into a single 0/1/2 deficit score per cell, per mode.
6. `scoring.sanitize_for_export()` converts internal `inf` sentinel
   values (unreachable cells) to `NaN` for clean GeoJSON export, must
   be called AFTER scoring is complete, not before (see the function's
   docstring and the two regression tests locking this ordering in).

## Inputs

Road, building, and facility GeoDataFrames, normally produced by the
sibling `lga-osm-extractor` package, but any GeoDataFrame with matching
column names/CRS (EPSG:32631) will work. This package never queries
OSM directly.

## Outputs

A scored grid GeoDataFrame with per-mode travel times and deficit
scores, ready for `dashboard/app.py` to present or for export to
GeoJSON.

## Related

- `../notebooks/`, the orchestration layer that calls these
  functions in sequence against real Akure North/South data
- `../dashboard/app.py`, the Streamlit presentation layer
- `../tests/`, unit tests, including equivalence tests proving the
  performance-optimized routing/completeness functions give identical
  results to their original, simpler implementations
- `../../lga-osm-extractor/`, the sibling repo that produces this
  package's expected input data
