# Methodology (condensed)

See the full methodology document delivered alongside this repository
(`Mapkathon2026_Project_Methodology_and_Workflow.docx`) for complete
detail. This file is a quick-reference condensed version for GitHub
viewers.

1. **Extraction** — `lga_extractor` pulls roads, buildings, waterways,
   land use, health facilities, and schools per LGA.
2. **Completeness check** — `akure_access/completeness/grid_check.py` flags
   settled grid cells with no nearby OSM facility tag.
3. **Accessibility analysis** — `akure_access/accessibility/` builds a mode-specific
   walking, okada (motorcycle taxi), or driving network graph, computes
   nearest-facility distance and travel time per cell for each mode, and
   derives a 0–2 access-deficit score per mode (each with its own
   time budget, since 30 minutes covers very different ground on foot
   vs. by okada vs. by car).
4. **Dashboard** — `dashboard/app.py` (Streamlit + leafmap) presents an
   interactive map, ranked underserved list, and health/education toggle.
5. **StoryMap** — ArcGIS StoryMap (linked separately) presents the
   narrative writeup and comparative findings.

## Key assumptions and limitations

- Building density is a population proxy, not actual population counts.
- Three transport modes are modeled with assumed average speeds:
  walking (5 km/h), okada/motorcycle taxi (25 km/h), and private/shared
  vehicle (35 km/h). Okada and drive share the OSM "drive" road network
  (OSM has no distinct motorcycle network type) and differ only in
  assumed speed, which does not capture okada-specific behavior such as
  weaving through traffic or using informal shortcuts.
- OSM facility completeness is not independently ground-truthed.
- "Underserved" findings may partly reflect OSM under-mapping rather
  than confirmed lack of physical service access — see the completeness
  flag columns for cells where this ambiguity applies.
- The completeness check (`akure_access/completeness/grid_check.py`) tests
  each settled cell's centroid against the facility layer using
  `geopandas.sjoin_nearest()`, which builds a spatial index (STRtree)
  over the facility points once and queries it per cell — O(n log m)
  rather than a per-cell linear distance scan against every facility
  (O(n × m)). This was rewritten from an earlier linear-scan
  implementation once the approach was in place; a dedicated
  equivalence test (`test_spatial_index_flagging_matches_naive_linear_scan`)
  confirms the rewrite produces identical results to the original scan.
- The accessibility-deficit scoring uses exact network shortest-path
  distances/times (`nearest_facility_distance_and_time()` /
  `batch_nearest_facility_distances()`), not convex-hull isochrone
  polygons. Isochrones (`compute_isochrone_polygon()`,
  `build_isochrones_for_facilities()`) are used elsewhere in the
  project -- as an illustrative "walking catchment" overlay in the
  Streamlit dashboard (see `notebooks/03_accessibility_analysis.ipynb`,
  Section 5.2, which precomputes 15/30/45-minute health-facility
  catchments; `dashboard/app.py` loads and displays them as an optional
  toggle) -- but deliberately NOT for the project's actual
  access-deficit scoring, since a convex hull can overstate true
  reachable area (real street networks are rarely convex). Keep this
  distinction in mind if reading the dashboard: the catchment overlay
  is an approximate visual aid, while the underlying access-deficit
  scores and rankings are based on exact routing.
- Boundary resolution and CRS handling inherit the companion
  `lga-osm-extractor` repository's behavior (see its README for detail):
  every resolved boundary is checked against Nigeria's approximate
  bounding box and a plausible area range before extraction proceeds,
  and the correct UTM zone is auto-selected from each LGA's boundary
  centroid (EPSG:32631 for this project's Ondo State study areas
  specifically, since Akure North/South both fall in UTM Zone 31N).
  This is not a full `admin_level`/relation-type verification against
  an authoritative Nigerian administrative boundary dataset -- see the
  extractor's README for what the current checks do and don't catch.
