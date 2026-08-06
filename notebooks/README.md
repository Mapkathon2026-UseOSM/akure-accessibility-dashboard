# notebooks/

## Purpose

This folder contains the full, worked analysis pipeline for the project,
as five sequential Jupyter notebooks. Together they take raw OSM data
(from the companion `lga-osm-extractor` repository) through to a
scored, exportable accessibility dataset for Akure North and Akure
South LGAs.

## Why notebooks, not scripts?

The analysis benefits from being inspectable step-by-step: intermediate
outputs (grid geometry, completeness flags, routing graphs) are worth
looking at directly while developing or auditing the methodology,
which a plain script would hide. The reusable, tested logic itself
lives in `akure_access/` (imported by these notebooks); the notebooks
are the orchestration + narrative layer on top of it, not where the
core algorithms are implemented.

## Contents, in execution order

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_data_extraction.ipynb` | Runs `lga_extractor` for both LGAs, saves raw layers to `data/processed/` | `lga-osm-extractor` (sibling repo) |
| `02_completeness_assessment.ipynb` | Builds the analysis grid, flags likely OSM data gaps | Output of 01 |
| `03_accessibility_analysis.ipynb` | Builds routable road networks, scores travel time to health/education facilities for all 3 modes, precomputes walking isochrones, and generates every publication-styled static map/chart + caption per LGA | Output of 02 |
| `04_results_summary.ipynb` | Cross-mode and cross-LGA comparison tables, exported as CSVs, plus the combined two-LGA scored dataset | Output of 03 |
| `05_kepler_visualization.ipynb` | Exports 3 standalone interactive kepler.gl HTML maps (combined deficit, mode-comparison, completeness) | Output of 03/04 |

## Workflow

Run strictly in order (01 → 05) on a fresh setup. Each notebook reads
files written by the previous one from `data/processed/`; none of them
regenerate earlier steps' outputs implicitly. If you change an earlier
notebook's parameters (e.g. grid cell size), re-run every notebook
after it.

## Running in Google Colab vs. locally

Every notebook's first code cell detects whether it's running in Colab
(`IN_COLAB` check) and adjusts paths accordingly (mounting Google
Drive, installing dependencies) or skips that setup entirely when run
locally. No manual editing should be needed to switch between the two.

## Dependencies

See `../requirements.txt` (or `../requirements-lock.txt` for exact
pinned versions). Notebook 03 additionally requires the companion
`lga-osm-extractor` package to be installed if you're re-running
Notebook 01 from scratch.

## Outputs

| Path | Written by | Contents |
|---|---|---|
| `../data/processed/{lga}/` | 01, 02, 03 | Raw + cleaned layers, the scored grid, isochrones |
| `../data/processed/combined_access_scored.geojson` | 04 | Both LGAs' scored grids combined into one file |
| `../visuals/{lga}/*.jpg`, `web/*.jpg`, `captions.json` | 03 | Print + web tier static maps/charts, per LGA, with generated captions |
| `../visuals/akure_access_static_maps.zip` | 03 | Zip of the above, for download (generated locally, not committed to the repo) |
| `../visuals/*.html` | 05 | 3 standalone kepler.gl interactive maps |
| `../reports/*.csv` | 04 | Cross-mode / cross-LGA comparison tables |

See each notebook's own "Export" section for exact filenames.

## Notes

- Notebook 03 is the most compute-intensive step (builds road network
  graphs and runs shortest-path routing per grid cell). As of the
  latest version, this uses a multi-source Dijkstra approach
  (`akure_access.accessibility.batch_nearest_facility_distances`)
  rather than one shortest-path search per cell, measured ~370x
  faster than the original per-cell approach at this project's scale.
- If you see a `ModuleNotFoundError` for `akure_access`, the notebook's
  setup cell couldn't find the package on `sys.path`, check that
  you're running from within the repo, or that the Colab Drive mount
  points at a folder named exactly "Akure Access Dashboard".
