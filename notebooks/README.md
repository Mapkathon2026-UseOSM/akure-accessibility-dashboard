# notebooks/

## Purpose

This folder contains the full, worked analysis pipeline for the project,
as five sequential Jupyter notebooks. Together they take raw OSM data
(from the companion `lga-osm-extractor` repository) through to a
scored, exportable accessibility dataset for Akure North and Akure
South LGAs.

## Contents, in execution order

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_data_extraction.ipynb` | Runs `lga_extractor` for both LGAs, saves raw layers to `data/processed/` | `lga-osm-extractor` (sibling repo) |
| `02_completeness_assessment.ipynb` | Builds the analysis grid, flags likely OSM data gaps | Output of 01 |
| `03_accessibility_analysis.ipynb` | Builds routable road networks, scores travel time to health/education facilities for all 3 modes | Output of 02 |
| `04_results_summary.ipynb` | Cross-mode and cross-LGA comparison tables/charts | Output of 03 |
| `05_kepler_visualization.ipynb` | Exports standalone interactive kepler.gl HTML maps | Output of 03/04 |

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

All notebooks write to `../data/processed/{lga}/`, `../reports/`, or
`../visuals/`, see each notebook's own "Export" section for exact
filenames.

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
