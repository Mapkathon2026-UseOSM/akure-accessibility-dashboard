# tests/

## Purpose

Automated tests for the `akure_access` package. These are unit and
integration tests against synthetic data, fast, deterministic, and
runnable without network access or real OSM data, distinct from
running the actual notebooks against real Akure North/South data.

## Contents

| File | Covers |
|---|---|
| `test_scoring.py` | Grid generation, building density, access-deficit scoring, export sanitization |
| `test_completeness.py` | OSM completeness flagging, including the spatial-index rewrite's equivalence with the original linear-scan approach |
| `test_network_graph.py` | Road network construction, nearest-facility routing (both the naive per-pair approach and the faster multi-source Dijkstra batch approach, and a test proving they agree) |
| `test_insights.py` | Every `insights.describe_*()` caption function: real numbers appear correctly in generated text, missing/all-NaN columns don't crash, directional-skew phrasing, tie-handling in mode comparisons, the interactive-view dispatcher routes to the right caption for each "Access view" choice, a style-guard test that no generated caption contains an em dash, and a test that `DEFAULT_THRESHOLDS_MIN` stays in sync with Notebook 03's actual threshold configuration |
| `test_static_maps.py` | `_ensure_lonlat()`'s reprojection logic (including the specific projected-CRS-treated-as-degrees bug this guards against, see the module's own docstring), every `plot_*()` function saves a real file and raises a clear error on bad input rather than a cryptic one, and `generate_all_static_outputs()` produces the expected file count, correct title convention, and skips cleanly when web-tier output is disabled or a service layer is entirely NaN |
| `test_cross_repo_integration.py` | Verifies `lga_extractor`'s real output schema is exactly what this package's functions expect, catches breakage between the two repos that neither repo's own isolated tests could detect |

## Running

```bash
# Fast, offline tests only (default; what CI runs on every push)
pytest -m "not integration"

# Everything, including tests that hit live OSM/Overpass (slow, needs network)
pytest -m integration

# Cross-repo test specifically (needs lga_extractor installed, see below)
pytest tests/test_cross_repo_integration.py
```

## Notes

- `test_cross_repo_integration.py` uses `pytest.importorskip("lga_extractor")`,
  so it's automatically skipped (not failed) if the sibling
  `lga-osm-extractor` package isn't installed. Install it with
  `pip install -e ../lga-osm-extractor` to run this test locally; CI
  runs it in a dedicated workflow
  (`.github/workflows/cross-repo-integration.yml`) that checks out
  both repos.
- Tests marked `@pytest.mark.integration` make real network calls to
  OSM/Overpass and are excluded by default (see `pytest.ini`) to keep
  the regular test suite fast and independent of external services.
  Run them at least once before a release/submission to confirm the
  live code paths still work, since they're never exercised by CI.
