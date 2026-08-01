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
