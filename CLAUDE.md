# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Indoor localisation and fingerprint analysis from Wi-Fi CSI (Channel State Information), captured by ESP32 anchors on 2.4 GHz and 5 GHz. `README.md` contains a per-function reference for `utils/`.

The repo holds three projects. **Only the root is active work.**

- **Root (current work):** `Fingerprint_Analysis.ipynb` (domain change / fingerprint analysis for a new article), plus `utils/` (full pipeline: data, features, ML, DL) and `results/` (this article's outputs only). New code goes in the root `utils/`.
- **`thesis/` (frozen):** the dissertation's `ML.ipynb`, `DL.ipynb`, its own `utils/` snapshot and `results/` (72 runs). Git tag `thesis-final`.
- **`isac-paper/` (frozen):** `paper.ipynb`, `paper_utils/`, and a `utils/` snapshot from commit `ab77738`. Git tag `isac-paper`.
- **`legacy/`:** early exploration notebooks and scripts with no data. Not part of any pipeline.
- **`scripts/`:**
  - `udp_server.py`: CSI acquisition, mapping ESP MAC addresses to ESP ids and writing CSVs.
  - `manage_cache.py`: lists and cleans `.cache/` and `results/`.
- **`tests/`:** pytest tests for the root `utils/`.

**Isolation mechanism:** each project's `utils/config.py` sets `PROJECT_ROOT = Path(__file__).parent.parent`. A notebook kernel starts in the notebook's own folder, so each notebook imports its sibling `utils/` and writes to its sibling `results/` and `.cache/`. Don't edit the `utils/` copies in `thesis/` or `isac-paper/` unless explicitly asked. Running a root script from another folder imports the wrong `utils`.

## Environment & commands

Experiments run on a Linux server (32 cores, 64 GB RAM, NVIDIA RTX PRO 6000 Blackwell, sm_120) that the user reaches over remote desktop from a Windows 11 PC; some settings exist for both machines.

- **Interpreter:** use the conda env **`thesis`** (`~/miniconda3/envs/thesis/bin/python`, Python 3.12, torch 2.11+cu128). `requirements*.txt` pin exactly what is installed there.
- **Raw data:** `/disco1500gb/Pedro_Data` (`DATA_DIR`). It is never committed. `*.csv` is git-ignored except under `**/results/`.

```bash
PY=~/miniconda3/envs/thesis/bin/python
$PY -m pytest                                    # all tests (config in pyproject.toml)
$PY -m pytest tests/test_lovo_split.py::test_split_lovo_folds_holds_out_one_user_at_a_time

ruff check . && ruff format .                    # ruff "ALL", line length 100; thesis/, isac-paper/, legacy/ excluded

$PY scripts/manage_cache.py list                 # also: clean [--all | --older-than DAYS]
```

**Parallelism uses the whole server.** CSV parsing uses `"max_workers": None` (all cores) in the notebook option dicts. `GRID_SEARCH_N_JOBS = -1` is set in `utils/config.py`, and `N_JOBS = -1` in `thesis/ML.ipynb`. None of these are part of run IDs. DataLoader settings (`cuda_num_workers`, batch size, etc.) **are** hashed into DL run IDs through `DEFAULT_CNN_PARAMS`, so changing them orphans existing DL checkpoints and predictions. Feature building (`feature_pipeline`) is serial Python and is the slowest step when rebuilding the cache (~1.5 min for the full dataset).

Ruff's `ALL` rule set is strict. The existing code silences specific rules inline (e.g. `# noqa: PLR0913`) rather than loosening the config; follow that pattern.

## Architecture

**Configuration** lives in `utils/config.py`. Per-experiment options (calibration mode, normalization, etc.) are set in the notebooks' config cells and passed down as dicts. The config holds:

- the data path, and the `.cache/` and `results/` locations;
- the ESP-id → band mapping: 2.4 GHz = ESPs 1–5 and 7–10, 5 GHz = 11–20, "Fusion" = both;
- the room ↔ anchor-pair mapping and the expected subcarrier counts (50 / 56);
- window defaults (60 packets, no overlap) and seeds (42, 43, 44);
- the frozen CNN hyperparameters.

**Data files** are named `<scenario>_<location>_<user>_<esp>_<trial>_<timestamp>.csv` (regex `PATTERN` in `utils/import_data.py`). Locations are grid cells `A-1`…`G-14`. `Z-0` is the empty room: it's used only as a normalization baseline and never as a class.

**Pipeline** (each stage is in its own module):

1. `import_data.get_csv_files` builds a nested file map: scenario/location/user/ESP/trial.
2. `load_csi.process_csv_files` parses the bracketed I/Q payload plus RSSI into complex CSI, with per-file parsing diagnostics. It returns the magnitude map, optionally followed by the raw map, stats and diagnostics (`return_raw_magnitude`, `return_stats`, `return_diagnostics`).
3. `csi_processing` does FFT-shift and active-subcarrier selection, calibration (`none` | `packet_norm` | `rssi`), magnitude, and normalization (`none` | `zscore` | `minmax` | `packet_minmax` | `empty_baseline`, the last using the `Z-0` recordings).
4. ML path: `feature_pipeline` turns aligned multi-ESP windows into per-subcarrier statistical feature DataFrames, one per band plus fusion. DL path: `DL/dl_pipeline` writes raw windows as float16 memory-mapped arrays.
5. `ML/ml_pipeline.py` and `DL/dl_pipeline.py` run experiments under three split protocols:
   - block-based `grid_search`, with hyperparameter tuning;
   - `cross_session`: train on trial `01`, test on later trials;
   - LOVO (leave-one-volunteer-out): one fold per held-out user.

   Metrics include position accuracy, room accuracy, macro-F1 and metric distance errors. `ml_pipeline.load_raw_csi_data` and `load_feature_dataframes` are the loaders the Fingerprint notebook uses.
6. `results.py` and `plots.py` write the outputs.

**Caching is content-addressed and pervasive.** `utils/cache.py` keys parsed CSI, feature DataFrames, window arrays and predictions by hashes of the processing options (`make_preproc_key`, `make_feat_key`, `make_run_id`). Changing any processing option produces a new cache entry instead of invalidating the old one. Changing processing *code* without changing options can serve stale results, so pass `force_reprocess` or clear the relevant cache. Run IDs look like `ml__rf__2_4ghz__grid_search__ebl-session__s42__<hash>`.

**Results:** in each project's `results/`, `runs.csv` (plus `runs_folds.csv` for per-fold LOVO rows) is the experiment registry, with one row per run and a wide superset of ML and DL columns. Alongside it:

- `manifests/` holds the run configs;
- `predictions/` holds the cached parquet predictions;
- `checkpoints/` holds the CNN weights;
- `tuning/` holds the grid-search outputs;
- `tables/` and `plots/` hold the publication-ready outputs.

Notebooks reload cached predictions instead of retraining when a matching run exists.
