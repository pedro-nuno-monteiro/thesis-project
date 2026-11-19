# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc thesis: indoor localisation from Wi-Fi CSI (Channel State Information) captured by ESP32 anchors on 2.4 GHz and 5 GHz. Work is driven from Jupyter notebooks at the repo root; all reusable logic lives in the `utils/` package, which the notebooks import as `from utils... import ...` (so run everything with the repo root as the working directory). `README.md` contains a per-function reference for `utils/`.

- `ML.ipynb`: classical baselines (RF, kNN, SVM, ...) on window-level statistical features.
- `DL.ipynb`: CNNs (`DualBandCNN` / `RoomStackedCNN`) on raw magnitude windows (PyTorch).
- `Fingerprint_Analysis.ipynb`: inspects the ML feature fingerprints; deliberately adds no extra normalization or model.
- `udp_server.py`: CSI acquisition (receives ESP UDP packets, maps MAC → ESP id, writes CSVs).
- `thesis-final/`: frozen snapshot of the notebooks + `utils/` as submitted. Edit the root `utils/`, not this copy, unless asked.
- `code/` (legacy exploration) and `isac paper/` (separate paper notebook with its own `paper_utils/`) are not part of the active pipeline.

## Environment & commands

Experiments run on a Linux server (32 cores, 64 GB RAM, NVIDIA RTX PRO 6000 Blackwell, sm_120) that the user reaches over remote desktop from a Windows 11 PC; some settings exist for both machines. On the server, the notebooks use the conda env **`thesis`** (`~/miniconda3/envs/thesis/bin/python`, Python 3.12, torch 2.11+cu128). The project-local `.conda/` (Python 3.14) is empty: don't use it. `requirements.txt` pins older versions (numpy 1.26, pandas 2.2, sklearn 1.4) than the env actually has (numpy 2.5, pandas 3.0, sklearn 1.9). The installed env is what produced `results/`.

```bash
PY=~/miniconda3/envs/thesis/bin/python
# tests (pytest is not installed in the env: $PY -m pip install pytest)
$PY -m pytest extras/tests
$PY -m pytest extras/tests/test_lovo_split.py::test_split_lovo_folds_holds_out_one_user_at_a_time

# lint/format (config in pyproject.toml: ruff "ALL" rules, line length 100, notebooks included)
ruff check .
ruff format .

# cache / results housekeeping
$PY extras/scripts/manage_cache.py list
$PY extras/scripts/manage_cache.py clean --older-than 30
```

Some tests in `extras/tests/test_thesis_csv_processing_calibration.py` are outdated: they unpack an `agc_gain_data` return value that `process_csv_files` no longer returns.

Parallelism is set to use the whole server: CSV parsing `"max_workers": None` (all cores) in the notebook option dicts, `GRID_SEARCH_N_JOBS = -1` in `utils/config.py`, and `N_JOBS = -1` in `ML.ipynb`. None of these are part of run IDs. DataLoader settings (`cuda_num_workers`, batch size, etc.) **are** hashed into DL run IDs through `DEFAULT_CNN_PARAMS`, so changing them orphans existing DL checkpoints and predictions. Feature building (`feature_pipeline`) is serial Python and is the slowest step when rebuilding the cache (~1.5 min for the full dataset).

Ruff's `ALL` rule set is strict; the existing code silences specific rules inline (e.g. `# noqa: PLR0913`) rather than loosening the config. Follow that pattern.

## Architecture

**Configuration** lives in `utils/config.py`: data path (`DATA_DIR = /disco1500gb/Pedro_Data`, an external disk), `.cache/` and `results/` locations, the ESP-id → band mapping (2.4 GHz = ESPs 1–5, 7–10; 5 GHz = 11–20; "Fusion" = both), room ↔ anchor-pair mapping, expected subcarrier counts (50 / 56), window defaults (60 packets, no overlap), seeds (42, 43, 44), and the frozen CNN hyperparameters. Per-experiment options (calibration mode, normalization, etc.) are set in the notebooks' config cells and passed down as dicts.

**Data files** are named `<scenario>_<location>_<user>_<esp>_<trial>_<timestamp>.csv` (regex `PATTERN` in `utils/import_data.py`). Locations are grid cells `A-1`…`G-14`; `Z-0` is the empty room, used only as a normalization baseline and never as a class.

**Pipeline** (each stage is in its own module):

1. `import_data.get_csv_files` builds a nested file map scenario/location/user/ESP/trial.
2. `load_csi.process_csv_files` parses the bracketed I/Q payload + RSSI into complex CSI, with per-file parsing diagnostics.
3. `csi_processing` does FFT-shift and active-subcarrier selection, calibration (`none` | `packet_norm` | `rssi`), magnitude, and normalization (`none` | `zscore` | `minmax` | `packet_minmax` | `empty_baseline`, the latter using `Z-0` recordings).
4. ML path: `feature_pipeline` turns aligned multi-ESP windows into per-subcarrier statistical feature DataFrames (one per band plus fusion). DL path: `DL/dl_pipeline` writes raw windows as float16 memory-mapped arrays.
5. `ML/ml_pipeline.py` / `DL/dl_pipeline.py` run experiments under three split protocols: block-based `grid_search` (with hyperparameter tuning), `cross_session` (train on trial `01`, test on later trials), and LOVO (leave-one-volunteer-out, one fold per held-out user). Metrics include position accuracy, room accuracy, macro-F1, and metric distance errors.
6. `results.py` / `plots.py` write outputs.

**Caching is content-addressed and pervasive.** `utils/cache.py` keys parsed CSI, feature DataFrames, window arrays, and predictions by hashes of the processing options (`make_preproc_key`, `make_feat_key`, `make_run_id`). Changing any processing option produces a new cache entry instead of invalidating the old one. Changing processing *code* without changing options can serve stale results, so pass `force_reprocess` or clear the relevant cache. Run IDs look like `ml__rf__2_4ghz__grid_search__ebl-session__s42__<hash>`.

**Results:** `results/runs.csv` (and `runs_folds.csv` for per-fold LOVO rows) is the canonical experiment registry, with one row per run and a wide superset of ML and DL columns. `manifests/` holds run configs, `predictions/` cached parquet predictions, `tuning/` grid-search outputs, and `tables/` and `plots/` the thesis-ready artifacts. Notebooks reload cached predictions instead of retraining when a matching run exists.
