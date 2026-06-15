# Wi‑Fi Sensing — Thesis Project

This repository contains the code, notebooks, and CSI datasets used in the current thesis project on Wi‑Fi sensing.

## Current repository organization

### `/CSI DATA/`
Contains the **current CSI datasets used in the thesis** (raw collection CSV files).

### `/images/`
Reserved for graphs and figures generated from notebook analysis (when exported).

### `/utils/`
Supporting Python modules used by the notebooks:
- `utils/__init__.py` — package marker for importing utilities.
- `utils/import_data.py` — scans CSV files, validates filename metadata, builds the nested file map used by the notebooks, and prints user/location coverage tables.
- `utils/thesis_csv_processing.py` ? thesis CSI CSV processing pipeline: parses raw 2.4 GHz and 5 GHz CSI payloads, builds CSI magnitudes, preserves 5 GHz AGC gain data, supports parallel processing, and caches processed outputs.
- `utils/csi_228_csv_processing.py` ? 228-notebook CSI CSV processing pipeline: parses 5 GHz CSI payloads, separates 114-length and 228-length vectors, builds the 228 comparison magnitude maps, supports parallel processing, and caches processed outputs.
- `utils/feature_pipeline.py` — feature-engineering layer: applies windowing/overlap logic, computes statistical features, assigns room labels, and builds model-ready dataframes for 2.4 GHz, 5 GHz, and fusion scenarios.
- `utils/graphs.py` — plotting and visualization toolkit: CSI profile/surface plots, interactive trial/location selection, Random Forest metric + confusion matrix plots, and I/Q constellation visualization helpers.

### `/code/`
Legacy and preliminary work not used in the current final pipeline:

#### `/code/first touches/` *(currently not used)*
Preliminary experiments for the **first part of the thesis** (small-room dataset and early analysis ideas).
- `code/first touches/Data analysis.ipynb` — exploratory notebook for early processing, feature extraction, and baseline Random Forest experiments on small-room data.
- `code/first touches/Graph Analysis.ipynb` — exploratory notebook focused on graph-based CSI visualization and scenario comparison.
- `code/first touches/what.ipynb` — early neural-network exploration for user presence detection.
- `code/first touches/csv_import_data_analysis.py` — helper module for parsing the early filename convention and organizing CSV paths by scenario/user/activity/ESP/trial.
- `code/first touches/graphs.py` — plotting helpers used in early experiments (3D magnitude surfaces, spectrograms, subcarrier comparisons, time-domain traces).
- `code/first touches/rename_copy_files.py` — utility script to copy and rename CSV files into a normalized naming format.

#### `/code/old/` *(currently not used)*
Archive of much older notebooks/scripts from previous authors, with only minor comment adjustments and accompanying historical data.
- `code/old/General-Data-Collecting-[meu].ipynb` — legacy notebook for early end-to-end CSI preprocessing and ML classification tests.
- `code/old/Generalistic_WIFI_Sensing.ipynb` — older generalized Random Forest experimentation notebook.
- `code/old/Position-Selection-[meu].ipynb` — legacy indoor localization workflow using CSI-derived features and position models.
- `code/old/Wi-Fi-Sensing-[meu].ipynb` — legacy full Wi‑Fi sensing notebook with preprocessing, normalization, dataset creation, and classification experiments.
- `code/old/csv_import.py` — deprecated CSV import/organization utility for old naming schemas and old experiment structures.
- `code/old/deles/Wi-Fi_Sensing-Miguel-[cópia].ipynb` — backup copy of an older collaborator notebook.
- `code/old/deles/feature_extraction_and_selection.py` — exported Colab-era experimental script for feature extraction, selection, and classification benchmarking.

## Root-level files (main working files)

- `thesis.ipynb` — **main notebook of the current thesis**. It executes the full workflow: data import, CSI processing, feature engineering, machine-learning training/evaluation, and result visualization.
- `udp_server.py` — UDP data-collection server used on the remote machine to receive CSI packets from ESP devices and save them into correctly named CSV files.
- `228.ipynb` — focused experimental notebook studying different CSI vector sizes (including 228-length handling variants for 5 GHz data).
- `README.md` — repository documentation and file organization reference.
- `requirements.txt` — pinned Python dependency list for notebook and utility execution.
- `pyproject.toml` — project metadata and Ruff lint/format configuration.
- `.gitignore` — ignore rules for Python artifacts, virtual environments, caches, and local/generated files.
- `.vscode/settings.json` — local VS Code workspace setting for Python environment manager behavior.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

## Usage

1. Collect data (if needed) with `udp_server.py`.
2. Place/verify current data inside `CSI DATA/`.
3. Run `thesis.ipynb` for the complete current pipeline.
4. Use `228.ipynb` only for the CSI vector-size experiment.

## Running the experiments

Before the first run, clear any stale cache and results:

```powershell
Remove-Item -Recurse -Force .cache, results -ErrorAction SilentlyContinue
```

Then open and run `paper.ipynb` top-to-bottom.

- **First run**: feature dataframes are built and written to `.cache/dataframes/`, summary tables and plots are written to `results/`.
- **Subsequent runs with the same options**: the slow build step is skipped (you will see `[cache hit]` log lines) and the notebook completes much faster.
- **Changing an option** (e.g., `window_size` in `FEATURE_EXTRACTION_OPTIONS`): a `[cache miss]` is logged and a new folder with the updated key is created under `.cache/` and `results/`.

To inspect or clean the cache:

```powershell
python scripts/manage_cache.py list
python scripts/manage_cache.py clean --older-than 30   # remove entries older than 30 days
python scripts/manage_cache.py clean --all              # wipe everything
```

---

*This README was updated by GitHub Copilot to reflect the current file organization.*
