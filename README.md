# Wi‑Fi Sensing — Thesis Project

This repository contains code, notebooks and data used for a Wi‑Fi sensing experiment (thesis project).

## Notebooks

### `General-Data-Collecting-[meu].ipynb`
The primary working notebook for user identification in indoor environments using Wi-Fi CSI data. This notebook combines and improves upon previous work from Miguel's and Diana's projects, implementing a complete machine learning pipeline for detecting which user is present in a room. Includes data preprocessing, feature extraction, normalization, and Random Forest classification with hyperparameter tuning.

### `Wi-Fi-Sensing-[meu].ipynb`
A refactored version of Miguel's original project with cleaner code structure, improved readability, and detailed comments explaining the CSI processing steps and machine learning workflow.

### `Position-Selection-[meu].ipynb`
An enhanced version of Diana's position detection notebook, reorganized for better clarity with improved comments and code structure to make the analysis more accessible.

### `Generalistic_WIFI_Sensing.ipynb`
An early experimental notebook started by Óscar that was not continued in later development.

## Contents
- `data/` — raw and processed CSV files collected from the CSI device(s). Subfolders group datasets (e.g. `CSI DATA RENAMED [alterado]`).
- `utils/` — helper scripts used by the notebooks (CSV importer, file renaming, etc.).

## Requirements
- Python 3.8+
- See `requirements.txt` for exact Python packages and versions.

Recommended: create and use a virtual environment before installing requirements.

## Installation
```powershell
# from repository root (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

## Usage
- Open and run the notebooks in a Jupyter environment (e.g. JupyterLab or VS Code).
- Primary notebook: `Wi‑Fi-Sensing-[meu].ipynb` — runs the full pipeline: read CSVs → process CSI → extract features → train/evaluate models.
- Helper scripts in `utils/` can be imported from notebooks (they are used in the notebooks already).

## Data layout
- Place your CSV datasets under the `data/` folder. The notebooks expect the directory structure present in the repository (see the `data/` subfolders).

## Notes
- This repository is part of an academic thesis. Some notebooks and scripts are exploratory and contain inline comments in Portuguese.
- If you plan to reproduce results, check and adapt the absolute paths used in notebooks (they reference local OneDrive paths). Prefer using relative paths or set the `path` variable at the top of each notebook.

## Author
- Pedro Nuno Monteiro

## License
See `LICENSE` if present. No license specified by default.

---

*This README was created and maintained with assistance from GitHub Copilot.*

