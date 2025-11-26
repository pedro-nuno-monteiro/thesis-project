# Wi‑Fi Sensing — Thesis Project

This repository contains code, notebooks and data used for a Wi‑Fi sensing experiment (thesis project).

## Contents
- `Wi‑Fi-Sensing-[meu].ipynb` — main analysis and ML pipeline (data processing → features → Random Forest).
- `General-Data-Collecting-[meu].ipynb`, `Position-Selection-[meu].ipynb` — additional notebooks used during data collection and analysis.
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

