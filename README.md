# Wi-Fi CSI Localisation Thesis

This repository contains the classical machine-learning and deep-learning
experiments used for indoor localisation from Wi-Fi CSI measurements.

## Main files

- `ML.ipynb` contains the classical ML experiment workflow and exposes the
  selected dataset, preprocessing, split, RF/SVM/KNN models, and result plots.
- `DL.ipynb` contains the deep-learning experiment workflow and exposes the
  selected dataset, splits, CNN settings, bands, anchors, and training options.
- `udp_server.py` handles CSI acquisition.

All active Python implementation is inside `utils/`:

- `utils/import_data.py` discovers input files and extracts dataset metadata.
- `utils/load_csi.py` opens CSV files, parses CSI rows, and assembles CSI arrays.
- `utils/csi_processing.py` processes CSI magnitude, calibration,
  normalisation, subcarriers, and windows.
- `utils/feature_pipeline.py` creates the ML feature DataFrames.
- `utils/cache.py` loads and saves reusable CSI, feature, prediction, and
  window-array caches.
- `utils/ML/` contains the classical estimators and the ML experiment pipeline.
- `utils/DL/` contains the CNN models and the DL experiment pipeline.
- `utils/results.py` contains shared metrics, prediction tables, and result
  handling.
- `utils/plots.py` contains shared CSI, evaluation, and training plots.

## Processing pipeline

```text
data discovery
-> CSI loading
-> CSI preprocessing
-> feature DataFrame or DL tensor creation
-> ML or DL model
-> metrics and plots
```

## Environment

Install `requirements.txt` for the standard environment. GPU hosts can use
`requirements-gpu.txt` for the deep-learning notebook.
