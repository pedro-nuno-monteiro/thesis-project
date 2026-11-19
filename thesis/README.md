# Thesis archive (frozen)

This folder holds the code and results of the MSc dissertation as they were when it was finished. The same state is also saved as the git tag `thesis-final`. It's a record, not active work: new code goes in the root `utils/`.

- `ML.ipynb`: classical baselines (RF, kNN, SVM) under block, cross-session and leave-one-volunteer-out (LOVO) splits.
- `DL.ipynb`: band-branch and room-stacked CNNs under the same splits.
- `utils/`: the pipeline code used by these two notebooks (a snapshot, independent of the root `utils/`).
- `results/`: everything the dissertation reports.
  - `runs.csv` (72 runs) and `runs_folds.csv` (per-fold LOVO rows) are the run registry.
  - `checkpoints/`, `predictions/` and `manifests/` are keyed by run ID.
  - `tuning/` holds the grid-search logs, and `tables/` the LaTeX tables with their CSVs.
  - `plots/` holds the figures (PDF; PNG only where no PDF exists).
- `scripts/height.py`: regenerates the volunteer height/weight figure into `results/plots/`.

## Re-opening

Open the notebooks from this folder. The kernel then imports `thesis/utils` and reads and writes `thesis/results` and `thesis/.cache`.

- Both notebooks have `FORCE_RETRAIN = False`, so re-running them loads the saved predictions instead of retraining.
- The raw data is read from `/disco1500gb/Pedro_Data`.
- The feature cache is rebuilt on first use (about 2–3 minutes on the server).
