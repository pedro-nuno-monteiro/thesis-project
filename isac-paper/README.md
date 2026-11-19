# ISAC paper archive (frozen)

This folder holds the notebook and helpers for the ISAC paper. It's a record, not active work.

- `paper.ipynb`: the paper's experiments (hierarchical room→position classifier, window-size sensitivity).
- `paper_utils/`: helpers used only by the paper.
- `utils/`: a snapshot of the shared pipeline code from commit `ab77738` (31 Aug 2026), the last commit that changed the paper. The same state is saved as the git tag `isac-paper`. It's independent of the root `utils/`.
- `images/`: figures used in the paper.

## Re-opening

Open `paper.ipynb` from this folder. The kernel then imports `isac-paper/utils` and `isac-paper/paper_utils`, and writes to `isac-paper/results` and `isac-paper/.cache` (both git-ignored). The raw data is read from `/disco1500gb/Pedro_Data`.

The saved outputs in the notebook come from a run on the Windows PC, which is why they show `C:\...` cache paths.
