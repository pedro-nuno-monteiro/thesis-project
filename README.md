# Wi-Fi CSI Localisation

Indoor localisation and fingerprint analysis with Wi-Fi CSI (Channel State Information) captured by ESP32 anchors on 2.4 GHz and 5 GHz.

The **current work** (the domain-change / fingerprint-analysis article) lives at the repository root. Finished projects are kept as **frozen, self-contained archives**, each with its own copy of the code and its own results.

## Project organization

```text
.
├── Fingerprint_Analysis.ipynb   # CURRENT WORK: domain change and fingerprint analysis
├── utils/                       # Current pipeline code (data, features, ML, DL)
├── results/                     # Results of the current work only
├── tests/                       # Tests for utils/ (pytest)
├── scripts/
│   ├── udp_server.py            # UDP CSI acquisition from the ESP32 anchors
│   └── manage_cache.py          # List/clean .cache/ and results/
├── thesis/                      # FROZEN: MSc dissertation (ML.ipynb, DL.ipynb, utils/, results/)
├── isac-paper/                  # FROZEN: ISAC paper (paper.ipynb, paper_utils/, utils/, images/)
├── legacy/                      # Early exploration notebooks and scripts (no data)
├── requirements.txt             # Pinned Python dependencies
├── requirements-gpu.txt         # + torch for the CUDA 12.8 / Blackwell GPU
└── requirements-dev.txt         # + pytest
```

### How the projects stay separate

Each project folder (root, `thesis/`, `isac-paper/`) has its own `utils/`, whose `config.py` sets `PROJECT_ROOT` to that folder. A notebook kernel starts in the notebook's own folder, so:

| Notebook | Imports | Writes results to | Cache |
|---|---|---|---|
| `Fingerprint_Analysis.ipynb` | `utils/` | `results/` | `.cache/` |
| `thesis/ML.ipynb`, `thesis/DL.ipynb` | `thesis/utils/` | `thesis/results/` | `thesis/.cache/` |
| `isac-paper/paper.ipynb` | `isac-paper/utils/` | `isac-paper/results/` | `isac-paper/.cache/` |

Always open a notebook from its own folder (the VS Code default). Don't edit the archived `utils/` copies. New code goes in the root `utils/`.

### Archives

- `thesis/`: the code and results exactly as they were when the dissertation was finished (git tag `thesis-final`). `thesis/results/runs.csv` is the registry of the 72 thesis runs, with `checkpoints/`, `predictions/`, `manifests/`, `tables/`, `tuning/` and `plots/` next to it. `thesis/scripts/height.py` regenerates the volunteer figure.
- `isac-paper/`: the ISAC paper notebook with a `utils/` snapshot from commit `ab77738` (git tag `isac-paper`), the last commit that changed the paper.

## Data

The raw CSI recordings are on the server at `/disco1500gb/Pedro_Data` (`DATA_DIR` in `utils/config.py`) and are never committed. Files are named `<scenario>_<location>_<user>_<esp>_<trial>_<timestamp>.csv`; `Z-0` is the empty room.

## Utils package reference

Reference for the current `utils/` at the repository root.

### `utils/import_data.py`
- `sort_meta_info`: Parse and sort metadata from valid CSV filenames.
- `get_csv_files`: Discover valid CSV files and index them by scenario/location/user/ESP/trial.

### `utils/load_csi.py`
- `extract_csi_numbers`: Parse CSI numeric payload inside bracketed CSV entries.
- `read_csi_columns`: Read CSI and RSSI columns required from one CSV file.
- `parse_valid_packet_rows`: Keep complete CSI packets and aligned RSSI values.
- `load_raw_complex_csi`: Parse one file to complex CSI before FFT shift and subcarrier selection.
- `iq_values_to_complex`: Convert interleaved I/Q CSI values into complex arrays.
- `processing_diagnostics_frame`: Build one diagnostics row per processed file.
- `summarize_processing_diagnostics`: Summarize packet parsing/retention diagnostics by band.
- `lowest_packet_count_files`: Return files with the fewest retained packets.
- `process_csv_files`: Main CSV loader that returns CSI maps and optional aligned pre-calibration magnitudes.

### `utils/csi_processing.py`
- `select_active_subcarriers`: Apply FFT-shift and keep active subcarriers.
- `process_complex_csi`: End-to-end subcarrier selection, filtering, calibration, and magnitude conversion.
- `filter_packets_for_calibration`: Remove packets incompatible with requested calibration.
- `valid_packet_mask`: Build mask for packets valid for calibration.
- `calibrate_complex_csi`: Apply packet-power or RSSI magnitude compensation.
- `complex_csi_power`: Compute per-packet complex CSI total power.
- `calibration_mode_error`: Standard error for invalid calibration mode.
- `window_count_for_magnitude`: Count complete sliding windows in one magnitude matrix.
- `iter_magnitude_windows`: Iterate complete magnitude windows in packet order.
- `validate_window_parameters`: Validate magnitude/window shape constraints.
- `normalize_magnitude`: Apply selected normalization mode.
- `set_processed_magnitude_entry`: Insert a processed magnitude array into nested CSI maps.
- `process_magnitude_data`: Normalize occupied recordings and return processing summaries.
- `build_empty_baseline_tables`: Build empty-room baseline lookup tables.
- `print_z0_inventory_report`: Print empty-room inventory coverage/diagnostics report.

### `utils/feature_pipeline.py`
- `room_label_for_location`: Map a location to its room label (excluding empty-room location).
- `compute_window_features`: Compute per-window statistical features per subcarrier.
- `build_frequency_feature_dataframe`: Build one window-level feature DataFrame for a frequency scenario.
- `build_frequency_feature_dataframes`: Build the 2.4 GHz / 5 GHz / fusion feature DataFrames.
- `iter_window_groups`: Iterate aligned recording groups that generate training windows.

### `utils/fingerprint.py`
- `FingerprintSelection`: One selected fingerprint (vector, organized table, recording rows) with `summary`, `title`, `label`, and filename slugs.
- `select_fingerprint`: Select one window (`individual`) or the recording mean (`mean`) for a band/position/user/trial.
- `select_recording_rows`: Return every window of one recording, sorted by `window_idx`.
- `fingerprint_vector`: Return one window's feature vector or the column-wise recording mean.
- `organize_fingerprint`: Reshape a flat feature vector into (ESP, subcarrier) x feature-family rows.
- `resolve_esps`: Validate and order the requested ESPs of an organized fingerprint.
- `fingerprint_table_preview`: Show the first N subcarriers of every ESP.
- `feature_family_table`: Pivot one feature family to subcarrier x ESP.
- `fingerprints_by_trial` / `TrialFingerprints`: Mean and SD (across windows) fingerprints per trial plus an inventory of windows used.
- `feature_inventory_table`: Fingerprints, recordings, positions, users, trials, and ESPs per band.
- `canonical_feature_family`, `fingerprint_feature_columns`, `normalize_position`, `normalize_two_digit_id`: Name and column helpers.

### `utils/data_quality.py`
- `subcarrier_tone_map`, `raw_tone_numbers`: Map raw CSI positions to physical tones and to retained indices of the actual selection.
- `pilot_retained_indices`, `pilot_retained_indices_by_esp`, `pilot_subcarrier_numbers_by_esp`, `pilot_neighbour_indices`: Pilot-tone (+/-7, +/-21) positions under the selection.
- `raw_tone_profile`, `raw_tone_profiles`, `diagnostic_file_path`: Per-tone raw magnitude/CV of recordings before selection.
- `pilot_excess_table`, `pilot_excess_summary`: Per-recording pilot-to-neighbour ratios, spike fractions, and pilot power share.
- `pilot_feature_ratio_table`: Pilot vs non-pilot feature magnitude (e.g. window std) per ESP and trial.
- `session_dates_table`: Recording dates per user/trial from file names.
- `diagnostics_by_trial`: Packet counts, parsing losses, and retention per band and trial.
- `missing_recordings_table`, `z0_reference_table`: Missing/empty recordings and Z-0 baseline coverage per session.
- `duplicate_recordings_table`, `duplicate_summary`: Byte-identical CSV files filed under different keys.
- `aligned_window_table`, `alignment_loss_table`: Aligned window counts and windows lost to multi-ESP alignment.

### `utils/domain_shift.py`
- `recording_profiles`, `within_recording_sd_table`: Per-recording mean/SD profiles and within-recording spread.
- `grid_neighbours`: 4-connected same-room neighbours of each position.
- `cross_session_similarity`: Per (user, position, ESP) Pearson r, RMSE, offset, scale ratio, SMD, neighbour RMSE, and shift ratio between two trials.
- `shift_summary_by_esp`, `location_shift_table`: Medians per ESP/user and per (user, position).
- `localization_by_location`, `shift_localization_table`: Per (user, position) accuracy/error and its join with shift measures.
- `shift_localization_correlations`, `esp_shift_localization_correlations`: Spearman correlations, pooled, per user, and per ESP.

### `utils/cache.py`
- `csi_cache_path`: Build content-addressed cache path for one CSI CSV file.
- `load_csi_cache`: Load one cached CSI payload (returns `None` when stale/missing).
- `save_csi_cache`: Atomically persist one CSI cache payload.
- `load_window_array_cache`: Load cached DL windows, metadata, and manifest.
- `open_window_array_writers`: Open float16 mmap writers for DL window arrays.
- `save_window_array_metadata`: Save window-array metadata and its manifest.
- `load_window_arrays`: Reopen persisted window arrays as read-only memory maps.
- `make_preproc_key`: Build readable key for preprocessing options.
- `make_feat_key`: Build readable key for feature options.
- `make_run_id`: Build readable content-addressed experiment run identifier.
- `parse_run_id`: Parse identifiers generated by `make_run_id`.
- `get_cache_path`: Get feature-cache directory for preprocessing/feature options.
- `get_results_path`: Get shared results root path.
- `predictions_path`: Build prediction-parquet path for model/band/split.
- `save_predictions`: Save predictions for reuse/analysis.
- `load_predictions`: Load cached predictions.
- `prediction_cache_metadata`: Build metadata used for prediction-cache validation.
- `get_all_dataframes`: Load/build cached feature DataFrames for all bands.
- `get_dataframe`: Load/build cached feature DataFrame for one band.

### `utils/results.py`
- `compute_localization_metrics`: Compute shared localisation metrics (position/room/F1/distance).
- `room_label_for_location`: Map position label to room label.
- `location_grid_coordinates`: Convert location label to grid coordinates.
- `location_distance_error`: Compute Euclidean distance between two location labels.
- `build_global_predictions_dataframe`: Build standard prediction table used by ML and DL.
- `majority_class_baselines`: Build constant baselines from training labels.
- `ensure_results_layout`: Ensure required result directories exist.
- `build_run_row`: Build normalized `runs.csv` row from one completed run.
- `upsert_run`: Insert/update a run in `runs.csv`.
- `upsert_fold_rows`: Insert/update LOVO or cross-session fold rows.
- `write_run_manifest`: Save full run configuration used to hash run IDs.
- `checkpoint_path`: Return checkpoint path for whole run or one LOVO fold.
- `derive_table_from_runs`: Generate CSV/LaTeX views from canonical `runs.csv`.
- `derive_seed_summary`: Summarize variation across independent seeds.
- `save_summary`: Save summary DataFrame to CSV/Markdown/LaTeX (Markdown needs `tabulate`, not installed in the `thesis` env).
- `save_table`: Save a table to CSV and LaTeX without the index.
- `write_manifest`: Save self-describing `manifest.json` in results directory.

### `utils/plots.py`
- `plot_localization_error_cdf_by_model`: Plot distance-error CDF per model.
- `plot_band_error_cdf`: Plot distance-error CDF per band for one model.
- `plot_model_band_error_boxplot`: Plot grouped distance-error boxplots by model and band.
- `plot_global_position_confusion_matrix`: Plot global true-vs-predicted position confusion matrix.
- `plot_position_confusion_by_true_room`: Plot one position confusion matrix per true room.
- `plot_floor_plan_heatmap`: Overlay accuracy and distance error on room layout.
- `plot_lovo_fold_spread`: Plot LOVO fold position-accuracy spread.
- `plot_block_vs_lovo_position_accuracy`: Compare block and LOVO accuracy.
- `save_training_curves`: Save CNN train/validation curves.
- `set_all_subcarrier_ticks`: Label each subcarrier tick on an axis.
- `set_sparse_index_ticks`: Add sparse/evenly spaced axis ticks.
- `magnitude_to_db`: Convert linear magnitude to decibel scale.
- `select_aligned_magnitude_interval`: Select aligned packet interval across CSI stages.
- `plot_csi_magnitude_stages`: Plot raw/calibrated/normalized CSI heatmaps and surfaces.
- `visualization_magnitude_db`: Convert magnitude to dB and filter by threshold for display.
- `plot_no_visible_magnitude_data`: Plot placeholder for empty/filtered magnitude views.
- `plot_blank_magnitude_slot`: Hide unused magnitude subplot slot.
- `normalize_location_input`: Normalize interactive location text to key.
- `normalize_esp_input`: Normalize interactive ESP text to key.
- `format_location_key`: Format location key for display.
- `format_esp_key`: Format ESP key for display.
- `sorted_location_keys`: Sort location keys by physical grid order.
- `sorted_esp_keys`: Sort ESP keys numerically.
- `paired_esp_keys`: Pair 2.4 GHz and 5 GHz ESP anchors.
- `get_available_location_keys`: List locations with available magnitude data.
- `get_available_esp_keys_for_location`: List ESPs available at one location.
- `prompt_yes_no`: Interactive yes/no prompt helper.
- `prompt_location_key`: Interactive location selection helper.
- `prompt_esp_keys`: Interactive ESP selection helper.
- `iter_selected_magnitude_groups`: Iterate selected location/ESP magnitude groups.
- `make_subplot_grid`: Compute compact subplot grid dimensions.
- `hide_unused_axes`: Hide unused subplot axes.
- `plot_selected_magnitude_profiles`: Plot selected packet-mean magnitude profiles.
- `plot_selected_magnitude_heatmaps`: Plot selected packet/subcarrier heatmaps.
- `stack_same_width_profiles`: Stack profiles sharing most common subcarrier width.
- `collect_user_mean_profiles_db`: Collect per-user mean dB profiles for one location/ESP.
- `empty_room_mean_profile_db`: Compute empty-room mean dB profile for one ESP.
- `plot_average_magnitude_profiles_across_users`: Compare average user profiles with optional empty-room baseline.
- `plot_magnitude_analysis_interactive`: Run interactive magnitude-analysis visual workflow.
- `plot_fingerprint_facets`: Plot one or more organized fingerprints with one panel per ESP (optional mean +/- SD bands and marked subcarriers).
- `plot_raw_tone_profiles`: Plot raw per-tone magnitude/CV with pilots and dropped tones marked.
- `plot_metric_by_esp`: Box plots of one metric per ESP, optionally split by trial or user.
- `plot_floor_plan_metric`: Draw any per-location metric on the floor plan.
- `plot_shift_vs_localization`: Scatter grid of shift measures against localization metrics with Spearman rho.
- `plot_fingerprint_esp_overlay`: Overlay every ESP of one organized fingerprint on a single axis.

### `utils/ML/models.py`
- `default_params_for`: Return default hyperparameters for one model/band pair.
- `build_estimator`: Build a configured classical estimator pipeline.

### `utils/ML/ml_pipeline.py`
- `load_raw_csi_data`: Load CSI and optional aligned pre-calibration arrays.
- `validate_cross_session_inventory`: Validate trial-02 compatibility for cross-session training.
- `select_cross_session_trials`: Keep only trials used by cross-session protocol.
- `build_preproc_opts`: Build cache-keyed preprocessing options (adds `baseline_scope` for `empty_baseline`).
- `load_feature_dataframes`: Load/build cached feature DataFrames for requested bands.
- `print_normalization_discriminability`: Print Fisher-ratio diagnostics for normalization quality.
- `load_or_build_none_reference_features`: Load/build normalization=`none` reference feature cache.
- `median_fisher_ratio`: Compute median Fisher ratio over feature columns.
- `load_params_lookup`: Resolve tuned/default parameter sets for experiments.
- `run_optional_grid_search`: Run inner tuning plus outer holdout evaluation.
- `run_global_baselines`: Execute configured ML baselines and persist results.
- `load_all_predictions`: Load all cached prediction files for analysis-only runs.
- `master_results_table`: Build master model×band metrics table.
- `per_room_position_accuracy_table`: Build per-room position-accuracy table.
- `save_analysis_tables`: Regenerate analysis tables from canonical `runs.csv`.
- `load_lovo_summary_tables`: Load LOVO summaries using canonical schema.
- `lovo_aggregated_analysis_table`: Aggregate LOVO metrics as mean±std table.
- `save_lovo_analysis_table`: Save LOVO aggregate table from `runs.csv`.
- `best_confusion_predictions`: Return configured/best predictions for confusion plotting.
- `feature_columns`: Return model feature columns (excluding metadata columns).
- `split_lovo_folds`: Build leave-one-volunteer-out folds.
- `split_cross_session`: Train on trial 01 and test known users in trial 02.
- `split_dataframe`: Split feature DataFrame by selected protocol.
- `run_global_position_experiment`: Train/evaluate or load one global 52-position baseline.
- `filter_training_protocol_trials`: Filter trials used by non-cross-session protocols.
- `run_global_lovo_experiment`: Run global LOVO experiment for one model/band.

### `utils/DL/models.py`
- CNN model definitions used by the DL pipeline:
  - `BandEncoder`
  - `RoomEncoder`
  - `DualBandCNN`
  - `RoomStackedCNN`

### `utils/DL/dl_pipeline.py`
- `resolve_dl_architecture`: Resolve array builder, model class, and labels for selected architecture.
- `prepare_dl_data`: Discover/load/preprocess/cache data for DL experiments.
- `create_position_label_encoder`: Fit and persist ordered position label encoder.
- `run_dl_experiments`: Execute configured CNN experiments across bands/protocols.
- `show_dl_results`: Load DL result views and selected RF-vs-CNN comparisons.
- `set_reproducible_seeds`: Seed Python/NumPy/torch (CPU+CUDA).
- `print_torch_environment`: Print and validate CUDA/torch environment.
- `parameter_count`: Count model parameters.
- `assert_window_identity`: Verify DL window order matches ML feature order.
- `assert_split_identity`: Verify split consistency between CNN and RF baselines.
- `train_evaluate_cnn`: Train/evaluate one seeded CNN run.
- `train_evaluate_cnn_multi_seed`: Execute repeated multi-seed CNN runs.
- `save_label_classes`: Save class ordering used by CNN heads.
- `resolved_dataloader_settings`: Resolve DataLoader settings for current hardware.
- `build_frequency_window_arrays`: Build/load mmap-backed raw CSI windows per band.
- `build_room_window_arrays`: Build room-pair tensors from fusion inventory.

## Processing flow

```text
data discovery
-> CSI loading/parsing
-> subcarrier selection, calibration, normalization
-> feature windows (ML) or raw windows/tensors (DL)
-> training/inference
-> metrics, manifests, tables, and plots in results/
```

## Environment

Experiments run on the Linux server in the conda env `thesis` (Python 3.12, torch 2.11 + CUDA 12.8):

```bash
conda activate thesis
pip install -r requirements-gpu.txt -r requirements-dev.txt   # the env already matches these pins
python -m pytest                                              # run the tests
python scripts/manage_cache.py list                           # inspect caches
```
