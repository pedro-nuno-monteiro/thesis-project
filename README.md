# Wi-Fi CSI Localisation Thesis

This repository contains the data-processing, classical machine-learning, and deep-learning pipelines used in the thesis project for indoor localisation with Wi-Fi CSI (Channel State Information).

## Final project organization

```text
.
├── ML.ipynb                          # Main notebook for classical ML experiments
├── DL.ipynb                          # Main notebook for deep-learning experiments
├── udp_server.py                     # UDP CSI acquisition script
├── utils/                            # Core reusable Python pipeline code
├── results/                          # Final generated artifacts (runs, tables, manifests, predictions, plots)
├── extras/                           # Support scripts and automated tests
├── code/                             # Legacy exploration notebooks/scripts used during development
├── isac paper/                       # ISAC paper-specific notebook, images, and utilities
├── requirements.txt                  # Standard Python dependencies
├── requirements-gpu.txt              # GPU-focused dependencies
└── pyproject.toml                    # Project/tool configuration
```

## What the other folders do

- `results/` stores experiment outputs used for reporting:
  - `runs.csv` is the canonical experiment registry.
  - `manifests/` stores run configurations.
  - `predictions/` stores cached prediction parquet files.
  - `tables/` stores publication-ready summary tables.
  - `plots/` stores generated figures.
  - `tuning/` stores grid-search outputs.
- `extras/` contains repository support material:
  - `scripts/manage_cache.py` for cache management.
  - `tests/` for regression/consistency checks.
- `code/` contains historical notebooks/scripts from earlier experimentation and preprocessing iterations.
- `isac paper/` contains paper-focused assets (`paper.ipynb`, `images/`, `paper_utils/`).

## Utils package reference

All active reusable implementation is in `utils/`.

### `utils/import_data.py`
- `sort_meta_info`: Parse and sort metadata from valid CSV filenames.
- `get_csv_files`: Discover valid CSV files and index them by scenario/location/user/ESP/trial.

### `utils/load_csi.py`
- `extract_csi_numbers`: Parse CSI numeric payload inside bracketed CSV entries.
- `read_csi_columns`: Read CSI and RSSI columns required from one CSV file.
- `parse_valid_packet_rows`: Keep complete CSI packets and aligned RSSI values.
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
- `save_summary`: Save summary DataFrame to CSV/Markdown/LaTeX.
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

### `utils/ML/models.py`
- `default_params_for`: Return default hyperparameters for one model/band pair.
- `build_estimator`: Build a configured classical estimator pipeline.

### `utils/ML/ml_pipeline.py`
- `load_raw_csi_data`: Load CSI and optional aligned pre-calibration arrays.
- `validate_cross_session_inventory`: Validate trial-02 compatibility for cross-session training.
- `select_cross_session_trials`: Keep only trials used by cross-session protocol.
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

- Standard environment: `requirements.txt`
- GPU environment (DL-focused): `requirements-gpu.txt`
