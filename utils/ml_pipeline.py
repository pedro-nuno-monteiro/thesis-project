from __future__ import annotations

import gc
import importlib.metadata
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.cache import (
    RESULTS_ROOT,
    get_all_dataframes,
    get_cache_path,
    load_predictions,
    make_run_id,
    predictions_path,
)
from utils.csi_preprocessing import process_magnitude_data
from utils.config import PLOT_DPI, PLOT_FORMAT
from utils.feature_pipeline import build_frequency_feature_dataframes, iter_window_groups
from utils.global_position_classifier import (
    feature_columns,
    location_distance_error,
    majority_class_baselines,
    room_label_for_location,
    run_global_lovo_experiment,
    run_global_position_experiment,
    split_lovo_folds,
    split_cross_session,
    split_dataframe,
)
from utils.import_data import get_csv_files, sort_meta_info
from utils.metrics import compute_localization_metrics
from utils.models import PARAM_GRIDS, build_estimator, default_params_for
from utils.results import (
    build_run_row,
    derive_table_from_runs,
    upsert_fold_rows,
    upsert_run,
    write_run_manifest,
)
from utils.thesis_csv_processing import process_csv_files


def load_raw_csi_data(
    data_dir: Path,
    *,
    calibration_mode: str,
    csv_options: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load CSI CSV files and return magnitude data plus diagnostics."""
    all_data_files = get_csv_files(str(data_dir))
    scenarios_id, locations_id, users_id, esps_id, _ = sort_meta_info(str(data_dir))
    print(f"Scenarios present: {', '.join(scenarios_id) or 'none'}")
    print(f"Locations: {len(locations_id)} | Users: {len(users_id)} | ESPs: {len(esps_id)}")
    magnitude_data, csv_diagnostics = process_csv_files(
        all_data_files,
        return_diagnostics=True,
        calibration_mode=calibration_mode,
        **csv_options,
    )
    return magnitude_data, csv_diagnostics


def validate_cross_session_inventory(csv_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Verify trial-02 anchor/subcarrier compatibility before cross-session training."""
    required = {"user", "trial", "location", "esp", "band", "subcarriers"}
    missing = required - set(csv_diagnostics.columns)
    if missing:
        msg = f"CSV diagnostics lack cross-session inventory columns: {sorted(missing)}"
        raise ValueError(msg)

    diagnostics = csv_diagnostics.copy()
    diagnostics["user"] = diagnostics["user"].astype(str).str.zfill(2)
    diagnostics["trial"] = diagnostics["trial"].astype(str).str.zfill(2)
    diagnostics["esp"] = diagnostics["esp"].astype(str).str.removeprefix("esp_").str.zfill(2)
    expected_by_band = {
        "2.4 GHz": ({"01", "02", "03", "04", "05", "07", "08", "09", "10"}, 50),
        "5 GHz": ({f"{esp:02d}" for esp in range(11, 21)}, 56),
    }
    trial_02 = diagnostics.loc[diagnostics["trial"] == "02"]
    test_users = sorted(trial_02["user"].unique())
    if not test_users:
        raise ValueError("No trial-02 CSV recordings were discovered.")
    print(f"[cross_session inventory] discovered trial-02 users: {', '.join(test_users)}")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for user in test_users:
        user_rows = diagnostics.loc[diagnostics["user"] == user]
        locations_01 = set(user_rows.loc[user_rows["trial"] == "01", "location"].astype(str))
        locations_02 = set(user_rows.loc[user_rows["trial"] == "02", "location"].astype(str))
        if locations_01 != locations_02 or len(locations_02) != 53:
            failures.append(
                f"user={user}: trial-01/trial-02 location inventories differ or do not "
                f"contain 52 positions plus Z-0 (trial01={len(locations_01)}, "
                f"trial02={len(locations_02)}, missing={sorted(locations_01 - locations_02)}, "
                f"extra={sorted(locations_02 - locations_01)})"
            )
        for band, (expected_anchors, expected_subcarriers) in expected_by_band.items():
            user_band = diagnostics.loc[
                (diagnostics["user"] == user) & (diagnostics["band"].astype(str) == band)
            ]
            anchors_01 = set(user_band.loc[user_band["trial"] == "01", "esp"].astype(str))
            anchors_02 = set(user_band.loc[user_band["trial"] == "02", "esp"].astype(str))
            subcarriers_01 = set(
                pd.to_numeric(
                    user_band.loc[user_band["trial"] == "01", "subcarriers"],
                    errors="coerce",
                ).dropna().astype(int)
            )
            subcarriers_02 = set(
                pd.to_numeric(
                    user_band.loc[user_band["trial"] == "02", "subcarriers"],
                    errors="coerce",
                ).dropna().astype(int)
            )
            compatible = (
                anchors_01 == expected_anchors
                and anchors_02 == expected_anchors
                and subcarriers_01 == {expected_subcarriers}
                and subcarriers_02 == {expected_subcarriers}
            )
            rows.append(
                {
                    "user": user,
                    "band": band,
                    "trial_01_anchor_count": len(anchors_01),
                    "trial_02_anchor_count": len(anchors_02),
                    "trial_01_subcarriers": sorted(subcarriers_01),
                    "trial_02_subcarriers": sorted(subcarriers_02),
                    "compatible": compatible,
                }
            )
            if not compatible:
                failures.append(
                    f"user={user} band={band}: expected anchors={sorted(expected_anchors)} "
                    f"and subcarriers={expected_subcarriers}; observed trial01 "
                    f"anchors={sorted(anchors_01)} subcarriers={sorted(subcarriers_01)}, "
                    f"trial02 anchors={sorted(anchors_02)} subcarriers={sorted(subcarriers_02)}"
                )
            for trial in ("01", "02"):
                trial_band = user_band.loc[user_band["trial"] == trial]
                for location, recording_group in trial_band.groupby("location", sort=True):
                    recording_anchors = set(recording_group["esp"].astype(str))
                    recording_widths = set(
                        pd.to_numeric(recording_group["subcarriers"], errors="coerce")
                        .dropna()
                        .astype(int)
                    )
                    if (
                        recording_anchors != expected_anchors
                        or recording_widths != {expected_subcarriers}
                    ):
                        failures.append(
                            f"user={user} trial={trial} location={location} band={band}: "
                            f"anchors={sorted(recording_anchors)}, "
                            f"subcarriers={sorted(recording_widths)}"
                        )

    inventory = pd.DataFrame(rows)
    print(inventory.to_string(index=False))
    trial_02_anchors = set(trial_02["esp"].astype(str))
    if len(trial_02_anchors) != 19:
        failures.append(
            f"trial 02 has {len(trial_02_anchors)} distinct anchors, expected 19: "
            f"{sorted(trial_02_anchors)}"
        )
    if failures:
        msg = "Cross-session inventory mismatch; stopping before training:\n" + "\n".join(failures)
        raise ValueError(msg)
    print("[cross_session inventory] PASS: same 19 anchors and matching subcarrier counts")
    return inventory


def load_feature_dataframes(
    magnitude_data: dict[str, Any],
    *,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    bands_to_run: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Build or load cached feature dataframes for every requested band."""

    def _build_feature_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        processed, _ = process_magnitude_data(magnitude_data, **preproc_opts)
        return build_frequency_feature_dataframes(processed, **feat_opts)

    expected_trials = {
        str(trial_key).removeprefix("trial_").zfill(2)
        for locations in magnitude_data.values()
        for users in locations.values()
        for esps in users.values()
        for trials in esps.values()
        for trial_key, magnitude in trials.items()
        if magnitude is not None
    }
    expected_window_inventory = {
        band: {
            group.group_id: int(group.min_windows)
            for group in iter_window_groups(
                magnitude_data,
                band,
                window_size=int(feat_opts.get("window_size", 60)),
                overlap_size=int(feat_opts.get("overlap_size", feat_opts.get("step", 30))),
                require_all_esps=bool(feat_opts.get("require_all_esps", False)),
            )
        }
        for band in bands_to_run
    }
    cache_path = get_cache_path(preproc_opts, feat_opts)
    print(f"[features] resolved cache path: {cache_path}")
    feature_dataframes = get_all_dataframes(
        preproc_opts,
        feat_opts,
        _build_feature_dataframes,
        expected_trials=expected_trials,
        expected_window_inventory=expected_window_inventory,
    )
    missing_bands = sorted(set(bands_to_run) - set(feature_dataframes))
    if missing_bands:
        msg = f"BANDS_TO_RUN contains bands with no dataframe: {missing_bands}"
        raise KeyError(msg)

    for band in bands_to_run:
        df = feature_dataframes[band]
        _assert_no_empty_room_rows(df, band=band)
        print(f"{band}: {df.shape[0]} windows, {df.shape[1]} columns")
    gc.collect()
    return feature_dataframes


def print_normalization_discriminability(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    normalization: str,
    reference_feature_dataframes: dict[str, pd.DataFrame] | None = None,
    bands_to_run: tuple[str, ...],
) -> pd.DataFrame:
    """Print median Fisher ratios for the active normalization and the none reference."""
    rows = []
    for band in bands_to_run:
        active_ratio = median_fisher_ratio(feature_dataframes[band])
        row = {
            "dataset": band,
            "normalization": normalization,
            "median_fisher_ratio": active_ratio,
        }
        print(
            f"[normalization diagnostic] {band}: normalization={normalization}, "
            f"median_fisher_ratio={active_ratio:.6g}"
        )
        if reference_feature_dataframes is not None and band in reference_feature_dataframes:
            reference_ratio = median_fisher_ratio(reference_feature_dataframes[band])
            row["none_median_fisher_ratio"] = reference_ratio
            print(
                f"[normalization diagnostic] {band}: normalization=none reference, "
                f"median_fisher_ratio={reference_ratio:.6g}"
            )
        rows.append(row)

    print(
        "[normalization diagnostic] Interpretation: a median ratio near zero means the "
        "features no longer separate locations, so the normalization likely destroyed "
        "the fingerprint."
    )
    return pd.DataFrame(rows)


def load_or_build_none_reference_features(
    magnitude_data: dict[str, Any],
    *,
    active_preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load or compute the normalization='none' feature cache for diagnostics."""
    none_preproc_opts = {
        key: value
        for key, value in active_preproc_opts.items()
        if key not in {"normalization", "baseline_scope"}
    }
    none_preproc_opts["normalization"] = "none"
    none_preproc_opts.setdefault("epsilon", active_preproc_opts.get("epsilon", 1e-8))
    none_cache_path = get_cache_path(none_preproc_opts, feat_opts)
    print(f"[normalization diagnostic] none reference cache: {none_cache_path}")

    def _build_none_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        processed, _ = process_magnitude_data(magnitude_data, **none_preproc_opts)
        return build_frequency_feature_dataframes(processed, **feat_opts)

    return get_all_dataframes(none_preproc_opts, feat_opts, _build_none_dataframes)


def median_fisher_ratio(df: pd.DataFrame) -> float:
    """Median over feature columns of between-location variance / within-location variance."""
    if df.empty:
        return float("nan")
    columns = feature_columns(df)
    if not columns:
        return float("nan")

    grouped = df.groupby("location", sort=True)[columns]
    location_means = grouped.mean()
    location_variances = grouped.var(ddof=1)
    numerator = location_means.var(axis=0, ddof=1)
    denominator = location_variances.mean(axis=0)
    ratios = numerator / denominator.replace(0, np.nan)
    return float(pd.to_numeric(ratios, errors="coerce").median(skipna=True))


def load_params_lookup(
    tuned_summary_path: Path,
    *,
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    svm_kernel: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load tuned params when available; otherwise use model defaults."""
    tuned = None
    if tuned_summary_path.exists():
        tuned = pd.read_csv(tuned_summary_path)
        print(f"Loaded tuned hyperparameters from {tuned_summary_path}")
    else:
        print(f"WARNING: {tuned_summary_path} not found. Using DEFAULT_PARAMS.")

    lookup = {}
    for model in models_to_run:
        for band in bands_to_run:
            params = default_params_for(model, band)
            if model == "SVM":
                params["kernel"] = svm_kernel
            if tuned is not None:
                row = tuned.loc[(tuned["dataset"] == band) & (tuned["model"] == model)]
                if not row.empty:
                    params.update(_params_from_tuned_row(row.iloc[0]))
            lookup[(model, band)] = params
    return lookup


def run_optional_grid_search(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    run_grid_search: bool,
    tuned_summary_path: Path,
    grid_log_path: Path,
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    svm_kernel: str,
    test_size: float,
    random_state: int,
    n_blocks: int,
    n_jobs: int,
    row_spacing: float,
    column_spacing: float,
) -> bool:
    """Run the gated direct test-set grid search and return True when it ran."""
    if not run_grid_search:
        return False

    all_grid_rows = []
    best_rows = []
    for band in bands_to_run:
        if band not in feature_dataframes:
            msg = f"No feature dataframe found for band {band!r}."
            raise KeyError(msg)
        train_df, test_df = _single_split(
            feature_dataframes[band],
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
        )
        columns = feature_columns(train_df)

        for model in models_to_run:
            model_rows = _grid_search_model_band(
                train_df,
                test_df,
                columns,
                band=band,
                model=model,
                svm_kernel=svm_kernel,
                random_state=random_state,
                n_jobs=n_jobs,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            all_grid_rows.extend(model_rows)
            best_rows.append(max(model_rows, key=lambda row: row["position_accuracy"]))
            tuning_path = (
                RESULTS_ROOT
                / "tuning"
                / f"{model.lower()}__{band.lower().replace('.', '_').replace(' ', '')}__grid.csv"
            )
            tuning_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(model_rows).to_csv(tuning_path, index=False)
            print(f"Wrote model/band grid log to {tuning_path}")

    grid_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_grid_rows).to_csv(grid_log_path, index=False)
    pd.DataFrame(best_rows).to_csv(tuned_summary_path, index=False)
    print(f"Wrote full grid log to {grid_log_path}")
    print(f"Wrote tuned summary to {tuned_summary_path}")
    return True


def run_global_baselines(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    params_lookup: dict[tuple[str, str], dict[str, Any]],
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    split_modes: tuple[str, ...],
    test_size: float,
    random_state: int,
    n_blocks: int,
    n_jobs: int,
    results_dir: Path,
    summary_dir: Path,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    force_retrain: bool,
    save_predictions: bool,
    svm_fallback_seconds: float,
    row_spacing: float,
    column_spacing: float,
    frozen_block_results_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    """Run requested ML baselines and persist the flat, global result contract."""
    del summary_dir
    predictions_by_key = {}
    summary_rows = []
    lovo_per_fold_rows = []
    lovo_summary_rows = []
    lovo_fold_user_ids: dict[str, list[str]] = {}

    for band in bands_to_run:
        if band not in feature_dataframes:
            msg = f"No feature dataframe found for band {band!r}."
            raise KeyError(msg)
        dataframe = feature_dataframes[band]
        for model_name in models_to_run:
            params = params_lookup[(model_name, band)]
            for split_mode in split_modes:
                seed = None if model_name.upper() == "KNN" else int(random_state)
                if seed is None:
                    print(
                        f"[seed] {band} / KNN / {split_mode}: deterministic; "
                        "no fabricated seed is stored."
                    )
                full_config = {
                    "preproc_opts": preproc_opts,
                    "feat_opts": feat_opts,
                    "model_params": params,
                    "seed": seed,
                    "split_params": {
                        "test_size": test_size,
                        "random_state": random_state,
                        "n_blocks": n_blocks,
                        "row_spacing": row_spacing,
                        "column_spacing": column_spacing,
                    },
                }
                run_id = make_run_id(
                    family="ml",
                    model=model_name,
                    band=band,
                    split=split_mode,
                    normalization=str(preproc_opts.get("normalization", "none")),
                    baseline_scope=preproc_opts.get("baseline_scope"),
                    seed=seed,
                    config=full_config,
                )
                write_run_manifest(run_id, full_config, results_root=results_dir)
                if split_mode == "lovo":
                    predictions, per_fold_metrics, metrics = run_global_lovo_experiment(
                        dataframe,
                        dataset_name=band,
                        model_name=model_name,
                        params=params,
                        random_state=random_state,
                        row_spacing=row_spacing,
                        column_spacing=column_spacing,
                        n_jobs=n_jobs,
                        results_dir=results_dir,
                        force_retrain=force_retrain,
                        save_prediction_cache=save_predictions,
                        svm_fallback_seconds=svm_fallback_seconds,
                        run_id=run_id,
                    )
                    per_fold_metrics = per_fold_metrics.copy()
                    per_fold_metrics.insert(0, "dataset", band)
                    per_fold_metrics.insert(0, "model", model_name)
                    lovo_per_fold_rows.extend(per_fold_metrics.to_dict("records"))
                    lovo_summary_rows.append(
                        {
                            "dataset": band,
                            "model": model_name,
                            **params,
                            **metrics,
                        }
                    )
                    lovo_fold_user_ids[band] = [
                        str(test_df["user"].iloc[0])
                        for _, test_df in split_lovo_folds(dataframe)
                    ]
                else:
                    _, predictions, metrics = run_global_position_experiment(
                        dataframe,
                        dataset_name=band,
                        model_name=model_name,
                        params=params,
                        split_mode=split_mode,
                        test_size=test_size,
                        random_state=random_state,
                        row_spacing=row_spacing,
                        column_spacing=column_spacing,
                        n_blocks=n_blocks,
                        n_jobs=n_jobs,
                        results_dir=results_dir,
                        force_retrain=force_retrain,
                        save_prediction_cache=save_predictions,
                        svm_fallback_seconds=svm_fallback_seconds,
                        run_id=run_id,
                    )
                predictions_by_key[(model_name, band, split_mode)] = predictions
                if {
                    "majority_position_accuracy",
                    "majority_room_accuracy",
                } <= set(metrics):
                    print(
                        f"[baseline] {band} / {model_name} / {split_mode}: "
                        f"majority_position_accuracy={metrics['majority_position_accuracy']:.4f}, "
                        f"majority_room_accuracy={metrics['majority_room_accuracy']:.4f}"
                    )
                summary_rows.append(
                    {
                        "dataset": band,
                        "model": model_name,
                        "split": split_mode,
                        **params,
                        **metrics,
                    }
                )

                protocol_split = split_dataframe(
                    dataframe,
                    test_size=test_size,
                    random_state=random_state,
                    split_mode=split_mode,
                    stratify_column="location",
                    n_blocks=n_blocks,
                )
                protocol_folds = (
                    protocol_split if isinstance(protocol_split, list) else [protocol_split]
                )
                trials_used = sorted(
                    {
                        str(trial).removeprefix("trial_").zfill(2)
                        for train_df, test_df in protocol_folds
                        for frame in (train_df, test_df)
                        for trial in frame["trial"].dropna().unique()
                    }
                )
                run_metrics = dict(metrics)
                if split_mode == "lovo":
                    for metric_name in (
                        "position_accuracy",
                        "macro_f1",
                        "room_accuracy",
                        "mean_distance_error",
                        "median_distance_error",
                        "rmse_distance_error",
                        "p90_distance_error",
                    ):
                        run_metrics.setdefault(
                            metric_name,
                            run_metrics.get(f"{metric_name}_mean", np.nan),
                        )
                run_row = build_run_row(
                    run_id=run_id,
                    preproc_opts=preproc_opts,
                    feat_opts=feat_opts,
                    hyperparameters=params,
                    metrics=run_metrics,
                    trials_used=trials_used,
                    n_train=sum(len(train_df) for train_df, _ in protocol_folds),
                    n_test=sum(len(test_df) for _, test_df in protocol_folds),
                    n_classes=int(
                        pd.concat(
                            [frame for fold in protocol_folds for frame in fold],
                            ignore_index=True,
                        )["location"].nunique()
                    ),
                    device="cpu",
                )
                upsert_run(
                    run_row,
                    hyperparameter_columns=params.keys(),
                    results_root=results_dir,
                )
                if split_mode == "lovo":
                    fold_metrics_by_user = {
                        str(row["held_out_user"]): row
                        for row in per_fold_metrics.to_dict("records")
                    }
                    upsert_fold_rows(
                        [
                            {
                                "run_id": run_id,
                                "fold": str(test_df["user"].iloc[0]),
                                "held_out_user": str(test_df["user"].iloc[0]),
                                "validation_user": None,
                                "n_train_windows": len(train_df),
                                "n_test_windows": len(test_df),
                                "trials_used": ",".join(trials_used),
                                **fold_metrics_by_user[str(test_df["user"].iloc[0])],
                            }
                            for train_df, test_df in protocol_folds
                        ],
                        results_root=results_dir,
                    )
                elif split_mode == "cross_session":
                    train_df, _ = protocol_folds[0]
                    cross_rows = []
                    for user, user_predictions in predictions.groupby("user", sort=True):
                        user_metrics = compute_localization_metrics(user_predictions)
                        user_test = user_predictions[["true_position"]].rename(
                            columns={"true_position": "location"}
                        )
                        user_metrics.update(majority_class_baselines(train_df, user_test))
                        cross_rows.append(
                            {
                                "run_id": run_id,
                                "fold": str(user),
                                "held_out_user": str(user),
                                "validation_user": None,
                                "n_train_windows": len(train_df),
                                "n_test_windows": len(user_predictions),
                                "trials_used": ",".join(trials_used),
                                **user_metrics,
                            }
                        )
                    upsert_fold_rows(cross_rows, results_root=results_dir)

    summary = pd.DataFrame(summary_rows)
    derive_table_from_runs("global_summary", results_root=results_dir)
    if "cross_session" in split_modes and frozen_block_results_dir is not None:
        print(
            "[cross_session] per-user rows are stored in runs_folds.csv; no independent "
            "summary tables are written because runs.csv is the single source of truth."
        )
    return summary, predictions_by_key


def _write_cross_session_reports(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    predictions_by_key: dict[tuple[str, str, str], pd.DataFrame],
    summary: pd.DataFrame,
    summary_dir: Path,
    block_results_dir: Path,
    row_spacing: float,
    column_spacing: float,
) -> None:
    """Persist the protocol-level, per-user, and paired cross-session tables."""
    cross_summary = summary.loc[summary["split"] == "cross_session"].copy()
    cross_summary.to_csv(summary_dir / "cross_session_summary.csv", index=False)

    per_user_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for (model, band, split_mode), predictions in predictions_by_key.items():
        if split_mode != "cross_session":
            continue
        train_df, test_df = split_cross_session(feature_dataframes[band])
        block_path = predictions_path(block_results_dir, model, band, "block")
        block_predictions = pd.read_parquet(block_path) if block_path.exists() else None
        if block_predictions is None:
            print(
                f"[cross_session paired warning] frozen block predictions not found: {block_path}"
            )

        for test_user, user_predictions in predictions.groupby("user", sort=True):
            user_test_df = test_df.loc[test_df["user"].astype(str) == str(test_user)]
            user_metrics = compute_localization_metrics(user_predictions)
            user_metrics.update(
                majority_class_baselines(
                    train_df,
                    user_test_df,
                    row_spacing=row_spacing,
                    column_spacing=column_spacing,
                )
            )
            per_user_rows.append(
                {
                    "model": model,
                    "dataset": band,
                    "test_user": str(test_user).zfill(2),
                    **user_metrics,
                    "n_test_windows": int(len(user_predictions)),
                }
            )
            if block_predictions is not None:
                user_block = block_predictions.loc[
                    block_predictions["user"].astype(str).str.zfill(2)
                    == str(test_user).zfill(2)
                ]
                if user_block.empty:
                    print(
                        f"[cross_session paired warning] no frozen block rows for user={test_user}, "
                        f"model={model}, dataset={band}"
                    )
                    continue
                block_accuracy = compute_localization_metrics(user_block)["position_accuracy"]
                cross_accuracy = user_metrics["position_accuracy"]
                paired_rows.append(
                    {
                        "model": model,
                        "dataset": band,
                        "test_user": str(test_user).zfill(2),
                        "block_position_accuracy": block_accuracy,
                        "cross_session_position_accuracy": cross_accuracy,
                        "delta_block_minus_cross_session": block_accuracy - cross_accuracy,
                    }
                )

    pd.DataFrame(per_user_rows).to_csv(
        summary_dir / "cross_session_per_user.csv",
        index=False,
    )
    pd.DataFrame(paired_rows).to_csv(
        summary_dir / "cross_session_paired_per_user.csv",
        index=False,
    )
    print(f"[cross_session] reports saved under {summary_dir}")


def load_all_predictions(
    results_dir: Path,
    *,
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    split_modes: tuple[str, ...],
) -> pd.DataFrame:
    """Load all persisted prediction parquet files for analysis-only notebook runs."""
    runs_path = Path(results_dir) / "runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        requested_models = {str(model).lower() for model in models_to_run}
        requested_bands = {
            band.lower().replace(".", "_").replace(" ", "") for band in bands_to_run
        }
        requested_splits = set(split_modes)
        selected = runs.loc[
            runs["model"].astype(str).str.lower().isin(requested_models)
            & runs["band"].astype(str).isin(requested_bands)
            & runs["split"].astype(str).isin(requested_splits)
        ]
        if selected.empty:
            raise FileNotFoundError(
                "runs.csv contains no rows matching the requested models/bands/splits."
            )
        frames = []
        missing = []
        for run_id in selected["run_id"].astype(str):
            path = Path(results_dir) / "predictions" / f"{run_id}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
            else:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "runs.csv references missing prediction parquets:\n" + "\n".join(missing)
            )
        return pd.concat(frames, ignore_index=True)

    frames = []
    missing = []
    for model in models_to_run:
        for band in bands_to_run:
            for split_mode in split_modes:
                predictions = load_predictions(results_dir, model, band, split_mode)
                if predictions is None:
                    missing.append(str(predictions_path(results_dir, model, band, split_mode)))
                    continue
                frames.append(predictions)
    if missing:
        msg = "Missing prediction parquet files. Run the Global baseline cell first.\n"
        raise FileNotFoundError(msg + "\n".join(missing))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def master_results_table(
    predictions: pd.DataFrame,
    *,
    summary_path: Path | None = None,
) -> pd.DataFrame:
    """Build the model x band master metrics table from predictions."""
    rows = []
    for (model, band, split_mode), group in predictions.groupby(
        ["model", "dataset", "split_mode"],
        sort=False,
    ):
        rows.append(
            {
                "model": model,
                "dataset": band,
                "split": split_mode,
                **_metrics_for_prediction_group(group, split_mode=str(split_mode)),
            }
        )
    table = pd.DataFrame(rows)
    if summary_path is not None and summary_path.exists() and not table.empty:
        timings = pd.read_csv(summary_path)
        if "run_id" in timings.columns and "band" in timings.columns:
            band_labels = {
                "2_4ghz": "2.4 GHz",
                "5ghz": "5 GHz",
                "fusion": "Fusion",
            }
            timings = timings.copy()
            timings["dataset"] = timings["band"].astype(str).map(band_labels)
            timings["model"] = timings["model"].astype(str).str.upper()
        timing_cols = [
            col
            for col in [
                "model",
                "dataset",
                "split",
                "position_accuracy_mean",
                "position_accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "room_accuracy_mean",
                "room_accuracy_std",
                "mean_distance_error_mean",
                "mean_distance_error_std",
                "median_distance_error_mean",
                "median_distance_error_std",
                "rmse_distance_error_mean",
                "rmse_distance_error_std",
                "p90_distance_error_mean",
                "p90_distance_error_std",
                "position_accuracy_pooled",
                "majority_position_accuracy",
                "majority_room_accuracy",
                "fit_seconds",
                "predict_seconds",
                "wall_seconds",
                "used_estimator",
            ]
            if col in timings.columns
        ]
        table = table.merge(timings[timing_cols], on=["model", "dataset", "split"], how="left")
        table = _format_lovo_master_rows(table)
    return table.sort_values(["dataset", "model", "split"]).reset_index(drop=True)


def per_room_position_accuracy_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build per-room position accuracy rows for every model and band."""
    rows = []
    for (model, band, split_mode, room), group in predictions.groupby(
        ["model", "dataset", "split_mode", "true_room"],
        sort=False,
    ):
        rows.append(
            {
                "model": model,
                "dataset": band,
                "split": split_mode,
                "true_room": room,
                "samples": len(group),
                "position_accuracy": float(
                    (group["true_position"] == group["pred_position"]).mean()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model", "split", "true_room"])
        .reset_index(drop=True)
    )


def save_analysis_tables(
    master_table: pd.DataFrame,
    per_room_table: pd.DataFrame,
    *,
    tables_dir: Path,
) -> None:
    """Regenerate the global ML view from runs.csv, the sole table source."""
    del master_table, per_room_table
    derive_table_from_runs(
        "global_ml_baseline",
        query="family == 'ml'",
        results_root=tables_dir.parent,
    )
    print("[tables] per-room predictions remain analysis-only and are not a runs.csv view.")


def load_lovo_summary_tables(summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LOVO rows from the two authoritative global CSV files."""
    results_root = summary_dir.parent if summary_dir.name in {"summary", "tables"} else summary_dir
    per_fold_path = results_root / "runs_folds.csv"
    summary_path = results_root / "runs.csv"
    missing = [str(path) for path in (per_fold_path, summary_path) if not path.exists()]
    if missing:
        msg = "Missing LOVO summary files. Run the Global baseline cell first.\n"
        raise FileNotFoundError(msg + "\n".join(missing))
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary["split"].astype(str) == "lovo"].copy()
    summary["dataset"] = summary["band"].astype(str).map(
        {"2_4ghz": "2.4 GHz", "5ghz": "5 GHz", "fusion": "Fusion"}
    )
    summary["model"] = summary["model"].astype(str).str.upper()
    run_ids = set(summary["run_id"].astype(str))
    per_fold = pd.read_csv(per_fold_path)
    per_fold = per_fold.loc[per_fold["run_id"].astype(str).isin(run_ids)].copy()
    per_fold = per_fold.merge(
        summary[["run_id", "model", "dataset"]],
        on="run_id",
        how="left",
    )
    return per_fold, summary


def lovo_aggregated_analysis_table(lovo_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the model x band LOVO table with mean +/- std metric strings."""
    metrics = [
        ("position_accuracy", "position_accuracy_mean_std"),
        ("macro_f1", "macro_f1_mean_std"),
        ("room_accuracy", "room_accuracy_mean_std"),
        ("mean_distance_error", "mean_distance_error_mean_std"),
        ("median_distance_error", "median_distance_error_mean_std"),
        ("rmse_distance_error", "rmse_distance_error_mean_std"),
        ("p90_distance_error", "p90_distance_error_mean_std"),
        ("majority_position_accuracy", "majority_position_accuracy_mean_std"),
        ("majority_room_accuracy", "majority_room_accuracy_mean_std"),
    ]
    rows = []
    for _, row in lovo_summary.iterrows():
        output_row = {
            "model": row["model"],
            "dataset": row.get("dataset", row.get("band")),
        }
        for metric, output_col in metrics:
            output_row[output_col] = _format_mean_std(
                row.get(f"{metric}_mean", np.nan),
                row.get(f"{metric}_std", np.nan),
            )
        rows.append(output_row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def save_lovo_analysis_table(table: pd.DataFrame, *, tables_dir: Path) -> None:
    """Regenerate the LOVO view directly from runs.csv."""
    del table
    derive_table_from_runs(
        "lovo_aggregated_table",
        query="split == 'lovo'",
        results_root=tables_dir.parent,
    )


def plot_lovo_fold_spread(
    lovo_per_fold: pd.DataFrame,
    *,
    bands: tuple[str, ...],
    model: str = "RF",
    save_path: Path | None = None,
) -> None:
    """Plot held-out-user position accuracy spread for the selected model."""
    import matplotlib.pyplot as plt

    filtered = lovo_per_fold.loc[lovo_per_fold["model"] == model]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = []
    labels = []
    positions = []
    for index, band in enumerate(bands, start=1):
        band_values = pd.to_numeric(
            filtered.loc[filtered["dataset"] == band, "position_accuracy"],
            errors="coerce",
        ).dropna()
        if band_values.empty:
            continue
        values.append(band_values.to_numpy(dtype=float))
        labels.append(band)
        positions.append(index)

    if values:
        ax.boxplot(values, positions=positions, labels=labels, widths=0.45)
        for position, band_values in zip(positions, values):
            offsets = np.linspace(-0.08, 0.08, num=len(band_values))
            ax.scatter(
                np.full(len(band_values), position) + offsets,
                band_values,
                color="black",
                s=28,
                zorder=3,
            )
    else:
        ax.text(0.5, 0.5, "No LOVO fold metrics", ha="center", va="center")
    ax.set_ylabel("Position accuracy")
    ax.set_title(f"LOVO held-out-user spread - {model}")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    _save_plot(fig, save_path)


def plot_block_vs_lovo_position_accuracy(
    global_summary: pd.DataFrame,
    lovo_summary: pd.DataFrame,
    *,
    bands: tuple[str, ...],
    model: str = "RF",
    save_path: Path | None = None,
) -> None:
    """Plot block position accuracy beside LOVO fold-mean accuracy."""
    import matplotlib.pyplot as plt

    rows = []
    for band in bands:
        block_row = global_summary.loc[
            (global_summary["model"] == model)
            & (global_summary["dataset"] == band)
            & (global_summary["split"] == "block")
        ]
        lovo_row = lovo_summary.loc[
            (lovo_summary["model"] == model) & (lovo_summary["dataset"] == band)
        ]
        if block_row.empty or lovo_row.empty:
            continue
        rows.append(
            {
                "dataset": band,
                "block": float(block_row.iloc[0]["position_accuracy"]),
                "lovo": float(lovo_row.iloc[0]["position_accuracy_mean"]),
            }
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if rows:
        x = np.arange(len(rows))
        width = 0.35
        block_values = np.array([row["block"] for row in rows])
        lovo_values = np.array([row["lovo"] for row in rows])
        ax.bar(x - width / 2, block_values, width, label="Block")
        ax.bar(x + width / 2, lovo_values, width, label="LOVO mean")
        for idx, (block_value, lovo_value) in enumerate(zip(block_values, lovo_values)):
            gap = block_value - lovo_value
            ax.text(
                idx,
                max(block_value, lovo_value) + 0.025,
                f"gap {gap:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([row["dataset"] for row in rows])
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Need both block and LOVO RF results", ha="center", va="center")
    ax.set_ylabel("Position accuracy")
    ax.set_title(f"Block vs LOVO cross-user generalization cost - {model}")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    _save_plot(fig, save_path)


def best_confusion_predictions(
    predictions: pd.DataFrame,
    results_table: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    split: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Return predictions for the configured or best-performing confusion model."""
    if model == "best":
        candidates = results_table.loc[results_table["dataset"] == dataset]
        if candidates.empty:
            msg = f"No metrics found for dataset={dataset!r}."
            raise ValueError(msg)
        best_row = candidates.sort_values("position_accuracy", ascending=False).iloc[0]
        model_name = str(best_row["model"])
        split_name = str(best_row["split"])
    else:
        model_name = model
        split_candidates = predictions.loc[
            (predictions["dataset"] == dataset) & (predictions["model"] == model_name),
            "split_mode",
        ].astype(str)
        if split is not None:
            split_name = split
        elif "block" in set(split_candidates):
            split_name = "block"
        elif not split_candidates.empty:
            split_name = str(split_candidates.iloc[0])
        else:
            split_name = ""
    filtered = predictions.loc[
        (predictions["dataset"] == dataset)
        & (predictions["model"] == model_name)
        & (predictions["split_mode"].astype(str) == split_name)
    ].copy()
    if filtered.empty:
        msg = (
            f"No predictions for dataset={dataset!r}, model={model_name!r}, "
            f"split={split_name!r}."
        )
        raise ValueError(msg)
    return model_name, filtered


def _metrics_for_prediction_group(group: pd.DataFrame, *, split_mode: str) -> dict[str, float]:
    if split_mode == "lovo" and "held_out_user" in group.columns:
        fold_rows = [
            compute_localization_metrics(fold_group)
            for _, fold_group in group.groupby("held_out_user", sort=True)
        ]
        metrics: dict[str, float] = {}
        for metric_name in [
            "position_accuracy",
            "macro_f1",
            "room_accuracy",
            "mean_distance_error",
            "median_distance_error",
            "rmse_distance_error",
            "p90_distance_error",
        ]:
            values = pd.to_numeric(
                pd.Series([row[metric_name] for row in fold_rows]),
                errors="coerce",
            )
            metrics[metric_name] = float(values.mean())
        metrics["samples"] = float(len(group))
        return metrics
    return compute_localization_metrics(group)


def _assert_no_empty_room_rows(df: pd.DataFrame, *, band: str) -> None:
    if "location" not in df.columns:
        return
    empty_mask = df["location"].astype(str).str.upper() == "Z-0"
    if empty_mask.any():
        msg = (
            f"Z-0 empty-room calibration rows leaked into the {band} feature dataframe: "
            f"{int(empty_mask.sum())} rows."
        )
        raise ValueError(msg)


def _format_lovo_master_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "split" not in table.columns:
        return table
    metric_names = [
        "position_accuracy",
        "macro_f1",
        "room_accuracy",
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
        "p90_distance_error",
    ]
    lovo_mask = table["split"].astype(str) == "lovo"
    if not lovo_mask.any():
        return table
    table = table.copy()
    for metric_name in metric_names:
        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col in table.columns and std_col in table.columns and metric_name in table.columns:
            table[metric_name] = table[metric_name].astype(object)
            table.loc[lovo_mask, metric_name] = [
                _format_mean_std(mean_value, std_value)
                for mean_value, std_value in zip(
                    table.loc[lovo_mask, mean_col],
                    table.loc[lovo_mask, std_col],
                )
            ]
    return table


def _build_lovo_manifest_metadata(
    fold_user_ids: dict[str, list[str]],
    *,
    params_lookup: dict[tuple[str, str], dict[str, Any]],
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    return {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "sklearn_version": _package_version("scikit-learn"),
        "fold_user_ids": fold_user_ids,
        "classifier_hyperparameters": {
            f"{model}:{band}": params_lookup[(model, band)]
            for model in models_to_run
            for band in bands_to_run
        },
        "hyperparameter_provenance": (
            "LOVO reuses block-split tuned/default hyperparameters; no nested CV."
        ),
    }


def _format_mean_std(mean_value: object, std_value: object) -> str:
    mean_float = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    std_float = pd.to_numeric(pd.Series([std_value]), errors="coerce").iloc[0]
    if pd.isna(mean_float) or pd.isna(std_float):
        return "nan"
    return f"{float(mean_float):.4f} +/- {float(std_float):.4f}"


def _save_plot(fig, save_path: Path | None) -> None:
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path).with_suffix(f".{PLOT_FORMAT}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            bbox_inches="tight",
            dpi=PLOT_DPI,
            format=PLOT_FORMAT,
        )
    import matplotlib.pyplot as plt

    plt.show()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _grid_search_model_band(  # noqa: PLR0913
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    *,
    band: str,
    model: str,
    svm_kernel: str,
    random_state: int,
    n_jobs: int,
    row_spacing: float,
    column_spacing: float,
) -> list[dict[str, Any]]:
    base_params = default_params_for(model, band)
    if model == "SVM":
        base_params["kernel"] = svm_kernel
    grid = PARAM_GRIDS[model]
    param_names = list(grid)
    combos = list(itertools.product(*[grid[name] for name in param_names]))
    print(f"[grid] {band} / {model}: {len(combos)} combinations")

    rows = []
    for combo_idx, values in enumerate(combos, start=1):
        params = {**base_params, **dict(zip(param_names, values))}
        estimator = build_estimator(model, params, random_state, n_jobs)
        if model == "SVM":
            print("[SVM] sklearn SVC is single-threaded; n_jobs is not used by SVC.")
        started_at = time.perf_counter()
        estimator.fit(train_df[columns], train_df["location"].astype(str))
        fit_seconds = time.perf_counter() - started_at
        pred_positions = estimator.predict(test_df[columns])
        pred_rooms = np.array([room_label_for_location(pos) for pos in pred_positions])
        distances = [
            location_distance_error(
                true_pos,
                pred_pos,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            for true_pos, pred_pos in zip(test_df["location"], pred_positions)
        ]
        pred_df = pd.DataFrame(
            {
                "true_position": test_df["location"].astype(str).to_numpy(),
                "pred_position": pred_positions,
                "true_room": np.array(
                    [room_label_for_location(pos) for pos in test_df["location"]]
                ),
                "pred_room": pred_rooms,
                "distance_error": distances,
            }
        )
        row = {
            "dataset": band,
            "model": model,
            **params,
            **compute_localization_metrics(pred_df),
            "fit_seconds": fit_seconds,
        }
        rows.append(row)
        print(
            f"[grid {combo_idx}/{len(combos)}] {band} / {model} "
            f"acc={row['position_accuracy']:.4f} fit={fit_seconds:.1f}s"
        )
    return rows


def _single_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_result = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
        split_mode="block",
        stratify_column="location",
        n_blocks=n_blocks,
    )
    if isinstance(split_result, list):
        msg = "Grid search expects one train/test split, not LOVO folds."
        raise RuntimeError(msg)
    return split_result


def _params_from_tuned_row(row: pd.Series) -> dict[str, Any]:
    metric_columns = {
        "dataset",
        "model",
        "position_accuracy",
        "macro_f1",
        "room_accuracy",
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
        "p90_distance_error",
        "samples",
        "fit_seconds",
        "used_estimator",
    }
    params = {}
    for key, value in row.items():
        if key in metric_columns:
            continue
        coerced = _coerce_param_value(value)
        if coerced is not None:
            params[key] = coerced
    return params


def _coerce_param_value(value: Any) -> Any:
    if value in (None, "", "None"):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    return value
