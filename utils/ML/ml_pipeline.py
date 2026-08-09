from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    ParameterGrid,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from utils.cache import (
    get_all_dataframes,
    get_cache_path,
    get_results_path,
    load_predictions,
    make_run_id,
    prediction_cache_metadata,
    predictions_path,
    save_predictions,
)
from utils.config import (
    GRID_INNER_N_BLOCKS,
    GRID_SEARCH_N_JOBS,
    TRIALS_FOR_TRAINING_PROTOCOLS,
)
from utils.csi_processing import process_magnitude_data
from utils.feature_pipeline import build_frequency_feature_dataframes, iter_window_groups
from utils.import_data import get_csv_files, sort_meta_info
from utils.load_csi import process_csv_files
from utils.ML.models import PARAM_GRIDS, build_estimator, default_params_for
from utils.results import (
    build_global_predictions_dataframe,
    build_run_row,
    compute_localization_metrics,
    derive_table_from_runs,
    majority_class_baselines,
    upsert_fold_rows,
    upsert_run,
    write_run_manifest,
)


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

    # Compare the two session inventories at both user-band and individual
    # recording levels so a mismatch is caught before model fitting.
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


def select_cross_session_trials(magnitude_data: dict[str, Any]) -> dict[str, Any]:
    """Keep only trials 01 and 02 used by the cross-session protocol."""
    selected: dict[str, Any] = {}
    excluded = 0
    for scenario_key, locations in magnitude_data.items():
        selected_locations = {}
        for location_key, users in locations.items():
            selected_users = {}
            for user_key, esps in users.items():
                selected_esps = {}
                for esp_key, trials in esps.items():
                    selected_trials = {}
                    for trial_key, magnitude in trials.items():
                        trial = str(trial_key).removeprefix("trial_").zfill(2)
                        if trial in {"01", "02"}:
                            selected_trials[trial_key] = magnitude
                        else:
                            excluded += 1
                    if selected_trials:
                        selected_esps[esp_key] = selected_trials
                if selected_esps:
                    selected_users[user_key] = selected_esps
            if selected_users:
                selected_locations[location_key] = selected_users
        if selected_locations:
            selected[scenario_key] = selected_locations
    print(f"[cross_session] excluded {excluded} recording(s) outside trials 01/02")
    return selected


def load_feature_dataframes(
    magnitude_data: dict[str, Any],
    *,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    bands_to_run: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Build or load cached feature dataframes for every requested band."""

    def _build_feature_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process magnitude data and build all frequency feature DataFrames."""
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
        """Build reference DataFrames without magnitude normalization."""
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


def load_params_lookup(  # noqa: PLR0913
    results_dir: Path,
    *,
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    require_tuned_params: bool,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load exact grid-search parameters from runs.csv or explicitly use defaults."""
    runs_path = Path(results_dir) / "runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for model in models_to_run:
        model_name = model.upper()
        for band in bands_to_run:
            expected_run_id, _ = _grid_search_identity(
                model=model_name,
                band=band,
                preproc_opts=preproc_opts,
                feat_opts=feat_opts,
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
            )
            matches = (
                runs.loc[runs["run_id"].astype(str) == expected_run_id].copy()
                if "run_id" in runs.columns
                else pd.DataFrame()
            )
            if matches.empty:
                message = (
                    "No tuned grid-search parameters exist for the exact current "
                    f"configuration: model={model_name}, band={band}, "
                    f"expected_run_id={expected_run_id}, runs_csv={runs_path}."
                )
                if require_tuned_params:
                    raise LookupError(f"STRICT TUNED-PARAMETER LOOKUP FAILED: {message}")
                border = "!" * 96
                print(f"\n{border}")
                print("!!! NO EXACT TUNED PARAMETERS - USING DEFAULT_PARAMS !!!")
                print(message)
                print(f"{border}\n")
                lookup[(model, band)] = default_params_for(model_name, band)
                continue

            if "timestamp" in matches.columns:
                matches = matches.sort_values("timestamp", kind="stable")
            matched_row = matches.iloc[-1]
            lookup[(model, band)] = _params_from_grid_search_row(
                matched_row,
                model=model_name,
                band=band,
            )
            print(
                f"[tuned params] exact runs.csv match: {model_name} / {band} / "
                f"run_id={expected_run_id}"
            )
    return lookup


def run_optional_grid_search(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    run_grid_search: bool,
    results_dir: Path,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    test_size: float,
    random_state: int,
    n_blocks: int,
    row_spacing: float,
    column_spacing: float,
) -> bool:
    """Tune on one inner block split and evaluate on the outer block holdout."""
    if not run_grid_search:
        return False
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    for band in bands_to_run:
        if band not in feature_dataframes:
            raise KeyError(f"No feature dataframe found for band {band!r}.")

        # Preserve the established temporal block split as the untouched outer
        # train/test boundary.
        protocol_df = filter_training_protocol_trials(
            feature_dataframes[band],
            split_mode="grid_search",
        )
        outer_train, outer_test = _split_dataframe_by_blocks(
            protocol_df,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
        )
        _print_protocol_split("grid_search_outer_block", outer_train, outer_test)
        _assert_block_split_integrity(
            protocol_df,
            outer_train,
            outer_test,
            band=band,
            split_label="outer",
        )

        inner_validation_size = test_size
        inner_random_state = random_state + 1
        inner_train, validation = _split_dataframe_by_blocks(
            outer_train,
            test_size=inner_validation_size,
            random_state=inner_random_state,
            n_blocks=GRID_INNER_N_BLOCKS,
        )
        _print_protocol_split("grid_search_inner_block", inner_train, validation)
        _assert_block_split_integrity(
            outer_train,
            inner_train,
            validation,
            band=band,
            split_label="inner",
        )
        columns = feature_columns(inner_train)

        for model in models_to_run:
            _grid_search_model_band(
                outer_train,
                outer_test,
                inner_train,
                validation,
                columns,
                band=band,
                model=model.upper(),
                preproc_opts=preproc_opts,
                feat_opts=feat_opts,
                results_dir=Path(results_dir),
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
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
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    force_retrain: bool,
    save_predictions: bool,
    svm_fallback_seconds: float,
    row_spacing: float,
    column_spacing: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    """Run requested ML baselines and persist the flat, global result contract."""
    predictions_by_key = {}
    summary_rows = []
    lovo_per_fold_rows = []
    lovo_summary_rows = []
    lovo_fold_user_ids: dict[str, list[str]] = {}

    # Coordinate each requested band/model/protocol combination while delegating
    # fitting and fold logic to the experiment helpers below.
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
                    # LOVO owns its fold loop and returns both fold-level and
                    # aggregate metrics for the shared result contract.
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

                # Reconstruct protocol folds to record exact train/test counts and
                # trial provenance beside the already computed metrics.
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
    return summary, predictions_by_key


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
    # Recompute scientific metrics from prediction rows; the optional run table
    # contributes timing and fold-spread metadata only.
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
    per_fold = per_fold.drop(columns=["model", "dataset"], errors="ignore")
    per_fold = per_fold.merge(
        summary[["run_id", "model", "dataset"]],
        on="run_id",
        how="left",
    )
    assert "model" in per_fold.columns
    assert "dataset" in per_fold.columns
    assert per_fold["model"].notna().all()
    assert per_fold["dataset"].notna().all()
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
    """Compute shared metrics, averaging LOVO folds without pooling their weights."""
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
    """Ensure Z-0 calibration recordings did not enter an ML feature table."""
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
    """Format LOVO mean and standard deviation pairs for the master result table."""
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


def _format_mean_std(mean_value: object, std_value: object) -> str:
    """Format a numeric mean and standard deviation for result tables."""
    mean_float = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    std_float = pd.to_numeric(pd.Series([std_value]), errors="coerce").iloc[0]
    if pd.isna(mean_float) or pd.isna(std_float):
        return "nan"
    return f"{float(mean_float):.4f} +/- {float(std_float):.4f}"


def _grid_search_model_band(  # noqa: PLR0913, PLR0914
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    inner_train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    *,
    band: str,
    model: str,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    results_dir: Path,
    test_size: float,
    random_state: int,
    n_blocks: int,
    row_spacing: float,
    column_spacing: float,
) -> dict[str, Any]:
    """Select on one inner block split, then refit and test the winner once."""
    grid = PARAM_GRIDS[model]
    candidates = list(ParameterGrid(grid))
    candidate_count = len(candidates)
    selection_fit_count = candidate_count
    refit_count = 1
    total_fit_count = selection_fit_count + refit_count
    print(
        f"[grid start] {band} / {model}: candidate_count={candidate_count}, "
        f"selection_fit_count={selection_fit_count}, n_jobs={GRID_SEARCH_N_JOBS}"
    )
    if model == "SVM" and band == "Fusion":
        print(
            "WARNING: a single Fusion SVM fit has previously taken 15+ minutes; "
            "this parameter search may take a long time. Interrupt it manually if needed."
        )

    best_params: dict[str, Any] | None = None
    best_validation_accuracy = float("-inf")
    grid_rows: list[dict[str, Any]] = []
    selection_started = time.perf_counter()
    for candidate_index, candidate_params in enumerate(candidates, start=1):
        print(
            f"[grid fit START] {band} / {model}: "
            f"candidate={candidate_index}/{candidate_count}, params={candidate_params}",
            flush=True,
        )
        candidate_estimator = build_estimator(
            model,
            candidate_params,
            random_state,
            GRID_SEARCH_N_JOBS,
        )
        fit_started = time.perf_counter()
        candidate_estimator.fit(
            inner_train[columns],
            inner_train["location"].astype(str),
        )
        candidate_fit_seconds = time.perf_counter() - fit_started
        validation_predict_started = time.perf_counter()
        validation_predictions = candidate_estimator.predict(validation[columns])
        validation_predict_seconds = time.perf_counter() - validation_predict_started
        validation_accuracy = float(
            np.mean(validation_predictions == validation["location"].astype(str).to_numpy())
        )
        grid_rows.append(
            {
                **candidate_params,
                "validation_position_accuracy": validation_accuracy,
                "fit_seconds": float(candidate_fit_seconds),
                "predict_seconds": float(validation_predict_seconds),
            }
        )
        print(
            f"[grid fit COMPLETE] {band} / {model}: "
            f"candidate={candidate_index}/{candidate_count}, "
            f"validation_position_accuracy={validation_accuracy:.6f}, "
            f"fit_seconds={candidate_fit_seconds:.3f}, "
            f"predict_seconds={validation_predict_seconds:.3f}",
            flush=True,
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_params = dict(candidate_params)
    selection_wall_seconds = time.perf_counter() - selection_started

    if best_params is None:
        raise ValueError(f"Parameter grid for {model} produced no candidates.")

    tuning_path = (
        Path(results_dir)
        / "tuning"
        / f"{model.lower()}__{band.lower().replace('.', '_').replace(' ', '')}__grid.csv"
    )
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).to_csv(tuning_path, index=False)
    print(f"[grid log] wrote {tuning_path}")

    # Hyperparameters are fixed solely by inner-validation position accuracy.
    # Construct a fresh estimator and refit it on every outer-training window.
    final_estimator = build_estimator(
        model,
        best_params,
        random_state,
        GRID_SEARCH_N_JOBS,
    )
    refit_started = time.perf_counter()
    final_estimator.fit(
        outer_train[columns],
        outer_train["location"].astype(str),
    )
    refit_time = time.perf_counter() - refit_started
    print(
        f"[grid complete] {band} / {model}: refit_count={refit_count}, "
        f"total_fit_count={total_fit_count}, "
        f"selection_wall_seconds={selection_wall_seconds:.1f}, "
        f"refit_time={refit_time:.3f}, best_params={best_params}, "
        f"validation_position_accuracy={best_validation_accuracy:.6f}"
    )

    # The outer test set is first predicted here, after selection and full refit.
    prediction_started = time.perf_counter()
    pred_positions = final_estimator.predict(outer_test[columns])
    predict_seconds = time.perf_counter() - prediction_started
    predictions = build_global_predictions_dataframe(
        outer_test,
        pred_positions,
        dataset_name=band,
        model_name=model,
        split_mode="grid_search",
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    metrics = compute_localization_metrics(predictions)
    metrics.update(
        majority_class_baselines(
            outer_train,
            outer_test,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
    )
    metrics.update(
        {
            "test_position_accuracy": metrics["position_accuracy"],
            "validation_position_accuracy": best_validation_accuracy,
            "candidate_count": int(candidate_count),
            "selection_fit_count": int(selection_fit_count),
            "refit_count": refit_count,
            "total_fit_count": int(total_fit_count),
            "mean_fit_time": float(np.mean([row["fit_seconds"] for row in grid_rows])),
            "refit_time": refit_time,
            "fit_seconds": float(selection_wall_seconds + refit_time),
            "predict_seconds": float(predict_seconds),
            "wall_seconds": float(selection_wall_seconds + refit_time + predict_seconds),
            "used_estimator": _estimator_name(final_estimator),
        }
    )
    parameter_names = list(grid)
    run_id, search_config = _grid_search_identity(
        model=model,
        band=band,
        preproc_opts=preproc_opts,
        feat_opts=feat_opts,
        test_size=test_size,
        random_state=random_state,
        n_blocks=n_blocks,
    )
    write_run_manifest(run_id, search_config, results_root=Path(results_dir))
    trials_used = sorted(
        {
            str(trial).removeprefix("trial_").zfill(2)
            for frame in (outer_train, outer_test)
            for trial in frame["trial"].dropna().unique()
        }
    )
    run_row = build_run_row(
        run_id=run_id,
        preproc_opts=preproc_opts,
        feat_opts=feat_opts,
        hyperparameters=best_params,
        metrics=metrics,
        trials_used=trials_used,
        n_train=len(outer_train),
        n_test=len(outer_test),
        n_classes=pd.concat([outer_train, outer_test], ignore_index=True)[
            "location"
        ].nunique(),
        device="cpu",
    )
    upsert_run(
        run_row,
        hyperparameter_columns=parameter_names,
        results_root=Path(results_dir),
    )

    return run_row


def _grid_search_identity(  # noqa: PLR0913
    *,
    model: str,
    band: str,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> tuple[str, dict[str, Any]]:
    """Return the winner-independent config and deterministic grid-search run ID."""
    model_name = model.upper()
    inner_validation_size = test_size
    inner_random_state = random_state + 1
    search_config = {
        "model": model_name,
        "band": band,
        "preproc_opts": preproc_opts,
        "feat_opts": feat_opts,
        "parameter_grid": PARAM_GRIDS[model_name],
        "tuning_protocol": "single_inner_block_split",
        "inner_validation_size": inner_validation_size,
        "inner_random_state": inner_random_state,
        "n_blocks": n_blocks,
        "outer_split": {
            "type": "block",
            "test_size": test_size,
            "random_state": random_state,
            "n_blocks": n_blocks,
            "trials": list(TRIALS_FOR_TRAINING_PROTOCOLS),
        },
        "inner_split": {
            "type": "block",
            "validation_size": inner_validation_size,
            "random_state": inner_random_state,
            "n_blocks": GRID_INNER_N_BLOCKS,
        },
        "selection_metric": "position_accuracy",
        "final_refit": "all_outer_train",
        "random_state": random_state,
    }
    run_id = make_run_id(
        family="ml",
        model=model_name,
        band=band,
        split="grid_search",
        normalization=str(preproc_opts.get("normalization", "none")),
        baseline_scope=preproc_opts.get("baseline_scope"),
        seed=random_state,
        config=search_config,
    )
    return run_id, search_config


def _assert_block_split_integrity(
    source_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    band: str,
    split_label: str = "outer",
) -> None:
    """Assert exact-window separation and excluded block-boundary neighbors."""
    identity_columns = ["group_id", "window_idx"]
    _validate_required_columns(source_df, set(identity_columns))

    train_ids = set(train_df[identity_columns].itertuples(index=False, name=None))
    test_ids = set(test_df[identity_columns].itertuples(index=False, name=None))
    overlap = train_ids & test_ids
    exact_pass = not overlap
    print(
        f"[{split_label} block assertion] {band} (a) exact window separation: "
        f"{'PASS' if exact_pass else 'FAIL'}"
    )

    boundary_violations: list[str] = []
    for group_id, group in source_df.groupby("group_id", sort=False):
        ordered = group.sort_values("window_idx", kind="stable")
        states = []
        for identity in ordered[identity_columns].itertuples(index=False, name=None):
            if identity in train_ids:
                states.append("train")
            elif identity in test_ids:
                states.append("test")
            else:
                states.append("excluded")
        retained = [(index, state) for index, state in enumerate(states) if state != "excluded"]
        for (left_index, left_state), (right_index, right_state) in zip(
            retained,
            retained[1:],
        ):
            if left_state == right_state:
                continue
            between = states[left_index + 1 : right_index]
            if len(between) < 2 or any(state != "excluded" for state in between):
                boundary_violations.append(str(group_id))
                break

    boundary_pass = not boundary_violations
    print(
        f"[{split_label} block assertion] {band} (b) boundary-adjacent windows excluded: "
        f"{'PASS' if boundary_pass else 'FAIL'}"
    )
    if not exact_pass or not boundary_pass:
        raise ValueError(
            f"{split_label.capitalize()} block integrity FAIL for {band}: "
            f"exact_overlap_count={len(overlap)}, "
            f"boundary_violation_groups={sorted(boundary_violations)[:10]}."
        )


def _params_from_grid_search_row(
    row: pd.Series,
    *,
    model: str,
    band: str,
) -> dict[str, Any]:
    """Extract only parameters belonging to the matched model's search space."""
    model_name = model.upper()
    parameter_space = PARAM_GRIDS.get(model_name)
    if parameter_space is None:
        parameter_space = default_params_for(model_name, band)
    parameter_names = list(parameter_space)
    missing_parameters = [name for name in parameter_names if name not in row.index]
    if missing_parameters:
        raise ValueError(
            f"Matched grid-search row for {model}/{band} lacks parameters: "
            f"{missing_parameters}."
        )
    loaded_params = {
        parameter: _coerce_param_value(row[parameter])
        for parameter in parameter_names
    }
    return _normalize_loaded_tuned_params(loaded_params, model=model_name)


def _normalize_loaded_tuned_params(
    params: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Restore model-specific Python types after pandas CSV deserialization."""
    normalized = dict(params)
    integer_parameters = {
        "RF": ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"),
        "KNN": ("n_neighbors",),
    }
    for parameter in integer_parameters.get(model, ()):
        value = normalized.get(parameter)
        normalized[parameter] = None if value is None else int(value)

    if model == "RF":
        class_weight = normalized.get("class_weight")
        normalized["class_weight"] = (
            None if class_weight is None or pd.isna(class_weight) else class_weight
        )
    return normalized


def _coerce_param_value(value: Any) -> Any:
    """Restore booleans and numeric values read from a CSV parameter cell."""
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


# Global position-classification experiment implementation.
METADATA_COLUMNS = {
    "frequency_scenario",
    "scenario",
    "location",
    "user",
    "trial",
    "group_id",
    "window_idx",
    "label",
}

MIN_GROUP_SPLIT_COUNT = 2
DEFAULT_BLOCK_COUNT = 10
DEFAULT_ROW_SPACING = 1.0
DEFAULT_COLUMN_SPACING = 1.0
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return CSI feature columns by excluding metadata columns."""
    return [col for col in df.columns if col not in METADATA_COLUMNS]

def split_lovo_folds(
    df: pd.DataFrame,
    *,
    trials_for_training_protocols: tuple[str, ...] = TRIALS_FOR_TRAINING_PROTOCOLS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return leave-one-volunteer-out folds using the user column."""
    _validate_required_columns(df, {"user"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
    if "trial" in df.columns:
        df = filter_training_protocol_trials(
            df,
            trials=trials_for_training_protocols,
            split_mode="lovo",
        )
    users = sorted(df["user"].dropna().unique())
    if len(users) < 2:
        msg = "LOVO split requires at least two users."
        raise ValueError(msg)

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for user in users:
        test_mask = df["user"] == user
        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()
        if not train_df.empty and not test_df.empty:
            _print_protocol_split("lovo", train_df, test_df, fold=user)
            folds.append((train_df, test_df))
    return folds


def split_cross_session(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on trial 01 and evaluate the same known users in trial 02."""
    _validate_required_columns(df, {"trial", "user", "location"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)

    trials = _normalized_trial_values(df["trial"])
    train_df = df.loc[trials == "01"].copy()
    test_df = df.loc[trials == "02"].copy()
    if train_df.empty:
        raise ValueError("cross_session split has no trial-01 training windows.")
    if test_df.empty:
        raise ValueError("cross_session split has no trial-02 test windows.")

    train_trials = _normalized_trial_values(train_df["trial"])
    test_trials = _normalized_trial_values(test_df["trial"])
    assert (test_trials == "02").all(), "cross_session test contains a non-trial-02 window"
    assert not (train_trials == "02").any(), "trial-02 window leaked into training"

    train_users = sorted(train_df["user"].astype(str).unique())
    test_users = sorted(test_df["user"].astype(str).unique())
    unknown_test_users = sorted(set(test_users) - set(train_users))
    if unknown_test_users:
        print(
            "[cross_session warning] trial-02 contains user(s) absent from trial 01: "
            f"{', '.join(unknown_test_users)}. Evaluation remains valid as cross-user plus "
            "cross-session transfer."
        )
    train_positions = set(train_df["location"].astype(str))
    test_positions = set(test_df["location"].astype(str))
    missing_positions = sorted(test_positions - train_positions)
    print(f"[cross_session] discovered trial-02 users: {', '.join(test_users)}")
    _print_protocol_split("cross_session", train_df, test_df)
    print(
        f"[cross_session] n_train={len(train_df)} users={train_users} "
        f"positions={len(train_positions)}"
    )
    print(
        f"[cross_session] n_test={len(test_df)} users={test_users} "
        f"positions={len(test_positions)}"
    )
    print("[cross_session] assertion passed: no trial-02 window is in training")
    if missing_positions:
        print(
            f"[cross_session warning] {len(missing_positions)} test positions are absent "
            f"from training and therefore unlearnable: {', '.join(missing_positions)}"
        )
    else:
        print("[cross_session] all test positions are represented in training")
    return train_df, test_df


def split_dataframe(
    df: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    split_mode: Literal["group", "random", "block", "lovo", "cross_session"] = "group",
    stratify_column: str | None = None,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    trials_for_training_protocols: tuple[str, ...] = TRIALS_FOR_TRAINING_PROTOCOLS,
) -> tuple[pd.DataFrame, pd.DataFrame] | list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split a feature dataframe for global position classification."""
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)

    # Cross-session and LOVO define complete protocols and therefore dispatch
    # before the single-split strategies below.
    if split_mode == "cross_session":
        return split_cross_session(df)

    df = filter_training_protocol_trials(
        df,
        trials=trials_for_training_protocols,
        split_mode=split_mode,
    )

    if split_mode == "lovo":
        return split_lovo_folds(
            df,
            trials_for_training_protocols=trials_for_training_protocols,
        )

    _validate_required_columns(df, {"group_id"})

    # The remaining branches differ only in how row indices are selected; each
    # returns the same train/test DataFrame contract.
    if split_mode == "group":
        _validate_required_columns(df, {"label"})
        if df["group_id"].nunique() < MIN_GROUP_SPLIT_COUNT:
            msg = "Group split requires at least two unique group_id values."
            raise ValueError(msg)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["group_id"]))
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        _print_protocol_split(split_mode, train_df, test_df)
        return train_df, test_df

    if split_mode == "random":
        stratify = None
        if stratify_column is not None and stratify_column in df.columns:
            col = df[stratify_column]
            if col.nunique() >= 2 and col.value_counts().min() >= 2:
                stratify = col
        row_indices = list(range(len(df)))
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        _print_protocol_split(split_mode, train_df, test_df)
        return train_df, test_df

    if split_mode == "block":
        train_df, test_df = _split_dataframe_by_blocks(
            df,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
        )
        _print_protocol_split(split_mode, train_df, test_df)
        return train_df, test_df

    msg = (
        f"Unknown split_mode {split_mode!r}. Must be 'group', 'random', 'block', "
        "'lovo', or 'cross_session'."
    )
    raise ValueError(msg)


def run_global_position_experiment(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    params: dict,
    split_mode: Literal["group", "random", "block", "lovo", "cross_session"] = "block",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_jobs: int = -1,
    results_dir: Path | str | None = None,
    force_retrain: bool = False,
    save_prediction_cache: bool = True,
    svm_fallback_seconds: float = 30.0 * 60.0,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    run_id: str | None = None,
) -> tuple[Pipeline | None, pd.DataFrame, dict[str, float]]:
    """Train/evaluate or load a global 52-position classical baseline."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label"})
    resolved_model_name = model_name.upper()
    resolved_params = dict(params)

    # LOVO requires its own fold loop and cache aggregation path.
    if split_mode == "lovo":
        predictions, _, metrics = run_global_lovo_experiment(
            df,
            dataset_name=dataset_name,
            model_name=resolved_model_name,
            params=resolved_params,
            random_state=random_state,
            n_jobs=n_jobs,
            results_dir=results_dir,
            force_retrain=force_retrain,
            save_prediction_cache=save_prediction_cache,
            svm_fallback_seconds=svm_fallback_seconds,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            run_id=run_id,
        )
        return None, predictions, metrics

    # Resolve the protocol split before checking caches because the fingerprints
    # are part of cache validity.
    split_result = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    if isinstance(split_result, list):
        msg = "Unexpected multi-fold split outside the LOVO dispatch path."
        raise RuntimeError(msg)
    train_df, test_df = split_result
    train_fingerprint = _window_identity_fingerprint(train_df)
    test_fingerprint = _window_identity_fingerprint(test_df)
    majority_metrics = majority_class_baselines(
        train_df,
        test_df,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )

    # Reuse predictions only when estimator settings and both split fingerprints
    # match the current experiment exactly.
    if results_dir is not None and not force_retrain:
        expected_metadata = prediction_cache_metadata(
            model=resolved_model_name,
            band=dataset_name,
            split_mode=split_mode,
            params=resolved_params,
            random_state=random_state,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            train_fingerprint=train_fingerprint,
            test_fingerprint=test_fingerprint,
        )
        cached_predictions = load_predictions(
            Path(results_dir),
            resolved_model_name,
            dataset_name,
            split_mode,
            expected_metadata=expected_metadata,
            run_id=run_id,
        )
        if cached_predictions is not None:
            metrics = compute_localization_metrics(cached_predictions)
            metrics.update(majority_metrics)
            metrics.update(
                {
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "used_estimator": "cached_predictions",
                }
            )
            return None, cached_predictions, metrics

    columns = feature_columns(train_df)

    # Construct and fit the requested classical estimator on feature columns only.
    model = build_estimator(resolved_model_name, resolved_params, random_state, n_jobs)
    if resolved_model_name == "SVM":
        print("[SVM] sklearn SVC is single-threaded; n_jobs is not used by SVC.")

    started_at = time.perf_counter()
    fit_started_at = time.perf_counter()
    model.fit(train_df[columns], train_df["location"].astype(str))
    fit_seconds = time.perf_counter() - fit_started_at
    used_estimator = _estimator_name(model)

    if (
        resolved_model_name == "SVM"
        and dataset_name == "Fusion"
        and resolved_params.get("kernel", "rbf") == "rbf"
        and fit_seconds > svm_fallback_seconds
    ):
        fallback_params = {**resolved_params, "kernel": "linear_svc"}
        print(
            "[SVM] Fusion RBF fit exceeded "
            f"{svm_fallback_seconds:.0f}s; refitting with LinearSVC fallback."
        )
        model = build_estimator(resolved_model_name, fallback_params, random_state, n_jobs)
        fit_started_at = time.perf_counter()
        model.fit(train_df[columns], train_df["location"].astype(str))
        fit_seconds = time.perf_counter() - fit_started_at
        used_estimator = "LinearSVC_fallback"

    predict_started_at = time.perf_counter()

    # inference
    pred_positions = model.predict(test_df[columns])
    predict_seconds = time.perf_counter() - predict_started_at
    predictions = build_global_predictions_dataframe(
        test_df,
        pred_positions,
        dataset_name=dataset_name,
        model_name=resolved_model_name,
        split_mode=split_mode,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    metrics = compute_localization_metrics(predictions)
    metrics.update(majority_metrics)
    metrics.update(
        {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "wall_seconds": float(time.perf_counter() - started_at),
            "used_estimator": used_estimator,
        }
    )

    if results_dir is not None and save_prediction_cache:
        save_predictions(
            predictions,
            Path(results_dir),
            resolved_model_name,
            dataset_name,
            split_mode,
            run_id=run_id,
            metadata=prediction_cache_metadata(
                model=resolved_model_name,
                band=dataset_name,
                split_mode=split_mode,
                params=resolved_params,
                random_state=random_state,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                train_fingerprint=train_fingerprint,
                test_fingerprint=test_fingerprint,
            ),
        )

    # returns the fitted model, predictions, and metrics for the current split
    return model, predictions, metrics


def _window_identity_fingerprint(df: pd.DataFrame) -> str:
    """Hash the exact ordered set of windows used by a prediction cache."""
    _validate_required_columns(df, {"group_id", "window_idx", "user", "trial"})
    identities = (
        df[["group_id", "window_idx", "user", "trial"]]
        .astype(str)
        .sort_values(["group_id", "window_idx", "user", "trial"], kind="stable")
    )
    payload = identities.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_training_protocol_trials(
    df: pd.DataFrame,
    *,
    trials: tuple[str, ...] = TRIALS_FOR_TRAINING_PROTOCOLS,
    split_mode: str,
) -> pd.DataFrame:
    """Restrict non-cross-session protocols to their configured recording trials."""
    _validate_required_columns(df, {"trial"})
    normalized_trials = tuple(_normalized_trial_values(pd.Series(trials)).tolist())
    if not normalized_trials:
        raise ValueError("TRIALS_FOR_TRAINING_PROTOCOLS cannot be empty.")
    observed = _normalized_trial_values(df["trial"])
    filtered = df.loc[observed.isin(normalized_trials)].copy()
    if filtered.empty:
        raise ValueError(
            f"{split_mode} trial filter emptied the dataframe; requested trials="
            f"{list(normalized_trials)}, observed trials={sorted(observed.unique())}."
        )
    print(
        f"[trial filter] split={split_mode} trials={list(normalized_trials)} "
        f"kept={len(filtered)}/{len(df)}"
    )
    return filtered


def run_global_lovo_experiment(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    df: pd.DataFrame,
    *,
    dataset_name: str,
    model_name: str,
    params: dict[str, Any],
    random_state: int = 42,
    n_jobs: int = -1,
    results_dir: Path | str | None = None,
    force_retrain: bool = False,
    save_prediction_cache: bool = True,
    svm_fallback_seconds: float = 30.0 * 60.0,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run leave-one-volunteer-out global classification for one model/band pair."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label", "user"})

    resolved_model_name = model_name.upper()
    resolved_params = dict(params)
    # Build the complete user folds once so cache fingerprints, warnings, and
    # aggregation all refer to the same protocol.
    folds = split_lovo_folds(df)
    protocol_df = pd.concat([test_df for _, test_df in folds], axis=0).sort_index()
    held_out_users = [_held_out_user(test_df) for _, test_df in folds]
    result_path = Path(results_dir) if results_dir is not None else None

    _print_lovo_honesty_warnings(protocol_df, folds, dataset_name=dataset_name)
    print(
        "[LOVO] Hyperparameter provenance: reusing block-split tuned/default "
        "hyperparameters; no nested CV is run for LOVO."
    )

    # Accept cached predictions only when the complete train/test fold collection
    # matches the current dataset.
    if result_path is not None and not force_retrain:
        cached_predictions = _load_lovo_cached_predictions(
            result_path,
            resolved_model_name,
            dataset_name,
            resolved_params,
            held_out_users,
            folds=folds,
            random_state=random_state,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            run_id=run_id,
            save_prediction_cache=save_prediction_cache,
        )
        if cached_predictions is not None:
            per_fold_metrics, aggregated_metrics = _lovo_metrics_from_predictions(
                cached_predictions,
                folds=folds,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            aggregated_metrics.update(
                {
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "used_estimator": "cached_predictions",
                }
            )
            return cached_predictions, per_fold_metrics, aggregated_metrics

    if resolved_model_name == "SVM":
        print("[SVM] sklearn SVC is single-threaded; n_jobs is not used by SVC.")

    fold_prediction_frames: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    total_started_at = time.perf_counter()

    # Fit one independent estimator per held-out user and retain fold-level
    # predictions so variation is not hidden by pooling.
    for fold_index, ((train_df, test_df), held_out_user) in enumerate(
        zip(folds, held_out_users),
        start=1,
    ):
        fold_label = _lovo_fold_label(held_out_user)
        train_fingerprint = _window_identity_fingerprint(train_df)
        test_fingerprint = _window_identity_fingerprint(test_df)
        print(
            f"[LOVO] {dataset_name} / {resolved_model_name} fold "
            f"{fold_index}/{len(folds)}: held_out_user={held_out_user}"
        )
        columns = feature_columns(train_df)
        model = build_estimator(resolved_model_name, resolved_params, random_state, n_jobs)

        fold_started_at = time.perf_counter()
        fit_started_at = time.perf_counter()
        model.fit(train_df[columns], train_df["location"].astype(str))
        fit_seconds = time.perf_counter() - fit_started_at
        used_estimator = _estimator_name(model)

        if (
            resolved_model_name == "SVM"
            and dataset_name == "Fusion"
            and resolved_params.get("kernel", "rbf") == "rbf"
            and fit_seconds > svm_fallback_seconds
        ):
            fallback_params = {**resolved_params, "kernel": "linear_svc"}
            print(
                "[SVM] Fusion RBF fit exceeded "
                f"{svm_fallback_seconds:.0f}s; refitting with LinearSVC fallback."
            )
            model = build_estimator(resolved_model_name, fallback_params, random_state, n_jobs)
            fit_started_at = time.perf_counter()
            model.fit(train_df[columns], train_df["location"].astype(str))
            fit_seconds = time.perf_counter() - fit_started_at
            used_estimator = "LinearSVC_fallback"

        predict_started_at = time.perf_counter()
        pred_positions = model.predict(test_df[columns])
        predict_seconds = time.perf_counter() - predict_started_at
        fold_predictions = build_global_predictions_dataframe(
            test_df,
            pred_positions,
            dataset_name=dataset_name,
            model_name=resolved_model_name,
            split_mode="lovo",
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        fold_predictions["held_out_user"] = held_out_user
        fold_wall_seconds = time.perf_counter() - fold_started_at
        fold_metrics = compute_localization_metrics(fold_predictions)
        fold_metrics.update(
            majority_class_baselines(
                train_df,
                test_df,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            ),
        )
        fold_metric_rows.append(
            {
                "held_out_user": held_out_user,
                **fold_metrics,
                "n_test_windows": int(len(fold_predictions)),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "wall_seconds": float(fold_wall_seconds),
                "used_estimator": used_estimator,
            }
        )
        fold_prediction_frames.append(fold_predictions)

        if result_path is not None and save_prediction_cache:
            save_predictions(
                fold_predictions,
                result_path,
                resolved_model_name,
                dataset_name,
                "lovo",
                fold=fold_label,
                run_id=run_id,
                metadata=prediction_cache_metadata(
                    model=resolved_model_name,
                    band=dataset_name,
                    split_mode="lovo",
                    params=resolved_params,
                    fold=fold_label,
                    random_state=random_state,
                    row_spacing=row_spacing,
                    column_spacing=column_spacing,
                    train_fingerprint=train_fingerprint,
                    test_fingerprint=test_fingerprint,
                ),
            )

        print(
            f"[LOVO] fold {fold_index}/{len(folds)} done: "
            f"acc={fold_metrics['position_accuracy']:.4f} "
            f"fit={fit_seconds:.1f}s predict={predict_seconds:.1f}s "
            f"wall={fold_wall_seconds:.1f}s"
        )
        del model
        gc.collect()

    predictions = pd.concat(fold_prediction_frames, ignore_index=True)
    if len(predictions) != len(protocol_df):
        msg = (
            f"LOVO predictions should cover {len(protocol_df)} filtered windows, "
            f"got {len(predictions)}."
        )
        raise RuntimeError(msg)

    # Aggregate fold statistics only after verifying every filtered window has one
    # prediction.
    per_fold_metrics = pd.DataFrame(fold_metric_rows)
    aggregated_metrics = _aggregate_lovo_metrics(
        per_fold_metrics,
        pooled_predictions=predictions,
    )
    aggregated_metrics.update(
        {
            "fit_seconds": float(per_fold_metrics["fit_seconds"].sum()),
            "predict_seconds": float(per_fold_metrics["predict_seconds"].sum()),
            "wall_seconds": float(time.perf_counter() - total_started_at),
            "used_estimator": ",".join(sorted(set(per_fold_metrics["used_estimator"]))),
        }
    )

    if result_path is not None and save_prediction_cache:
        save_predictions(
            predictions,
            result_path,
            resolved_model_name,
            dataset_name,
            "lovo",
            run_id=run_id,
            metadata=prediction_cache_metadata(
                model=resolved_model_name,
                band=dataset_name,
                split_mode="lovo",
                params=resolved_params,
                random_state=random_state,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                train_fingerprint=_fold_collection_fingerprint(folds, side="train"),
                test_fingerprint=_fold_collection_fingerprint(folds, side="test"),
            ),
        )

    print(
        f"[LOVO] {dataset_name} / {resolved_model_name} complete: "
        f"position_accuracy={aggregated_metrics['position_accuracy_mean']:.4f} +/- "
        f"{aggregated_metrics['position_accuracy_std']:.4f}, "
        f"total_wall={aggregated_metrics['wall_seconds']:.1f}s"
    )
    return predictions, per_fold_metrics, aggregated_metrics


def _load_lovo_cached_predictions(
    results_dir: Path,
    model_name: str,
    dataset_name: str,
    params: dict[str, Any],
    held_out_users: list[object],
    *,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    random_state: int,
    row_spacing: float,
    column_spacing: float,
    run_id: str | None,
    save_prediction_cache: bool,
) -> pd.DataFrame | None:
    """Load a valid concatenated LOVO cache or reconstruct it from fold caches."""
    concat_metadata = prediction_cache_metadata(
        model=model_name,
        band=dataset_name,
        split_mode="lovo",
        params=params,
        random_state=random_state,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
        train_fingerprint=_fold_collection_fingerprint(folds, side="train"),
        test_fingerprint=_fold_collection_fingerprint(folds, side="test"),
    )
    # Prefer the single concatenated cache because its fingerprints cover the
    # complete fold collection.
    cached_concat = load_predictions(
        results_dir,
        model_name,
        dataset_name,
        "lovo",
        expected_metadata=concat_metadata,
        run_id=run_id,
    )
    if cached_concat is not None:
        if "held_out_user" not in cached_concat.columns:
            print("[predictions cache stale] LOVO concat lacks held_out_user.")
            return None
        return cached_concat

    # Fall back only when every individual fold cache is present and valid.
    fold_frames = []
    for held_out_user, (train_df, test_df) in zip(held_out_users, folds):
        fold_label = _lovo_fold_label(held_out_user)
        fold_predictions = load_predictions(
            results_dir,
            model_name,
            dataset_name,
            "lovo",
            fold=fold_label,
            run_id=run_id,
            expected_metadata=prediction_cache_metadata(
                model=model_name,
                band=dataset_name,
                split_mode="lovo",
                params=params,
                fold=fold_label,
                random_state=random_state,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                train_fingerprint=_window_identity_fingerprint(train_df),
                test_fingerprint=_window_identity_fingerprint(test_df),
            ),
        )
        if fold_predictions is None:
            return None
        if "held_out_user" not in fold_predictions.columns:
            print(f"[predictions cache stale] LOVO fold {fold_label} lacks held_out_user.")
            return None
        fold_frames.append(fold_predictions)

    predictions = pd.concat(fold_frames, ignore_index=True)
    if save_prediction_cache:
        save_predictions(
            predictions,
            results_dir,
            model_name,
            dataset_name,
            "lovo",
            run_id=run_id,
            metadata=concat_metadata,
        )
    return predictions


def _lovo_metrics_from_predictions(
    predictions: pd.DataFrame,
    *,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reconstruct per-fold and aggregate LOVO metrics from prediction rows."""
    _validate_required_columns(predictions, {"held_out_user"})
    majority_by_user: dict[object, dict[str, float]] = {}
    if folds is not None:
        majority_by_user = {
            _held_out_user(test_df): majority_class_baselines(
                train_df,
                test_df,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            for train_df, test_df in folds
        }
    rows = []
    for held_out_user, group in predictions.groupby("held_out_user", sort=True):
        metrics = compute_localization_metrics(group)
        metrics.update(majority_by_user.get(held_out_user, {}))
        rows.append(
            {
                "held_out_user": held_out_user,
                **metrics,
                "n_test_windows": int(len(group)),
            }
        )
    per_fold_metrics = pd.DataFrame(rows)
    return per_fold_metrics, _aggregate_lovo_metrics(
        per_fold_metrics,
        pooled_predictions=predictions,
    )


def _aggregate_lovo_metrics(
    per_fold_metrics: pd.DataFrame,
    *,
    pooled_predictions: pd.DataFrame,
) -> dict[str, float]:
    """Summarize fold metrics and retain a separate pooled accuracy diagnostic."""
    metric_names = [
        "position_accuracy",
        "macro_f1",
        "room_accuracy",
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
        "p90_distance_error",
        "samples",
        "majority_position_accuracy",
        "majority_room_accuracy",
    ]
    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name not in per_fold_metrics.columns:
            continue
        values = pd.to_numeric(per_fold_metrics[metric_name], errors="coerce")
        aggregated[f"{metric_name}_mean"] = float(values.mean())
        aggregated[f"{metric_name}_std"] = float(values.std(ddof=1))
        aggregated[f"{metric_name}_min"] = float(values.min())
        aggregated[f"{metric_name}_max"] = float(values.max())
        if metric_name in {"majority_position_accuracy", "majority_room_accuracy"}:
            aggregated[metric_name] = aggregated[f"{metric_name}_mean"]

    sample_values = pd.to_numeric(per_fold_metrics["samples"], errors="coerce")
    aggregated["samples"] = float(sample_values.sum())
    aggregated["n_folds"] = float(len(per_fold_metrics))
    aggregated["n_test_windows"] = float(per_fold_metrics["n_test_windows"].sum())
    pooled_metrics = compute_localization_metrics(pooled_predictions)
    aggregated["position_accuracy_pooled"] = float(pooled_metrics["position_accuracy"])
    return aggregated


def _print_lovo_honesty_warnings(
    df: pd.DataFrame,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    dataset_name: str,
) -> None:
    """Report positions whose user coverage limits honest LOVO evaluation."""
    location_user_counts = df.groupby("location", sort=True)["user"].nunique()
    singleton_positions = sorted(location_user_counts[location_user_counts == 1].index.astype(str))
    if singleton_positions:
        affected_windows = int(df["location"].astype(str).isin(singleton_positions).sum())
        print(
            f"[LOVO warning] {dataset_name}: {len(singleton_positions)} positions are "
            f"covered by exactly one user ({affected_windows} windows): "
            f"{', '.join(singleton_positions)}"
        )
        print(
            "[LOVO warning] Under block split these positions can inflate results via "
            "user identity; under LOVO they have zero training examples in the held-out "
            "user's fold."
        )
    else:
        print(f"[LOVO check] {dataset_name}: every observed position has at least two users.")

    all_positions = set(df["location"].astype(str))
    position_count = len(all_positions)
    for _, test_df in folds:
        held_out_user = _held_out_user(test_df)
        train_df = df.loc[df["user"] != held_out_user]
        train_positions = set(train_df["location"].astype(str))
        missing_positions = sorted(all_positions - train_positions)
        singleton_test_windows = int(
            test_df["location"].astype(str).isin(singleton_positions).sum()
        )
        print(
            f"[LOVO check] {dataset_name}: held_out_user={held_out_user} training covers "
            f"{len(train_positions)}/{position_count} observed positions; "
            f"singleton-position test windows={singleton_test_windows}."
        )
        if missing_positions:
            print(
                f"[LOVO warning] held_out_user={held_out_user} missing training positions: "
                f"{', '.join(missing_positions)}"
            )


def _held_out_user(test_df: pd.DataFrame) -> object:
    """Return the single user represented by a LOVO test fold."""
    users = test_df["user"].dropna().unique()
    if len(users) != 1:
        msg = f"Expected exactly one held-out user, got {users!r}."
        raise ValueError(msg)
    return users[0]


def _lovo_fold_label(held_out_user: object) -> str:
    """Build the stable cache label for a held-out user."""
    return f"user-{held_out_user}"


def _fold_collection_fingerprint(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    side: Literal["train", "test"],
) -> str:
    """Fingerprint all train or test window identities in a fold collection."""
    frames = [train_df if side == "train" else test_df for train_df, test_df in folds]
    return _window_identity_fingerprint(pd.concat(frames, ignore_index=True))


def _split_dataframe_by_blocks(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each recording into deterministic train/test temporal blocks."""
    if n_blocks < 1:
        msg = "n_blocks must be at least 1 for block split mode."
        raise ValueError(msg)
    _validate_required_columns(df, {"window_idx"})
    group_ids = df["group_id"].to_numpy()
    window_idxs = df["window_idx"].to_numpy()
    n_test_blocks = min(n_blocks, max(1, round(test_size * n_blocks)))

    train_locs: list[int] = []
    test_locs: list[int] = []

    # Assign blocks independently within each recording and remove windows at
    # train/test boundaries to reduce overlap leakage.
    for group_id in df["group_id"].unique():
        session_pos = np.flatnonzero(group_ids == group_id)
        sort_order = np.argsort(window_idxs[session_pos], kind="stable")
        session_pos_sorted = session_pos[sort_order]
        n_windows = len(session_pos_sorted)

        if n_windows < n_blocks:
            train_locs.extend(session_pos_sorted.tolist())
            continue

        block_assignment = (np.arange(n_windows) * n_blocks) // n_windows
        gid_int = int.from_bytes(hashlib.md5(str(group_id).encode()).digest()[:4], "big")
        rng = np.random.RandomState((gid_int + random_state) % (2**31))
        test_block_set = set(rng.choice(n_blocks, size=n_test_blocks, replace=False).tolist())
        is_test = np.array([block in test_block_set for block in block_assignment])

        is_boundary = np.zeros(n_windows, dtype=bool)
        if n_windows > 1:
            is_boundary[:-1] |= is_test[:-1] != is_test[1:]
            is_boundary[1:] |= is_test[1:] != is_test[:-1]

        for pos, test_block, boundary in zip(session_pos_sorted, is_test, is_boundary):
            if boundary:
                continue
            (test_locs if test_block else train_locs).append(int(pos))

    if not train_locs or not test_locs:
        msg = "Block split produced an empty train or test set."
        raise ValueError(msg)
    return df.iloc[train_locs].copy(), df.iloc[test_locs].copy()


def _estimator_name(model: Pipeline) -> str:
    """Return the fitted classifier class name from a scikit-learn pipeline."""
    classifier = model.named_steps.get("classifier")
    return type(classifier).__name__ if classifier is not None else type(model).__name__


def _validate_grid_spacing(*, row_spacing: float, column_spacing: float) -> None:
    """Require finite, positive physical spacing along both grid axes."""
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        msg = "row_spacing must be a finite positive value."
        raise ValueError(msg)
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        msg = "column_spacing must be a finite positive value."
        raise ValueError(msg)


def _validate_training_dataframe(df: pd.DataFrame) -> None:
    """Validate metadata, rows, and feature columns before model fitting."""
    _validate_required_columns(df, METADATA_COLUMNS)
    if df.empty:
        msg = "Cannot train on an empty dataframe."
        raise ValueError(msg)
    if not feature_columns(df):
        msg = "No CSI feature columns found."
        raise ValueError(msg)


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error when an ML DataFrame lacks required columns."""
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        msg = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)


def _normalized_trial_values(values: pd.Series) -> pd.Series:
    """Normalize trial identifiers to two-digit strings."""
    return (
        values.astype(str)
        .str.removeprefix("trial_")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )


def _print_protocol_split(
    split_mode: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    fold: object | None = None,
) -> None:
    """Print the users, trials, and sizes defining an experiment split."""
    combined = pd.concat([train_df, test_df], ignore_index=True)
    trials = (
        sorted(_normalized_trial_values(combined["trial"]).unique())
        if "trial" in combined.columns
        else ["unavailable"]
    )
    train_users = sorted(train_df["user"].dropna().astype(str).unique())
    test_users = sorted(test_df["user"].dropna().astype(str).unique())
    fold_text = f" fold={fold}" if fold is not None else ""
    print(
        f"[protocol] split={split_mode}{fold_text} trials_used={trials} "
        f"n_train={len(train_df)} n_test={len(test_df)} "
        f"users={sorted(set(train_users) | set(test_users))} "
        f"train_users={train_users} test_users={test_users}"
    )
