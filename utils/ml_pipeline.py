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
    get_all_dataframes,
    get_cache_path,
    load_predictions,
    predictions_path,
    write_manifest,
)
from utils.csi_preprocessing import process_magnitude_data
from utils.feature_pipeline import build_frequency_feature_dataframes
from utils.global_position_classifier import (
    feature_columns,
    location_distance_error,
    room_label_for_location,
    run_global_lovo_experiment,
    run_global_position_experiment,
    split_lovo_folds,
    split_dataframe,
)
from utils.import_data import get_csv_files, sort_meta_info
from utils.metrics import compute_localization_metrics
from utils.models import PARAM_GRIDS, build_estimator, default_params_for
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

    feature_dataframes = get_all_dataframes(preproc_opts, feat_opts, _build_feature_dataframes)
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
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    """Run every requested global ML baseline and persist summary/manifest files."""
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

    summary = pd.DataFrame(summary_rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "global_summary.csv", index=False)
    if lovo_per_fold_rows:
        pd.DataFrame(lovo_per_fold_rows).to_csv(summary_dir / "lovo_per_fold.csv", index=False)
    if lovo_summary_rows:
        pd.DataFrame(lovo_summary_rows).to_csv(summary_dir / "lovo_summary.csv", index=False)
    write_manifest(
        results_dir,
        preproc_opts,
        feat_opts,
        classifier_params={
            f"{model}:{band}": params_lookup[(model, band)]
            for model in models_to_run
            for band in bands_to_run
        },
        splits=list(split_modes),
        test_size=test_size,
        feature_dataframes={band: feature_dataframes[band] for band in bands_to_run},
        lovo_metadata=_build_lovo_manifest_metadata(
            lovo_fold_user_ids,
            params_lookup=params_lookup,
            models_to_run=models_to_run,
            bands_to_run=bands_to_run,
            enabled="lovo" in split_modes,
        ),
    )
    return summary, predictions_by_key


def load_all_predictions(
    results_dir: Path,
    *,
    models_to_run: tuple[str, ...],
    bands_to_run: tuple[str, ...],
    split_modes: tuple[str, ...],
) -> pd.DataFrame:
    """Load all persisted prediction parquet files for analysis-only notebook runs."""
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
    """Write CSV and LaTeX analysis tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    master_table.to_csv(tables_dir / "global_ml_baseline.csv", index=False)
    (tables_dir / "global_ml_baseline.tex").write_text(
        master_table.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    per_room_table.to_csv(tables_dir / "global_ml_baseline_per_room.csv", index=False)
    (tables_dir / "global_ml_baseline_per_room.tex").write_text(
        per_room_table.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )


def load_lovo_summary_tables(summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load persisted LOVO summary tables without triggering training."""
    per_fold_path = summary_dir / "lovo_per_fold.csv"
    summary_path = summary_dir / "lovo_summary.csv"
    missing = [str(path) for path in (per_fold_path, summary_path) if not path.exists()]
    if missing:
        msg = "Missing LOVO summary files. Run the Global baseline cell first.\n"
        raise FileNotFoundError(msg + "\n".join(missing))
    return pd.read_csv(per_fold_path), pd.read_csv(summary_path)


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
        output_row = {"model": row["model"], "dataset": row["dataset"]}
        for metric, output_col in metrics:
            output_row[output_col] = _format_mean_std(
                row.get(f"{metric}_mean", np.nan),
                row.get(f"{metric}_std", np.nan),
            )
        rows.append(output_row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def save_lovo_analysis_table(table: pd.DataFrame, *, tables_dir: Path) -> None:
    """Write the persisted LOVO analysis table to CSV and LaTeX."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "lovo_aggregated_table.csv", index=False)
    (tables_dir / "lovo_aggregated_table.tex").write_text(
        table.to_latex(index=False, escape=False),
        encoding="utf-8",
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
        fig.savefig(save_path, bbox_inches="tight")
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
