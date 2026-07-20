from __future__ import annotations

import gc
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from utils.cache import load_predictions, prediction_cache_metadata, save_predictions
from utils.metrics import compute_localization_metrics
from utils.models import build_estimator

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

EMPTY_ROOM_LABEL = 0
EMPTY_ROOM_LOCATION = "Z-0"
MIN_GROUP_SPLIT_COUNT = 2
DEFAULT_BLOCK_COUNT = 10
DEFAULT_ROW_SPACING = 1.0
DEFAULT_COLUMN_SPACING = 1.0
GRID_ROW_ORIGIN = ord("A")
GRID_COLUMN_ORIGIN = 1
LOCATION_PATTERN = re.compile(r"^(?P<row>[A-Z])[-_ ]?(?P<column>\d+)$")
ROOM_1_COLUMNS = range(1, 10)
ROOM_2_A_COLUMNS = {13, 14}
ROOM_2_BC_COLUMNS = range(10, 15)
ROOM_3_EF_COLUMNS = range(10, 14)

GLOBAL_PREDICTION_COLUMNS = [
    "window_id",
    "user",
    "trial",
    "true_position",
    "pred_position",
    "true_room",
    "pred_room",
    "true_x",
    "true_y",
    "pred_x",
    "pred_y",
    "distance_error",
    "dataset",
    "model",
    "split_mode",
    "split",
    "true_location",
    "pred_location",
    "scenario",
    "group_id",
    "window_idx",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return CSI feature columns by excluding metadata columns."""
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def room_label_for_location(location: object) -> int | None:
    """Map a 52-class reference point label to its room label."""
    location_label = _normalize_location_label(location)
    if location_label == EMPTY_ROOM_LOCATION:
        return EMPTY_ROOM_LABEL

    match = LOCATION_PATTERN.fullmatch(location_label)
    if match is None:
        return None

    row = match.group("row")
    column = int(match.group("column"))
    if row in "ABCDEF" and column in ROOM_1_COLUMNS:
        return 1
    if (row == "A" and column in ROOM_2_A_COLUMNS) or (
        row in "BC" and column in ROOM_2_BC_COLUMNS
    ):
        return 2
    if row in "EF" and column in ROOM_3_EF_COLUMNS:
        return 3
    return None


def location_grid_coordinates(
    location: object,
    *,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[float, float] | None:
    """Convert labels such as A-1 into physical row/column coordinates."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    location_label = _normalize_location_label(location)
    if location_label == EMPTY_ROOM_LOCATION:
        return None

    match = LOCATION_PATTERN.fullmatch(location_label)
    if match is None:
        return None

    row_index = ord(match.group("row")) - GRID_ROW_ORIGIN
    column_index = int(match.group("column")) - GRID_COLUMN_ORIGIN
    return row_index * row_spacing, column_index * column_spacing


def location_distance_error(
    true_location: object,
    pred_location: object,
    *,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> float:
    """Euclidean distance between two non-empty reference point labels."""
    true_coordinates = location_grid_coordinates(
        true_location,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    pred_coordinates = location_grid_coordinates(
        pred_location,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    if true_coordinates is None or pred_coordinates is None:
        return np.nan

    row_error = true_coordinates[0] - pred_coordinates[0]
    column_error = true_coordinates[1] - pred_coordinates[1]
    return float(np.hypot(row_error, column_error))


def split_lovo_folds(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return leave-one-volunteer-out folds using the user column."""
    _validate_required_columns(df, {"user"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
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
            folds.append((train_df, test_df))
    return folds


def split_cross_session(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on trial 01 and evaluate the same known users in trial 02."""
    _validate_required_columns(df, {"trial", "user", "location"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)

    trials = df["trial"].astype(str).str.removeprefix("trial_").str.zfill(2)
    train_df = df.loc[trials == "01"].copy()
    test_df = df.loc[trials == "02"].copy()
    if train_df.empty:
        raise ValueError("cross_session split has no trial-01 training windows.")
    if test_df.empty:
        raise ValueError("cross_session split has no trial-02 test windows.")

    train_trials = train_df["trial"].astype(str).str.removeprefix("trial_").str.zfill(2)
    test_trials = test_df["trial"].astype(str).str.removeprefix("trial_").str.zfill(2)
    assert (test_trials == "02").all(), "cross_session test contains a non-trial-02 window"
    assert not (train_trials == "02").any(), "trial-02 window leaked into training"

    train_users = sorted(train_df["user"].astype(str).unique())
    test_users = sorted(test_df["user"].astype(str).unique())
    unknown_test_users = sorted(set(test_users) - set(train_users))
    if unknown_test_users:
        msg = (
            "cross_session requires every trial-02 user to be present in trial 01; "
            f"unknown test users: {', '.join(unknown_test_users)}"
        )
        raise ValueError(msg)
    train_positions = set(train_df["location"].astype(str))
    test_positions = set(test_df["location"].astype(str))
    missing_positions = sorted(test_positions - train_positions)
    print(f"[cross_session] discovered trial-02 users: {', '.join(test_users)}")
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
) -> tuple[pd.DataFrame, pd.DataFrame] | list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split a feature dataframe for global position classification."""
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)

    if split_mode == "lovo":
        return split_lovo_folds(df)

    if split_mode == "cross_session":
        return split_cross_session(df)

    _validate_required_columns(df, {"group_id"})

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
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

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
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    if split_mode == "block":
        return _split_dataframe_by_blocks(
            df,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
        )

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
) -> tuple[Pipeline | None, pd.DataFrame, dict[str, float]]:
    """Train/evaluate or load a global 52-position classical baseline."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label"})
    resolved_model_name = model_name.upper()
    resolved_params = dict(params)

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
        )
        return None, predictions, metrics

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
    data_fingerprint = _window_identity_fingerprint(test_df)
    majority_metrics = majority_class_baselines(
        train_df,
        test_df,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )

    if results_dir is not None and not force_retrain:
        expected_metadata = None
        if split_mode == "cross_session":
            expected_metadata = prediction_cache_metadata(
                model=resolved_model_name,
                band=dataset_name,
                split_mode=split_mode,
                params=resolved_params,
                random_state=random_state,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                data_fingerprint=data_fingerprint,
            )
        cached_predictions = load_predictions(
            Path(results_dir),
            resolved_model_name,
            dataset_name,
            split_mode,
            expected_metadata=expected_metadata,
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
            metadata=prediction_cache_metadata(
                model=resolved_model_name,
                band=dataset_name,
                split_mode=split_mode,
                params=resolved_params,
                random_state=random_state,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                data_fingerprint=data_fingerprint,
            ),
        )

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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run leave-one-volunteer-out global classification for one model/band pair."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label", "user"})

    resolved_model_name = model_name.upper()
    resolved_params = dict(params)
    folds = split_lovo_folds(df)
    held_out_users = [_held_out_user(test_df) for _, test_df in folds]
    result_path = Path(results_dir) if results_dir is not None else None

    _print_lovo_honesty_warnings(df, folds, dataset_name=dataset_name)
    print(
        "[LOVO] Hyperparameter provenance: reusing block-split tuned/default "
        "hyperparameters; no nested CV is run for LOVO."
    )

    if result_path is not None and not force_retrain:
        cached_predictions = _load_lovo_cached_predictions(
            result_path,
            resolved_model_name,
            dataset_name,
            resolved_params,
            held_out_users,
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

    for fold_index, ((train_df, test_df), held_out_user) in enumerate(
        zip(folds, held_out_users),
        start=1,
    ):
        fold_label = _lovo_fold_label(held_out_user)
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
                metadata=prediction_cache_metadata(
                    model=resolved_model_name,
                    band=dataset_name,
                    split_mode="lovo",
                    params=resolved_params,
                    fold=fold_label,
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
    if len(predictions) != len(df):
        msg = f"LOVO predictions should cover {len(df)} windows, got {len(predictions)}."
        raise RuntimeError(msg)

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
            metadata=prediction_cache_metadata(
                model=resolved_model_name,
                band=dataset_name,
                split_mode="lovo",
                params=resolved_params,
            ),
        )

    print(
        f"[LOVO] {dataset_name} / {resolved_model_name} complete: "
        f"position_accuracy={aggregated_metrics['position_accuracy_mean']:.4f} +/- "
        f"{aggregated_metrics['position_accuracy_std']:.4f}, "
        f"total_wall={aggregated_metrics['wall_seconds']:.1f}s"
    )
    return predictions, per_fold_metrics, aggregated_metrics


def build_global_predictions_dataframe(  # noqa: PLR0913
    test_df: pd.DataFrame,
    pred_positions: np.ndarray,
    *,
    dataset_name: str,
    model_name: str,
    split_mode: str,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> pd.DataFrame:
    """Build the model-agnostic predictions dataframe used by all ML analysis."""
    pred_rooms = np.asarray(
        [room_label_for_location(location) for location in pred_positions],
        dtype=object,
    )
    true_rooms = np.asarray(
        [room_label_for_location(location) for location in test_df["location"]],
        dtype=object,
    )
    distance_errors = np.asarray(
        [
            location_distance_error(
                true_position,
                pred_position,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            for true_position, pred_position in zip(test_df["location"], pred_positions)
        ],
        dtype=float,
    )
    true_coordinates = [
        location_grid_coordinates(
            location,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        for location in test_df["location"]
    ]
    pred_coordinates = [
        location_grid_coordinates(
            location,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        for location in pred_positions
    ]

    return pd.DataFrame(
        {
            "window_id": (
                test_df["group_id"].astype(str) + "_window_" + test_df["window_idx"].astype(str)
            ).to_numpy(),
            "user": test_df["user"].to_numpy(),
            "trial": test_df["trial"].to_numpy(),
            "true_position": test_df["location"].astype(str).to_numpy(),
            "pred_position": pred_positions,
            "true_room": true_rooms,
            "pred_room": pred_rooms,
            "true_x": [_coord_component(coord, "x") for coord in true_coordinates],
            "true_y": [_coord_component(coord, "y") for coord in true_coordinates],
            "pred_x": [_coord_component(coord, "x") for coord in pred_coordinates],
            "pred_y": [_coord_component(coord, "y") for coord in pred_coordinates],
            "distance_error": distance_errors,
            "dataset": dataset_name,
            "model": model_name,
            "split_mode": split_mode,
            "split": split_mode,
            "true_location": test_df["location"].astype(str).to_numpy(),
            "pred_location": pred_positions,
            "scenario": test_df["scenario"].to_numpy(),
            "group_id": test_df["group_id"].to_numpy(),
            "window_idx": test_df["window_idx"].to_numpy(),
        },
        columns=GLOBAL_PREDICTION_COLUMNS,
    )


def majority_class_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> dict[str, float]:
    """Accuracy of constant predictors using the most frequent training labels."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_required_columns(train_df, {"location"})
    _validate_required_columns(test_df, {"location"})
    if train_df.empty or test_df.empty:
        return {
            "majority_position_accuracy": np.nan,
            "majority_room_accuracy": np.nan,
        }

    majority_position = str(train_df["location"].astype(str).mode(dropna=True).iloc[0])
    train_rooms = pd.Series(
        [room_label_for_location(location) for location in train_df["location"]],
    )
    majority_room = train_rooms.mode(dropna=True).iloc[0]
    true_rooms = pd.Series([room_label_for_location(location) for location in test_df["location"]])

    return {
        "majority_position_accuracy": float(
            (test_df["location"].astype(str).to_numpy() == majority_position).mean()
        ),
        "majority_room_accuracy": float((true_rooms.to_numpy() == majority_room).mean()),
    }


def _load_lovo_cached_predictions(
    results_dir: Path,
    model_name: str,
    dataset_name: str,
    params: dict[str, Any],
    held_out_users: list[object],
    *,
    save_prediction_cache: bool,
) -> pd.DataFrame | None:
    concat_metadata = prediction_cache_metadata(
        model=model_name,
        band=dataset_name,
        split_mode="lovo",
        params=params,
    )
    cached_concat = load_predictions(
        results_dir,
        model_name,
        dataset_name,
        "lovo",
        expected_metadata=concat_metadata,
    )
    if cached_concat is not None:
        if "held_out_user" not in cached_concat.columns:
            print("[predictions cache stale] LOVO concat lacks held_out_user.")
            return None
        return cached_concat

    fold_frames = []
    for held_out_user in held_out_users:
        fold_label = _lovo_fold_label(held_out_user)
        fold_predictions = load_predictions(
            results_dir,
            model_name,
            dataset_name,
            "lovo",
            fold=fold_label,
            expected_metadata=prediction_cache_metadata(
                model=model_name,
                band=dataset_name,
                split_mode="lovo",
                params=params,
                fold=fold_label,
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
    users = test_df["user"].dropna().unique()
    if len(users) != 1:
        msg = f"Expected exactly one held-out user, got {users!r}."
        raise ValueError(msg)
    return users[0]


def _lovo_fold_label(held_out_user: object) -> str:
    return f"user-{held_out_user}"


def _split_dataframe_by_blocks(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_blocks < 1:
        msg = "n_blocks must be at least 1 for block split mode."
        raise ValueError(msg)
    _validate_required_columns(df, {"window_idx"})
    group_ids = df["group_id"].to_numpy()
    window_idxs = df["window_idx"].to_numpy()
    n_test_blocks = min(n_blocks, max(1, round(test_size * n_blocks)))

    train_locs: list[int] = []
    test_locs: list[int] = []

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


def _coord_component(coordinates: tuple[float, float] | None, axis: Literal["x", "y"]) -> float:
    if coordinates is None:
        return np.nan
    row_coordinate, column_coordinate = coordinates
    return float(column_coordinate if axis == "x" else row_coordinate)


def _estimator_name(model: Pipeline) -> str:
    classifier = model.named_steps.get("classifier")
    return type(classifier).__name__ if classifier is not None else type(model).__name__


def _normalize_location_label(location: object) -> str:
    return str(location).strip().upper().removeprefix("LOCATION_")


def _validate_grid_spacing(*, row_spacing: float, column_spacing: float) -> None:
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        msg = "row_spacing must be a finite positive value."
        raise ValueError(msg)
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        msg = "column_spacing must be a finite positive value."
        raise ValueError(msg)


def _validate_training_dataframe(df: pd.DataFrame) -> None:
    _validate_required_columns(df, METADATA_COLUMNS)
    if df.empty:
        msg = "Cannot train on an empty dataframe."
        raise ValueError(msg)
    if not feature_columns(df):
        msg = "No CSI feature columns found."
        raise ValueError(msg)


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        msg = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)
