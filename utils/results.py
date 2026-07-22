"""Flat, content-addressed experiment result storage and reporting."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from utils.cache import RESULTS_ROOT, _lib_version, parse_run_id
from utils.config import (
    EMPTY_ROOM_LOCATION,
    PROJECT_ROOT,
    ROOM_1_COLUMNS,
    ROOM_2_A_COLUMNS,
    ROOM_2_BC_COLUMNS,
    ROOM_3_EF_COLUMNS,
)

LOCATION_PATTERN = re.compile(r"^(?P<row>[A-Z])[-_ ]?(?P<column>\d+)$")
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

RUN_PREFIX_COLUMNS = [
    "run_id",
    "timestamp",
    "family",
    "model",
    "band",
    "split",
    "seed",
    "normalization",
    "baseline_scope",
    "window_size",
    "overlap_size",
    "require_all_esps",
    "trials_used",
    "n_train",
    "n_test",
    "n_classes",
]

RUN_METRIC_COLUMNS = [
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
    "position_accuracy_mean",
    "position_accuracy_std",
    "position_accuracy_pooled",
    "n_folds",
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
    "majority_position_accuracy_mean",
    "majority_position_accuracy_std",
    "majority_room_accuracy_mean",
    "majority_room_accuracy_std",
    "fit_seconds",
    "predict_seconds",
    "wall_seconds",
    "used_estimator",
    "parameter_count",
    "best_val_accuracy",
    "best_val_accuracy_mean",
    "best_val_accuracy_std",
    "best_epoch",
    "best_epoch_mean",
    "best_epoch_std",
    "mean_seconds_per_epoch",
    "mean_seconds_per_epoch_mean",
    "mean_seconds_per_epoch_std",
    "stopped_epoch",
    "patience_triggered",
    "peak_cuda_memory_bytes",
    "device",
    "torch_version",
    "sklearn_version",
    "numpy_version",
]

FOLD_PREFIX_COLUMNS = [
    "run_id",
    "fold",
    "held_out_user",
    "validation_user",
    "n_train_windows",
    "n_test_windows",
    "trials_used",
]
FOLD_METRIC_COLUMNS = [
    "position_accuracy",
    "macro_f1",
    "room_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "p90_distance_error",
    "majority_position_accuracy",
    "majority_room_accuracy",
    "fit_seconds",
    "predict_seconds",
    "wall_seconds",
    "parameter_count",
    "best_val_accuracy",
    "best_epoch",
    "mean_seconds_per_epoch",
]


def compute_localization_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    """Compute the shared position, room, F1, and distance metrics."""
    true_position = _metric_column(predictions_df, "true_position", "true_location")
    pred_position = _metric_column(predictions_df, "pred_position", "pred_location")
    true_room = _metric_column(predictions_df, "true_room")
    pred_room = _metric_column(predictions_df, "pred_room")

    if predictions_df.empty:
        return {
            "position_accuracy": np.nan,
            "macro_f1": np.nan,
            "room_accuracy": np.nan,
            "mean_distance_error": np.nan,
            "median_distance_error": np.nan,
            "rmse_distance_error": np.nan,
            "p90_distance_error": np.nan,
            "samples": 0.0,
        }

    distance_errors = pd.to_numeric(
        predictions_df["distance_error"],
        errors="coerce",
    ).dropna()
    error_values = distance_errors.to_numpy(dtype=float)
    return {
        "position_accuracy": _accuracy(true_position, pred_position),
        "macro_f1": float(
            f1_score(
                true_position.astype(str),
                pred_position.astype(str),
                average="macro",
                zero_division=0,
            )
        ),
        "room_accuracy": _accuracy(true_room, pred_room),
        "mean_distance_error": float(np.mean(error_values)) if error_values.size else np.nan,
        "median_distance_error": (
            float(np.median(error_values)) if error_values.size else np.nan
        ),
        "rmse_distance_error": (
            float(np.sqrt(np.mean(np.square(error_values))))
            if error_values.size
            else np.nan
        ),
        "p90_distance_error": (
            float(np.percentile(error_values, 90)) if error_values.size else np.nan
        ),
        "samples": float(len(predictions_df)),
    }


def _metric_column(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first available metric column from a list of aliases."""
    for name in names:
        if name in df.columns:
            return df[name]
    raise ValueError(f"Missing required column. Expected one of: {', '.join(names)}")


def _accuracy(left: pd.Series, right: pd.Series) -> float:
    """Return equality accuracy for two aligned Series, or NaN when empty."""
    if left.empty:
        return np.nan
    return float((left.to_numpy() == right.to_numpy()).mean())


def room_label_for_location(location: object) -> int | None:
    """Map a reference-point label to its room label."""
    label = _normalize_location_label(location)
    if label == EMPTY_ROOM_LOCATION:
        return 0
    match = LOCATION_PATTERN.fullmatch(label)
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
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
) -> tuple[float, float] | None:
    """Convert a location label to physical row/column coordinates."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    label = _normalize_location_label(location)
    if label == EMPTY_ROOM_LOCATION:
        return None
    match = LOCATION_PATTERN.fullmatch(label)
    if match is None:
        return None
    row_index = ord(match.group("row")) - ord("A")
    column_index = int(match.group("column")) - 1
    return row_index * row_spacing, column_index * column_spacing


def location_distance_error(
    true_location: object,
    pred_location: object,
    *,
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
) -> float:
    """Return Euclidean distance between two reference-point labels."""
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
    return float(
        np.hypot(
            true_coordinates[0] - pred_coordinates[0],
            true_coordinates[1] - pred_coordinates[1],
        )
    )


def build_global_predictions_dataframe(  # noqa: PLR0913
    test_df: pd.DataFrame,
    pred_positions: np.ndarray,
    *,
    dataset_name: str,
    model_name: str,
    split_mode: str,
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
) -> pd.DataFrame:
    """Build the shared prediction table used by ML and DL evaluation."""
    # Derive room labels and physical errors from the position predictions so all
    # model families use the same evaluation definitions.
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
                test_df["group_id"].astype(str)
                + "_window_"
                + test_df["window_idx"].astype(str)
            ).to_numpy(),
            "user": test_df["user"].to_numpy(),
            "trial": test_df["trial"].to_numpy(),
            "true_position": test_df["location"].astype(str).to_numpy(),
            "pred_position": pred_positions,
            "true_room": true_rooms,
            "pred_room": pred_rooms,
            "true_x": [_coord_component(coordinates, "x") for coordinates in true_coordinates],
            "true_y": [_coord_component(coordinates, "y") for coordinates in true_coordinates],
            "pred_x": [_coord_component(coordinates, "x") for coordinates in pred_coordinates],
            "pred_y": [_coord_component(coordinates, "y") for coordinates in pred_coordinates],
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
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
) -> dict[str, float]:
    """Return constant position and room baselines from the training labels."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _require_columns(train_df, {"location"})
    _require_columns(test_df, {"location"})
    if train_df.empty or test_df.empty:
        return {"majority_position_accuracy": np.nan, "majority_room_accuracy": np.nan}
    majority_position = str(train_df["location"].astype(str).mode(dropna=True).iloc[0])
    train_rooms = pd.Series(
        [room_label_for_location(location) for location in train_df["location"]]
    )
    majority_room = train_rooms.mode(dropna=True).iloc[0]
    true_rooms = pd.Series(
        [room_label_for_location(location) for location in test_df["location"]]
    )
    return {
        "majority_position_accuracy": float(
            (test_df["location"].astype(str).to_numpy() == majority_position).mean()
        ),
        "majority_room_accuracy": float((true_rooms.to_numpy() == majority_room).mean()),
    }


def _coord_component(
    coordinates: tuple[float, float] | None,
    axis: str,
) -> float:
    """Extract an x or y value from row/column coordinates."""
    if coordinates is None:
        return np.nan
    row_coordinate, column_coordinate = coordinates
    return float(column_coordinate if axis == "x" else row_coordinate)


def _normalize_location_label(location: object) -> str:
    """Normalize a location value to its uppercase grid label."""
    return str(location).strip().upper().removeprefix("LOCATION_")


def _validate_grid_spacing(*, row_spacing: float, column_spacing: float) -> None:
    """Require finite, positive physical spacing along both grid axes."""
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        raise ValueError("row_spacing must be a finite positive value.")
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        raise ValueError("column_spacing must be a finite positive value.")


def _require_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error when an evaluation DataFrame lacks columns."""
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def ensure_results_layout(results_root: Path = RESULTS_ROOT) -> None:
    """Create the directories in the flat results contract."""
    for relative in ("predictions", "plots", "tables", "tuning", "manifests", "checkpoints"):
        (Path(results_root) / relative).mkdir(parents=True, exist_ok=True)


def build_run_row(  # noqa: PLR0913
    *,
    run_id: str,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    trials_used: Iterable[object],
    n_train: int,
    n_test: int,
    n_classes: int,
    device: str = "cpu",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one normalized row for ``runs.csv`` from a completed run."""
    parsed = parse_run_id(run_id)
    scope = preproc_opts.get("baseline_scope")
    if scope is not None:
        scope = str(scope).removeprefix("per_")
    row = {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "family": parsed["family"],
        "model": parsed["model"],
        "band": parsed["band"],
        "split": parsed["split"],
        "seed": parsed["seed"],
        "normalization": preproc_opts.get("normalization", "none"),
        "baseline_scope": scope,
        "window_size": feat_opts.get("window_size", 60),
        "overlap_size": feat_opts.get("overlap_size", feat_opts.get("step", 30)),
        "require_all_esps": bool(feat_opts.get("require_all_esps", False)),
        "trials_used": ",".join(sorted({str(value).zfill(2) for value in trials_used})),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_classes": int(n_classes),
        **_csv_safe_mapping(hyperparameters),
        **metrics,
        "device": device,
        "torch_version": metrics.get("torch_version", _lib_version("torch")),
        "sklearn_version": metrics.get("sklearn_version", _lib_version("scikit-learn")),
        "numpy_version": metrics.get("numpy_version", _lib_version("numpy")),
    }
    return row


def upsert_run(
    row: dict[str, Any],
    *,
    hyperparameter_columns: Iterable[str] = (),
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Append one run, replacing an existing row with the same run_id."""
    if not row.get("run_id"):
        raise ValueError("A non-empty run_id is required for runs.csv.")
    ensure_results_layout(results_root)
    return _upsert_rows(
        Path(results_root) / "runs.csv",
        [row],
        key_columns=["run_id"],
        prefix_columns=RUN_PREFIX_COLUMNS,
        preferred_middle=list(hyperparameter_columns),
        suffix_columns=RUN_METRIC_COLUMNS,
    )


def upsert_fold_rows(
    rows: Iterable[dict[str, Any]],
    *,
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Upsert LOVO/cross-session rows by ``(run_id, fold)``."""
    normalized_rows = list(rows)
    if not normalized_rows:
        path = Path(results_root) / "runs_folds.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    ensure_results_layout(results_root)
    return _upsert_rows(
        Path(results_root) / "runs_folds.csv",
        normalized_rows,
        key_columns=["run_id", "fold"],
        prefix_columns=FOLD_PREFIX_COLUMNS,
        preferred_middle=[],
        suffix_columns=FOLD_METRIC_COLUMNS,
    )


def write_run_manifest(
    run_id: str,
    config: dict[str, Any],
    *,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """Write the full config used to hash a run identifier."""
    parsed = parse_run_id(run_id)
    ensure_results_layout(results_root)
    normalized_config = json.loads(json.dumps(config, sort_keys=True, default=str))
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parsed_run_id": parsed,
        "config": normalized_config,
    }
    path = Path(results_root) / "manifests" / f"{run_id}.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate existing run manifest {path}: {exc}") from exc
        if existing.get("config") != normalized_config:
            raise RuntimeError(
                f"run_id collision detected for {run_id}: the existing full config differs."
            )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def checkpoint_path(
    run_id: str,
    *,
    fold: object | None = None,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """Return the flat checkpoint path for a whole run or one LOVO fold."""
    parse_run_id(run_id)
    ensure_results_layout(results_root)
    suffix = "" if fold is None else f"__fold-{str(fold).removeprefix('user-')}"
    return Path(results_root) / "checkpoints" / f"{run_id}{suffix}.pt"


def derive_table_from_runs(
    basename: str,
    *,
    columns: Iterable[str] | None = None,
    query: str | None = None,
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    """Create CSV/LaTeX views exclusively from the global ``runs.csv`` table."""
    runs_path = Path(results_root) / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Cannot derive tables before {runs_path} exists.")
    table = pd.read_csv(runs_path)
    if query:
        table = table.query(query)
    if columns is not None:
        selected = [column for column in columns if column in table.columns]
        table = table.loc[:, selected]
    tables_dir = Path(results_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / f"{basename}.csv", index=False)
    (tables_dir / f"{basename}.tex").write_text(
        table.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    return table


def derive_seed_summary(*, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Report across-seed variation without conflating it with LOVO fold variation."""
    runs = pd.read_csv(Path(results_root) / "runs.csv")
    seeded = runs.loc[runs["seed"].notna()].copy()
    if seeded.empty:
        return pd.DataFrame()
    group_columns = [
        "family",
        "model",
        "band",
        "split",
        "normalization",
        "baseline_scope",
        "window_size",
        "overlap_size",
        "require_all_esps",
    ]
    rows: list[dict[str, Any]] = []
    # Aggregate one metric per seed; LOVO rows already contain their fold-level
    # mean and standard deviation and must not be pooled as independent samples.
    for keys, group in seeded.groupby(group_columns, dropna=False, sort=True):
        split = str(group["split"].iloc[0])
        basis = "fold_mean_per_seed" if split == "lovo" else "run_metric_per_seed"
        metric_column = (
            "position_accuracy_mean"
            if split == "lovo" and "position_accuracy_mean" in group
            else "position_accuracy"
        )
        values = pd.to_numeric(group[metric_column], errors="coerce")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "seed_aggregation_basis": basis,
                "n_seeds": int(values.notna().sum()),
                "position_accuracy_seed_mean": float(values.mean()),
                "position_accuracy_seed_std": float(values.std(ddof=1)),
                "fold_position_accuracy_std_mean_across_seeds": (
                    float(pd.to_numeric(group["position_accuracy_std"], errors="coerce").mean())
                    if split == "lovo" and "position_accuracy_std" in group
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    tables_dir = Path(results_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "seed_summary.csv", index=False)
    (tables_dir / "seed_summary.tex").write_text(
        summary.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    return summary


def _upsert_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    key_columns: list[str],
    prefix_columns: list[str],
    preferred_middle: list[str],
    suffix_columns: list[str],
) -> pd.DataFrame:
    """Replace rows with matching keys and atomically rewrite the CSV table."""
    incoming = pd.DataFrame(rows)
    existing_middle: list[str] = []
    # Preserve the existing table's metric-column order while replacing rows
    # identified by the supplied stable keys.
    if path.exists():
        string_columns = {
            column: "string"
            for column in (
                "run_id",
                "trials_used",
                "fold",
                "held_out_user",
                "validation_user",
            )
        }
        header_order = list(pd.read_csv(path, nrows=0).columns)
        header_columns = set(header_order)
        suffix_positions = [
            header_order.index(column)
            for column in suffix_columns
            if column in header_columns
        ]
        if suffix_positions:
            first_suffix = min(suffix_positions)
            existing_middle = [
                column
                for column in header_order[len(prefix_columns) : first_suffix]
                if column not in prefix_columns
            ]
        existing = pd.read_csv(
            path,
            dtype={
                key: value
                for key, value in string_columns.items()
                if key in header_columns
            },
        )
        for key in key_columns:
            if key not in existing or key not in incoming:
                raise ValueError(f"Missing upsert key column {key!r} in {path}.")
        old_keys = existing[key_columns].astype(str).agg("\x1f".join, axis=1)
        new_keys = set(incoming[key_columns].astype(str).agg("\x1f".join, axis=1))
        existing = existing.loc[~old_keys.isin(new_keys)]
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    # Rebuild a deterministic schema: identifiers, metrics, provenance, then any
    # previously unknown columns.
    preferred_middle = _unique(existing_middle + preferred_middle)
    known = set(prefix_columns) | set(preferred_middle) | set(suffix_columns)
    additional_columns = [column for column in combined.columns if column not in known]
    ordered = _unique(prefix_columns + preferred_middle + suffix_columns + additional_columns)
    combined = combined.reindex(columns=ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"[results upsert] {path}: {len(combined)} row(s)")
    return combined


def _csv_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Serialize nested values so a mapping can be stored safely in CSV cells."""
    return {
        key: (json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else value)
        for key, value in values.items()
    }


def _unique(values: Iterable[str]) -> list[str]:
    """Remove duplicate strings while preserving their original order."""
    return list(dict.fromkeys(values))


# Legacy paper result exports retained in the shared result module.

def save_summary(df: pd.DataFrame, results_dir: Path, basename: str = "summary") -> None:
    """Write the summary dataframe to CSV, Markdown, and LaTeX in results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / f"{basename}.csv")

    md = df.to_markdown(index=True)
    (results_dir / f"{basename}.md").write_text(md or "", encoding="utf-8")

    tex = df.to_latex(
        index=True,
        escape=False,
        float_format="%.4f",
    )
    (results_dir / f"{basename}.tex").write_text(tex, encoding="utf-8")

    print(f"[results] Summary saved to {results_dir}/{basename}.*")


# ── Reproducibility manifest ──────────────────────────────────────────────────

def _git_commit() -> str | None:
    """Return the current short Git commit when it can be determined."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def write_manifest(
    results_dir: Path,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    classifier_params: dict[str, Any],
    splits: list[str],
    test_size: float,
    feature_dataframes: dict[str, pd.DataFrame],
    tuned_hyperparameters: dict[str, Any] | None = None,
    tuned_hyperparameters_direct: dict[str, Any] | None = None,
    tuned_hyperparameters_v1: dict[str, Any] | None = None,
    tuned_hyperparameters_v2: dict[str, Any] | None = None,
    lovo_metadata: dict[str, Any] | None = None,
) -> None:
    """Write a self-describing manifest.json to results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Capture input shapes and experiment settings before merging optional
    # tuning or LOVO metadata into the persisted manifest.
    dataset_sizes = {
        f"{_band_stem(band)}_windows": len(df)
        for band, df in feature_dataframes.items()
    }
    feature_matrix_shapes = {
        band: {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_columns": int(
                len([col for col in df.columns if col not in _EXPECTED_METADATA_COLS])
            ),
        }
        for band, df in feature_dataframes.items()
    }
    config_payload = {
        "preprocessing_options": preproc_opts,
        "feature_extraction_options": feat_opts,
        "classifier_hyperparameters": classifier_params,
        "splits": splits,
        "test_size": test_size,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    manifest_path = results_dir / "manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "key_library_versions": {
            "numpy": _lib_version("numpy"),
            "pandas": _lib_version("pandas"),
            "scikit-learn": _lib_version("scikit-learn"),
        },
        "preprocessing_options": preproc_opts,
        "feature_extraction_options": feat_opts,
        "classifier_hyperparameters": classifier_params,
        "config_hash": config_hash,
        "splits": splits,
        "test_size": test_size,
        "dataset_sizes": dataset_sizes,
        "feature_matrix_shapes": feature_matrix_shapes,
    }
    if tuned_hyperparameters is None:
        tuned_hyperparameters = existing_manifest.get("tuned_hyperparameters")
    if tuned_hyperparameters is not None:
        manifest["tuned_hyperparameters"] = tuned_hyperparameters
    if tuned_hyperparameters_direct is not None:
        manifest["tuned_hyperparameters_direct"] = tuned_hyperparameters_direct
    if tuned_hyperparameters_v1 is None:
        tuned_hyperparameters_v1 = existing_manifest.get("tuned_hyperparameters_v1")
    if tuned_hyperparameters_v1 is not None:
        manifest["tuned_hyperparameters_v1"] = tuned_hyperparameters_v1
    if tuned_hyperparameters_v2 is None:
        tuned_hyperparameters_v2 = existing_manifest.get("tuned_hyperparameters_v2")
    if tuned_hyperparameters_v2 is not None:
        manifest["tuned_hyperparameters_v2"] = tuned_hyperparameters_v2
    if lovo_metadata is None:
        lovo_metadata = existing_manifest.get("lovo")
    if lovo_metadata is not None:
        manifest["lovo"] = lovo_metadata

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[results] Manifest written to {manifest_path}")
