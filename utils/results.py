"""Flat, content-addressed experiment result storage and reporting."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils.cache import RESULTS_ROOT, parse_run_id

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
        "torch_version": metrics.get("torch_version", _version("torch")),
        "sklearn_version": metrics.get("sklearn_version", _version("scikit-learn")),
        "numpy_version": metrics.get("numpy_version", np.__version__),
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
    incoming = pd.DataFrame(rows)
    existing_middle: list[str] = []
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
    return {
        key: (json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else value)
        for key, value in values.items()
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
