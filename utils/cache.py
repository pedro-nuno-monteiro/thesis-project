from __future__ import annotations

import importlib.metadata
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataframes"
RESULTS_ROOT = PROJECT_ROOT / "results"

_EXPECTED_METADATA_COLS = {
    "frequency_scenario",
    "scenario",
    "location",
    "user",
    "trial",
    "group_id",
    "window_idx",
    "label",
}
_PREDICTION_METADATA_KEY = b"thesis_prediction_cache_metadata"


class _StaleFeatureCache(ValueError):
    """Raised when an existing cache predates the current feature schema."""


# ── Key generation ────────────────────────────────────────────────────────────

def make_preproc_key(opts: dict[str, Any]) -> str:
    """Encode magnitude-processing options as a short, human-readable key."""
    parts: list[str] = []
    norm = str(opts.get("normalization", "none")).lower()
    parts.append(f"norm-{norm or 'none'}")
    if norm == "empty_baseline":
        scope = str(opts.get("baseline_scope", "per_session")).lower()
        parts.append(f"scope-{scope}")
    return "_".join(sorted(parts))


def make_feat_key(opts: dict[str, Any]) -> str:
    """Encode feature-extraction options as a short, human-readable key."""
    parts: list[str] = []
    win = opts.get("window_size", 60)
    step = opts.get("step", opts.get("overlap_size", 30))
    parts.append(f"win{win}-step{step}")
    if opts.get("require_all_esps", False):
        parts.append("allesps-on")
    return "_".join(sorted(parts))


def get_cache_path(preproc_opts: dict[str, Any], feat_opts: dict[str, Any]) -> Path:
    return (
        CACHE_DIR
        / f"preproc={make_preproc_key(preproc_opts)}"
        / f"feat={make_feat_key(feat_opts)}"
    )


def get_results_path(preproc_opts: dict[str, Any], feat_opts: dict[str, Any]) -> Path:
    return (
        RESULTS_ROOT
        / f"preproc={make_preproc_key(preproc_opts)}"
        / f"feat={make_feat_key(feat_opts)}"
    )


def predictions_path(
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
    *,
    fold: str | None = None,
) -> Path:
    """Return the parquet path for a model/band/split prediction dataframe."""
    stem = f"{_band_stem(band)}__{_band_stem(model)}__{_band_stem(split_mode)}"
    if fold is not None:
        stem = f"{stem}__{_band_stem(fold)}"
    return (
        Path(results_dir)
        / "predictions"
        / f"{stem}.parquet"
    )


def save_predictions(
    df: pd.DataFrame,
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
    *,
    fold: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist prediction rows as parquet for analysis without retraining."""
    path = predictions_path(results_dir, model, band, split_mode, fold=fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    if metadata is None:
        df.to_parquet(tmp, index=False)
    else:
        _write_prediction_parquet_with_metadata(df, tmp, metadata)
    tmp.replace(path)
    if metadata is not None:
        _write_prediction_metadata(path, metadata)
    print(f"[predictions] Saved {path}")


def load_predictions(
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
    *,
    fold: str | None = None,
    expected_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Load persisted predictions, returning None when no cache file exists."""
    path = predictions_path(results_dir, model, band, split_mode, fold=fold)
    if not path.exists():
        return None
    if expected_metadata is not None and not _prediction_metadata_matches(
        path,
        expected_metadata,
    ):
        print(f"[predictions cache stale] {path}")
        return None
    print(f"[predictions cache hit] {path}")
    return pd.read_parquet(path)


def prediction_cache_metadata(
    *,
    model: str,
    band: str,
    split_mode: str,
    params: dict[str, Any],
    fold: str | None = None,
    random_state: int | None = None,
    row_spacing: float | None = None,
    column_spacing: float | None = None,
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return the metadata payload used to validate prediction caches."""
    normalized_params = _json_normalized(params)
    metadata = {
        "schema_version": 1,
        "model": model,
        "band": band,
        "split_mode": split_mode,
        "fold": fold,
        "params": normalized_params,
        "params_hash": hashlib.sha256(
            json.dumps(normalized_params, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "sklearn_version": _lib_version("scikit-learn"),
    }
    if data_fingerprint is not None:
        metadata.update(
            {
                "schema_version": 2,
                "random_state": random_state,
                "row_spacing": row_spacing,
                "column_spacing": column_spacing,
                "data_fingerprint": data_fingerprint,
            }
        )
    return metadata


# ── Internal helpers ──────────────────────────────────────────────────────────

def _band_stem(band: str) -> str:
    """'2.4 GHz' → '2_4ghz',  '5 GHz' → '5ghz',  'Fusion' → 'fusion'."""
    return band.lower().replace(".", "_").replace(" ", "")


def _prediction_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _write_prediction_metadata(path: Path, metadata: dict[str, Any]) -> None:
    metadata_path = _prediction_metadata_path(path)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_normalized(metadata), indent=2), encoding="utf-8")
    tmp.replace(metadata_path)


def _write_prediction_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        df.to_parquet(path, index=False)
        return

    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[_PREDICTION_METADATA_KEY] = json.dumps(
        _json_normalized(metadata),
        sort_keys=True,
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(schema_metadata), path)


def _read_prediction_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = _prediction_metadata_path(path)
    if not metadata_path.exists():
        return _read_prediction_parquet_metadata(path)
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _read_prediction_parquet_metadata(path)


def _read_prediction_parquet_metadata(path: Path) -> dict[str, Any] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    try:
        parquet_metadata = pq.read_metadata(path).metadata or {}
        raw_payload = parquet_metadata.get(_PREDICTION_METADATA_KEY)
        if raw_payload is None:
            return None
        return json.loads(raw_payload.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _prediction_metadata_matches(path: Path, expected_metadata: dict[str, Any]) -> bool:
    observed = _read_prediction_metadata(path)
    if observed is None:
        return False
    expected = _json_normalized(expected_metadata)
    return all(observed.get(key) == value for key, value in expected.items())


def _json_normalized(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _check_schema(df: pd.DataFrame, path: Path) -> None:
    missing = _EXPECTED_METADATA_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Cache schema mismatch in {path} -- missing columns: {sorted(missing)}.\n"
            "Delete the cache folder and rerun to rebuild."
        )

    feature_cols = [col for col in df.columns if col not in _EXPECTED_METADATA_COLS]
    non_float32_cols = [
        col for col in feature_cols if str(df[col].dtype) != "float32"
    ]
    if non_float32_cols:
        raise _StaleFeatureCache(
            f"Cache schema mismatch in {path} -- feature columns must be float32; "
            f"stale columns include {non_float32_cols[:5]}."
        )


# ── Feature dataframe cache ───────────────────────────────────────────────────

def get_all_dataframes(
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    builder: Callable[[], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    expected_trials: set[str] | None = None,
    expected_window_inventory: dict[str, dict[str, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return feature dataframes for all three bands, using the on-disk cache.

    If all three parquet files exist under .cache/dataframes/<keys>/, loads and
    returns them. Otherwise calls builder() — which must return (df_24ghz,
    df_5ghz, df_fusion) — saves each file atomically, and returns the result.
    """
    cache_dir = get_cache_path(preproc_opts, feat_opts)
    band_stems: dict[str, str] = {
        "2.4 GHz": "2_4ghz",
        "5 GHz": "5ghz",
        "Fusion": "fusion",
    }
    paths = {name: cache_dir / f"{stem}.parquet" for name, stem in band_stems.items()}

    if all(p.exists() for p in paths.values()):
        result: dict[str, pd.DataFrame] = {}
        try:
            for name, path in paths.items():
                print(f"[cache hit] {path}")
                df = pd.read_parquet(path)
                _check_schema(df, path)
                result[name] = df
            stale_reasons = _feature_cache_stale_reasons(
                result,
                expected_trials=expected_trials,
                expected_window_inventory=expected_window_inventory,
            )
            if stale_reasons:
                print(
                    "!!! LOUD WARNING: cached feature dataframes are stale: "
                    f"{'; '.join(stale_reasons)}. Rebuilding all feature dataframes. !!!"
                )
                raise _StaleFeatureCache("cached windows do not match raw CSI inventory")
            return result
        except _StaleFeatureCache as exc:
            print(f"[cache stale] {exc} Rebuilding cached feature dataframes.")

    print(f"[cache miss] {cache_dir}, computing...")
    df_24ghz, df_5ghz, df_fusion = builder()
    dataframes: dict[str, pd.DataFrame] = {
        "2.4 GHz": df_24ghz,
        "5 GHz": df_5ghz,
        "Fusion": df_fusion,
    }
    stale_reasons = _feature_cache_stale_reasons(
        dataframes,
        expected_trials=expected_trials,
        expected_window_inventory=expected_window_inventory,
    )
    if stale_reasons:
        msg = "Built feature dataframes do not match raw CSI data: " + "; ".join(stale_reasons)
        raise ValueError(msg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, df in dataframes.items():
        dest = paths[name]
        tmp = dest.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(dest)

    options_path = cache_dir / "options.json"
    options_path.write_text(
        json.dumps(
            {"preprocessing": preproc_opts, "feature_extraction": feat_opts},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return dataframes


def _feature_cache_stale_reasons(
    dataframes: dict[str, pd.DataFrame],
    *,
    expected_trials: set[str] | None,
    expected_window_inventory: dict[str, dict[str, int]] | None,
) -> list[str]:
    reasons: list[str] = []
    for band, frame in dataframes.items():
        observed_trials = {str(value).zfill(2) for value in frame["trial"].dropna().unique()}
        missing_trials = set(expected_trials or ()) - observed_trials
        if missing_trials:
            reasons.append(f"{band} lacks trial(s) {sorted(missing_trials)}")
        if expected_window_inventory is None or band not in expected_window_inventory:
            continue
        observed_inventory = {
            str(group_id): int(count)
            for group_id, count in frame.groupby("group_id", sort=True).size().items()
        }
        expected_inventory = expected_window_inventory[band]
        if observed_inventory == expected_inventory:
            continue
        missing_groups = sorted(set(expected_inventory) - set(observed_inventory))
        extra_groups = sorted(set(observed_inventory) - set(expected_inventory))
        changed_groups = sorted(
            group_id
            for group_id in set(expected_inventory) & set(observed_inventory)
            if expected_inventory[group_id] != observed_inventory[group_id]
        )
        reasons.append(
            f"{band} window inventory differs "
            f"(missing={missing_groups[:3]}, extra={extra_groups[:3]}, "
            f"changed={changed_groups[:3]})"
        )
    return reasons


def get_dataframe(
    band: str,
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Return the cached feature dataframe for a single band.

    Loads from cache if the parquet file exists; otherwise calls builder(),
    saves atomically, and returns the result.
    """
    cache_dir = get_cache_path(preproc_opts, feat_opts)
    path = cache_dir / f"{_band_stem(band)}.parquet"

    if path.exists():
        print(f"[cache hit] {path}")
        df = pd.read_parquet(path)
        try:
            _check_schema(df, path)
            return df
        except _StaleFeatureCache as exc:
            print(f"[cache stale] {exc} Rebuilding cached feature dataframe.")

    print(f"[cache miss] {path}, computing...")
    df = builder()

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(path)

    options_path = cache_dir / "options.json"
    if not options_path.exists():
        options_path.write_text(
            json.dumps(
                {"preprocessing": preproc_opts, "feature_extraction": feat_opts},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return df


# ── Summary table saving ──────────────────────────────────────────────────────

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


def _lib_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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
