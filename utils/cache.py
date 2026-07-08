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


class _StaleFeatureCache(ValueError):
    """Raised when an existing cache predates the current feature schema."""


# ── Key generation ────────────────────────────────────────────────────────────

def make_preproc_key(opts: dict[str, Any]) -> str:
    """Encode magnitude-processing options as a short, human-readable key."""
    parts: list[str] = []
    agc = opts.get("apply_agc_compensation", False)
    parts.append(f"agc-{'on' if agc else 'off'}")
    filt = str(opts.get("filter_method", "none")).lower()
    if filt not in ("none", ""):
        parts.append(f"filt-{filt}")
    norm = str(opts.get("normalization", "none")).lower()
    if norm not in ("none", ""):
        parts.append(f"norm-{norm}")
    return "_".join(sorted(parts))


def make_feat_key(opts: dict[str, Any]) -> str:
    """Encode feature-extraction options as a short, human-readable key."""
    parts: list[str] = []
    win = opts.get("window_size", 60)
    step = opts.get("step", opts.get("overlap_size", 30))
    parts.append(f"win{win}-step{step}")
    if opts.get("calibrate", False):
        parts.append("cal-on")
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


def predictions_path(results_dir: Path, model: str, band: str, split_mode: str) -> Path:
    """Return the parquet path for a model/band/split prediction dataframe."""
    return (
        Path(results_dir)
        / "predictions"
        / f"{_band_stem(band)}__{_band_stem(model)}__{_band_stem(split_mode)}.parquet"
    )


def save_predictions(
    df: pd.DataFrame,
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
) -> None:
    """Persist prediction rows as parquet for analysis without retraining."""
    path = predictions_path(results_dir, model, band, split_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"[predictions] Saved {path}")


def load_predictions(
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
) -> pd.DataFrame | None:
    """Load persisted predictions, returning None when no cache file exists."""
    path = predictions_path(results_dir, model, band, split_mode)
    if not path.exists():
        return None
    print(f"[predictions cache hit] {path}")
    return pd.read_parquet(path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _band_stem(band: str) -> str:
    """'2.4 GHz' → '2_4ghz',  '5 GHz' → '5ghz',  'Fusion' → 'fusion'."""
    return band.lower().replace(".", "_").replace(" ", "")


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

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[results] Manifest written to {manifest_path}")
