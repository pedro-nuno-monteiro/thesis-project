from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.config import CACHE_DIR as PROJECT_CACHE_DIR
from utils.config import RESULTS_DIR

CACHE_DIR = PROJECT_CACHE_DIR / "dataframes"
CSI_CACHE_DIR = PROJECT_CACHE_DIR / "csi_processing"
RESULTS_ROOT = RESULTS_DIR

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
_RUN_ID_PATTERN = re.compile(
    r"^(?P<family>ml|dl)__(?P<model>[a-z0-9_]+)__(?P<band>2_4ghz|5ghz|fusion)"
    r"__(?P<split>block|lovo|cross_session|group|random)"
    r"__(?P<normalization>none|zscore|minmax|packet_minmax|ebl-(?:session|user|global))"
    r"(?:__s(?P<seed>\d+))?__(?P<hash6>[0-9a-f]{6})$"
)


class _StaleFeatureCache(ValueError):
    """Raised when an existing cache predates the current feature schema."""


def csi_cache_path(
    file_path: str | Path,
    processor_version: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the content-addressed cache path for one CSI CSV file."""
    path = Path(file_path)
    stat = path.stat()
    identity = "|".join(
        [
            processor_version,
            os.path.normcase(str(path.resolve())),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ]
    )
    cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    root = CSI_CACHE_DIR if cache_dir is None else Path(cache_dir)
    return root / processor_version / f"{cache_key}.pkl"


def load_csi_cache(cache_file: Path) -> object | None:
    """Load one cached CSI-processing payload, returning ``None`` if it is stale."""
    try:
        with cache_file.open("rb") as file:
            return pickle.load(file)  # noqa: S301
    except (EOFError, ImportError, ModuleNotFoundError, OSError, pickle.PickleError):
        return None


def save_csi_cache(cache_file: Path, payload: object) -> None:
    """Atomically save one CSI-processing payload."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{cache_file.stem}.",
        suffix=".tmp",
        dir=cache_file.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(cache_file)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_window_array_cache(
    array_paths: dict[str, Path],
    meta_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]] | None:
    """Load cached DL windows, metadata, and their manifest when complete."""
    if not (
        meta_path.exists()
        and manifest_path.exists()
        and all(path.exists() for path in array_paths.values())
    ):
        return None
    arrays = {band: np.load(path, mmap_mode="r") for band, path in array_paths.items()}
    metadata = pd.read_parquet(meta_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return arrays, metadata, manifest


def open_window_array_writers(
    array_paths: dict[str, Path],
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    """Open float16 memory-mapped writers for DL window arrays."""
    return {
        band: np.lib.format.open_memmap(
            array_paths[band],
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        for band, shape in shapes.items()
    }


def save_window_array_metadata(
    metadata: pd.DataFrame,
    meta_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    """Atomically save DL window metadata and write its descriptive manifest."""
    temporary_meta = meta_path.with_suffix(".parquet.tmp")
    metadata.to_parquet(temporary_meta, index=False)
    temporary_meta.replace(meta_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_window_arrays(array_paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Reopen saved DL window arrays as read-only memory maps."""
    return {band: np.load(path, mmap_mode="r") for band, path in array_paths.items()}


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


def make_run_id(
    *,
    family: str,
    model: str,
    band: str,
    split: str,
    normalization: str,
    config: dict[str, Any],
    baseline_scope: str | None = None,
    seed: int | None = None,
) -> str:
    """Return the readable, content-addressed identifier for one experiment run."""
    normalized_family = family.strip().lower()
    if normalized_family not in {"ml", "dl"}:
        raise ValueError("family must be 'ml' or 'dl'.")
    normalized_model = _band_stem(model)
    normalized_band = _band_stem(band)
    if normalized_band not in {"2_4ghz", "5ghz", "fusion"}:
        raise ValueError(f"Unsupported band for run_id: {band!r}.")
    normalized_split = split.strip().lower()
    if normalized_split not in {"block", "lovo", "cross_session", "group", "random"}:
        raise ValueError(f"Unsupported split for run_id: {split!r}.")
    normalization_stem = _normalization_stem(normalization, baseline_scope)
    config_hash = hashlib.sha256(
        json.dumps(_json_normalized(config), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:6]
    seed_stem = f"__s{int(seed)}" if seed is not None else ""
    return (
        f"{normalized_family}__{normalized_model}__{normalized_band}__{normalized_split}"
        f"__{normalization_stem}{seed_stem}__{config_hash}"
    )


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Parse a run identifier created by :func:`make_run_id`."""
    match = _RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError(f"Invalid run_id: {run_id!r}.")
    parsed: dict[str, Any] = match.groupdict()
    parsed["seed"] = int(parsed["seed"]) if parsed["seed"] is not None else None
    normalization = str(parsed.pop("normalization"))
    if normalization.startswith("ebl-"):
        parsed["normalization"] = "empty_baseline"
        parsed["baseline_scope"] = normalization.removeprefix("ebl-")
    else:
        parsed["normalization"] = normalization
        parsed["baseline_scope"] = None
    return parsed


def get_cache_path(preproc_opts: dict[str, Any], feat_opts: dict[str, Any]) -> Path:
    """Return the feature-cache directory for preprocessing and feature options."""
    return (
        CACHE_DIR
        / f"preproc={make_preproc_key(preproc_opts)}"
        / f"feat={make_feat_key(feat_opts)}"
    )


def get_results_path() -> Path:
    """Return the shared results root."""
    return RESULTS_ROOT


def predictions_path(
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
    *,
    fold: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Return the parquet path for a model/band/split prediction dataframe."""
    if run_id is not None:
        parse_run_id(run_id)
        stem = run_id
        if fold is not None:
            fold_value = str(fold).removeprefix("user-")
            stem = f"{stem}__fold-{_band_stem(fold_value)}"
    else:
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
    run_id: str | None = None,
) -> Path:
    """Persist prediction rows as parquet for analysis without retraining."""
    path = predictions_path(
        results_dir, model, band, split_mode, fold=fold, run_id=run_id
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    if metadata is None:
        df.to_parquet(tmp, index=False)
    else:
        _write_prediction_parquet_with_metadata(df, tmp, metadata)
    tmp.replace(path)
    if metadata is not None and run_id is None:
        _write_prediction_metadata(path, metadata)
    print(f"[predictions] Saved {path}")
    return path


def load_predictions(
    results_dir: Path,
    model: str,
    band: str,
    split_mode: str,
    *,
    fold: str | None = None,
    expected_metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> pd.DataFrame | None:
    """Load persisted predictions, returning None when no cache file exists."""
    path = predictions_path(
        results_dir, model, band, split_mode, fold=fold, run_id=run_id
    )
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
    train_fingerprint: str | None = None,
    test_fingerprint: str | None = None,
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
    if train_fingerprint is not None or test_fingerprint is not None:
        if train_fingerprint is None or test_fingerprint is None:
            raise ValueError("train_fingerprint and test_fingerprint must be provided together.")
        metadata.update(
            {
                "schema_version": 3,
                "random_state": random_state,
                "row_spacing": row_spacing,
                "column_spacing": column_spacing,
                "train_fingerprint": train_fingerprint,
                "test_fingerprint": test_fingerprint,
            }
        )
    return metadata


# ── Internal helpers ──────────────────────────────────────────────────────────

def _band_stem(band: str) -> str:
    """'2.4 GHz' → '2_4ghz',  '5 GHz' → '5ghz',  'Fusion' → 'fusion'."""
    return band.lower().replace(".", "_").replace(" ", "")


def _normalization_stem(normalization: str, baseline_scope: str | None) -> str:
    """Encode normalization and optional baseline scope in a run identifier."""
    normalized = str(normalization or "none").strip().lower()
    if normalized == "empty_baseline":
        scope = str(baseline_scope or "session").strip().lower().removeprefix("per_")
        if scope not in {"session", "user", "global"}:
            raise ValueError(
                "baseline_scope must be session, user, or global for empty_baseline."
            )
        return f"ebl-{scope}"
    if normalized not in {"none", "zscore", "minmax", "packet_minmax"}:
        raise ValueError(f"Unsupported normalization for run_id: {normalization!r}.")
    return normalized


def _prediction_metadata_path(path: Path) -> Path:
    """Return the JSON sidecar path for a prediction parquet file."""
    return path.with_suffix(path.suffix + ".metadata.json")


def _write_prediction_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically write prediction-cache metadata to its JSON sidecar."""
    metadata_path = _prediction_metadata_path(path)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_normalized(metadata), indent=2), encoding="utf-8")
    tmp.replace(metadata_path)


def _write_prediction_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write predictions with embedded metadata when PyArrow is available."""
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
    """Read prediction metadata from the JSON sidecar or parquet schema."""
    metadata_path = _prediction_metadata_path(path)
    if not metadata_path.exists():
        return _read_prediction_parquet_metadata(path)
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _read_prediction_parquet_metadata(path)


def _read_prediction_parquet_metadata(path: Path) -> dict[str, Any] | None:
    """Read embedded prediction metadata from a parquet schema."""
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
    """Return whether cached prediction metadata matches all expected values."""
    observed = _read_prediction_metadata(path)
    if observed is None:
        return False
    expected = _json_normalized(expected_metadata)
    mismatches = [
        key for key, value in expected.items() if observed.get(key) != value
    ]
    for side in ("train", "test"):
        key = f"{side}_fingerprint"
        if key in mismatches:
            print(f"[predictions cache mismatch] {side} fingerprint changed: {path}")
    other_mismatches = [
        key for key in mismatches if key not in {"train_fingerprint", "test_fingerprint"}
    ]
    if other_mismatches:
        print(
            f"[predictions cache mismatch] metadata changed ({', '.join(other_mismatches)}): "
            f"{path}"
        )
    return not mismatches


def _json_normalized(payload: Any) -> Any:
    """Convert a payload to a stable JSON-compatible value."""
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _check_schema(df: pd.DataFrame, path: Path) -> None:
    """Validate metadata columns and feature dtypes in a cached DataFrame."""
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

    # A cache hit is accepted only when all bands exist and their schemas and
    # window inventories still match the discovered raw data.
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

    # Rebuild the related DataFrames together so fusion and single-band caches
    # cannot silently represent different preprocessing runs.
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
    """Describe trial or window-inventory differences in feature caches."""
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

    # Validate a candidate cache before returning it to the ML pipeline.
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

# ── Reproducibility manifest ──────────────────────────────────────────────────
