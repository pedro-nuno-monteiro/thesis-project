from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from utils.cache import get_results_path
from utils.feature_pipeline import (
    METADATA_COLUMNS,
    CsiMap,
    FeatureScenario,
    _esp_keys_for_scenario,
    iter_window_groups,
)

BandName = Literal["2.4 GHz", "5 GHz", "Fusion"]

BAND_TO_SCENARIO: dict[BandName, FeatureScenario] = {
    "2.4 GHz": "2.4ghz",
    "5 GHz": "5ghz",
    "Fusion": "fusion",
}
BAND_ANCHOR_RANGES: dict[str, tuple[str, ...]] = {
    "2.4 GHz": _esp_keys_for_scenario("2.4ghz"),
    "5 GHz": _esp_keys_for_scenario("5ghz"),
}
DEFAULT_PREPROC_OPTS = {
    "normalization": "empty_baseline",
    "baseline_scope": "per_user",
}


def build_frequency_window_arrays(
    magnitude_data: CsiMap,
    frequency_scenario: BandName | FeatureScenario,
    *,
    window_size: int = 60,
    overlap_size: int = 30,
    require_all_esps: bool = True,
    preproc_opts: dict[str, Any] | None = None,
    force_rebuild: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build or load mmap-backed raw CSI amplitude windows for one band config."""
    band_name, scenario_key = _normalize_frequency_scenario(frequency_scenario)
    feat_opts = {
        "window_size": window_size,
        "overlap_size": overlap_size,
        "require_all_esps": require_all_esps,
    }
    results_dir = get_results_path(preproc_opts or DEFAULT_PREPROC_OPTS, feat_opts)
    cache_dir = results_dir / "window_arrays" / _band_stem(band_name)
    print(f"[window arrays] cache path: {cache_dir.resolve()}")

    bands = ["2.4 GHz", "5 GHz"] if band_name == "Fusion" else [band_name]
    array_paths = {band: cache_dir / f"{_band_stem(band)}.npy" for band in bands}
    meta_path = cache_dir / "meta.parquet"
    manifest_path = cache_dir / "manifest.json"

    if (
        not force_rebuild
        and meta_path.exists()
        and manifest_path.exists()
        and all(path.exists() for path in array_paths.values())
    ):
        arrays = {band: np.load(path, mmap_mode="r") for band, path in array_paths.items()}
        meta = pd.read_parquet(meta_path)
        _print_loaded_arrays(arrays, manifest_path)
        return arrays, meta

    groups = list(
        iter_window_groups(
            magnitude_data,
            scenario_key,
            window_size=window_size,
            overlap_size=overlap_size,
            require_all_esps=require_all_esps,
        )
    )
    total_windows = int(sum(group.min_windows for group in groups))
    if total_windows == 0:
        msg = f"No windows found for {band_name}."
        raise ValueError(msg)

    anchor_order = {band: sorted(BAND_ANCHOR_RANGES[band]) for band in bands}
    subcarrier_counts = _subcarrier_counts(groups, anchor_order)
    shapes = {
        band: (
            total_windows,
            len(anchor_order[band]),
            subcarrier_counts[band],
            window_size,
        )
        for band in bands
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        band: np.lib.format.open_memmap(
            array_paths[band],
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        for band, shape in shapes.items()
    }
    rows: list[dict[str, object]] = []
    step = window_size - overlap_size
    row_idx = 0

    for group in groups:
        for window_idx in range(group.min_windows):
            start = window_idx * step
            for band in bands:
                for anchor_idx, esp_key in enumerate(anchor_order[band]):
                    magnitude = group.magnitudes_by_esp[esp_key]
                    window = magnitude[start : start + window_size]
                    writers[band][row_idx, anchor_idx] = window.T.astype(
                        np.float16,
                        copy=False,
                    )
            rows.append(
                {
                    "frequency_scenario": scenario_key,
                    "scenario": group.scenario_key.removeprefix("scenario_"),
                    "location": group.location_key.removeprefix("location_"),
                    "user": group.user_key.removeprefix("user_"),
                    "trial": group.trial_key.removeprefix("trial_"),
                    "group_id": group.group_id,
                    "window_idx": window_idx,
                    "label": group.label,
                }
            )
            row_idx += 1

    for writer in writers.values():
        writer.flush()
        del writer

    meta = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
    tmp_meta = meta_path.with_suffix(".parquet.tmp")
    meta.to_parquet(tmp_meta, index=False)
    tmp_meta.replace(meta_path)

    manifest = {
        "frequency_scenario": band_name,
        "feature_scenario": scenario_key,
        "window_size": window_size,
        "overlap_size": overlap_size,
        "require_all_esps": require_all_esps,
        "preprocessing": preproc_opts or DEFAULT_PREPROC_OPTS,
        "anchor_order": anchor_order,
        "subcarrier_counts": subcarrier_counts,
        "shapes": {band: list(shape) for band, shape in shapes.items()},
        "dtype": "float16",
        "meta_rows": int(len(meta)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    arrays = {band: np.load(path, mmap_mode="r") for band, path in array_paths.items()}
    _print_array_summary(arrays, anchor_order)
    return arrays, meta


def _normalize_frequency_scenario(
    frequency_scenario: BandName | FeatureScenario,
) -> tuple[BandName, FeatureScenario]:
    aliases: dict[str, tuple[BandName, FeatureScenario]] = {
        "2.4 ghz": ("2.4 GHz", "2.4ghz"),
        "2.4ghz": ("2.4 GHz", "2.4ghz"),
        "5 ghz": ("5 GHz", "5ghz"),
        "5ghz": ("5 GHz", "5ghz"),
        "fusion": ("Fusion", "fusion"),
    }
    key = str(frequency_scenario).strip().lower()
    try:
        return aliases[key]
    except KeyError as exc:
        msg = f"Unknown frequency scenario {frequency_scenario!r}."
        raise ValueError(msg) from exc


def _subcarrier_counts(
    groups: list,
    anchor_order: dict[str, list[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for band, esp_keys in anchor_order.items():
        for esp_key in esp_keys:
            for group in groups:
                magnitude = group.magnitudes_by_esp.get(esp_key)
                if magnitude is not None:
                    counts[band] = int(magnitude.shape[1])
                    break
            if band in counts:
                break
        if band not in counts:
            msg = f"Could not infer subcarrier count for {band}."
            raise ValueError(msg)
    return counts


def _print_loaded_arrays(arrays: dict[str, np.ndarray], manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[window arrays cache hit] {manifest_path.parent.resolve()}")
    _print_array_summary(arrays, manifest["anchor_order"])


def _print_array_summary(
    arrays: dict[str, np.ndarray],
    anchor_order: dict[str, list[str]],
) -> None:
    for band, array in arrays.items():
        print(f"[window arrays] {band}: shape={tuple(array.shape)}, dtype={array.dtype}")
        print(f"[window arrays] {band} anchors: {anchor_order[band]}")


def _band_stem(band: str) -> str:
    return band.lower().replace(".", "_").replace(" ", "")
