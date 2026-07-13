from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
BaselineKey = tuple[str, ...]

DEFAULT_EPSILON = 1e-8
EXPECTED_MAGNITUDE_DIMS = 2
EMPTY_ROOM_LOCATION = "Z-0"
VALID_BASELINE_SCOPES = ("per_user", "global")


@dataclass(frozen=True)
class EmptyBaselineTables:
    mean: dict[BaselineKey, np.ndarray]
    divisor: dict[BaselineKey, np.ndarray]
    clip_counts: dict[BaselineKey, int]
    file_counts: dict[BaselineKey, int]
    trial_counts: dict[tuple[str, str, str], int]
    total_files: int


def _norm_none(
    magnitude: np.ndarray,
    *,
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    return magnitude


def _norm_zscore(
    magnitude: np.ndarray,
    *,
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    return (magnitude - magnitude.mean(axis=0, keepdims=True)) / (
        magnitude.std(axis=0, keepdims=True) + epsilon
    )


def _norm_minmax(
    magnitude: np.ndarray,
    *,
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    min_values = magnitude.min(axis=0, keepdims=True)
    max_values = magnitude.max(axis=0, keepdims=True)
    return (magnitude - min_values) / (max_values - min_values + epsilon)


def _norm_packet_minmax(
    magnitude: np.ndarray,
    *,
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    min_values = magnitude.min(axis=1, keepdims=True)
    max_values = magnitude.max(axis=1, keepdims=True)
    return (magnitude - min_values) / (max_values - min_values + epsilon)


def _norm_empty_baseline(
    magnitude: np.ndarray,
    *,
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    if baseline_mean is None or baseline_absmax is None:
        msg = "empty_baseline normalization requires baseline_mean and baseline_absmax."
        raise ValueError(msg)

    expected_shape = (magnitude.shape[1],)
    if baseline_mean.shape != expected_shape or baseline_absmax.shape != expected_shape:
        msg = (
            "empty_baseline shape mismatch: "
            f"expected {expected_shape}, got baseline_mean={baseline_mean.shape} "
            f"and baseline_absmax={baseline_absmax.shape}."
        )
        raise ValueError(msg)

    return (magnitude - baseline_mean[None, :]) / (baseline_absmax[None, :] + epsilon)


NORMALIZERS: dict[str, Callable[..., np.ndarray]] = {
    "none": _norm_none,
    "zscore": _norm_zscore,
    "minmax": _norm_minmax,
    "packet_minmax": _norm_packet_minmax,
    "empty_baseline": _norm_empty_baseline,
}


def normalize_magnitude(
    magnitude: np.ndarray,
    method: str,
    *,
    baseline_mean: np.ndarray | None = None,
    baseline_absmax: np.ndarray | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    method_key = method.lower()
    if method_key not in NORMALIZERS:
        valid = ", ".join(NORMALIZERS)
        msg = f"Unknown normalization {method!r}. Valid methods: {valid}."
        raise ValueError(msg)
    if magnitude.size == 0:
        return magnitude

    working = np.asarray(magnitude, dtype=np.float32)
    normalized = NORMALIZERS[method_key](
        working,
        baseline_mean=baseline_mean,
        baseline_absmax=baseline_absmax,
        epsilon=epsilon,
    )
    return np.asarray(normalized, dtype=np.float32)


def set_processed_magnitude_entry(  # noqa: PLR0913
    processed_data: CsiMap,
    scenario_key: str,
    location_key: str,
    user_key: str,
    esp_key: str,
    trial_key: str,
    magnitude: np.ndarray,
) -> None:
    processed_data.setdefault(scenario_key, {})
    processed_data[scenario_key].setdefault(location_key, {})
    processed_data[scenario_key][location_key].setdefault(user_key, {})
    processed_data[scenario_key][location_key][user_key].setdefault(esp_key, {})
    processed_data[scenario_key][location_key][user_key][esp_key][trial_key] = magnitude


def process_magnitude_data(
    raw_magnitude_data: CsiMap,
    *,
    normalization: str = "none",
    baseline_scope: str = "per_user",
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[CsiMap, pd.DataFrame]:
    normalization_key = normalization.lower()
    baseline_tables: EmptyBaselineTables | None = None
    if normalization_key == "empty_baseline":
        if baseline_scope == "global":
            print(
                "WARNING: normalization='empty_baseline' with baseline_scope='global' "
                "is expected to produce results identical to normalization='none' for "
                "RF, KNN, and SVM. Run it only as a no-op control."
            )
        baseline_tables = build_empty_baseline_tables(
            raw_magnitude_data,
            baseline_scope=baseline_scope,
            print_inventory=True,
        )
    elif baseline_scope not in VALID_BASELINE_SCOPES:
        _raise_baseline_scope_error(baseline_scope)

    processed_data: CsiMap = {}
    summary_rows: list[dict[str, object]] = []

    for scenario_key, locations_map in raw_magnitude_data.items():
        for location_key, users_map in locations_map.items():
            if _normalized_key(location_key) == EMPTY_ROOM_LOCATION:
                continue

            for user_key, esps_map in users_map.items():
                for esp_key, trials_map in esps_map.items():
                    for trial_key, magnitude in trials_map.items():
                        if magnitude is None:
                            continue

                        trial_magnitude = np.asarray(magnitude, dtype=np.float32).copy()
                        if trial_magnitude.ndim != EXPECTED_MAGNITUDE_DIMS:
                            msg = "Magnitude entries must be 2D arrays."
                            raise ValueError(msg)

                        baseline_mean, baseline_absmax = _baseline_for_recording(
                            baseline_tables,
                            baseline_scope=baseline_scope,
                            user_key=user_key,
                            esp_key=esp_key,
                            magnitude=trial_magnitude,
                        )
                        trial_magnitude = normalize_magnitude(
                            trial_magnitude,
                            normalization,
                            baseline_mean=baseline_mean,
                            baseline_absmax=baseline_absmax,
                            epsilon=epsilon,
                        )
                        set_processed_magnitude_entry(
                            processed_data,
                            scenario_key,
                            location_key,
                            user_key,
                            esp_key,
                            trial_key,
                            trial_magnitude,
                        )
                        summary_rows.append(
                            {
                                "scenario": _normalized_key(scenario_key),
                                "location": _normalized_key(location_key),
                                "user": _normalized_key(user_key),
                                "esp": _normalized_key(esp_key),
                                "trial": _normalized_key(trial_key),
                                "samples": trial_magnitude.shape[0],
                                "subcarriers": trial_magnitude.shape[1],
                                "normalization": normalization_key,
                                "baseline_scope": (
                                    baseline_scope if normalization_key == "empty_baseline" else ""
                                ),
                            },
                        )

    return processed_data, pd.DataFrame(summary_rows)


def build_empty_baseline_tables(
    raw_magnitude_data: CsiMap,
    *,
    baseline_scope: str,
    print_inventory: bool = False,
) -> EmptyBaselineTables:
    if baseline_scope not in VALID_BASELINE_SCOPES:
        _raise_baseline_scope_error(baseline_scope)

    empty_arrays: dict[BaselineKey, list[np.ndarray]] = {}
    occupied_pairs: set[BaselineKey] = set()
    file_counts: dict[BaselineKey, int] = {}
    trial_counts: dict[tuple[str, str, str], int] = {}
    total_files = 0

    for _, locations_map in raw_magnitude_data.items():
        for location_key, users_map in locations_map.items():
            is_empty_room = _normalized_key(location_key) == EMPTY_ROOM_LOCATION
            for user_key, esps_map in users_map.items():
                user = _normalized_key(user_key)
                for esp_key, trials_map in esps_map.items():
                    esp = _normalized_key(esp_key)
                    if not is_empty_room:
                        occupied_pairs.add(_baseline_key(baseline_scope, user_key, esp_key))
                        continue

                    key = _baseline_key(baseline_scope, user_key, esp_key)
                    for trial_key, magnitude in trials_map.items():
                        if magnitude is None:
                            continue
                        magnitude_array = np.asarray(magnitude, dtype=np.float32)
                        if magnitude_array.ndim != EXPECTED_MAGNITUDE_DIMS:
                            msg = "Empty-room baseline entries must be 2D arrays."
                            raise ValueError(msg)
                        empty_arrays.setdefault(key, []).append(magnitude_array)
                        file_counts[key] = file_counts.get(key, 0) + 1
                        total_files += 1
                        trial = _normalized_key(trial_key)
                        trial_key_tuple = (user, esp, trial)
                        trial_counts[trial_key_tuple] = trial_counts.get(trial_key_tuple, 0) + 1

    missing = sorted(occupied_pairs - set(empty_arrays))
    if missing:
        if baseline_scope == "per_user":
            missing_text = ", ".join(f"(user={key[0]}, esp={key[1]})" for key in missing)
        else:
            missing_text = ", ".join(f"(esp={key[0]})" for key in missing)
        msg = f"Missing empty-room Z-0 baselines for {missing_text}."
        raise ValueError(msg)

    mean: dict[BaselineKey, np.ndarray] = {}
    divisor: dict[BaselineKey, np.ndarray] = {}
    clip_counts: dict[BaselineKey, int] = {}
    for key, arrays in empty_arrays.items():
        subcarrier_counts = {array.shape[1] for array in arrays}
        if len(subcarrier_counts) != 1:
            msg = f"Z-0 baseline arrays for key {key} have mismatched shapes: {subcarrier_counts}."
            raise ValueError(msg)
        concatenated = np.concatenate(arrays, axis=0)
        raw_absmax = np.max(np.abs(concatenated), axis=0).astype(np.float32)
        floor = float(np.percentile(raw_absmax, 1))
        clipped_mask = raw_absmax < floor
        mean[key] = concatenated.mean(axis=0).astype(np.float32)
        divisor[key] = np.maximum(raw_absmax, floor).astype(np.float32)
        clip_counts[key] = int(np.count_nonzero(clipped_mask))

    if print_inventory:
        print_z0_inventory_report(
            raw_magnitude_data,
            baseline_scope=baseline_scope,
            baseline_tables=EmptyBaselineTables(
                mean=mean,
                divisor=divisor,
                clip_counts=clip_counts,
                file_counts=file_counts,
                trial_counts=trial_counts,
                total_files=total_files,
            ),
        )

    return EmptyBaselineTables(
        mean=mean,
        divisor=divisor,
        clip_counts=clip_counts,
        file_counts=file_counts,
        trial_counts=trial_counts,
        total_files=total_files,
    )


def print_z0_inventory_report(
    raw_magnitude_data: CsiMap,
    *,
    baseline_scope: str,
    baseline_tables: EmptyBaselineTables,
) -> None:
    occupied_pairs = sorted(_occupied_baseline_keys(raw_magnitude_data, baseline_scope))
    available_keys = set(baseline_tables.mean)
    satisfiable = all(key in available_keys for key in occupied_pairs)

    print("[Z-0 inventory]")
    print(f"  total Z-0 files found: {baseline_tables.total_files}")
    if baseline_tables.trial_counts:
        print("  breakdown by user / esp / trial:")
        for (user, esp, trial), count in sorted(baseline_tables.trial_counts.items()):
            print(f"    user={user} esp={esp} trial={trial}: {count}")
    else:
        print("  breakdown by user / esp / trial: none")

    print("  baseline coverage for occupied recordings:")
    for key in occupied_pairs:
        if baseline_scope == "per_user":
            label = f"user={key[0]} esp={key[1]}"
        else:
            label = f"esp={key[0]}"
        print(f"    {label}: {'yes' if key in available_keys else 'no'}")

    print(f"  per_user scope fully satisfiable: {'yes' if satisfiable else 'no'}")
    print("  deep-fade clipped subcarriers:")
    for key, count in sorted(baseline_tables.clip_counts.items()):
        label = f"user={key[0]} esp={key[1]}" if len(key) == 2 else f"esp={key[0]}"
        print(f"    {label}: {count}")


def _baseline_for_recording(
    baseline_tables: EmptyBaselineTables | None,
    *,
    baseline_scope: str,
    user_key: str,
    esp_key: str,
    magnitude: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if baseline_tables is None:
        return None, None

    key = _baseline_key(baseline_scope, user_key, esp_key)
    if key not in baseline_tables.mean:
        label = f"user={_normalized_key(user_key)} esp={_normalized_key(esp_key)}"
        msg = f"Missing empty-room Z-0 baseline for {label}."
        raise ValueError(msg)

    baseline_mean = baseline_tables.mean[key]
    baseline_absmax = baseline_tables.divisor[key]
    expected_shape = (magnitude.shape[1],)
    if baseline_mean.shape != expected_shape or baseline_absmax.shape != expected_shape:
        msg = (
            "Empty-room baseline subcarrier mismatch for "
            f"esp={_normalized_key(esp_key)}: recording shape={magnitude.shape}, "
            f"baseline_mean shape={baseline_mean.shape}, "
            f"baseline_absmax shape={baseline_absmax.shape}."
        )
        raise ValueError(msg)
    return baseline_mean, baseline_absmax


def _occupied_baseline_keys(raw_magnitude_data: CsiMap, baseline_scope: str) -> set[BaselineKey]:
    occupied: set[BaselineKey] = set()
    for _, locations_map in raw_magnitude_data.items():
        for location_key, users_map in locations_map.items():
            if _normalized_key(location_key) == EMPTY_ROOM_LOCATION:
                continue
            for user_key, esps_map in users_map.items():
                for esp_key in esps_map:
                    occupied.add(_baseline_key(baseline_scope, user_key, esp_key))
    return occupied


def _baseline_key(baseline_scope: str, user_key: str, esp_key: str) -> BaselineKey:
    user = _normalized_key(user_key)
    esp = _normalized_key(esp_key)
    if baseline_scope == "per_user":
        return (user, esp)
    if baseline_scope == "global":
        return (esp,)
    _raise_baseline_scope_error(baseline_scope)


def _raise_baseline_scope_error(baseline_scope: str) -> None:
    valid = ", ".join(VALID_BASELINE_SCOPES)
    msg = f"baseline_scope must be one of: {valid}. Got {baseline_scope!r}."
    raise ValueError(msg)


def _normalized_key(value: str) -> str:
    normalized = value
    for prefix in ("scenario_", "location_", "user_", "esp_", "trial_"):
        normalized = normalized.removeprefix(prefix)
    return normalized
