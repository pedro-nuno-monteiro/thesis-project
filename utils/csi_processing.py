from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
BaselineKey = tuple[str, ...]
CalibrationMode = Literal["none", "packet_norm", "rssi"]

DEFAULT_EPSILON = 1e-8
EXPECTED_MAGNITUDE_DIMS = 2
EMPTY_ROOM_LOCATION = "Z-0"
VALID_BASELINE_SCOPES = ("per_session", "per_user", "global")
VALID_CALIBRATION_MODES: tuple[CalibrationMode, ...] = ("none", "packet_norm", "rssi")


def select_active_subcarriers(
    complex_csi: np.ndarray,
    *,
    its5ghz: bool,
) -> np.ndarray:
    """Apply the existing FFT shift and active-subcarrier selection."""
    if its5ghz:
        return np.delete(complex_csi, [28], axis=1)

    fft_csi = np.fft.fftshift(complex_csi, axes=1)
    active_subcarriers = fft_csi[:, 6:58]
    return np.delete(active_subcarriers, [26, 27], axis=1)


def calculate_csi_magnitude(complex_csi: np.ndarray) -> np.ndarray:
    """Return CSI magnitude without changing packet or subcarrier ordering."""
    return np.abs(complex_csi)


def process_complex_csi(
    complex_csi: np.ndarray,
    rssi_dbm: np.ndarray,
    *,
    its5ghz: bool,
    calibration_mode: CalibrationMode = "none",
    min_rssi_dbm: float = -95.0,
    calibration_eps: float = 1e-12,
) -> tuple[np.ndarray, int, bool]:
    """Select subcarriers, filter invalid packets, calibrate, and take magnitude."""
    selected_csi = select_active_subcarriers(complex_csi, its5ghz=its5ghz)
    filtered_csi, filtered_rssi, invalid_packets_removed = filter_packets_for_calibration(
        selected_csi,
        rssi_dbm,
        calibration_mode=calibration_mode,
        min_rssi_dbm=min_rssi_dbm,
        eps=calibration_eps,
    )
    calibrated_csi, calibration_applied = calibrate_complex_csi(
        filtered_csi,
        calibration_mode=calibration_mode,
        rssi_dbm=filtered_rssi if calibration_mode == "rssi" else None,
        eps=calibration_eps,
    )
    magnitude = calculate_csi_magnitude(calibrated_csi)
    return magnitude, invalid_packets_removed, calibration_applied and magnitude.shape[0] > 0


def filter_packets_for_calibration(
    complex_csi: np.ndarray,
    rssi_dbm: np.ndarray,
    *,
    calibration_mode: CalibrationMode,
    min_rssi_dbm: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove packets that cannot support the requested calibration."""
    if calibration_mode == "none":
        return complex_csi, rssi_dbm, 0

    valid_mask = valid_packet_mask(
        complex_csi,
        rssi_dbm=rssi_dbm,
        calibration_mode=calibration_mode,
        min_rssi_dbm=min_rssi_dbm,
        eps=eps,
    )
    removed = int(valid_mask.size - np.count_nonzero(valid_mask))
    return complex_csi[valid_mask], rssi_dbm[valid_mask], removed


def valid_packet_mask(
    complex_csi: np.ndarray,
    *,
    rssi_dbm: np.ndarray | None,
    calibration_mode: CalibrationMode,
    min_rssi_dbm: float,
    eps: float,
) -> np.ndarray:
    """Return packets suitable for calibration."""
    if calibration_mode not in VALID_CALIBRATION_MODES:
        raise calibration_mode_error()
    if calibration_mode == "rssi" and rssi_dbm is None:
        raise ValueError("RSSI values are required when calibration_mode='rssi'.")
    if rssi_dbm is not None and complex_csi.shape[0] != rssi_dbm.shape[0]:
        msg = (
            "Packet filtering requires one RSSI value per CSI packet: "
            f"got {complex_csi.shape[0]} CSI packets and {rssi_dbm.shape[0]} RSSI values."
        )
        raise ValueError(msg)

    csi_power = complex_csi_power(complex_csi)
    mask = np.isfinite(complex_csi).all(axis=1)
    mask &= np.isfinite(csi_power)
    mask &= csi_power > eps
    if calibration_mode == "rssi" and rssi_dbm is not None:
        mask &= np.isfinite(rssi_dbm)
        mask &= rssi_dbm >= min_rssi_dbm
    return mask


def calibrate_complex_csi(
    complex_csi: np.ndarray,
    *,
    calibration_mode: CalibrationMode,
    rssi_dbm: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Apply packet-power or RSSI-based magnitude compensation."""
    if calibration_mode == "none":
        return complex_csi, False
    if calibration_mode == "packet_norm":
        return _normalize_complex_csi_per_packet(complex_csi, eps=eps), True
    if calibration_mode == "rssi":
        if rssi_dbm is None:
            raise ValueError("RSSI values are required when calibration_mode='rssi'.")
        return _calibrate_complex_csi_with_rssi(complex_csi, rssi_dbm, eps=eps), True
    raise calibration_mode_error()


def complex_csi_power(complex_csi: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(complex_csi) ** 2, axis=1)


def _normalize_complex_csi_per_packet(
    complex_csi: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    norm = np.sqrt(complex_csi_power(complex_csi) + eps)
    return complex_csi / norm[:, None]


def _calibrate_complex_csi_with_rssi(
    complex_csi: np.ndarray,
    rssi_dbm: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    if complex_csi.shape[0] != rssi_dbm.shape[0]:
        msg = (
            "RSSI calibration requires one RSSI value per CSI packet: "
            f"got {complex_csi.shape[0]} CSI packets and {rssi_dbm.shape[0]} RSSI values."
        )
        raise ValueError(msg)
    rssi_mw = 10 ** (rssi_dbm / 10.0)
    scale = np.sqrt(rssi_mw / (complex_csi_power(complex_csi) + eps))
    return complex_csi * scale[:, None]


def calibration_mode_error() -> ValueError:
    return ValueError("calibration_mode must be one of: 'none', 'packet_norm', 'rssi'")


def window_count_for_magnitude(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
) -> int:
    """Return the number of complete sliding windows in a magnitude array."""
    validate_window_parameters(
        magnitude,
        window_size=window_size,
        overlap_size=overlap_size,
    )
    if magnitude.shape[0] < window_size or magnitude.shape[1] == 0:
        return 0
    return 1 + (magnitude.shape[0] - window_size) // (window_size - overlap_size)


def iter_magnitude_windows(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
) -> Iterator[np.ndarray]:
    """Yield complete magnitude windows in packet order."""
    validate_window_parameters(
        magnitude,
        window_size=window_size,
        overlap_size=overlap_size,
    )
    step = window_size - overlap_size
    for start in range(0, magnitude.shape[0] - window_size + 1, step):
        yield magnitude[start : start + window_size]


def validate_window_parameters(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
) -> None:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative")
    if overlap_size >= window_size:
        raise ValueError("overlap_size must be smaller than window_size")
    if magnitude.ndim != EXPECTED_MAGNITUDE_DIMS:
        raise ValueError("magnitude must be a 2D array: samples x subcarriers")


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
    baseline_scope: str = "per_session",
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
                            trial_key=trial_key,
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
                        for trial_key, magnitude in trials_map.items():
                            if magnitude is not None:
                                occupied_pairs.add(
                                    _baseline_key(
                                        baseline_scope,
                                        user_key,
                                        trial_key,
                                        esp_key,
                                    )
                                )
                        continue

                    for trial_key, magnitude in trials_map.items():
                        if magnitude is None:
                            continue
                        magnitude_array = np.asarray(magnitude, dtype=np.float32)
                        if magnitude_array.ndim != EXPECTED_MAGNITUDE_DIMS:
                            msg = "Empty-room baseline entries must be 2D arrays."
                            raise ValueError(msg)
                        key = _baseline_key(
                            baseline_scope,
                            user_key,
                            trial_key,
                            esp_key,
                        )
                        empty_arrays.setdefault(key, []).append(magnitude_array)
                        file_counts[key] = file_counts.get(key, 0) + 1
                        total_files += 1
                        trial = _normalized_key(trial_key)
                        trial_key_tuple = (user, esp, trial)
                        trial_counts[trial_key_tuple] = trial_counts.get(trial_key_tuple, 0) + 1

    missing = sorted(occupied_pairs - set(empty_arrays))
    missing_error: str | None = None
    if missing:
        if baseline_scope == "per_session":
            missing_text = ", ".join(
                f"(user={key[0]}, trial={key[1]}, esp={key[2]})" for key in missing
            )
        elif baseline_scope == "per_user":
            missing_text = ", ".join(f"(user={key[0]}, esp={key[1]})" for key in missing)
        else:
            missing_text = ", ".join(f"(esp={key[0]})" for key in missing)
        missing_error = f"Missing empty-room Z-0 baselines for {missing_text}."

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

    if missing_error is not None:
        raise ValueError(missing_error)

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
        print("  Z-0 count per (user, trial):")
        per_session_counts: dict[tuple[str, str], int] = {}
        for (user, _, trial), count in baseline_tables.trial_counts.items():
            session_key = (user, trial)
            per_session_counts[session_key] = per_session_counts.get(session_key, 0) + count
        for (user, trial), count in sorted(per_session_counts.items()):
            print(f"    (user={user}, trial={trial}): {count}")
        print("  breakdown by user / esp / trial:")
        for (user, esp, trial), count in sorted(baseline_tables.trial_counts.items()):
            print(f"    user={user} esp={esp} trial={trial}: {count}")
    else:
        print("  breakdown by user / esp / trial: none")

    print("  baseline coverage for occupied recordings:")
    for key in occupied_pairs:
        if baseline_scope == "per_session":
            label = f"user={key[0]} trial={key[1]} esp={key[2]}"
        elif baseline_scope == "per_user":
            label = f"user={key[0]} esp={key[1]}"
        else:
            label = f"esp={key[0]}"
        print(f"    {label}: {'yes' if key in available_keys else 'no'}")

    print(f"  {baseline_scope} scope fully satisfiable: {'yes' if satisfiable else 'no'}")
    print("  deep-fade clipped subcarriers:")
    for key, count in sorted(baseline_tables.clip_counts.items()):
        if baseline_scope == "per_session":
            label = f"user={key[0]} trial={key[1]} esp={key[2]}"
        elif baseline_scope == "per_user":
            label = f"user={key[0]} esp={key[1]}"
        else:
            label = f"esp={key[0]}"
        print(f"    {label}: {count}")


def _baseline_for_recording(
    baseline_tables: EmptyBaselineTables | None,
    *,
    baseline_scope: str,
    user_key: str,
    trial_key: str,
    esp_key: str,
    magnitude: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if baseline_tables is None:
        return None, None

    key = _baseline_key(baseline_scope, user_key, trial_key, esp_key)
    if key not in baseline_tables.mean:
        label = (
            f"user={_normalized_key(user_key)} trial={_normalized_key(trial_key)} "
            f"esp={_normalized_key(esp_key)}"
        )
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
                for esp_key, trials_map in esps_map.items():
                    for trial_key, magnitude in trials_map.items():
                        if magnitude is not None:
                            occupied.add(
                                _baseline_key(
                                    baseline_scope,
                                    user_key,
                                    trial_key,
                                    esp_key,
                                )
                            )
    return occupied


def _baseline_key(
    baseline_scope: str,
    user_key: str,
    trial_key: str,
    esp_key: str,
) -> BaselineKey:
    user = _normalized_key(user_key)
    trial = _normalized_key(trial_key)
    esp = _normalized_key(esp_key)
    if baseline_scope == "per_session":
        return (user, trial, esp)
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
