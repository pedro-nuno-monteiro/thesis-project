from __future__ import annotations

import numpy as np
import pandas as pd

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
AgcGainMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]

DEFAULT_EPSILON = 1e-8
EXPECTED_MAGNITUDE_DIMS = 2


def get_agc_gains_for_trial(  # noqa: PLR0913
    agc_data: AgcGainMap,
    scenario_key: str,
    location_key: str,
    user_key: str,
    esp_key: str,
    trial_key: str,
) -> np.ndarray | None:
    return (
        agc_data.get(scenario_key, {})
        .get(location_key, {})
        .get(user_key, {})
        .get(esp_key, {})
        .get(trial_key)
    )


def agc_reference_gain(agc_gains: np.ndarray, reference: str | float) -> float:
    if isinstance(reference, (int, float)):
        return float(reference)

    reference = reference.lower()
    if reference == "median":
        return float(np.median(agc_gains))
    if reference == "mean":
        return float(np.mean(agc_gains))
    if reference == "max":
        return float(np.max(agc_gains))
    if reference == "min":
        return float(np.min(agc_gains))

    msg = "agc_reference must be median, mean, max, min, or a numeric value."
    raise ValueError(msg)


def compensate_agc_gain(
    magnitude: np.ndarray,
    agc_gains: np.ndarray | None,
    reference: str | float,
) -> tuple[np.ndarray, bool]:
    if agc_gains is None or len(agc_gains) == 0 or magnitude.size == 0:
        return magnitude, False

    compensated = magnitude.copy()
    valid_count = min(compensated.shape[0], len(agc_gains))
    gains = np.asarray(agc_gains[:valid_count], dtype=float)
    reference_gain = agc_reference_gain(gains, reference)
    scale = 10 ** ((reference_gain - gains) / 20.0)
    compensated[:valid_count] = compensated[:valid_count] * scale[:, np.newaxis]
    return compensated, True


def filter_magnitude(magnitude: np.ndarray, method: str, window: int) -> np.ndarray:
    method = method.lower()
    if method == "none" or window <= 1 or magnitude.size == 0:
        return magnitude

    if window % 2 == 0:
        window += 1

    magnitude_frame = pd.DataFrame(magnitude)
    if method == "moving_average":
        return magnitude_frame.rolling(window, center=True, min_periods=1).mean().to_numpy()
    if method == "median":
        return magnitude_frame.rolling(window, center=True, min_periods=1).median().to_numpy()

    msg = "filter_method must be none, moving_average, or median."
    raise ValueError(msg)


def normalize_magnitude(
    magnitude: np.ndarray,
    method: str,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    method = method.lower()
    if method == "none" or magnitude.size == 0:
        return magnitude

    if method == "center":
        return magnitude - np.mean(magnitude, axis=0, keepdims=True)
    if method == "zscore":
        return (magnitude - np.mean(magnitude, axis=0, keepdims=True)) / (
            np.std(magnitude, axis=0, keepdims=True) + epsilon
        )
    if method == "minmax":
        min_values = np.min(magnitude, axis=0, keepdims=True)
        max_values = np.max(magnitude, axis=0, keepdims=True)
        return (magnitude - min_values) / (max_values - min_values + epsilon)
    if method == "robust":
        median_values = np.median(magnitude, axis=0, keepdims=True)
        q1_values = np.percentile(magnitude, 25, axis=0, keepdims=True)
        q3_values = np.percentile(magnitude, 75, axis=0, keepdims=True)
        return (magnitude - median_values) / (q3_values - q1_values + epsilon)

    msg = "normalization must be none, center, zscore, minmax, or robust."
    raise ValueError(msg)


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


def process_magnitude_data(  # noqa: PLR0913
    raw_magnitude_data: CsiMap,
    agc_data: AgcGainMap,
    *,
    apply_agc_compensation: bool = False,
    agc_reference: str | float = "median",
    filter_method: str = "none",
    filter_window: int = 5,
    normalization: str = "none",
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[CsiMap, pd.DataFrame]:
    processed_data: CsiMap = {}
    summary_rows: list[dict[str, object]] = []

    for scenario_key, locations_map in raw_magnitude_data.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                for esp_key, trials_map in esps_map.items():
                    for trial_key, magnitude in trials_map.items():
                        if magnitude is None:
                            continue

                        trial_magnitude = np.asarray(magnitude, dtype=float).copy()
                        if trial_magnitude.ndim != EXPECTED_MAGNITUDE_DIMS:
                            msg = "Magnitude entries must be 2D arrays."
                            raise ValueError(msg)

                        agc_gains = get_agc_gains_for_trial(
                            agc_data,
                            scenario_key,
                            location_key,
                            user_key,
                            esp_key,
                            trial_key,
                        )
                        if apply_agc_compensation:
                            trial_magnitude, agc_used = compensate_agc_gain(
                                trial_magnitude,
                                agc_gains,
                                agc_reference,
                            )
                        else:
                            agc_used = False

                        trial_magnitude = filter_magnitude(
                            trial_magnitude,
                            filter_method,
                            filter_window,
                        )
                        trial_magnitude = normalize_magnitude(
                            trial_magnitude,
                            normalization,
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
                                "scenario": scenario_key.removeprefix("scenario_"),
                                "location": location_key.removeprefix("location_"),
                                "user": user_key.removeprefix("user_"),
                                "esp": esp_key.removeprefix("esp_"),
                                "trial": trial_key.removeprefix("trial_"),
                                "samples": trial_magnitude.shape[0],
                                "subcarriers": trial_magnitude.shape[1],
                                "agc_compensated": agc_used,
                                "filter": filter_method,
                                "normalization": normalization,
                            },
                        )

    return processed_data, pd.DataFrame(summary_rows)
