from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
FeatureScenario = Literal["2.4ghz", "5ghz", "fusion"]

FEATURE_NAMES = ("mean", "std", "variance", "max", "min", "energy")
METADATA_COLUMNS = (
    "frequency_scenario",
    "scenario",
    "location",
    "user",
    "trial",
    "group_id",
    "window_idx",
    "label",
)
EXPECTED_MAGNITUDE_DIMS = 2
ROOM_1_COLUMNS = range(1, 10)
ROOM_2_A_COLUMNS = {13, 14}
ROOM_2_BC_COLUMNS = range(10, 15)
ROOM_3_EF_COLUMNS = range(10, 14)
ESP_IDS_BY_SCENARIO: dict[FeatureScenario, tuple[int, ...]] = {
    "2.4ghz": tuple(range(1, 11)),
    "5ghz": tuple(range(11, 21)),
    "fusion": tuple(range(1, 21)),
}



# This function maps a location key to a room label (0, 1, 2, or 3)
# based on the location's row and column.
def room_label_for_location(location_key: str) -> int | None:
    location = location_key.removeprefix("location_").upper()
    if location == "Z-0":
        return 0

    try:
        row, column_text = location.split("-", maxsplit=1)
        column = int(column_text)
    except ValueError:
        return None

    if row in "ABCDEF" and column in ROOM_1_COLUMNS:
        return 1
    if (row == "A" and column in ROOM_2_A_COLUMNS) or (
        row in "BC"
        and column in ROOM_2_BC_COLUMNS
    ):
        return 2
    if row in "EF" and column in ROOM_3_EF_COLUMNS:
        return 3

    return None


# This function computes statistical features for the given
# magnitude data using a sliding window approach.
def compute_window_features(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
    calibrate: bool = False,
) -> np.ndarray:
    _validate_window_parameters(magnitude, window_size=window_size, overlap_size=overlap_size)

    subcarrier_count = magnitude.shape[1]
    window_count = window_count_for_magnitude(
        magnitude,
        window_size=window_size,
        overlap_size=overlap_size,
    )
    if window_count == 0:
        return np.empty((0, subcarrier_count * len(FEATURE_NAMES)), dtype=float)

    if calibrate:
        magnitude = magnitude - np.mean(magnitude, axis=0, keepdims=True)

    step = window_size - overlap_size
    features = np.empty((window_count, subcarrier_count * len(FEATURE_NAMES)), dtype=float)

    for window_idx, start in enumerate(range(0, magnitude.shape[0] - window_size + 1, step)):
        window = magnitude[start : start + window_size]
        features[window_idx] = np.concatenate(
            [
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.var(window, axis=0),
                np.max(window, axis=0),
                np.min(window, axis=0),
                np.sum(window**2, axis=0),
            ],
        )

    return features


def window_count_for_magnitude(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
) -> int:
    _validate_window_parameters(magnitude, window_size=window_size, overlap_size=overlap_size)
    if magnitude.shape[0] < window_size or magnitude.shape[1] == 0:
        return 0

    step = window_size - overlap_size
    return 1 + (magnitude.shape[0] - window_size) // step


# This function builds a feature dataframe for a given
# frequency scenario by iterating through the magnitude
# data and extracting features for each trial that matches the scenario.
def build_frequency_feature_dataframe(  # noqa: C901, PLR0913
    magnitude_data: CsiMap,
    frequency_scenario: FeatureScenario,
    *,
    window_size: int = 60,
    overlap_size: int = 30,
    calibrate: bool = False,
    require_all_esps: bool = True,
) -> pd.DataFrame:
    esp_keys = _esp_keys_for_scenario(frequency_scenario)
    feature_columns: list[str] = []
    known_feature_columns: set[str] = set()
    rows: list[dict[str, object]] = []

    for scenario_key, locations_map in magnitude_data.items():
        for location_key, users_map in locations_map.items():
            label = room_label_for_location(location_key)
            if label is None:
                continue

            for user_key, esps_map in users_map.items():
                selected_esp_keys = [esp_key for esp_key in esp_keys if esp_key in esps_map]
                if require_all_esps and len(selected_esp_keys) != len(esp_keys):
                    continue
                if not selected_esp_keys:
                    continue

                trial_sets = [
                    {
                        trial_key
                        for trial_key, magnitude in esps_map[esp_key].items()
                        if magnitude is not None
                    }
                    for esp_key in selected_esp_keys
                ]
                common_trials = sorted(set.intersection(*trial_sets))

                for trial_key in common_trials:
                    esp_features = _features_by_esp(
                        esps_map,
                        selected_esp_keys,
                        trial_key,
                        window_size=window_size,
                        overlap_size=overlap_size,
                        calibrate=calibrate,
                    )
                    if require_all_esps and len(esp_features) != len(selected_esp_keys):
                        continue
                    if not esp_features:
                        continue

                    # determine the minimum number of windows across all ESPs for this trial
                    min_windows = min(features.shape[0] for features in esp_features.values())
                    if min_windows == 0:
                        continue

                    # extend the feature columns list with any new
                    # feature columns from this trial's ESPs
                    _extend_feature_columns(
                        feature_columns,
                        known_feature_columns,
                        _feature_columns_for_group(esp_keys, esp_features),
                    )
                    rows.extend(
                        _rows_for_group(
                            esp_keys,
                            esp_features,
                            min_windows,
                            frequency_scenario=frequency_scenario,
                            scenario_key=scenario_key,
                            location_key=location_key,
                            user_key=user_key,
                            trial_key=trial_key,
                            label=label,
                        ),
                    )

    df = pd.DataFrame(rows, columns=[*METADATA_COLUMNS, *feature_columns])
    if feature_columns:
        df[feature_columns] = df[feature_columns].astype("float32")
    return df


# function that is called by the main script
# builds three dataframes, one for each frequency scenario, and returns them as a tuple
def build_frequency_feature_dataframes(
    magnitude_data: CsiMap,
    *,
    window_size: int = 60,
    overlap_size: int = 30,
    calibrate: bool = False,
    require_all_esps: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_24ghz = build_frequency_feature_dataframe(
        magnitude_data,
        "2.4ghz",
        window_size=window_size,
        overlap_size=overlap_size,
        calibrate=calibrate,
        require_all_esps=require_all_esps,
    )
    df_5ghz = build_frequency_feature_dataframe(
        magnitude_data,
        "5ghz",
        window_size=window_size,
        overlap_size=overlap_size,
        calibrate=calibrate,
        require_all_esps=require_all_esps,
    )
    df_fusion = build_frequency_feature_dataframe(
        magnitude_data,
        "fusion",
        window_size=window_size,
        overlap_size=overlap_size,
        calibrate=calibrate,
        require_all_esps=require_all_esps,
    )
    return df_24ghz, df_5ghz, df_fusion


def _esp_keys_for_scenario(frequency_scenario: FeatureScenario) -> tuple[str, ...]:
    return tuple(f"esp_{esp_id:02d}" for esp_id in ESP_IDS_BY_SCENARIO[frequency_scenario])


def _extend_feature_columns(
    feature_columns: list[str],
    known_feature_columns: set[str],
    new_feature_columns: list[str],
) -> None:
    for feature_column in new_feature_columns:
        if feature_column not in known_feature_columns:
            feature_columns.append(feature_column)
            known_feature_columns.add(feature_column)


def _feature_columns_for_group(
    esp_keys: tuple[str, ...],
    esp_features: dict[str, np.ndarray],
) -> list[str]:
    feature_columns: list[str] = []

    for esp_key in esp_keys:
        features = esp_features.get(esp_key)
        if features is None:
            continue

        feature_columns.extend(
            _feature_columns_for_esp(esp_key, _subcarrier_count_from_features(features)),
        )

    return feature_columns


def _feature_columns_for_esp(esp_key: str, subcarrier_count: int) -> list[str]:
    return [
        f"{esp_key}_{feature_name}_sc_{subcarrier_idx:02d}"
        for feature_name in FEATURE_NAMES
        for subcarrier_idx in range(subcarrier_count)
    ]


def _subcarrier_count_from_features(features: np.ndarray) -> int:
    feature_count = len(FEATURE_NAMES)
    if features.shape[1] % feature_count != 0:
        msg = "feature width must be divisible by the number of feature names"
        raise ValueError(msg)
    return features.shape[1] // feature_count


def _validate_window_parameters(
    magnitude: np.ndarray,
    *,
    window_size: int,
    overlap_size: int,
) -> None:
    if window_size <= 0:
        msg = "window_size must be greater than zero"
        raise ValueError(msg)
    if overlap_size < 0:
        msg = "overlap_size cannot be negative"
        raise ValueError(msg)
    if overlap_size >= window_size:
        msg = "overlap_size must be smaller than window_size"
        raise ValueError(msg)
    if magnitude.ndim != EXPECTED_MAGNITUDE_DIMS:
        msg = "magnitude must be a 2D array: samples x subcarriers"
        raise ValueError(msg)

# This function extracts features for each ESP in the given trial and returns a dictionary
# mapping ESP keys to their corresponding feature arrays. The features are computed using the
# compute_window_features function, which applies a sliding window to the magnitude data and
# computes statistical features for each window. The function takes parameters for window size,
# overlap size, and whether to calibrate the magnitude data before feature extraction.
def _features_by_esp(  # noqa: PLR0913
    esps_map: dict[str, dict[str, np.ndarray]],
    selected_esp_keys: list[str],
    trial_key: str,
    *,
    window_size: int,
    overlap_size: int,
    calibrate: bool,
) -> dict[str, np.ndarray]:
    esp_features: dict[str, np.ndarray] = {}

    for esp_key in selected_esp_keys:
        magnitude = esps_map[esp_key].get(trial_key)
        if magnitude is None:
            continue

        features = compute_window_features(
            magnitude,
            window_size=window_size,
            overlap_size=overlap_size,
            calibrate=calibrate,
        )
        if features.shape[0] > 0:
            esp_features[esp_key] = features

    return esp_features

# This function builds a list of rows for a given group of features
# corresponding to a specific trial.
# Each row corresponds to a window of features and includes metadata
# such as frequency scenario, scenario, location, user, trial, group_id, window index, and label.
# The function iterates over the windows of features for each ESP and constructs a dictionary
# for each window, which is then added to the list of rows.
# The feature values for each ESP are included in the row with
# column names that indicate the ESP, feature name, and sub
def _rows_for_group(  # noqa: PLR0913
    esp_keys: tuple[str, ...],
    esp_features: dict[str, np.ndarray],
    min_windows: int,
    *,
    frequency_scenario: FeatureScenario,
    scenario_key: str,
    location_key: str,
    user_key: str,
    trial_key: str,
    label: int,
) -> list[dict[str, object]]:
    rows = []
    group_id = f"{scenario_key}_{location_key}_{user_key}_{trial_key}"

    for window_idx in range(min_windows):
        row = {
            "frequency_scenario": frequency_scenario,
            "scenario": scenario_key.removeprefix("scenario_"),
            "location": location_key.removeprefix("location_"),
            "user": user_key.removeprefix("user_"),
            "trial": trial_key.removeprefix("trial_"),
            "group_id": group_id,
            "window_idx": window_idx,
            "label": label,
        }
        for esp_key in esp_keys:
            features = esp_features.get(esp_key)
            if features is None:
                continue

            subcarrier_count = _subcarrier_count_from_features(features)
            for feature_idx, feature_name in enumerate(FEATURE_NAMES):
                feature_start = feature_idx * subcarrier_count
                feature_values = features[
                    window_idx,
                    feature_start : feature_start + subcarrier_count,
                ]
                for subcarrier_idx, feature_value in enumerate(feature_values):
                    row[f"{esp_key}_{feature_name}_sc_{subcarrier_idx:02d}"] = feature_value

        rows.append(row)

    return rows
