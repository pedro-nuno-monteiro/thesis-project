from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from utils.config import (
    EMPTY_ROOM_LOCATION,
    ESP_IDS_BY_BAND,
    ROOM_1_COLUMNS,
    ROOM_2_A_COLUMNS,
    ROOM_2_BC_COLUMNS,
    ROOM_3_EF_COLUMNS,
)
from utils.csi_processing import (
    iter_magnitude_windows,
    validate_window_parameters,
    window_count_for_magnitude,
)

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
FeatureScenario = Literal["2.4 GHz", "5 GHz", "Fusion"]

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


@dataclass(frozen=True)
class WindowGroup:
    scenario_key: str
    location_key: str
    user_key: str
    trial_key: str
    selected_esp_keys: list[str]
    magnitudes_by_esp: dict[str, np.ndarray]
    min_windows: int
    label: int
    group_id: str


# This function maps a location key to a room label (0, 1, 2, or 3)
# based on the location's row and column.
def room_label_for_location(location_key: str) -> int | None:
    location = location_key.removeprefix("location_").upper()
    if location == EMPTY_ROOM_LOCATION:
        return None

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
) -> np.ndarray:
    validate_window_parameters(magnitude, window_size=window_size, overlap_size=overlap_size)

    subcarrier_count = magnitude.shape[1]
    window_count = window_count_for_magnitude(
        magnitude,
        window_size=window_size,
        overlap_size=overlap_size,
    )
    if window_count == 0:
        return np.empty((0, subcarrier_count * len(FEATURE_NAMES)), dtype=float)

    features = np.empty((window_count, subcarrier_count * len(FEATURE_NAMES)), dtype=float)

    for window_idx, window in enumerate(
        iter_magnitude_windows(
            magnitude,
            window_size=window_size,
            overlap_size=overlap_size,
        )
    ):
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


# This function builds a feature dataframe for a given
# frequency scenario by iterating through the magnitude
# data and extracting features for each trial that matches the scenario.
def build_frequency_feature_dataframe(  # noqa: C901, PLR0913
    magnitude_data: CsiMap,
    frequency_scenario: FeatureScenario,
    *,
    window_size: int = 60,
    overlap_size: int = 30,
    require_all_esps: bool = True,
) -> pd.DataFrame:
    esp_keys = _esp_keys_for_scenario(frequency_scenario)
    feature_columns: list[str] = []
    known_feature_columns: set[str] = set()
    rows: list[dict[str, object]] = []

    for group in iter_window_groups(
        magnitude_data,
        frequency_scenario,
        window_size=window_size,
        overlap_size=overlap_size,
        require_all_esps=require_all_esps,
    ):
        esp_features = {
            esp_key: compute_window_features(
                magnitude,
                window_size=window_size,
                overlap_size=overlap_size,
            )
            for esp_key, magnitude in group.magnitudes_by_esp.items()
        }

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
                group.min_windows,
                frequency_scenario=frequency_scenario,
                scenario_key=group.scenario_key,
                location_key=group.location_key,
                user_key=group.user_key,
                trial_key=group.trial_key,
                label=group.label,
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
    require_all_esps: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_24ghz = build_frequency_feature_dataframe(
        magnitude_data,
        "2.4 GHz",
        window_size=window_size,
        overlap_size=overlap_size,
        require_all_esps=require_all_esps,
    )
    df_5ghz = build_frequency_feature_dataframe(
        magnitude_data,
        "5 GHz",
        window_size=window_size,
        overlap_size=overlap_size,
        require_all_esps=require_all_esps,
    )
    df_fusion = build_frequency_feature_dataframe(
        magnitude_data,
        "Fusion",
        window_size=window_size,
        overlap_size=overlap_size,
        require_all_esps=require_all_esps,
    )
    return df_24ghz, df_5ghz, df_fusion


def _esp_keys_for_scenario(frequency_scenario: FeatureScenario) -> tuple[str, ...]:
    if frequency_scenario not in ESP_IDS_BY_BAND:
        msg = (
            f"Unknown frequency_scenario {frequency_scenario!r}. "
            f"Expected one of {sorted(ESP_IDS_BY_BAND)}. "
            "Note: display names ('2.4 GHz') are scenario identifiers; "
            "slugs ('2_4ghz') are for cache paths only."
        )
        raise ValueError(msg)
    return tuple(f"esp_{esp_id:02d}" for esp_id in ESP_IDS_BY_BAND[frequency_scenario])


def iter_window_groups(
    magnitude_data: CsiMap,
    frequency_scenario: FeatureScenario,
    *,
    window_size: int,
    overlap_size: int,
    require_all_esps: bool,
) -> Iterator[WindowGroup]:
    esp_keys = _esp_keys_for_scenario(frequency_scenario)

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
                    magnitudes_by_esp: dict[str, np.ndarray] = {}
                    window_counts: list[int] = []
                    for esp_key in selected_esp_keys:
                        magnitude = esps_map[esp_key].get(trial_key)
                        if magnitude is None:
                            continue
                        window_count = window_count_for_magnitude(
                            magnitude,
                            window_size=window_size,
                            overlap_size=overlap_size,
                        )
                        if window_count > 0:
                            magnitudes_by_esp[esp_key] = magnitude
                            window_counts.append(window_count)

                    if require_all_esps and len(magnitudes_by_esp) != len(selected_esp_keys):
                        continue
                    if not magnitudes_by_esp:
                        continue

                    min_windows = min(window_counts)
                    if min_windows == 0:
                        continue

                    group_id = f"{scenario_key}_{location_key}_{user_key}_{trial_key}"
                    yield WindowGroup(
                        scenario_key=scenario_key,
                        location_key=location_key,
                        user_key=user_key,
                        trial_key=trial_key,
                        selected_esp_keys=selected_esp_keys,
                        magnitudes_by_esp=magnitudes_by_esp,
                        min_windows=min_windows,
                        label=label,
                        group_id=group_id,
                    )


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


# This function extracts features for each ESP in the given trial and returns a dictionary
# mapping ESP keys to their corresponding feature arrays. The features are computed using the
# compute_window_features function, which applies a sliding window to the magnitude data and
# computes statistical features for each window. The function takes parameters for window size,
# overlap size.
def _features_by_esp(  # noqa: PLR0913
    esps_map: dict[str, dict[str, np.ndarray]],
    selected_esp_keys: list[str],
    trial_key: str,
    *,
    window_size: int,
    overlap_size: int,
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
