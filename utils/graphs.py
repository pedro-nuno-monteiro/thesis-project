from __future__ import annotations

import re
from collections.abc import Mapping as MappingABC
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]
CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
MagnitudeSetMap = dict[str, CsiMap]
PairEntry = tuple[str, str, np.ndarray, np.ndarray]
TrialPairGroup = tuple[str, str, str, str, list[PairEntry]]
SelectedEntry = tuple[str, np.ndarray | None]
SelectedTrialGroup = tuple[str, str, str, str, list[SelectedEntry]]
SelectedFileEntry = tuple[str, str]
SelectedFileGroup = tuple[str, str, str, str, list[SelectedFileEntry]]
SelectedMagnitudeSetEntry = tuple[str, np.ndarray | None]
SelectedMagnitudeSetGroup = tuple[str, str, str, str, str, list[SelectedMagnitudeSetEntry]]

LOCATION_PATTERN = re.compile(r"^(?:(?P<letter>[A-G])(?:-)?(?P<number>[1-9]|1[0-4])|Z-?0)$")
RF_METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
RF_SPLIT_ORDER = ("random", "group")
RF_DATASET_ORDER = ("2.4 GHz", "5 GHz", "Fusion")
RF_FEATURE_STATISTIC_ORDER = ("mean", "std", "variance", "max", "min", "energy")
RF_BAND_ORDER = ("2.4 GHz", "5 GHz")
FEATURE_IMPORTANCE_TOP_N = 20
FEATURE_IMPORTANCE_COLUMN_PATTERN = re.compile(
    r"^(?P<esp>esp_\d{2})_(?P<statistic>.+)_sc_(?P<subcarrier>\d+)$",
)
FEATURE_METADATA_COLUMNS = (
    "frequency_scenario",
    "scenario",
    "location",
    "user",
    "trial",
    "group_id",
    "window_idx",
    "label",
)
CONFUSION_MATRIX_DIMS = 2
LOW_FREQUENCY_ESP_IDS = range(1, 11)
HIGH_FREQUENCY_ESP_OFFSET = 10
SPARSE_3D_SUBCARRIER_TICK_COUNT = 8
DB_EPSILON = 1e-12
EMPTY_ROOM_LOCATION_KEY = "location_Z-0"
MAGNITUDE_DIMS = 2
VISUALIZATION_MIN_DB = -80.0
INVALID_AXIS_MESSAGE = "axis must be 'x' or 'y'."
INVALID_STRIDE_MESSAGE = "Strides must be at least 1."
TWO_GHZ_CSI_COLUMN = 24
FIVE_GHZ_CSI_COLUMN = 14
TWO_GHZ_RAW_LENGTH = 128
FIVE_GHZ_RAW_LENGTH = 114
FIVE_GHZ_MIN_ESP_ID = 11
FIVE_GHZ_MAX_ESP_ID = 20
MAX_IQ_LEGEND_ITEMS = 12
RF_REPORT_METRIC_ROWS = {
    "precision_macro": ("macro avg", "precision"),
    "recall_macro": ("macro avg", "recall"),
    "f1_macro": ("macro avg", "f1-score"),
    "precision_weighted": ("weighted avg", "precision"),
    "recall_weighted": ("weighted avg", "recall"),
    "f1_weighted": ("weighted avg", "f1-score"),
}


def infer_esp_pairs(esps_map: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, str]]:
    esp_by_id: dict[int, str] = {}

    for esp_key in esps_map:
        try:
            esp_id = int(esp_key.removeprefix("esp_"))
        except ValueError:
            continue

        esp_by_id[esp_id] = esp_key

    return [
        (esp_by_id[esp_id], esp_by_id[esp_id + HIGH_FREQUENCY_ESP_OFFSET])
        for esp_id in sorted(esp_by_id)
        if esp_id in LOW_FREQUENCY_ESP_IDS
        and esp_id + HIGH_FREQUENCY_ESP_OFFSET in esp_by_id
    ]


def iter_magnitude_pair_groups(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
) -> Iterator[TrialPairGroup]:
    for scenario_key, locations_map in magnitudes.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                current_pairs = infer_esp_pairs(esps_map) if esp_pairs is None else esp_pairs
                trial_keys = sorted(
                    {
                        trial_key
                        for esp_trials in esps_map.values()
                        for trial_key in esp_trials
                    },
                )

                for trial_key in trial_keys:
                    pair_entries: list[PairEntry] = []

                    for esp_key_a, esp_key_b in current_pairs:
                        if esp_key_a not in esps_map or esp_key_b not in esps_map:
                            continue
                        if (
                            trial_key not in esps_map[esp_key_a]
                            or trial_key not in esps_map[esp_key_b]
                        ):
                            continue

                        magnitude_a = np.asarray(esps_map[esp_key_a][trial_key])
                        magnitude_b = np.asarray(esps_map[esp_key_b][trial_key])

                        if magnitude_a.size == 0 or magnitude_b.size == 0:
                            continue

                        pair_entries.append((esp_key_a, esp_key_b, magnitude_a, magnitude_b))

                    if pair_entries:
                        yield scenario_key, location_key, user_key, trial_key, pair_entries


def set_all_subcarrier_ticks(ax: Axes, subcarrier_count: int) -> None:
    subcarrier_index = np.arange(subcarrier_count)
    ax.set_xticks(subcarrier_index)
    ax.set_xticklabels(subcarrier_index, rotation=90, fontsize=6)


def set_sparse_subcarrier_ticks(ax: Axes, subcarrier_count: int) -> None:
    if subcarrier_count <= 0:
        return

    tick_count = min(SPARSE_3D_SUBCARRIER_TICK_COUNT, subcarrier_count)
    tick_positions = np.linspace(0, subcarrier_count - 1, tick_count, dtype=int)
    tick_positions = np.unique(tick_positions)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)


def set_sparse_index_ticks(ax: Axes, values: np.ndarray, axis: str) -> None:
    if values.size == 0:
        return

    tick_count = min(SPARSE_3D_SUBCARRIER_TICK_COUNT, values.size)
    tick_positions = np.linspace(0, values.size - 1, tick_count, dtype=int)
    tick_positions = np.unique(tick_positions)

    if axis == "x":
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(values[tick_positions])
        return
    if axis == "y":
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(values[tick_positions])
        return

    raise ValueError(INVALID_AXIS_MESSAGE)


def magnitude_to_db(magnitude: np.ndarray, epsilon: float = DB_EPSILON) -> np.ndarray:
    magnitude_array = np.asarray(magnitude, dtype=float)
    return 20.0 * np.log10(np.clip(magnitude_array, epsilon, None))


def visualization_magnitude_db(
    magnitude: np.ndarray,
    min_db: float = VISUALIZATION_MIN_DB,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude_db = magnitude_to_db(magnitude)
    if magnitude_db.ndim != MAGNITUDE_DIMS or magnitude_db.size == 0:
        return magnitude_db, np.empty(0, dtype=int)

    visible_sample_mask = np.all(magnitude_db >= min_db, axis=1)
    return magnitude_db[visible_sample_mask], np.flatnonzero(visible_sample_mask)


def mean_magnitude_profile_db(magnitude: np.ndarray) -> np.ndarray:
    magnitude_db, _ = visualization_magnitude_db(magnitude)
    if magnitude_db.size == 0:
        return np.empty(magnitude_db.shape[1] if magnitude_db.ndim == MAGNITUDE_DIMS else 0)

    return magnitude_db.mean(axis=0)


def plot_no_visible_magnitude_data(
    ax: Axes,
    label: str,
    min_db: float = VISUALIZATION_MIN_DB,
) -> None:
    ax.set_title(f"{label} | no samples >= {min_db:g} dB")
    ax.text(
        0.5,
        0.5,
        f"No samples >= {min_db:g} dB",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()


def plot_blank_magnitude_slot(ax: Axes) -> None:
    ax.set_axis_off()


def plot_pair_magnitude_profiles(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
) -> None:
    plot_count = 0
    packet_colors = ("tab:blue", "tab:orange")

    for scenario_key, location_key, user_key, trial_key, pair_entries in iter_magnitude_pair_groups(
        magnitudes,
        esp_pairs,
    ):
        row_count = len(pair_entries)
        fig, axes = plt.subplots(
            row_count,
            2,
            figsize=(18, max(4.5, 4.2 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for row_index, (esp_key_a, esp_key_b, magnitude_a, magnitude_b) in enumerate(pair_entries):
            for ax, esp_key, magnitude, color in zip(
                axes[row_index],
                (esp_key_a, esp_key_b),
                (magnitude_a, magnitude_b),
                packet_colors,
            ):
                magnitude_db, _ = visualization_magnitude_db(magnitude)
                if magnitude_db.size == 0:
                    plot_no_visible_magnitude_data(ax, esp_key)
                    continue

                subcarrier_index = np.arange(magnitude.shape[1])
                ax.plot(
                    subcarrier_index,
                    magnitude_db.T,
                    color=color,
                    alpha=0.035,
                    linewidth=0.45,
                )
                ax.plot(
                    subcarrier_index,
                    magnitude_db.mean(axis=0),
                    color="black",
                    linewidth=3.0,
                    label="Mean",
                )
                ax.set_title(
                    f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
                )
                ax.set_xlabel("Subcarrier index")
                ax.set_ylabel("CSI magnitude (dB)")
                set_all_subcarrier_ticks(ax, magnitude.shape[1])
                ax.grid(alpha=0.25)
                ax.legend()

        fig.suptitle(f"CSI magnitude (dB) vs subcarrier | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No paired ESP magnitude trials found.")


def plot_pair_magnitude_surfaces_3d(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for scenario_key, location_key, user_key, trial_key, pair_entries in iter_magnitude_pair_groups(
        magnitudes,
        esp_pairs,
    ):
        row_count = len(pair_entries)
        fig = plt.figure(
            figsize=(18, max(5.5, 5.0 * row_count)),
            constrained_layout=True,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for row_index, (esp_key_a, esp_key_b, magnitude_a, magnitude_b) in enumerate(pair_entries):
            for column_index, (esp_key, magnitude) in enumerate(
                ((esp_key_a, magnitude_a), (esp_key_b, magnitude_b)),
            ):
                magnitude_plot = magnitude[::packet_stride, ::subcarrier_stride]
                packet_index = np.arange(magnitude.shape[0])[::packet_stride]
                subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]
                subcarrier_grid, packet_grid = np.meshgrid(subcarrier_index, packet_index)

                subplot_index = row_index * 2 + column_index + 1
                ax = fig.add_subplot(row_count, 2, subplot_index, projection="3d")
                surface = ax.plot_surface(
                    subcarrier_grid,
                    packet_grid,
                    magnitude_plot,
                    cmap="viridis",
                    linewidth=0,
                    antialiased=False,
                )
                ax.set_title(f"{esp_key} | {magnitude.shape[0]} packets")
                ax.set_xlabel("Subcarrier index")
                ax.set_ylabel("Packet number")
                ax.set_zlabel("CSI magnitude")
                set_sparse_subcarrier_ticks(ax, magnitude.shape[1])
                fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)

        fig.suptitle(f"3D CSI magnitude | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No paired ESP magnitude trials found.")


def plot_pair_magnitude_heatmaps(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for scenario_key, location_key, user_key, trial_key, pair_entries in iter_magnitude_pair_groups(
        magnitudes,
        esp_pairs,
    ):
        row_count = len(pair_entries)
        fig, axes = plt.subplots(
            row_count,
            2,
            figsize=(18, max(5.0, 4.6 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for row_index, (esp_key_a, esp_key_b, magnitude_a, magnitude_b) in enumerate(pair_entries):
            for ax, esp_key, magnitude in zip(
                axes[row_index],
                (esp_key_a, esp_key_b),
                (magnitude_a, magnitude_b),
            ):
                magnitude_db, packet_index = visualization_magnitude_db(magnitude)
                if magnitude_db.size == 0:
                    plot_no_visible_magnitude_data(ax, esp_key)
                    continue

                magnitude_plot = magnitude_db[::packet_stride, ::subcarrier_stride]
                packet_index = packet_index[::packet_stride]
                subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]

                image = ax.imshow(magnitude_plot, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(
                    f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
                )
                ax.set_xlabel("Subcarrier index")
                ax.set_ylabel("Packet number")
                set_sparse_index_ticks(ax, subcarrier_index, "x")
                set_sparse_index_ticks(ax, packet_index, "y")
                fig.colorbar(image, ax=ax, shrink=0.82, label="CSI magnitude (dB)")

        fig.suptitle(f"CSI magnitude (dB) heatmap | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No paired ESP magnitude trials found.")


def plot_magnitude_pair_analysis(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
    packet_stride_3d: int = 1,
    subcarrier_stride_3d: int = 1,
) -> None:
    plot_pair_magnitude_profiles(magnitudes, esp_pairs)
    plot_pair_magnitude_heatmaps(
        magnitudes,
        esp_pairs,
        packet_stride=packet_stride_3d,
        subcarrier_stride=subcarrier_stride_3d,
    )


def normalize_location_input(value: str) -> str | None:
    normalized_value = value.strip().replace(" ", "").upper()
    normalized_value = normalized_value.removeprefix("LOCATION_")
    match = LOCATION_PATTERN.fullmatch(normalized_value)

    if match is None:
        return None
    if normalized_value.replace("-", "") == "Z0":
        return "location_Z-0"

    return f"location_{match.group('letter')}-{match.group('number')}"


def normalize_esp_input(value: str) -> str | None:
    normalized_value = value.strip().lower().removeprefix("esp_")
    if not normalized_value.isdigit():
        return None

    return f"esp_{int(normalized_value):02d}"


def format_location_key(location_key: str) -> str:
    return location_key.removeprefix("location_")


def format_esp_key(esp_key: str) -> str:
    return esp_key.removeprefix("esp_")


def sorted_location_keys(location_keys: set[str]) -> list[str]:
    def sort_key(location_key: str) -> tuple[str, int]:
        location = format_location_key(location_key)
        if location == "Z-0":
            return ("Z", 0)

        letter, number = location.split("-", maxsplit=1)
        return (letter, int(number))

    return sorted(location_keys, key=sort_key)


def sorted_esp_keys(esp_keys: set[str]) -> list[str]:
    return sorted(esp_keys, key=lambda esp_key: int(format_esp_key(esp_key)))


def paired_esp_keys(esp_keys: list[str]) -> list[str]:
    esp_by_id: dict[int, str] = {}
    unordered_esp_keys = []

    for esp_key in esp_keys:
        try:
            esp_id = int(format_esp_key(esp_key))
        except ValueError:
            unordered_esp_keys.append(esp_key)
            continue

        esp_by_id[esp_id] = esp_key

    ordered_esp_keys = []
    used_ids: set[int] = set()

    for low_esp_id in LOW_FREQUENCY_ESP_IDS:
        high_esp_id = low_esp_id + HIGH_FREQUENCY_ESP_OFFSET

        for esp_id in (low_esp_id, high_esp_id):
            if esp_id in esp_by_id:
                ordered_esp_keys.append(esp_by_id[esp_id])
                used_ids.add(esp_id)

    remaining_ids = sorted(set(esp_by_id) - used_ids)
    ordered_esp_keys.extend(esp_by_id[esp_id] for esp_id in remaining_ids)
    ordered_esp_keys.extend(unordered_esp_keys)
    return ordered_esp_keys


def get_available_location_keys(magnitudes: CsiMap) -> list[str]:
    return sorted_location_keys(
        {
            location_key
            for locations_map in magnitudes.values()
            for location_key in locations_map
        },
    )


def get_available_esp_keys_for_location(magnitudes: CsiMap, location_key: str) -> list[str]:
    esp_keys: set[str] = set()

    for locations_map in magnitudes.values():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue

        for esps_map in users_map.values():
            esp_keys.update(esps_map)

    return sorted_esp_keys(esp_keys)


def prompt_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes", "s", "sim"}:
            return True
        if answer in {"n", "no", "nao", "não"}:
            return False

        print("Please answer yes or no.")


def prompt_location_key(magnitudes: CsiMap) -> str | None:
    available_locations = get_available_location_keys(magnitudes)
    if not available_locations:
        print("No locations are available in magnitude_data.")
        return None

    location_text = ", ".join(
        format_location_key(location_key) for location_key in available_locations
    )
    print(f"Available locations: {location_text}")

    while True:
        answer = input("Which position/location do you want to analyse? ").strip()
        location_key = normalize_location_input(answer)

        if location_key in available_locations:
            return location_key

        print("Invalid location. Use one of the available locations, e.g. A1 or A-1.")


def prompt_esp_keys(magnitudes: CsiMap, location_key: str) -> list[str]:
    available_esps = get_available_esp_keys_for_location(magnitudes, location_key)
    if not available_esps:
        print(f"No ESPs are available for {format_location_key(location_key)}.")
        return []

    esp_text = ", ".join(format_esp_key(esp_key) for esp_key in available_esps)
    print(f"Available ESPs for {format_location_key(location_key)}: {esp_text}")
    print("Type ESP IDs separated by spaces or commas, or type all.")

    while True:
        answer = input("Which ESPs do you want to analyse? ").strip()
        if answer.lower() == "all":
            return paired_esp_keys(available_esps)

        esp_keys = [
            esp_key
            for raw_esp in re.split(r"[\s,;]+", answer)
            if raw_esp
            for esp_key in [normalize_esp_input(raw_esp)]
            if esp_key is not None
        ]
        invalid_esps = [esp_key for esp_key in esp_keys if esp_key not in available_esps]

        if esp_keys and not invalid_esps:
            return list(dict.fromkeys(esp_keys))

        print("Invalid ESP selection. Use available ESP IDs, e.g. 08 18 or all.")


def iter_selected_magnitude_groups(
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
) -> Iterator[SelectedTrialGroup]:
    for scenario_key, locations_map in magnitudes.items():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue

        for user_key, esps_map in users_map.items():
            trial_keys = sorted(
                {
                    trial_key
                    for esp_key in esp_keys
                    for trial_key in esps_map.get(esp_key, {})
                },
            )

            for trial_key in trial_keys:
                entries: list[SelectedEntry] = []
                has_visible_entry = False

                for esp_key in esp_keys:
                    if trial_key not in esps_map.get(esp_key, {}):
                        entries.append((esp_key, None))
                        continue

                    magnitude = np.asarray(esps_map[esp_key][trial_key])
                    if magnitude.size == 0:
                        entries.append((esp_key, None))
                        continue

                    has_visible_entry = True
                    entries.append((esp_key, magnitude))

                if has_visible_entry:
                    yield scenario_key, location_key, user_key, trial_key, entries


def make_subplot_grid(entry_count: int, column_count: int) -> tuple[int, int]:
    row_count = max(1, (entry_count + column_count - 1) // column_count)
    return row_count, column_count


def hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for ax in axes.ravel()[used_count:]:
        ax.set_visible(False)


def order_existing_values(
    values: Sequence[str],
    preferred_order: Sequence[str],
) -> list[str]:
    value_set = set(values)
    ordered_values = [value for value in preferred_order if value in value_set]
    ordered_values.extend(sorted(value_set - set(ordered_values)))
    return ordered_values


def plot_random_forest_metric_comparison(
    results: pd.DataFrame,
    metric_columns: Sequence[str] = RF_METRIC_COLUMNS,
    split_order: Sequence[str] = RF_SPLIT_ORDER,
    dataset_order: Sequence[str] = RF_DATASET_ORDER,
) -> None:
    if results.empty:
        print("No Random Forest results to plot.")
        return

    available_metrics = [metric for metric in metric_columns if metric in results.columns]
    if not available_metrics:
        print("No Random Forest metric columns found.")
        return

    datasets = order_existing_values(results["dataset"].astype(str).unique(), dataset_order)
    splits = order_existing_values(results["split"].astype(str).unique(), split_order)
    row_count, column_count = make_subplot_grid(len(available_metrics), 2)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.5 * column_count, 4.4 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    x_positions = np.arange(len(datasets))
    bar_width = min(0.36, 0.80 / max(1, len(splits)))

    for ax, metric in zip(axes.ravel(), available_metrics):
        for split_idx, split_name in enumerate(splits):
            values = [
                _random_forest_metric_value(results, dataset_name, split_name, metric)
                for dataset_name in datasets
            ]
            offset = (split_idx - (len(splits) - 1) / 2) * bar_width
            bars = ax.bar(
                x_positions + offset,
                values,
                width=bar_width,
                label=split_name,
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x_positions)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Split")

    hide_unused_axes(axes, len(available_metrics))
    fig.suptitle("Random Forest Metrics by Dataset and Split")
    plt.show()


def add_random_forest_report_metrics(
    results: pd.DataFrame,
    classification_reports: Mapping[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    enriched_results = results.copy()

    for row_idx, row in enriched_results.iterrows():
        model_key = (str(row["dataset"]), str(row["split"]))
        report = classification_reports.get(model_key)
        if report is None:
            continue

        for metric_column, (report_row, report_column) in RF_REPORT_METRIC_ROWS.items():
            if metric_column not in enriched_results.columns:
                enriched_results[metric_column] = np.nan
            if not _has_report_metric(report, report_row, report_column):
                continue

            enriched_results.loc[row_idx, metric_column] = float(
                report.loc[report_row, report_column],
            )

    return enriched_results


def _has_report_metric(report: pd.DataFrame, report_row: str, report_column: str) -> bool:
    return report_row in report.index and report_column in report.columns


def _random_forest_metric_value(
    results: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    metric: str,
) -> float:
    rows = results[
        (results["dataset"].astype(str) == dataset_name)
        & (results["split"].astype(str) == split_name)
    ]
    if rows.empty:
        return np.nan

    return float(rows.iloc[0][metric])


def plot_random_forest_confusion_matrices(
    confusion_matrices: Mapping[tuple[str, str], object],
    split_order: Sequence[str] = RF_SPLIT_ORDER,
    dataset_order: Sequence[str] = RF_DATASET_ORDER,
    *,
    normalize: bool = False,
    cmap: str = "Blues",
) -> None:
    if not confusion_matrices:
        print("No Random Forest confusion matrices to plot.")
        return

    datasets = order_existing_values(
        [dataset_name for dataset_name, _ in confusion_matrices],
        dataset_order,
    )
    splits = order_existing_values(
        [split_name for _, split_name in confusion_matrices],
        split_order,
    )
    fig, axes = plt.subplots(
        len(datasets),
        len(splits),
        figsize=(5.4 * len(splits), 4.6 * len(datasets)),
        constrained_layout=True,
        squeeze=False,
    )
    vmax = 1.0 if normalize else _max_confusion_matrix_value(confusion_matrices)
    images = []

    for row_idx, dataset_name in enumerate(datasets):
        for column_idx, split_name in enumerate(splits):
            ax = axes[row_idx, column_idx]
            matrix = confusion_matrices.get((dataset_name, split_name))

            if matrix is None:
                _plot_empty_confusion_matrix(ax, dataset_name, split_name)
                continue

            raw_values, actual_labels, predicted_labels = _confusion_matrix_values(matrix)
            values = _normalize_confusion_matrix(raw_values) if normalize else raw_values
            image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
            images.append(image)
            _annotate_confusion_matrix(
                ax,
                raw_values,
                values,
                normalize=normalize,
                threshold=vmax / 2,
            )
            ax.set_title(f"{dataset_name} | {split_name}")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_xticks(np.arange(len(predicted_labels)))
            ax.set_xticklabels(predicted_labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(actual_labels)))
            ax.set_yticklabels(actual_labels)

    if images:
        colorbar_label = "Row-normalized proportion" if normalize else "Count"
        fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.82, label=colorbar_label)

    title_suffix = "Normalized" if normalize else "Counts"
    fig.suptitle(f"Random Forest Confusion Matrices ({title_suffix})")
    plt.show()


def _max_confusion_matrix_value(confusion_matrices: Mapping[tuple[str, str], object]) -> float:
    max_values = []

    for matrix in confusion_matrices.values():
        values, _, _ = _confusion_matrix_values(matrix)
        if values.size:
            max_values.append(float(values.max()))

    if not max_values:
        return 1.0

    return max(1.0, *max_values)


def _confusion_matrix_values(matrix: object) -> tuple[np.ndarray, list[str], list[str]]:
    if hasattr(matrix, "to_numpy"):
        values = matrix.to_numpy(dtype=float)
        actual_labels = [str(label) for label in matrix.index]
        predicted_labels = [str(label) for label in matrix.columns]
        return values, actual_labels, predicted_labels

    values = np.asarray(matrix, dtype=float)
    label_count = values.shape[0] if values.ndim == CONFUSION_MATRIX_DIMS else 0
    labels = [str(label) for label in range(label_count)]
    return values, labels, labels


def _normalize_confusion_matrix(values: np.ndarray) -> np.ndarray:
    row_sums = values.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)


def _annotate_confusion_matrix(
    ax: Axes,
    raw_values: np.ndarray,
    values: np.ndarray,
    *,
    normalize: bool,
    threshold: float,
) -> None:
    for row_idx in range(values.shape[0]):
        for column_idx in range(values.shape[1]):
            value = values[row_idx, column_idx]
            label = f"{value:.2f}" if normalize else f"{raw_values[row_idx, column_idx]:.0f}"
            color = "white" if value > threshold else "black"
            ax.text(column_idx, row_idx, label, ha="center", va="center", color=color)


def _plot_empty_confusion_matrix(ax: Axes, dataset_name: str, split_name: str) -> None:
    ax.set_title(f"{dataset_name} | {split_name}")
    ax.text(0.5, 0.5, "No matrix", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def feature_columns_for_importance(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if column not in FEATURE_METADATA_COLUMNS
        and FEATURE_IMPORTANCE_COLUMN_PATTERN.fullmatch(str(column)) is not None
    ]


def random_forest_feature_importance_frame(
    models: Mapping[tuple[str, str], object],
    feature_dataframes: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for (dataset_name, split_name), model in models.items():
        dataframe = feature_dataframes.get(dataset_name)
        if dataframe is None or dataframe.empty:
            continue

        feature_columns = feature_columns_for_importance(dataframe)
        importances = _model_feature_importances(model)
        if importances is None:
            print(f"No feature_importances_ found for {dataset_name} / {split_name}.")
            continue
        if len(feature_columns) != len(importances):
            print(
                "Skipping feature importance for "
                f"{dataset_name} / {split_name}: "
                f"{len(feature_columns)} feature columns but {len(importances)} importances.",
            )
            continue

        for feature_column, importance in zip(feature_columns, importances):
            parsed_feature = parse_feature_importance_column(feature_column)
            if parsed_feature is None:
                continue

            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "feature": feature_column,
                    "importance": float(importance),
                    **parsed_feature,
                },
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "feature",
                "importance",
                "esp",
                "esp_id",
                "band",
                "statistic",
                "subcarrier",
            ],
        )

    return pd.DataFrame(rows)


def _model_feature_importances(model: object) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)

    named_steps = getattr(model, "named_steps", {})
    if not isinstance(named_steps, MappingABC):
        return None

    for step in reversed(tuple(named_steps.values())):
        if hasattr(step, "feature_importances_"):
            return np.asarray(step.feature_importances_, dtype=float)

    return None


def parse_feature_importance_column(feature_column: str) -> dict[str, object] | None:
    match = FEATURE_IMPORTANCE_COLUMN_PATTERN.fullmatch(feature_column)
    if match is None:
        return None

    esp_key = match.group("esp")
    esp_id = int(format_esp_key(esp_key))
    return {
        "esp": esp_key,
        "esp_id": esp_id,
        "band": band_for_esp_key(esp_key),
        "statistic": match.group("statistic"),
        "subcarrier": int(match.group("subcarrier")),
    }


def band_for_esp_key(esp_key: str) -> str:
    try:
        esp_id = int(format_esp_key(esp_key))
    except ValueError:
        return "Unknown"

    if esp_id in LOW_FREQUENCY_ESP_IDS:
        return "2.4 GHz"
    if FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID:
        return "5 GHz"

    return "Unknown"


def plot_random_forest_feature_importances(  # noqa: PLR0913
    models: Mapping[tuple[str, str], object],
    feature_dataframes: Mapping[str, pd.DataFrame],
    *,
    top_n: int = FEATURE_IMPORTANCE_TOP_N,
    dataset_order: Sequence[str] = RF_DATASET_ORDER,
    split_order: Sequence[str] = RF_SPLIT_ORDER,
) -> pd.DataFrame:
    importances = random_forest_feature_importance_frame(models, feature_dataframes)
    if importances.empty:
        print("No Random Forest feature importances to plot.")
        return importances

    datasets = order_existing_values(importances["dataset"].astype(str).unique(), dataset_order)
    splits = order_existing_values(importances["split"].astype(str).unique(), split_order)

    for dataset_name in datasets:
        for split_name in splits:
            model_importances = importances[
                (importances["dataset"].astype(str) == dataset_name)
                & (importances["split"].astype(str) == split_name)
            ]
            if model_importances.empty:
                continue

            fig, axes = plt.subplots(
                3,
                2,
                figsize=(18, 15),
                constrained_layout=True,
                squeeze=False,
            )
            _plot_top_feature_importances(axes[0, 0], model_importances, top_n)
            _plot_grouped_feature_importance(
                axes[0, 1],
                model_importances,
                "esp",
                "By ESP",
                order=_esp_importance_order(model_importances),
            )
            _plot_grouped_feature_importance(
                axes[1, 0],
                model_importances,
                "band",
                "By Band",
                order=RF_BAND_ORDER,
            )
            _plot_subcarrier_feature_importance(axes[1, 1], model_importances)
            _plot_grouped_feature_importance(
                axes[2, 0],
                model_importances,
                "statistic",
                "By Statistic",
                order=RF_FEATURE_STATISTIC_ORDER,
            )
            axes[2, 1].set_axis_off()

            fig.suptitle(f"Random Forest Feature Importance | {dataset_name} / {split_name}")
            plt.show()

    return importances


def _plot_top_feature_importances(
    ax: Axes,
    importances: pd.DataFrame,
    top_n: int,
) -> None:
    top_features = importances.nlargest(top_n, "importance").sort_values("importance")
    ax.barh(top_features["feature"], top_features["importance"], color="tab:blue")
    ax.set_title(f"Top {min(top_n, len(importances))} Columns")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)


def _plot_grouped_feature_importance(
    ax: Axes,
    importances: pd.DataFrame,
    group_column: str,
    title: str,
    *,
    order: Sequence[str] | None = None,
) -> None:
    grouped = _group_feature_importance(importances, group_column, order)
    if grouped.empty:
        ax.set_title(title)
        ax.set_axis_off()
        return

    x_positions = np.arange(len(grouped))
    ax.bar(x_positions, grouped["importance"], color="tab:green")
    ax.set_title(title)
    ax.set_ylabel("Importance")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(grouped[group_column], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)


def _plot_subcarrier_feature_importance(ax: Axes, importances: pd.DataFrame) -> None:
    grouped = (
        importances.groupby("subcarrier", as_index=False)["importance"]
        .sum()
        .sort_values("subcarrier")
    )
    if grouped.empty:
        ax.set_title("By Subcarrier")
        ax.set_axis_off()
        return

    x_positions = np.arange(len(grouped))
    ax.bar(x_positions, grouped["importance"], color="tab:orange")
    ax.set_title("By Subcarrier")
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("Importance")
    set_sparse_index_ticks(ax, grouped["subcarrier"].to_numpy(dtype=int), "x")
    ax.grid(axis="y", alpha=0.25)


def _group_feature_importance(
    importances: pd.DataFrame,
    group_column: str,
    order: Sequence[str] | None,
) -> pd.DataFrame:
    grouped = importances.groupby(group_column, as_index=False)["importance"].sum()
    if order is None:
        return grouped.sort_values("importance", ascending=False)

    ordered_values = order_existing_values(grouped[group_column].astype(str).unique(), order)
    grouped = grouped.assign(
        _order=grouped[group_column].astype(str).map(
            {value: idx for idx, value in enumerate(ordered_values)},
        ),
    )
    return grouped.sort_values("_order").drop(columns="_order")


def _esp_importance_order(importances: pd.DataFrame) -> list[str]:
    esp_keys = importances["esp"].astype(str).unique()
    return sorted_esp_keys(set(esp_keys))


def plot_selected_magnitude_profiles(
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    column_count: int = 2,
) -> None:
    plot_count = 0

    for scenario_key, _, user_key, trial_key, entries in iter_selected_magnitude_groups(
        magnitudes,
        location_key,
        esp_keys,
    ):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(9 * column_count, max(4.5, 4.2 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for ax, (esp_key, magnitude) in zip(axes.ravel(), entries):
            if magnitude is None:
                plot_blank_magnitude_slot(ax)
                continue

            magnitude_db, _ = visualization_magnitude_db(magnitude)
            if magnitude_db.size == 0:
                plot_no_visible_magnitude_data(ax, esp_key)
                continue

            subcarrier_index = np.arange(magnitude.shape[1])
            ax.plot(
                subcarrier_index,
                magnitude_db.T,
                color="tab:blue",
                alpha=0.035,
                linewidth=0.45,
            )
            ax.plot(
                subcarrier_index,
                magnitude_db.mean(axis=0),
                color="black",
                linewidth=3.0,
                label="Mean",
            )
            ax.set_title(
                f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
            )
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("CSI magnitude (dB)")
            set_all_subcarrier_ticks(ax, magnitude.shape[1])
            ax.grid(alpha=0.25)
            ax.legend()

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"CSI magnitude (dB) vs subcarrier | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_selected_magnitude_surfaces_3d(  # noqa: PLR0913
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for scenario_key, _, user_key, trial_key, entries in iter_selected_magnitude_groups(
        magnitudes,
        location_key,
        esp_keys,
    ):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig = plt.figure(
            figsize=(9 * column_count, max(5.5, 5.0 * row_count)),
            constrained_layout=True,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for index, (esp_key, magnitude) in enumerate(entries, start=1):
            ax = fig.add_subplot(row_count, column_count, index, projection="3d")
            if magnitude is None:
                plot_blank_magnitude_slot(ax)
                continue

            magnitude_plot = magnitude[::packet_stride, ::subcarrier_stride]
            packet_index = np.arange(magnitude.shape[0])[::packet_stride]
            subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]
            subcarrier_grid, packet_grid = np.meshgrid(subcarrier_index, packet_index)

            ax.plot_surface(
                subcarrier_grid,
                packet_grid,
                magnitude_plot,
                cmap="viridis",
                linewidth=0,
                antialiased=False,
            )
            ax.set_title(f"{esp_key} | {magnitude.shape[0]} packets")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Packet number")
            ax.set_zlabel("CSI magnitude")
            set_sparse_subcarrier_ticks(ax, magnitude.shape[1])

        fig.suptitle(f"3D CSI magnitude | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_selected_magnitude_heatmaps(  # noqa: PLR0913
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for scenario_key, _, user_key, trial_key, entries in iter_selected_magnitude_groups(
        magnitudes,
        location_key,
        esp_keys,
    ):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(9 * column_count, max(4.8, 4.5 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for ax, (esp_key, magnitude) in zip(axes.ravel(), entries):
            if magnitude is None:
                plot_blank_magnitude_slot(ax)
                continue

            magnitude_db, packet_index = visualization_magnitude_db(magnitude)
            if magnitude_db.size == 0:
                plot_no_visible_magnitude_data(ax, esp_key)
                continue

            magnitude_plot = magnitude_db[::packet_stride, ::subcarrier_stride]
            packet_index = packet_index[::packet_stride]
            subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]

            image = ax.imshow(magnitude_plot, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(
                f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
            )
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Packet number")
            set_sparse_index_ticks(ax, subcarrier_index, "x")
            set_sparse_index_ticks(ax, packet_index, "y")
            fig.colorbar(image, ax=ax, shrink=0.82, label="CSI magnitude (dB)")

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"CSI magnitude (dB) heatmap | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def stack_same_width_profiles(profiles: list[np.ndarray]) -> np.ndarray:
    if not profiles:
        return np.empty((0, 0), dtype=float)

    width_counts: dict[int, int] = {}
    for profile in profiles:
        width_counts[profile.shape[0]] = width_counts.get(profile.shape[0], 0) + 1

    target_width = max(width_counts, key=width_counts.get)
    matching_profiles = [profile for profile in profiles if profile.shape[0] == target_width]
    return np.vstack(matching_profiles)


def collect_user_mean_profiles_db(
    magnitudes: CsiMap,
    scenario_key: str,
    location_key: str,
    esp_key: str,
) -> dict[str, np.ndarray]:
    users_map = magnitudes.get(scenario_key, {}).get(location_key, {})
    user_profiles: dict[str, np.ndarray] = {}

    for user_key, esps_map in users_map.items():
        trial_profiles = []

        for magnitude in esps_map.get(esp_key, {}).values():
            magnitude_array = np.asarray(magnitude)
            if (
                magnitude_array.ndim != MAGNITUDE_DIMS
                or magnitude_array.size == 0
                or magnitude_array.shape[0] == 0
            ):
                continue

            magnitude_db, _ = visualization_magnitude_db(magnitude_array)
            if magnitude_db.size == 0:
                continue

            trial_profiles.append(magnitude_db.mean(axis=0))

        stacked_trials = stack_same_width_profiles(trial_profiles)
        if stacked_trials.size:
            user_profiles[user_key] = stacked_trials.mean(axis=0)

    return user_profiles


def empty_room_mean_profile_db(
    magnitudes: CsiMap,
    scenario_key: str,
    esp_key: str,
    target_shape: tuple[int, ...],
    empty_room_location_key: str = EMPTY_ROOM_LOCATION_KEY,
) -> np.ndarray | None:
    empty_user_profiles = collect_user_mean_profiles_db(
        magnitudes,
        scenario_key,
        empty_room_location_key,
        esp_key,
    )
    matching_profiles = [
        profile
        for profile in empty_user_profiles.values()
        if profile.shape == target_shape
    ]
    if not matching_profiles:
        return None

    return np.vstack(matching_profiles).mean(axis=0)


def plot_average_magnitude_profiles_across_users(
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    empty_room_location_key: str = EMPTY_ROOM_LOCATION_KEY,
    column_count: int = 2,
) -> None:
    plot_count = 0

    for scenario_key in sorted(magnitudes):
        entries = []
        has_average_entry = False

        for esp_key in esp_keys:
            user_profiles = collect_user_mean_profiles_db(
                magnitudes,
                scenario_key,
                location_key,
                esp_key,
            )
            stacked_users = stack_same_width_profiles(list(user_profiles.values()))
            if stacked_users.size == 0:
                entries.append((esp_key, None, None, None, None))
                continue

            mean_profile = stacked_users.mean(axis=0)
            std_profile = stacked_users.std(axis=0)
            empty_profile = empty_room_mean_profile_db(
                magnitudes,
                scenario_key,
                esp_key,
                mean_profile.shape,
                empty_room_location_key=empty_room_location_key,
            )
            entries.append(
                (
                    esp_key,
                    stacked_users.shape[0],
                    mean_profile,
                    std_profile,
                    empty_profile,
                ),
            )
            has_average_entry = True

        if not has_average_entry:
            continue

        row_count, grid_column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            grid_column_count,
            figsize=(9 * grid_column_count, max(4.6, 4.2 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )

        for ax, (esp_key, user_count, mean_profile, std_profile, empty_profile) in zip(
            axes.ravel(),
            entries,
        ):
            if mean_profile is None or std_profile is None:
                plot_blank_magnitude_slot(ax)
                continue

            subcarrier_index = np.arange(mean_profile.shape[0])
            ax.plot(
                subcarrier_index,
                mean_profile,
                color="black",
                linewidth=2.5,
                label="Mean across users",
            )
            ax.fill_between(
                subcarrier_index,
                mean_profile - std_profile,
                mean_profile + std_profile,
                color="tab:blue",
                alpha=0.22,
                label="+/-1 sigma across users",
            )
            if empty_profile is not None:
                ax.plot(
                    subcarrier_index,
                    empty_profile,
                    color="tab:red",
                    linestyle="--",
                    linewidth=2.0,
                    label="Empty-room mean",
                )

            ax.set_title(f"{esp_key} | {user_count} users")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("CSI magnitude (dB)")
            set_all_subcarrier_ticks(ax, mean_profile.shape[0])
            ax.grid(alpha=0.25)
            ax.legend()

        hide_unused_axes(axes, len(entries))
        fig.suptitle(
            "Average CSI magnitude (dB) across users | "
            f"{scenario_key} / {location_key}",
        )
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No user-average magnitude profiles found for the selected location/ESPs.")


def plot_magnitude_analysis_interactive(
    magnitudes: CsiMap,
    packet_stride_3d: int = 1,
    subcarrier_stride_3d: int = 1,
    empty_room_location_key: str = EMPTY_ROOM_LOCATION_KEY,
    column_count: int = 2,
) -> None:
    if not prompt_yes_no("Do you want to show graphs? [yes/no] "):
        print("Skipping graphs.")
        return

    location_key = prompt_location_key(magnitudes)
    if location_key is None:
        return

    esp_keys = prompt_esp_keys(magnitudes, location_key)
    if not esp_keys:
        return

    plot_selected_magnitude_profiles(
        magnitudes,
        location_key,
        esp_keys,
        column_count=column_count,
    )
    plot_selected_magnitude_heatmaps(
        magnitudes,
        location_key,
        esp_keys,
        packet_stride=packet_stride_3d,
        subcarrier_stride=subcarrier_stride_3d,
        column_count=column_count,
    )
    plot_average_magnitude_profiles_across_users(
        magnitudes,
        location_key,
        esp_keys,
        empty_room_location_key=empty_room_location_key,
        column_count=column_count,
    )


def get_available_location_keys_for_magnitude_sets(magnitude_sets: MagnitudeSetMap) -> list[str]:
    return sorted_location_keys(
        {
            location_key
            for magnitudes in magnitude_sets.values()
            for locations_map in magnitudes.values()
            for location_key in locations_map
        },
    )


def get_available_esp_keys_for_magnitude_sets(
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
) -> list[str]:
    esp_keys: set[str] = set()

    for magnitudes in magnitude_sets.values():
        for locations_map in magnitudes.values():
            users_map = locations_map.get(location_key)
            if users_map is None:
                continue

            for esps_map in users_map.values():
                esp_keys.update(esps_map)

    return sorted_esp_keys(esp_keys)


def prompt_location_key_for_magnitude_sets(magnitude_sets: MagnitudeSetMap) -> str | None:
    available_locations = get_available_location_keys_for_magnitude_sets(magnitude_sets)
    if not available_locations:
        print("No locations are available in magnitude data.")
        return None

    location_text = ", ".join(
        format_location_key(location_key) for location_key in available_locations
    )
    print(f"Available locations: {location_text}")

    while True:
        answer = input("Which position/location do you want to analyse? ").strip()
        location_key = normalize_location_input(answer)

        if location_key in available_locations:
            return location_key

        print("Invalid location. Use one of the available locations, e.g. A1 or A-1.")


def prompt_esp_keys_for_magnitude_sets(
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
) -> list[str]:
    available_esps = get_available_esp_keys_for_magnitude_sets(magnitude_sets, location_key)
    if not available_esps:
        print(f"No ESPs are available for {format_location_key(location_key)}.")
        return []

    esp_text = ", ".join(format_esp_key(esp_key) for esp_key in available_esps)
    print(f"Available ESPs for {format_location_key(location_key)}: {esp_text}")
    print("Type ESP IDs separated by spaces or commas, or type all.")

    while True:
        answer = input("Which ESPs do you want to analyse? ").strip()
        if answer.lower() == "all":
            return paired_esp_keys(available_esps)

        esp_keys = [
            esp_key
            for raw_esp in re.split(r"[\s,;]+", answer)
            if raw_esp
            for esp_key in [normalize_esp_input(raw_esp)]
            if esp_key is not None
        ]
        invalid_esps = [esp_key for esp_key in esp_keys if esp_key not in available_esps]

        if esp_keys and not invalid_esps:
            return list(dict.fromkeys(esp_keys))

        print("Invalid ESP selection. Use available ESP IDs, e.g. 08 18 or all.")


def get_magnitude_entry(  # noqa: PLR0913
    magnitudes: CsiMap,
    scenario_key: str,
    location_key: str,
    user_key: str,
    esp_key: str,
    trial_key: str,
) -> np.ndarray | None:
    trial_map = (
        magnitudes.get(scenario_key, {})
        .get(location_key, {})
        .get(user_key, {})
        .get(esp_key, {})
    )

    if trial_key not in trial_map:
        return None

    return np.asarray(trial_map[trial_key])


def iter_selected_magnitude_set_groups(
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
    esp_keys: list[str],
) -> Iterator[SelectedMagnitudeSetGroup]:
    scenario_keys = sorted(
        {
            scenario_key
            for magnitudes in magnitude_sets.values()
            for scenario_key in magnitudes
        },
    )

    for scenario_key in scenario_keys:
        user_keys = sorted(
            {
                user_key
                for magnitudes in magnitude_sets.values()
                for user_key in magnitudes.get(scenario_key, {}).get(location_key, {})
            },
        )

        for user_key in user_keys:
            trial_keys = sorted(
                {
                    trial_key
                    for magnitudes in magnitude_sets.values()
                    for esp_key in esp_keys
                    for trial_key in (
                        magnitudes.get(scenario_key, {})
                        .get(location_key, {})
                        .get(user_key, {})
                        .get(esp_key, {})
                    )
                },
            )

            for trial_key in trial_keys:
                for esp_key in esp_keys:
                    entries = [
                        (
                            label,
                            get_magnitude_entry(
                                magnitudes,
                                scenario_key,
                                location_key,
                                user_key,
                                esp_key,
                                trial_key,
                            ),
                        )
                        for label, magnitudes in magnitude_sets.items()
                    ]

                    if any(magnitude is not None and magnitude.size for _, magnitude in entries):
                        yield scenario_key, location_key, user_key, trial_key, esp_key, entries


def plot_no_magnitude_data(ax: Axes, label: str) -> None:
    ax.set_title(f"{label} | no data")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_selected_magnitude_set_profiles(
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
    esp_keys: list[str],
    column_count: int = 2,
) -> None:
    plot_count = 0

    for (
        scenario_key,
        _,
        user_key,
        trial_key,
        esp_key,
        entries,
    ) in iter_selected_magnitude_set_groups(magnitude_sets, location_key, esp_keys):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(9 * column_count, max(4.5, 4.2 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key} / {esp_key}"

        for ax, (label, magnitude) in zip(axes.ravel(), entries):
            if magnitude is None or magnitude.size == 0:
                plot_no_magnitude_data(ax, label)
                continue

            magnitude_db, _ = visualization_magnitude_db(magnitude)
            if magnitude_db.size == 0:
                plot_no_visible_magnitude_data(ax, label)
                continue

            subcarrier_index = np.arange(magnitude.shape[1])
            ax.plot(
                subcarrier_index,
                magnitude_db.T,
                color="tab:blue",
                alpha=0.035,
                linewidth=0.45,
            )
            ax.plot(
                subcarrier_index,
                magnitude_db.mean(axis=0),
                color="black",
                linewidth=3.0,
                label="Mean",
            )
            ax.set_title(
                f"{label} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
            )
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("CSI magnitude (dB)")
            set_all_subcarrier_ticks(ax, magnitude.shape[1])
            ax.grid(alpha=0.25)
            ax.legend()

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"CSI magnitude (dB) vs subcarrier | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_selected_magnitude_set_surfaces_3d(  # noqa: PLR0913
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for (
        scenario_key,
        _,
        user_key,
        trial_key,
        esp_key,
        entries,
    ) in iter_selected_magnitude_set_groups(magnitude_sets, location_key, esp_keys):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig = plt.figure(
            figsize=(9 * column_count, max(5.5, 5.0 * row_count)),
            constrained_layout=True,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key} / {esp_key}"

        for index, (label, magnitude) in enumerate(entries, start=1):
            ax = fig.add_subplot(row_count, column_count, index, projection="3d")

            if magnitude is None or magnitude.size == 0:
                ax.set_title(f"{label} | no data")
                ax.text2D(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            magnitude_plot = magnitude[::packet_stride, ::subcarrier_stride]
            packet_index = np.arange(magnitude.shape[0])[::packet_stride]
            subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]
            subcarrier_grid, packet_grid = np.meshgrid(subcarrier_index, packet_index)

            ax.plot_surface(
                subcarrier_grid,
                packet_grid,
                magnitude_plot,
                cmap="viridis",
                linewidth=0,
                antialiased=False,
            )
            ax.set_title(f"{label} | {magnitude.shape[0]} packets")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Packet number")
            ax.set_zlabel("CSI magnitude")
            set_sparse_subcarrier_ticks(ax, magnitude.shape[1])

        fig.suptitle(f"3D CSI magnitude | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_selected_magnitude_set_heatmaps(  # noqa: PLR0913
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError(INVALID_STRIDE_MESSAGE)

    plot_count = 0

    for (
        scenario_key,
        _,
        user_key,
        trial_key,
        esp_key,
        entries,
    ) in iter_selected_magnitude_set_groups(magnitude_sets, location_key, esp_keys):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(9 * column_count, max(4.8, 4.5 * row_count)),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key} / {esp_key}"

        for ax, (label, magnitude) in zip(axes.ravel(), entries):
            if magnitude is None or magnitude.size == 0:
                plot_no_magnitude_data(ax, label)
                continue

            magnitude_db, packet_index = visualization_magnitude_db(magnitude)
            if magnitude_db.size == 0:
                plot_no_visible_magnitude_data(ax, label)
                continue

            magnitude_plot = magnitude_db[::packet_stride, ::subcarrier_stride]
            packet_index = packet_index[::packet_stride]
            subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]

            image = ax.imshow(magnitude_plot, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(
                f"{label} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown",
            )
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("Packet number")
            set_sparse_index_ticks(ax, subcarrier_index, "x")
            set_sparse_index_ticks(ax, packet_index, "y")
            fig.colorbar(image, ax=ax, shrink=0.82, label="CSI magnitude (dB)")

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"CSI magnitude (dB) heatmap | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_magnitude_sets_analysis_interactive(
    magnitude_sets: MagnitudeSetMap,
    packet_stride_3d: int = 1,
    subcarrier_stride_3d: int = 1,
    column_count: int = 2,
) -> None:
    if not prompt_yes_no("Do you want to show graphs? [yes/no] "):
        print("Skipping graphs.")
        return

    location_key = prompt_location_key_for_magnitude_sets(magnitude_sets)
    if location_key is None:
        return

    esp_keys = prompt_esp_keys_for_magnitude_sets(magnitude_sets, location_key)
    if not esp_keys:
        return

    plot_selected_magnitude_set_profiles(
        magnitude_sets,
        location_key,
        esp_keys,
        column_count=column_count,
    )
    plot_selected_magnitude_set_heatmaps(
        magnitude_sets,
        location_key,
        esp_keys,
        packet_stride=packet_stride_3d,
        subcarrier_stride=subcarrier_stride_3d,
        column_count=column_count,
    )


def extract_complex_csi_from_file(file_path: str | Path, *, its5ghz: bool) -> np.ndarray:
    csi_column = FIVE_GHZ_CSI_COLUMN if its5ghz else TWO_GHZ_CSI_COLUMN
    expected_length = FIVE_GHZ_RAW_LENGTH if its5ghz else TWO_GHZ_RAW_LENGTH
    file_csv = pd.read_csv(str(file_path), header=None, usecols=[csi_column])
    csi_raw = file_csv.iloc[:, 0]
    valid_csi: list[list[float]] = []

    for entry in csi_raw.to_numpy():
        match = re.search(r"\[(.*?)\]", str(entry))
        if not match:
            continue

        nums = [float(number) for number in re.findall(r"-?\d+", match.group(1))]
        if len(nums) == expected_length:
            valid_csi.append(nums)

    if not valid_csi:
        return np.empty((0, 0), dtype=complex)

    valid_csi = np.array(valid_csi, dtype=float)
    if not its5ghz:
        real = valid_csi[:, 1::2]
        imag = valid_csi[:, ::2]
        complex_csi = real + 1j * imag
        fft_csi = np.fft.fftshift(complex_csi, axes=1)
        active_sc = fft_csi[:, 6:58]
        active_sc = np.delete(active_sc, [26, 27], axis=1)
    else:
        imag = valid_csi[:, ::2]
        real = valid_csi[:, 1::2]
        complex_csi = real + 1j * imag
        active_sc = np.delete(complex_csi, [28], axis=1)

    return active_sc


def iter_selected_file_groups(
    files: FileMap,
    location_key: str,
    esp_keys: list[str],
) -> Iterator[SelectedFileGroup]:
    for scenario_key, locations_map in files.items():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue

        for user_key, esps_map in users_map.items():
            trial_keys = sorted(
                {
                    trial_key
                    for esp_key in esp_keys
                    for trial_key in esps_map.get(esp_key, {})
                },
            )

            for trial_key in trial_keys:
                entries: list[SelectedFileEntry] = []
                for esp_key in esp_keys:
                    file_path = esps_map.get(esp_key, {}).get(trial_key)
                    if file_path is None:
                        continue
                    entries.append((esp_key, file_path))

                if entries:
                    yield (scenario_key, location_key, user_key, trial_key, entries)


def iq_subcarrier_indices(
    sc_indices: Sequence[int] | str | None,
    subcarrier_count: int,
) -> list[int]:
    if subcarrier_count <= 0:
        return []
    if sc_indices is None:
        return [0]
    if isinstance(sc_indices, str):
        if sc_indices.strip().lower() == "all":
            return list(range(subcarrier_count))
        sc_indices = [int(number) for number in re.findall(r"-?\d+", sc_indices)]

    return [
        sc
        for sc in dict.fromkeys(int(sc) for sc in sc_indices)
        if 0 <= sc < subcarrier_count
    ]


def plot_selected_iq_constellations(  # noqa: PLR0913, PLR0915
    files: FileMap,
    location_key: str,
    esp_keys: list[str],
    sc_indices: Sequence[int] | str | None = None,
    sample_stride: int = 1,
    column_count: int = 2,
) -> None:
    if sample_stride < 1:
        msg = "sample_stride must be at least 1."
        raise ValueError(msg)

    plot_count = 0

    for scenario_key, _, user_key, trial_key, entries in iter_selected_file_groups(
        files,
        location_key,
        esp_keys,
    ):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(5.5 * column_count, max(4.5, 4.0 * row_count)),
            squeeze=False,
            constrained_layout=True,
        )
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        for ax, (esp_key, file_path) in zip(axes.ravel(), entries):
            try:
                esp_id = int(esp_key.removeprefix("esp_"))
            except ValueError:
                esp_id = None
            its5ghz = (
                esp_id is not None and FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID
            )
            cplx = extract_complex_csi_from_file(file_path, its5ghz=its5ghz)
            if cplx.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"{esp_key} | no samples")
                ax.set_axis_off()
                continue

            selected_subcarriers = iq_subcarrier_indices(sc_indices, cplx.shape[1])
            if not selected_subcarriers:
                ax.text(0.5, 0.5, "No selected subcarriers", ha="center", va="center")
                ax.set_title(f"{esp_key} | no selected subcarriers")
                ax.set_axis_off()
                continue

            color_values = np.linspace(0, 1, len(selected_subcarriers))
            colors = plt.cm.viridis(color_values)
            show_legend = len(selected_subcarriers) <= MAX_IQ_LEGEND_ITEMS
            point_size = 6 if show_legend else 4
            point_alpha = 0.7 if show_legend else 0.35

            for color, sc in zip(colors, selected_subcarriers):
                samples = cplx[:, sc][::sample_stride]
                ax.scatter(
                    samples.real,
                    samples.imag,
                    s=point_size,
                    alpha=point_alpha,
                    color=color,
                    label=f"SC {sc}" if show_legend else None,
                )

            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{esp_key} | {trial_key} | {len(selected_subcarriers)} SCs")
            ax.set_xlabel("I (Real)")
            ax.set_ylabel("Q (Imag)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)

            if show_legend:
                ax.legend(fontsize=6)
            else:
                norm = plt.Normalize(
                    vmin=min(selected_subcarriers),
                    vmax=max(selected_subcarriers),
                )
                colorbar_source = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
                colorbar_source.set_array([])
                colorbar = fig.colorbar(
                    colorbar_source,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )
                colorbar.set_label("Subcarrier index")

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"I/Q Constellations | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No I/Q data found for the selected location/ESPs.")


def get_available_location_keys_from_files(files: FileMap) -> list[str]:
    return sorted_location_keys(
        {location_key for locations_map in files.values() for location_key in locations_map},
    )


def get_available_esp_keys_for_location_files(files: FileMap, location_key: str) -> list[str]:
    esp_keys: set[str] = set()

    for locations_map in files.values():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue

        for esps_map in users_map.values():
            esp_keys.update(esps_map)

    return sorted_esp_keys(esp_keys)


def prompt_esp_keys_from_files(files: FileMap, location_key: str) -> list[str]:
    available_esps = get_available_esp_keys_for_location_files(files, location_key)
    if not available_esps:
        print(f"No ESPs are available for {format_location_key(location_key)}.")
        return []

    esp_text = ", ".join(format_esp_key(esp_key) for esp_key in available_esps)
    print(f"Available ESPs for {format_location_key(location_key)}: {esp_text}")
    print("Type ESP IDs separated by spaces or commas, or type all.")

    while True:
        answer = input("Which ESPs do you want to analyse? ").strip()
        if answer.lower() == "all":
            return paired_esp_keys(available_esps)

        esp_keys = [
            esp_key
            for raw_esp in re.split(r"[\s,;]+", answer)
            if raw_esp
            for esp_key in [normalize_esp_input(raw_esp)]
            if esp_key is not None
        ]
        invalid_esps = [esp_key for esp_key in esp_keys if esp_key not in available_esps]

        if esp_keys and not invalid_esps:
            return list(dict.fromkeys(esp_keys))

        print("Invalid ESP selection. Use available ESP IDs, e.g. 08 18 or all.")


def plot_iq_interactive(files: FileMap, column_count: int = 2) -> None:
    if not prompt_yes_no("Do you want to show I/Q constellation plots? [yes/no] "):
        print("Skipping I/Q plots.")
        return

    available_locations = get_available_location_keys_from_files(files)
    if not available_locations:
        print("No locations available in files.")
        return

    location_text = ", ".join(
        format_location_key(location_key) for location_key in available_locations
    )
    print(f"Available locations: {location_text}")
    while True:
        answer = input("Which position/location do you want to analyse? ").strip()
        location_key = normalize_location_input(answer)
        if location_key in available_locations:
            break
        print("Invalid location. Use one of the available locations, e.g. A1 or A-1.")

    esp_keys = prompt_esp_keys_from_files(files, location_key)
    if not esp_keys:
        return

    sc_text = input(
        "Subcarrier indices to plot (comma-separated, e.g. 0 or 0,1,2, or all). Default 0: ",
    ).strip()
    if not sc_text:
        sc_indices = [0]
    elif sc_text.lower() == "all":
        sc_indices = "all"
    else:
        nums = re.findall(r"-?\d+", sc_text)
        sc_indices = [int(n) for n in nums] if nums else [0]

    plot_selected_iq_constellations(
        files,
        location_key,
        esp_keys,
        sc_indices=sc_indices,
        column_count=column_count,
    )
