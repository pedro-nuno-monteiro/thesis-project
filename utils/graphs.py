from __future__ import annotations

import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
SelectedEntry = tuple[str, np.ndarray | None]
SelectedTrialGroup = tuple[str, str, str, str, list[SelectedEntry]]

LOCATION_PATTERN = re.compile(r"^(?:(?P<letter>[A-G])(?:-)?(?P<number>[1-9]|1[0-4])|Z-?0)$")
LOW_FREQUENCY_ESP_IDS = range(1, 11)
HIGH_FREQUENCY_ESP_OFFSET = 10
DB_EPSILON = 1e-12
EMPTY_ROOM_LOCATION_KEY = "location_Z-0"
MAGNITUDE_DIMS = 2
VISUALIZATION_MIN_DB = -80.0
INVALID_AXIS_MESSAGE = "axis must be 'x' or 'y'."
INVALID_STRIDE_MESSAGE = "Strides must be at least 1."


def set_all_subcarrier_ticks(ax: Axes, subcarrier_count: int) -> None:
    subcarrier_index = np.arange(subcarrier_count)
    ax.set_xticks(subcarrier_index)
    ax.set_xticklabels(subcarrier_index, rotation=90, fontsize=6)


def set_sparse_index_ticks(ax: Axes, values: np.ndarray, axis: str) -> None:
    if values.size == 0:
        return

    tick_count = min(8, values.size)
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
