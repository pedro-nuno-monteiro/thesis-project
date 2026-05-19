from __future__ import annotations

from collections.abc import Iterator
import re

import matplotlib.pyplot as plt
import numpy as np

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
PairEntry = tuple[str, str, np.ndarray, np.ndarray]
TrialPairGroup = tuple[str, str, str, str, list[PairEntry]]
SelectedEntry = tuple[str, np.ndarray]
SelectedTrialGroup = tuple[str, str, str, str, list[SelectedEntry]]

LOCATION_PATTERN = re.compile(r"^(?:(?P<letter>[A-G])(?:-)?(?P<number>[1-9]|1[0-4])|Z-?0)$")


def infer_esp_pairs(esps_map: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, str]]:
    esp_by_id: dict[int, str] = {}

    for esp_key in esps_map:
        try:
            esp_id = int(esp_key.removeprefix("esp_"))
        except ValueError:
            continue

        esp_by_id[esp_id] = esp_key

    return [
        (esp_by_id[esp_id], esp_by_id[esp_id + 10])
        for esp_id in sorted(esp_by_id)
        if 1 <= esp_id <= 10 and esp_id + 10 in esp_by_id
    ]


def iter_magnitude_pair_groups(
    magnitudes: CsiMap,
    esp_pairs: list[tuple[str, str]] | None = None,
) -> Iterator[TrialPairGroup]:
    for scenario_key, locations_map in magnitudes.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                if esp_pairs is None:
                    current_pairs = infer_esp_pairs(esps_map)
                else:
                    current_pairs = esp_pairs
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


def set_all_subcarrier_ticks(ax, subcarrier_count: int) -> None:
    subcarrier_index = np.arange(subcarrier_count)
    ax.set_xticks(subcarrier_index)
    ax.set_xticklabels(subcarrier_index, rotation=90, fontsize=6)


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
                subcarrier_index = np.arange(magnitude.shape[1])
                ax.plot(subcarrier_index, magnitude.T, color=color, alpha=0.035, linewidth=0.45)
                ax.plot(
                    subcarrier_index,
                    magnitude.mean(axis=0),
                    color="black",
                    linewidth=3.0,
                    label="Mean",
                )
                ax.set_title(f"{esp_key} | {magnitude.shape[0]} packets")
                ax.set_xlabel("Subcarrier index")
                ax.set_ylabel("CSI magnitude")
                set_all_subcarrier_ticks(ax, magnitude.shape[1])
                ax.grid(alpha=0.25)
                ax.legend()

        fig.suptitle(f"CSI magnitude vs subcarrier | {title}")
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
        raise ValueError("Strides must be at least 1.")

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
                set_all_subcarrier_ticks(ax, magnitude.shape[1])
                fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)

        fig.suptitle(f"3D CSI magnitude | {title}")
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
    plot_pair_magnitude_surfaces_3d(
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
            return available_esps

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

                for esp_key in esp_keys:
                    if trial_key not in esps_map.get(esp_key, {}):
                        continue

                    magnitude = np.asarray(esps_map[esp_key][trial_key])
                    if magnitude.size == 0:
                        continue

                    entries.append((esp_key, magnitude))

                if entries:
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
            subcarrier_index = np.arange(magnitude.shape[1])
            ax.plot(subcarrier_index, magnitude.T, color="tab:blue", alpha=0.035, linewidth=0.45)
            ax.plot(
                subcarrier_index,
                magnitude.mean(axis=0),
                color="black",
                linewidth=3.0,
                label="Mean",
            )
            ax.set_title(f"{esp_key} | {magnitude.shape[0]} packets")
            ax.set_xlabel("Subcarrier index")
            ax.set_ylabel("CSI magnitude")
            set_all_subcarrier_ticks(ax, magnitude.shape[1])
            ax.grid(alpha=0.25)
            ax.legend()

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"CSI magnitude vs subcarrier | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_selected_magnitude_surfaces_3d(
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError("Strides must be at least 1.")

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
            magnitude_plot = magnitude[::packet_stride, ::subcarrier_stride]
            packet_index = np.arange(magnitude.shape[0])[::packet_stride]
            subcarrier_index = np.arange(magnitude.shape[1])[::subcarrier_stride]
            subcarrier_grid, packet_grid = np.meshgrid(subcarrier_index, packet_index)

            ax = fig.add_subplot(row_count, column_count, index, projection="3d")
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
            set_all_subcarrier_ticks(ax, magnitude.shape[1])

        fig.suptitle(f"3D CSI magnitude | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No magnitude data found for the selected location/ESPs.")


def plot_magnitude_analysis_interactive(
    magnitudes: CsiMap,
    packet_stride_3d: int = 1,
    subcarrier_stride_3d: int = 1,
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
    plot_selected_magnitude_surfaces_3d(
        magnitudes,
        location_key,
        esp_keys,
        packet_stride=packet_stride_3d,
        subcarrier_stride=subcarrier_stride_3d,
        column_count=column_count,
    )
