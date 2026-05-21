from __future__ import annotations

from collections.abc import Iterator
import re

import matplotlib.pyplot as plt
import numpy as np

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
MagnitudeSetMap = dict[str, CsiMap]
PairEntry = tuple[str, str, np.ndarray, np.ndarray]
TrialPairGroup = tuple[str, str, str, str, list[PairEntry]]
SelectedEntry = tuple[str, np.ndarray]
SelectedTrialGroup = tuple[str, str, str, str, list[SelectedEntry]]
SelectedMagnitudeSetEntry = tuple[str, np.ndarray | None]
SelectedMagnitudeSetGroup = tuple[str, str, str, str, str, list[SelectedMagnitudeSetEntry]]

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


def get_magnitude_entry(
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


def plot_no_magnitude_data(ax, label: str) -> None:
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

            subcarrier_index = np.arange(magnitude.shape[1])
            ax.plot(subcarrier_index, magnitude.T, color="tab:blue", alpha=0.035, linewidth=0.45)
            ax.plot(
                subcarrier_index,
                magnitude.mean(axis=0),
                color="black",
                linewidth=3.0,
                label="Mean",
            )
            ax.set_title(f"{label} | {magnitude.shape[0]} packets")
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


def plot_selected_magnitude_set_surfaces_3d(
    magnitude_sets: MagnitudeSetMap,
    location_key: str,
    esp_keys: list[str],
    packet_stride: int = 1,
    subcarrier_stride: int = 1,
    column_count: int = 2,
) -> None:
    if packet_stride < 1 or subcarrier_stride < 1:
        raise ValueError("Strides must be at least 1.")

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
            set_all_subcarrier_ticks(ax, magnitude.shape[1])

        fig.suptitle(f"3D CSI magnitude | {title}")
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
    plot_selected_magnitude_set_surfaces_3d(
        magnitude_sets,
        location_key,
        esp_keys,
        packet_stride=packet_stride_3d,
        subcarrier_stride=subcarrier_stride_3d,
        column_count=column_count,
    )


def extract_complex_csi_from_file(file_path: str | Path, its5ghz: bool) -> np.ndarray:
    import pandas as pd
    import re

    file_csv = pd.read_csv(str(file_path), header=None, usecols=[7, 14]) if its5ghz else pd.read_csv(str(file_path), header=None, usecols=[24])
    csi_raw = file_csv.iloc[:, 1] if its5ghz else file_csv.iloc[:, 0]

    total_values_2_4 = 128
    total_sc_5 = 114
    valid_csi = []

    for _, entry in csi_raw.items():
        match = re.search(r"\[(.*?)\]", str(entry))
        if not match:
            continue
        nums = [float(n) for n in re.findall(r"-?\d+", match.group(1))]
        if (its5ghz and len(nums) == total_sc_5) or (not its5ghz and len(nums) == total_values_2_4):
            valid_csi.append(nums)

    if len(valid_csi) == 0:
        return np.empty((0, 0), dtype=complex)

    valid_csi = np.array(valid_csi)
    if not its5ghz:
        real = valid_csi[:, 1::2]
        imag = valid_csi[:, ::2]
        complex_csi = real + 1j * imag
        fft_csi = np.fft.fftshift(complex_csi, axes=1)
        active_sc = fft_csi[:, 6:58]
        active_sc = np.delete(active_sc, [26, 27], axis=1)
    else:
        real = valid_csi[:, 1::2]
        imag = valid_csi[:, ::2]
        complex_csi = real + 1j * imag
        active_sc = np.delete(complex_csi, [28], axis=1)

    return active_sc


def iter_selected_file_groups(files: FileMap, location_key: str, esp_keys: list[str]) -> Iterator[SelectedTrialGroup]:
    for scenario_key, locations_map in files.items():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue

        for user_key, esps_map in users_map.items():
            trial_keys = sorted({trial_key for esp_key in esp_keys for trial_key in esps_map.get(esp_key, {})})

            for trial_key in trial_keys:
                entries: list[SelectedEntry] = []
                for esp_key in esp_keys:
                    file_path = esps_map.get(esp_key, {}).get(trial_key)
                    if file_path is None:
                        continue
                    entries.append((esp_key, file_path))

                if entries:
                    yield (scenario_key, location_key, user_key, trial_key, entries)


def plot_selected_iq_constellations(
    files: FileMap,
    location_key: str,
    esp_keys: list[str],
    sc_indices: list[int] | None = None,
    sample_stride: int = 1,
    column_count: int = 2,
) -> None:
    plot_count = 0

    for scenario_key, _, user_key, trial_key, entries in iter_selected_file_groups(files, location_key, esp_keys):
        row_count, column_count = make_subplot_grid(len(entries), column_count)
        fig, axes = plt.subplots(row_count, column_count, figsize=(5 * column_count, max(4.5, 4.0 * row_count)), squeeze=False, constrained_layout=True)
        title = f"{scenario_key} / {location_key} / {user_key} / {trial_key}"

        # default single subcarrier 0
        if sc_indices is None:
            sc_indices_local = [0]
        else:
            sc_indices_local = list(sc_indices)

        for ax, (esp_key, file_path) in zip(axes.ravel(), entries):
            try:
                esp_id = int(esp_key.removeprefix("esp_"))
            except Exception:
                esp_id = None
            its5ghz = esp_id is not None and 11 <= esp_id <= 20
            cplx = extract_complex_csi_from_file(file_path, its5ghz)
            if cplx.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"{esp_key} | no samples")
                ax.set_axis_off()
                continue

            # plot multiple subcarriers with color cycle and legend
            colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", plt.cm.tab10.colors)
            for idx, sc in enumerate(sc_indices_local):
                if sc < 0 or sc >= cplx.shape[1]:
                    continue
                samples = cplx[:, sc][::sample_stride]
                ax.scatter(samples.real, samples.imag, s=6, alpha=0.7, color=colors[idx % len(colors)], label=f"SC {sc}")

            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{esp_key} | {trial_key}")
            ax.set_xlabel("I (Real)")
            ax.set_ylabel("Q (Imag)")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=6)

        hide_unused_axes(axes, len(entries))
        fig.suptitle(f"I/Q Constellations | {title}")
        plt.show()
        plot_count += 1

    if plot_count == 0:
        print("No I/Q data found for the selected location/ESPs.")


def get_available_location_keys_from_files(files: FileMap) -> list[str]:
    return sorted_location_keys({location_key for locations_map in files.values() for location_key in locations_map})


def get_available_esp_keys_for_location_files(files: FileMap, location_key: str) -> list[str]:
    esp_keys: set[str] = set()
    for locations_map in files.values():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue
        for esps_map in users_map.values():
            esp_keys.update(esps_map)
    return sorted_esp_keys(esp_keys)


def plot_iq_interactive(files: FileMap, column_count: int = 2) -> None:
    if not prompt_yes_no("Do you want to show I/Q constellation plots? [yes/no] "):
        print("Skipping I/Q plots.")
        return

    available_locations = get_available_location_keys_from_files(files)
    if not available_locations:
        print("No locations available in files.")
        return

    location_text = ", ".join(format_location_key(l) for l in available_locations)
    print(f"Available locations: {location_text}")
    while True:
        answer = input("Which position/location do you want to analyse? ").strip()
        location_key = normalize_location_input(answer)
        if location_key in available_locations:
            break
        print("Invalid location. Use one of the available locations, e.g. A1 or A-1.")

    esp_keys = prompt_esp_keys(files, location_key) if callable(globals().get("prompt_esp_keys")) else get_available_esp_keys_for_location_files(files, location_key)
    if not esp_keys:
        return

    sc_text = input("Subcarrier indices to plot (comma-separated, e.g. 0 or 0,1,2). Default 0: ").strip()
    if not sc_text:
        sc_indices = [0]
    else:
        nums = re.findall(r"-?\d+", sc_text)
        sc_indices = [int(n) for n in nums] if nums else [0]

    plot_selected_iq_constellations(files, location_key, esp_keys, sc_indices=sc_indices, column_count=column_count)
