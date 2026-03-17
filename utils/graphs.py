import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

csi_map = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]

# Scenario ID decoder
# Example: 21313 -> "Scenario 2, 2.4 Ghz, Random, 1 meter, 3 ESPs"

SCENARIO_ID_MAPS = {
    "scenario_number": {
        "1": "Scen. 1",
        "2": "Scen. 2",
    },
    "frequency_band": {
        "1": "2.4 Ghz",
        "2": "5 Ghz",
        "3": "Mixed freq.",
    },
    "sensor_placement": {
        "1": "Lay Down",
        "2": "Standing",
        "3": "Random plac.",
    },
    "height": {
        "0": "floor",
        "1": "1 m",
        "2": "2 m",
        "3": "Random height",
    },
    "esp_count": {
        "1": "1 ESP",
        "2": "2 ESPs",
        "3": "3 ESPs",
        "4": "4 ESPs",
        "5": "5 ESPs",
    },
}


def decode_scenario_id(scenario_id: str | int) -> dict[str, str]:
    scenario_str = str(scenario_id)

    if scenario_str.startswith("scenario_"):
        scenario_str = scenario_str.split("_", 1)[1]

    if len(scenario_str) != 5 or not scenario_str.isdigit():
        raise ValueError(f"Invalid scenario_id '{scenario_id}'. Expected 5 digits, e.g. '21313'.")

    d1, d2, d3, d4, d5 = scenario_str

    return {
        "scenario_id": scenario_str,
        "scenario_number": SCENARIO_ID_MAPS["scenario_number"].get(d1, f"Unknown ({d1})"),
        "frequency_band": SCENARIO_ID_MAPS["frequency_band"].get(d2, f"Unknown ({d2})"),
        "sensor_placement": SCENARIO_ID_MAPS["sensor_placement"].get(d3, f"Unknown ({d3})"),
        "height": SCENARIO_ID_MAPS["height"].get(d4, f"Unknown ({d4})"),
        "esp_count": SCENARIO_ID_MAPS["esp_count"].get(d5, f"Unknown ({d5})"),
    }


def scenario_id_to_label(scenario_id: str | int) -> str:
    decoded = decode_scenario_id(scenario_id)
    return "; ".join(
        [
            decoded["scenario_number"],
            decoded["frequency_band"],
            decoded["sensor_placement"],
            decoded["height"],
            decoded["esp_count"],
        ],
    )


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "plot"


def _measurement_plot_filename(
    scenario: str,
    user: str,
    activity: str,
    esp: str,
    trial: str,
    suffix: str = "",
) -> str:
    base = "_".join(
        [
            _sanitize_filename_part(scenario),
            _sanitize_filename_part(user),
            _sanitize_filename_part(activity),
            _sanitize_filename_part(esp),
            _sanitize_filename_part(trial),
        ],
    )
    suffix_clean = _sanitize_filename_part(suffix) if suffix else ""
    if suffix_clean:
        return f"{base}_{suffix_clean}.png"
    return f"{base}.png"


def plot_csi_magnitude_3d(
    magnitude: np.ndarray,
    title: str,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 7),
    save_path: str | Path | None = None,
) -> None:
    if magnitude.size == 0:
        print(f"[SKIP] Empty magnitude matrix for: {title}")
        return

    n_ts, n_esop = magnitude.shape
    ts = np.arange(n_ts)
    esop = np.arange(n_esop)

    # Build a grid aligned with magnitude shape: (TS, ESOP).
    ts_grid, esop_grid = np.meshgrid(ts, esop, indexing="ij")

    # Create a figure with 3D surface and magnitude heatmap side by side
    fig = plt.figure(figsize=(figsize[0] * 2.0, figsize[1]))
    gs = gridspec.GridSpec(1, 2)

    # 3D surface
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    surface = ax3d.plot_surface(
        esop_grid,
        ts_grid,
        magnitude,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
    )

    ax3d.set_xlabel("Subcarrier index")
    ax3d.set_ylabel("Packet number")
    ax3d.set_zlabel("Magnitude")
    ax3d.set_title(title)
    fig.colorbar(surface, ax=ax3d, shrink=0.6, aspect=14, pad=0.1, label="Magnitude")

    # 2D heatmap: magnitude
    ax_mag = fig.add_subplot(gs[0, 1])
    im_mag = ax_mag.imshow(
        magnitude,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[esop.min(), esop.max(), ts.min(), ts.max()],
    )
    ax_mag.set_xlabel("Subcarrier index")
    ax_mag.set_ylabel("Packet number")
    ax_mag.set_title("Magnitude heatmap")
    fig.colorbar(im_mag, ax=ax_mag, shrink=0.7, pad=0.02, label="Magnitude")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.show()


def draw_3d_graphs_for_all_files(
    magnitude_data: csi_map,
    max_plots: int | None = None,
    scenarios: set[str] | str | None = None,
    users: set[str] | str | None = None,
    activities: set[str] | str | None = None,
    esps: set[str] | str | None = None,
    trials: set[str] | str | None = None,
    save_dir: str | Path | None = None,
) -> None:
    if isinstance(scenarios, str):
        scenarios = {scenarios}
    if isinstance(users, str):
        users = {users}
    if isinstance(activities, str):
        activities = {activities}
    if isinstance(esps, str):
        esps = {esps}
    if isinstance(trials, str):
        trials = {trials}

    if save_dir is None:
        save_dir = Path("images") / "saved_graphs" / "3D graph"

    plotted = 0

    for scenario_key, users_map in magnitude_data.items():
        if scenarios is not None and scenario_key not in scenarios:
            continue

        for user_key, activities_map in users_map.items():
            if users is not None and user_key not in users:
                continue

            for activity_key, esps_map in activities_map.items():
                if activities is not None and activity_key not in activities:
                    continue

                for esp_key, trials_map in esps_map.items():
                    if esps is not None and esp_key not in esps:
                        continue

                    for trial_key, magnitude in trials_map.items():
                        if trials is not None and trial_key not in trials:
                            continue

                        title = (
                            f"{scenario_id_to_label(scenario_key)}; User={user_key}; "
                            f"Activity={activity_key}; ESP={esp_key}; Trial={trial_key}"
                        )
                        save_path = None
                        if save_dir is not None:
                            save_path = Path(save_dir) / _measurement_plot_filename(
                                scenario_key,
                                user_key,
                                activity_key,
                                esp_key,
                                trial_key,
                                suffix="3d",
                            )
                        plot_csi_magnitude_3d(magnitude, title=title, save_path=save_path)

                        plotted += 1
                        if max_plots is not None and plotted >= max_plots:
                            print(f"Stopped after {plotted} plots (max_plots reached).")
                            return


def _reduce_profile(magnitude: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "mean":
        return np.mean(magnitude, axis=0)
    if reduction == "median":
        return np.median(magnitude, axis=0)
    if reduction == "max":
        return np.max(magnitude, axis=0)
    raise ValueError("reduction must be one of: 'mean', 'median', 'max'")


def _amplitude_to_db(amplitude: np.ndarray, db_floor: float = -120.0) -> np.ndarray:
    # Stable conversion for amplitude-like signals, with floor to avoid -inf.
    eps = np.finfo(float).tiny
    amplitude_safe = np.maximum(amplitude, eps)
    amplitude_db = 20 * np.log10(amplitude_safe)
    return np.maximum(amplitude_db, db_floor)


def plot_csi_magnitude_vs_subcarrier(
    magnitude: np.ndarray,
    title: str,
    figsize: tuple[int, int] = (10, 5),
    packet_idx: int | None = None,
    reduction: str = "mean",
    show_db: bool = True,
    db_floor: float = -120.0,
) -> None:
    if magnitude.size == 0:
        print(f"[SKIP] Empty magnitude matrix for: {title}")
        return

    n_packets, n_subcarriers = magnitude.shape
    subcarrier_idx = np.arange(n_subcarriers)

    if packet_idx is not None:
        if packet_idx < 0 or packet_idx >= n_packets:
            print(f"[SKIP] packet_idx must be in [0, {n_packets - 1}]")
            return
        profile = magnitude[packet_idx, :]
        profile_label = f"Packet {packet_idx}"
    else:
        try:
            profile = _reduce_profile(magnitude, reduction)
        except ValueError as exc:
            print(f"[SKIP] {exc}")
            return
        profile_label = f"{reduction.capitalize()} across packets"

    if show_db:
        fig, (ax_mag, ax_db) = plt.subplots(
            2,
            1,
            figsize=(figsize[0], max(6, int(figsize[1] * 1.6))),
            sharex=True,
        )

        ax_mag.plot(subcarrier_idx, profile, linewidth=2, color="steelblue")
        ax_mag.set_ylabel("CSI magnitude")
        ax_mag.set_title(f"{title} ({profile_label})")
        ax_mag.grid(alpha=0.3)

        profile_db = _amplitude_to_db(profile, db_floor=db_floor)
        ax_db.plot(subcarrier_idx, profile_db, linewidth=2, color="darkorange")
        ax_db.set_xlabel("Subcarrier index")
        ax_db.set_ylabel("Amplitude (dB)")
        ax_db.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        plt.plot(subcarrier_idx, profile, linewidth=2)
        plt.xlabel("Subcarrier index")
        plt.ylabel("CSI magnitude")
        plt.title(f"{title} ({profile_label})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_scenario_comparison_vs_subcarrier(
    magnitude_data: csi_map,
    scenario: str,
    selections: list[tuple[str, str, str, str]],
    reduction: str = "mean",
    figsize: tuple[int, int] = (12, 6),
    show_db: bool = True,
    db_floor: float = -120.0,
    save_path: str | Path | None = None,
) -> None:
    """Overlay multiple (user, activity, esp, trial) profiles in one graph for the same scenario."""
    if show_db:
        fig, (ax_mag, ax_db) = plt.subplots(
            2,
            1,
            figsize=(figsize[0], max(7, int(figsize[1] * 1.7))),
            sharex=True,
        )
    else:
        plt.figure(figsize=figsize)
        ax_mag = plt.gca()
        ax_db = None

    # Assign a consistent color per ESP, dashed line for user_01
    unique_esps = list(dict.fromkeys(esp for _, _, esp, _ in selections))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    esp_colors = {esp: color_cycle[i % len(color_cycle)] for i, esp in enumerate(unique_esps)}

    plotted = 0
    for user, activity, esp, trial in selections:
        try:
            magnitude = magnitude_data[scenario][user][activity][esp][trial]
        except KeyError:
            print(
                f"[SKIP] Missing data: scenario={scenario}, user={user}, activity={activity}, esp={esp}, trial={trial}",
            )
            continue

        if magnitude.size == 0:
            print(
                f"[SKIP] Empty magnitude: scenario={scenario}, user={user}, activity={activity}, esp={esp}, trial={trial}",
            )
            continue

        profile = _reduce_profile(magnitude, reduction)
        n_subcarriers = magnitude.shape[1]
        subcarrier_idx = np.arange(n_subcarriers)
        label = f"{user} | {activity} | {esp} | {trial}"
        color = esp_colors[esp]
        linestyle = "--" if user == "user_01" else "-"

        ax_mag.plot(
            subcarrier_idx, profile, linewidth=2, label=label, color=color, linestyle=linestyle
        )
        if show_db and ax_db is not None:
            profile_db = _amplitude_to_db(profile, db_floor=db_floor)
            ax_db.plot(
                subcarrier_idx,
                profile_db,
                linewidth=2,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        plotted += 1

    if plotted == 0:
        if show_db:
            plt.close(fig)
        else:
            plt.close()
        print("[SKIP] No valid series to plot.")
        return

    ax_mag.set_ylabel("CSI magnitude")
    ax_mag.set_title(f"{scenario_id_to_label(scenario)} - CSI vs subcarrier ({reduction})")
    ax_mag.grid(alpha=0.3)
    ax_mag.legend(loc="best", fontsize=9)

    if show_db and ax_db is not None:
        ax_db.set_xlabel("Subcarrier index")
        ax_db.set_ylabel("Amplitude (dB)")
        ax_db.grid(alpha=0.3)
        ax_db.legend(loc="best", fontsize=9)
    else:
        ax_mag.set_xlabel("Subcarrier index")

    if save_path is None:
        save_path = (
            Path("images")
            / "saved_graphs"
            / "2D graph magnitude comparison"
            / f"{_sanitize_filename_part(scenario)}_comparison_{_sanitize_filename_part(reduction)}_{len(selections)}series.png"
        )

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_to_save = fig if show_db else ax_mag.figure
        fig_to_save.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.show()


def plot_subcarrier_magnitude_vs_time(
    magnitude_data: csi_map,
    scenario: str,
    user: str,
    activity: str,
    esp: str,
    trial: str,
    subcarrier_idx: int,
    sampling_rate_hz: float = 1.0,
    figsize: tuple[int, int] = (12, 6),
    show_db: bool = True,
    db_floor: float = -120.0,
    linewidth: float = 1.5,
    save_path: str | Path | None = None,
) -> None:
    scenario_label = scenario_id_to_label(scenario)

    try:
        magnitude = magnitude_data[scenario][user][activity][esp][trial]
    except KeyError:
        print(
            f"[SKIP] Missing data: scenario={scenario_label}, user={user}, "
            f"activity={activity}, esp={esp}, trial={trial}",
        )
        return

    if magnitude is None or magnitude.size == 0:
        print("[SKIP] Empty magnitude matrix.")
        return

    if sampling_rate_hz <= 0:
        print("[SKIP] sampling_rate_hz must be > 0.")
        return

    n_packets, n_subcarriers = magnitude.shape

    if subcarrier_idx < 0 or subcarrier_idx >= n_subcarriers:
        print(f"[SKIP] subcarrier_idx must be in [0, {n_subcarriers - 1}]")
        return

    time_s = np.arange(n_packets) / sampling_rate_hz
    mag_series = magnitude[:, subcarrier_idx]

    title = (
        f"{scenario_label} | {user} | {activity} | {esp} | {trial} | Subcarrier={subcarrier_idx}"
    )

    if save_path is None:
        save_path = (
            Path("images")
            / "saved_graphs"
            / "2D graph subcarrier vs time"
            / _measurement_plot_filename(
                scenario,
                user,
                activity,
                esp,
                trial,
                suffix=f"subcarrier_{subcarrier_idx}_vs_time",
            )
        )

    if show_db:
        fig, (ax_mag, ax_db) = plt.subplots(
            2,
            1,
            figsize=(figsize[0], max(7, int(figsize[1] * 1.7))),
            sharex=True,
        )

        ax_mag.plot(time_s, mag_series, color="steelblue", linewidth=linewidth)
        ax_mag.set_ylabel("CSI magnitude")
        ax_mag.set_title(title)
        ax_mag.grid(alpha=0.3)

        db_series = _amplitude_to_db(mag_series, db_floor=db_floor)
        ax_db.plot(time_s, db_series, color="darkorange", linewidth=linewidth)
        ax_db.set_xlabel("Time (s)")
        ax_db.set_ylabel("Amplitude (dB)")
        ax_db.grid(alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
        plt.show()
    else:
        fig = plt.figure(figsize=figsize)
        plt.plot(time_s, mag_series, color="steelblue", linewidth=linewidth)
        plt.xlabel("Time (s)")
        plt.ylabel("CSI magnitude")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
        plt.show()


def plot_all_packets_vs_subcarrier(
    magnitude_data: csi_map,
    scenario: str,
    user: str,
    activity: str,
    esp: str,
    trial: str,
    figsize: tuple[int, int] = (12, 6),
    max_packets: int | None = 300,
    alpha: float = 0.08,
    linewidth: float = 0.8,
    x_tick_step: int = 2,
    show_db: bool = True,
    db_floor: float = -120.0,
    save_path: str | Path | None = None,
) -> None:
    try:
        magnitude = magnitude_data[scenario][user][activity][esp][trial]
    except KeyError:
        print(
            f"[SKIP] Missing data: scenario={scenario}, user={user}, "
            f"activity={activity}, esp={esp}, trial={trial}",
        )
        return

    if magnitude is None or magnitude.size == 0:
        print("[SKIP] Empty magnitude matrix.")
        return

    n_packets, n_subcarriers = magnitude.shape
    subcarrier_idx = np.arange(n_subcarriers)

    if max_packets is not None and n_packets > max_packets:
        step = int(np.ceil(n_packets / max_packets))
        packet_indices = np.arange(0, n_packets, step)
        print(f"Plotting {len(packet_indices)} packets (downsampled from {n_packets}, step={step})")
    else:
        packet_indices = np.arange(n_packets)
        print(f"Plotting all {n_packets} packets")

    x_tick_step = max(1, x_tick_step)

    if save_path is None:
        save_path = (
            Path("images")
            / "saved_graphs"
            / "2D graph all packets vs subcarrier"
            / _measurement_plot_filename(
                scenario,
                user,
                activity,
                esp,
                trial,
                suffix="all_packets_vs_subcarrier",
            )
        )

    # Aggregate statistics across packets (time dimension)
    mean_profile = np.mean(magnitude, axis=0)
    std_profile = np.std(magnitude, axis=0)
    var_profile = np.var(magnitude, axis=0)

    if show_db:
        # Three stacked panels:
        #  - top: all packet traces + mean ± 1 std band (magnitude)
        #  - middle: all traces + mean profile in dB
        #  - bottom: temporal variance across packets per subcarrier
        fig, (ax_mag, ax_db, ax_var) = plt.subplots(
            3,
            1,
            figsize=(figsize[0], max(9, int(figsize[1] * 2.0))),
            sharex=True,
        )

        for pkt_idx in packet_indices:
            packet_mag = magnitude[pkt_idx, :]
            ax_mag.plot(
                subcarrier_idx,
                packet_mag,
                color="steelblue",
                alpha=alpha,
                linewidth=linewidth,
            )
            packet_db = _amplitude_to_db(packet_mag, db_floor=db_floor)
            ax_db.plot(
                subcarrier_idx,
                packet_db,
                color="darkorange",
                alpha=alpha,
                linewidth=linewidth,
            )

        # Mean profile with ±1 std shaded band (magnitude)
        ax_mag.fill_between(
            subcarrier_idx,
            mean_profile - std_profile,
            mean_profile + std_profile,
            color="crimson",
            alpha=0.2,
            label="Mean ± 1 std",
        )
        ax_mag.plot(
            subcarrier_idx,
            mean_profile,
            color="crimson",
            linewidth=2.2,
            label="Mean profile",
        )

        # Mean profile in dB
        mean_profile_db = _amplitude_to_db(mean_profile, db_floor=db_floor)
        ax_db.plot(
            subcarrier_idx,
            mean_profile_db,
            color="firebrick",
            linewidth=2.2,
            label="Mean profile (dB)",
        )

        # Temporal variance profile
        ax_var.plot(
            subcarrier_idx,
            var_profile,
            color="purple",
            linewidth=2.0,
        )

        ax_mag.set_ylabel("CSI magnitude")
        ax_mag.set_title(
            f"All packets | {scenario_id_to_label(scenario)} | {user} | {activity} | {esp} | {trial}\n"
            f"shape={magnitude.shape}",
        )
        ax_mag.grid(alpha=0.3)
        ax_mag.legend(loc="best")

        ax_db.set_ylabel("Amplitude (dB)")
        ax_db.grid(alpha=0.3)
        ax_db.legend(loc="best")

        ax_var.set_xlabel("Subcarrier index")
        ax_var.set_ylabel("Variance across packets")
        ax_var.set_xticks(np.arange(0, n_subcarriers, x_tick_step))
        ax_var.grid(alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
        plt.show()
    else:
        fig = plt.figure(figsize=figsize)

        for pkt_idx in packet_indices:
            plt.plot(
                subcarrier_idx,
                magnitude[pkt_idx, :],
                color="steelblue",
                alpha=alpha,
                linewidth=linewidth,
            )

        # Mean profile with ±1 std shaded band
        plt.fill_between(
            subcarrier_idx,
            mean_profile - std_profile,
            mean_profile + std_profile,
            color="crimson",
            alpha=0.2,
            label="Mean ± 1 std",
        )
        plt.plot(
            subcarrier_idx,
            mean_profile,
            color="crimson",
            linewidth=2.2,
            label="Mean profile",
        )

        plt.xlabel("Subcarrier index")
        plt.ylabel("CSI magnitude")
        plt.title(
            f"All packets | {scenario_id_to_label(scenario)} | {user} | {activity} | {esp} | {trial}\n"
            f"shape={magnitude.shape}",
        )
        plt.xticks(np.arange(0, n_subcarriers, x_tick_step))
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot: {save_path}")
        plt.show()
