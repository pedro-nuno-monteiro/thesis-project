from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

from utils.config import PLOT_DPI, PLOT_FORMAT
from utils.csi_processing import process_magnitude_data

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

LOCATION_PATTERN = re.compile(r"^(?P<row>[A-Z])[-_ ]?(?P<column>\d+)$")
EMPTY_ROOM_LOCATION = "Z-0"
BAND_ORDER = ("2.4 GHz", "5 GHz", "Fusion")
FLOOR_PLAN_ROWS = "ABCDEF"
ROOM_PATCH_COLORS = {
    1: "#d6eaf8",
    2: "#d5f5e3",
    3: "#fef9e7",
    0: "#f2f3f4",
}

CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
CsiStageMaps = dict[str, CsiMap]
SelectedMagnitudeStages = dict[str, tuple[np.ndarray, np.ndarray]]
SelectedEntry = tuple[str, np.ndarray | None]
SelectedTrialGroup = tuple[str, str, str, str, list[SelectedEntry]]
CSI_LOCATION_PATTERN = re.compile(
    r"^(?:(?P<letter>[A-G])(?:-)?(?P<number>[1-9]|1[0-4])|Z-?0)$"
)
LOW_FREQUENCY_ESP_IDS = range(1, 11)
HIGH_FREQUENCY_ESP_OFFSET = 10
DB_EPSILON = 1e-12
EMPTY_ROOM_LOCATION_KEY = "location_Z-0"
MAGNITUDE_DIMS = 2
VISUALIZATION_MIN_DB = -80.0
INVALID_AXIS_MESSAGE = "axis must be 'x' or 'y'."
INVALID_STRIDE_MESSAGE = "Strides must be at least 1."


def plot_localization_error_cdf_by_model(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    save_path: str | Path | None = None,
) -> None:
    """Plot one distance-error CDF curve per model for a single band."""
    _validate_columns(predictions, {"dataset", "model", "distance_error"})
    dataset_predictions = predictions.loc[predictions["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for model_name, model_predictions in dataset_predictions.groupby("model", sort=False):
        errors = _numeric_distance_errors(model_predictions)
        if errors.empty:
            continue
        values = np.sort(errors.to_numpy(dtype=float))
        cdf = np.arange(1, values.size + 1) / values.size
        ax.plot(values, cdf, linewidth=2, label=str(model_name))
        plotted = True

    if plotted:
        ax.legend(title="Model")
    else:
        ax.text(0.5, 0.5, "No valid distance errors", ha="center", va="center")
    ax.set_title(f"{dataset} - localization error CDF by model")
    ax.set_xlabel("Distance error (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.grid(visible=True, alpha=0.3)
    _save_and_show(fig, save_path)


def plot_band_error_cdf(
    predictions: pd.DataFrame,
    *,
    model_label: str,
    split_modes: tuple[str, ...],
    band_order: Sequence[str] = BAND_ORDER,
    save_path: str | Path | None = None,
) -> None:
    """Plot one distance-error CDF curve per band for a single model."""
    _validate_columns(predictions, {"dataset", "split", "distance_error"})
    splits = [split for split in split_modes if split in set(predictions["split"].astype(str))]

    for split in splits:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        split_predictions = predictions.loc[predictions["split"] == split]
        plotted = False
        for band in [band for band in band_order if band in set(split_predictions["dataset"])]:
            errors = _numeric_distance_errors(
                split_predictions.loc[split_predictions["dataset"] == band]
            )
            if errors.empty:
                continue
            values = np.sort(errors.to_numpy(dtype=float))
            cdf = np.arange(1, values.size + 1) / values.size
            ax.plot(values, cdf, linewidth=2, label=band)
            plotted = True

        if plotted:
            ax.legend(title="Band")
        else:
            ax.text(0.5, 0.5, "No valid distance errors", ha="center", va="center")
        ax.set_title(f"{split} split")
        ax.set_xlabel("Distance error (m)")
        ax.set_ylabel("Cumulative probability")
        ax.set_ylim(0, 1.02)
        ax.grid(visible=True, alpha=0.3)
        fig.suptitle(f"Position error CDF by frequency band - {model_label}")
        fig.tight_layout()
        output = None
        if save_path is not None:
            output = (
                Path(save_path)
                / f"cdf_{_slugify(model_label)}_all-bands_{split}.{PLOT_FORMAT}"
            )
        _save_and_show(fig, output)


def plot_model_band_error_boxplot(
    predictions: pd.DataFrame,
    *,
    models: Sequence[str],
    bands: Sequence[str],
    save_path: str | Path | None = None,
) -> None:
    """Plot distance-error boxplots grouped by model and band."""
    _validate_columns(predictions, {"model", "dataset", "distance_error"})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels: list[str] = []
    values: list[np.ndarray] = []

    for model in models:
        for band in bands:
            errors = _numeric_distance_errors(
                predictions.loc[
                    (predictions["model"] == model) & (predictions["dataset"] == band)
                ]
            )
            if not errors.empty:
                labels.append(f"{model}\n{band}")
                values.append(errors.to_numpy(dtype=float))

    if values:
        ax.boxplot(values, labels=labels, showmeans=True, meanline=True)
    else:
        ax.text(0.5, 0.5, "No distance errors", ha="center", va="center")
    ax.set_ylabel("Distance error (m)")
    ax.grid(axis="y", alpha=0.3)
    _save_and_show(fig, save_path)


def plot_global_position_confusion_matrix(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    normalize: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot true vs predicted position labels for a global classifier."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    _validate_columns(predictions, {"dataset", "true_location", "pred_location"})
    filtered = predictions.loc[predictions["dataset"] == dataset]
    if filtered.empty:
        print(f"No predictions found for dataset {dataset!r}.")
        return

    labels = _location_values(filtered["true_location"], filtered["pred_location"])
    matrix = confusion_matrix(
        filtered["true_location"],
        filtered["pred_location"],
        labels=labels,
        normalize=normalize,
    )
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    width = max(5, min(18, 0.45 * len(labels) + 3))
    height = max(4, min(16, 0.45 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(width, height))
    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format=".2f" if normalize is not None else "d",
    )
    ax.set_title(f"{dataset} - global position confusion matrix")
    ax.tick_params(axis="x", labelrotation=45)
    _save_and_show(fig, save_path)


def plot_position_confusion_by_true_room(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    normalize: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot one position confusion matrix per true room."""
    _validate_columns(predictions, {"dataset", "true_room"})
    dataset_predictions = predictions.loc[predictions["dataset"] == dataset]
    output_dir = Path(save_path) if save_path is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for room in sorted(dataset_predictions["true_room"].dropna().unique()):
        room_predictions = dataset_predictions.loc[dataset_predictions["true_room"] == room]
        output = (
            output_dir / f"confusion_room_{room}.{PLOT_FORMAT}"
            if output_dir is not None
            else None
        )
        plot_global_position_confusion_matrix(
            room_predictions,
            dataset=dataset,
            normalize=normalize,
            save_path=output,
        )


def plot_floor_plan_heatmap(
    predictions: pd.DataFrame,
    *,
    title: str = "",
    annotate: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Overlay per-grid-point accuracy and mean distance error on the room layout."""
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    _validate_columns(predictions, {"true_location", "pred_location", "distance_error"})
    records = _floor_plan_records(predictions)
    if not records:
        print("No valid location data for floor plan heatmap.")
        return

    max_col = max(record["col_idx"] for record in records)
    n_rows = len(FLOOR_PLAN_ROWS)
    n_cols = max_col + 1
    all_errors = [record["mean_error"] for record in records if not np.isnan(record["mean_error"])]
    error_vmax = float(np.percentile(all_errors, 95)) if all_errors else 1.0
    metrics = [
        ("accuracy", "RdYlGn", 0.0, 1.0, "Accuracy", lambda value: f"{value:.0%}"),
        ("mean_error", "RdYlGn_r", 0.0, error_vmax, "Mean error (m)", lambda value: f"{value:.2f}"),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(14, n_cols * 1.1 * 2 + 2), max(5, n_rows * 1.2 + 1)),
    )

    for ax, (metric_key, cmap_name, vmin, vmax, metric_label, fmt_fn) in zip(axes, metrics):
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps[cmap_name]
        ax.set_xlim(-0.5, max_col + 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        _draw_floor_background(ax, n_rows, n_cols, mpatches)
        _draw_floor_metric_cells(ax, records, metric_key, cmap, norm, annotate, fmt_fn, mpatches)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([str(index + 1) for index in range(n_cols)])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(list(FLOOR_PLAN_ROWS))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(metric_label)
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable, ax=ax, label=metric_label, shrink=0.8)

    fig.suptitle(title or "Floor plan - per-position accuracy and mean distance error")
    _save_and_show(fig, save_path)


def _floor_plan_records(predictions: pd.DataFrame) -> list[dict[str, float]]:
    """Aggregate accuracy and distance error for each valid floor-plan position."""
    records = []
    for location, group in predictions.groupby("true_location"):
        match = LOCATION_PATTERN.fullmatch(_normalize_location_label(location))
        if match is None:
            continue
        row_letter = match.group("row")
        if row_letter not in FLOOR_PLAN_ROWS:
            continue
        errors = _numeric_distance_errors(group)
        records.append(
            {
                "row_idx": FLOOR_PLAN_ROWS.index(row_letter),
                "col_idx": int(match.group("column")) - 1,
                "accuracy": float((group["true_location"] == group["pred_location"]).mean()),
                "mean_error": float(errors.mean()) if not errors.empty else np.nan,
            }
        )
    return records


def _draw_floor_background(ax, n_rows: int, n_cols: int, mpatches) -> None:
    """Draw room-colored floor-plan cells and their grid lines."""
    for col_i in range(n_cols):
        for row_i in range(n_rows):
            room = _room_id(chr(ord("A") + row_i), col_i + 1)
            ax.add_patch(
                mpatches.Rectangle(
                    (col_i - 0.5, row_i - 0.5),
                    1.0,
                    1.0,
                    linewidth=0,
                    facecolor=ROOM_PATCH_COLORS.get(room, "#f2f3f4"),
                    zorder=0,
                )
            )
    for row_i in range(n_rows + 1):
        ax.axhline(row_i - 0.5, color="gray", linewidth=0.3, alpha=0.4, zorder=0)
    for col_i in range(n_cols + 1):
        ax.axvline(col_i - 0.5, color="gray", linewidth=0.3, alpha=0.4, zorder=0)


def _draw_floor_metric_cells(
    ax,
    records,
    metric_key,
    cmap,
    norm,
    annotate,
    fmt_fn,
    mpatches,
) -> None:
    """Overlay colored and optionally annotated metric cells on a floor plan."""
    for record in records:
        row_idx = record["row_idx"]
        col_idx = record["col_idx"]
        value = record[metric_key]
        face_color = (0.8, 0.8, 0.8, 1.0) if np.isnan(value) else tuple(cmap(norm(value)))
        ax.add_patch(
            mpatches.Rectangle(
                (col_idx - 0.45, row_idx - 0.45),
                0.9,
                0.9,
                linewidth=0.5,
                edgecolor="white",
                facecolor=face_color,
                zorder=1,
            )
        )
        if annotate and not np.isnan(value):
            brightness = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]
            ax.text(
                col_idx,
                row_idx,
                fmt_fn(value),
                ha="center",
                va="center",
                fontsize=7,
                color="black" if brightness > 0.5 else "white",
                zorder=2,
            )


def _room_id(row_letter: str, col_num: int) -> int:
    """Map a floor-plan grid coordinate to its room identifier."""
    if row_letter in "ABCDEF" and col_num in range(1, 10):
        return 1
    if (row_letter == "A" and col_num in {13, 14}) or (
        row_letter in "BC" and col_num in range(10, 15)
    ):
        return 2
    if row_letter in "EF" and col_num in range(10, 14):
        return 3
    return 0


def _location_values(*location_series: pd.Series) -> list[str]:
    """Return unique location labels in physical grid order."""
    locations = {
        str(location)
        for series in location_series
        for location in series.dropna().unique()
    }
    return sorted(locations, key=_location_sort_key)


def _location_sort_key(location: str) -> tuple[int, str, int | str]:
    """Return a stable grid-aware sort key for a location label."""
    normalized = _normalize_location_label(location)
    if normalized == EMPTY_ROOM_LOCATION:
        return (1, "Z", 0)
    match = LOCATION_PATTERN.fullmatch(normalized)
    if match is None:
        return (2, normalized, normalized)
    return (0, match.group("row"), int(match.group("column")))


def _normalize_location_label(location: object) -> str:
    """Normalize a location value to its uppercase grid label."""
    return str(location).strip().upper().removeprefix("LOCATION_")


def _numeric_distance_errors(predictions: pd.DataFrame) -> pd.Series:
    """Return valid numeric distance-error values from prediction rows."""
    return pd.to_numeric(predictions["distance_error"], errors="coerce").dropna()


def _slugify(value: str) -> str:
    """Convert a display value to a compact filename-safe slug."""
    return value.lower().replace(".", "-").replace(" ", "-").strip("-")


def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error when plotting data lacks required columns."""
    missing = sorted(required_columns - set(df.columns))
    if missing:
        msg = f"Missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def _save_and_show(fig, save_path: str | Path | None) -> None:
    """Finalize a figure, optionally save it, and display it."""
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path).with_suffix(f".{PLOT_FORMAT}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            bbox_inches="tight",
            dpi=PLOT_DPI,
            format=PLOT_FORMAT,
        )
        print(
            f"[plots] saved {output} at dpi={PLOT_DPI}. PNG is raster; set "
            "PLOT_FORMAT='pdf' for a LaTeX vector figure without other changes."
        )
    plt.show()


def plot_lovo_fold_spread(
    lovo_per_fold: pd.DataFrame,
    *,
    bands: Sequence[str],
    model: str = "RF",
    save_path: str | Path | None = None,
) -> None:
    """Plot held-out-user position accuracy spread for the selected model."""
    _validate_columns(
        lovo_per_fold,
        {"model", "dataset", "held_out_user", "position_accuracy"},
    )
    model_key = str(model).casefold()
    filtered = lovo_per_fold.loc[
        lovo_per_fold["model"].astype(str).str.casefold() == model_key
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = []
    labels = []
    positions = []
    for index, band in enumerate(bands, start=1):
        band_values = pd.to_numeric(
            filtered.loc[
                filtered["dataset"].astype(str).str.casefold() == str(band).casefold(),
                "position_accuracy",
            ],
            errors="coerce",
        ).dropna()
        if band_values.empty:
            continue
        values.append(band_values.to_numpy(dtype=float))
        labels.append(band)
        positions.append(index)

    if values:
        ax.boxplot(values, positions=positions, labels=labels, widths=0.45)
        for position, band_values in zip(positions, values):
            offsets = np.linspace(-0.08, 0.08, num=len(band_values))
            ax.scatter(
                np.full(len(band_values), position) + offsets,
                band_values,
                color="black",
                s=28,
                zorder=3,
            )
    else:
        ax.text(0.5, 0.5, "No LOVO fold metrics", ha="center", va="center")
    ax.set_ylabel("Position accuracy")
    ax.set_title(f"LOVO held-out-user spread - {model}")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    _save_and_show(fig, save_path)


def plot_block_vs_lovo_position_accuracy(
    global_summary: pd.DataFrame,
    lovo_summary: pd.DataFrame,
    *,
    bands: Sequence[str],
    model: str = "RF",
    save_path: str | Path | None = None,
) -> None:
    """Plot block position accuracy beside LOVO fold-mean accuracy."""
    model_key = str(model).casefold()
    rows = []
    for band in bands:
        block_row = global_summary.loc[
            (global_summary["model"].astype(str).str.casefold() == model_key)
            & (global_summary["dataset"] == band)
            & (global_summary["split"] == "block")
        ]
        lovo_row = lovo_summary.loc[
            (lovo_summary["model"].astype(str).str.casefold() == model_key)
            & (lovo_summary["dataset"] == band)
        ]
        if block_row.empty or lovo_row.empty:
            continue
        rows.append(
            {
                "dataset": band,
                "block": float(block_row.iloc[0]["position_accuracy"]),
                "lovo": float(lovo_row.iloc[0]["position_accuracy_mean"]),
            }
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if rows:
        x = np.arange(len(rows))
        width = 0.35
        block_values = np.array([row["block"] for row in rows])
        lovo_values = np.array([row["lovo"] for row in rows])
        ax.bar(x - width / 2, block_values, width, label="Block")
        ax.bar(x + width / 2, lovo_values, width, label="LOVO mean")
        for index, (block_value, lovo_value) in enumerate(zip(block_values, lovo_values)):
            gap = block_value - lovo_value
            ax.text(
                index,
                max(block_value, lovo_value) + 0.025,
                f"gap {gap:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([row["dataset"] for row in rows])
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Need both block and LOVO RF results", ha="center", va="center")
    ax.set_ylabel("Position accuracy")
    ax.set_title(f"Block vs LOVO cross-user generalization cost - {model}")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    _save_and_show(fig, save_path)


def save_training_curves(history: pd.DataFrame, save_path: str | Path, title: str) -> None:
    """Save the CNN training and validation loss/accuracy curves."""
    output = Path(save_path).with_suffix(f".{PLOT_FORMAT}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title(f"{title} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.savefig(output, dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close(fig)
    print(
        f"[plots] saved {output} at dpi={PLOT_DPI}. PNG is raster; set "
        "PLOT_FORMAT='pdf' for a LaTeX vector figure without other code changes."
    )


def set_all_subcarrier_ticks(ax: Axes, subcarrier_count: int) -> None:
    """Label every subcarrier index on an axis."""
    subcarrier_index = np.arange(subcarrier_count)
    ax.set_xticks(subcarrier_index)
    ax.set_xticklabels(subcarrier_index, rotation=90, fontsize=6)


def set_sparse_index_ticks(ax: Axes, values: np.ndarray, axis: str) -> None:
    """Add a small, evenly spaced set of index ticks to an axis."""
    if values.size == 0:
        return
    tick_count = min(8, values.size)
    tick_positions = np.unique(np.linspace(0, values.size - 1, tick_count, dtype=int))
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
    """Convert linear CSI magnitude to decibels with a numerical floor."""
    magnitude_array = np.asarray(magnitude, dtype=float)
    return 20.0 * np.log10(np.clip(magnitude_array, epsilon, None))


def select_aligned_magnitude_interval(  # noqa: PLR0913
    magnitude_stages: CsiStageMaps,
    *,
    position: str,
    user: str,
    trial: str,
    anchor_pair: int,
    packet_count: int,
    packet_selection: str,
) -> SelectedMagnitudeStages:
    """Select one shared ordinal packet interval from three aligned CSI stages."""
    required_stages = ("raw", "calibrated", "normalized")
    missing_stages = [stage for stage in required_stages if stage not in magnitude_stages]
    if missing_stages:
        msg = f"Magnitude-stage data is missing: {', '.join(missing_stages)}."
        raise ValueError(msg)
    if packet_count <= 0:
        raise ValueError("packet_count must be greater than zero.")

    location_key = normalize_location_input(str(position))
    if location_key is None:
        raise ValueError(f"Invalid CSI position: {position!r}.")
    user_key = _indexed_recording_key(user, "user")
    trial_key = _indexed_recording_key(trial, "trial")
    low_esp_key, high_esp_key = _anchor_pair_keys(anchor_pair)
    esp_keys = (low_esp_key, high_esp_key)
    scenario_key = _unique_recording_scenario(
        magnitude_stages["calibrated"],
        location_key=location_key,
        user_key=user_key,
        trial_key=trial_key,
        esp_keys=esp_keys,
    )

    calibrated_pair = tuple(
        _recording_magnitude(
            magnitude_stages["calibrated"],
            scenario_key=scenario_key,
            location_key=location_key,
            user_key=user_key,
            esp_key=esp_key,
            trial_key=trial_key,
        )
        for esp_key in esp_keys
    )
    available_by_anchor = [magnitude.shape[0] for magnitude in calibrated_pair]
    common_packet_count = min(available_by_anchor)
    if common_packet_count < packet_count:
        msg = (
            f"Requested {packet_count} common valid packets, but only "
            f"{common_packet_count} are available for {low_esp_key} and {high_esp_key} "
            f"(counts: {available_by_anchor[0]} and {available_by_anchor[1]})."
        )
        raise ValueError(msg)
    start = _packet_interval_start(
        common_packet_count,
        packet_count=packet_count,
        method=packet_selection,
    )
    packet_slice = slice(start, start + packet_count)

    selected: SelectedMagnitudeStages = {}
    for stage_name in required_stages:
        stage_pair = []
        for esp_index, esp_key in enumerate(esp_keys):
            magnitude = _recording_magnitude(
                magnitude_stages[stage_name],
                scenario_key=scenario_key,
                location_key=location_key,
                user_key=user_key,
                esp_key=esp_key,
                trial_key=trial_key,
            )
            expected_shape = calibrated_pair[esp_index].shape
            if magnitude.shape != expected_shape:
                msg = (
                    f"Packet alignment failed for {stage_name}/{esp_key}: expected shape "
                    f"{expected_shape}, got {magnitude.shape}."
                )
                raise ValueError(msg)
            stage_pair.append(magnitude[packet_slice])
        selected[stage_name] = (stage_pair[0], stage_pair[1])
    return selected


def plot_csi_magnitude_stages(  # noqa: PLR0913
    magnitude_stages: CsiStageMaps,
    *,
    position: str,
    user: str,
    trial: str,
    anchor_pair: int,
    packet_count: int,
    packet_selection: str,
    normalized_limit_percentile: float,
    surface_elevation: float,
    surface_azimuth: float,
    save: bool,
    output_directory: str | Path,
) -> None:
    """Plot paired heatmaps and 3D surfaces for raw, calibrated, and normalized CSI."""
    if not 0 < normalized_limit_percentile <= 100:
        raise ValueError("normalized_limit_percentile must be in (0, 100].")
    calibrated_magnitudes = magnitude_stages.get("calibrated")
    if calibrated_magnitudes is None:
        raise ValueError("Magnitude-stage data is missing the calibrated CSI map.")

    normalized_magnitudes, _ = process_magnitude_data(
        calibrated_magnitudes,
        normalization="empty_baseline",
        baseline_scope="per_session",
    )
    stages_with_normalized = {
        **magnitude_stages,
        "normalized": normalized_magnitudes,
    }
    selected = select_aligned_magnitude_interval(
        stages_with_normalized,
        position=position,
        user=user,
        trial=trial,
        anchor_pair=anchor_pair,
        packet_count=packet_count,
        packet_selection=packet_selection,
    )

    low_esp_key, high_esp_key = _anchor_pair_keys(anchor_pair)
    panel_titles = (
        f"(a) ESP {format_esp_key(low_esp_key)} — 2.4 GHz",
        f"(b) ESP {format_esp_key(high_esp_key)} — 5 GHz",
    )
    subtitle = (
        f"Position {format_location_key(normalize_location_input(str(position)) or '')} · "
        f"User {_display_recording_id(user)} · Trial {_display_recording_id(trial)} · "
        f"{packet_count} packets"
    )
    stage_settings = (
        ("raw", "Raw CSI magnitude", "csi_raw", "CSI magnitude (dB)", True),
        (
            "calibrated",
            "RSSI-calibrated CSI magnitude",
            "csi_calibrated",
            "CSI magnitude (dB)",
            True,
        ),
        (
            "normalized",
            "Empty-room-normalized CSI",
            "csi_normalized",
            "Normalized CSI value",
            False,
        ),
    )
    for stage_name, figure_title, filename_stem, value_label, convert_to_db in stage_settings:
        values = selected[stage_name]
        plot_values = (
            (magnitude_to_db(values[0]), magnitude_to_db(values[1]))
            if convert_to_db
            else values
        )
        cmap, norm = _magnitude_color_mapping(
            plot_values,
            normalized=not convert_to_db,
            normalized_limit_percentile=normalized_limit_percentile,
        )
        heatmap_figure = _plot_magnitude_heatmap_pair(
            plot_values,
            panel_titles=panel_titles,
            figure_title=figure_title,
            subtitle=subtitle,
            value_label=value_label,
            cmap=cmap,
            norm=norm,
        )
        _finish_magnitude_figure(
            heatmap_figure,
            save=save,
            output_directory=output_directory,
            filename=f"{filename_stem}_heatmap.png",
        )
        surface_figure = _plot_magnitude_surface_pair(
            plot_values,
            panel_titles=panel_titles,
            figure_title=figure_title,
            subtitle=subtitle,
            value_label=value_label,
            cmap=cmap,
            norm=norm,
            elevation=surface_elevation,
            azimuth=surface_azimuth,
        )
        _finish_magnitude_figure(
            surface_figure,
            save=save,
            output_directory=output_directory,
            filename=f"{filename_stem}_3d.png",
        )


def _indexed_recording_key(value: object, prefix: str) -> str:
    normalized = str(value).strip().lower().removeprefix(f"{prefix}_")
    if not normalized.isdigit():
        raise ValueError(f"Invalid {prefix} identifier: {value!r}.")
    return f"{prefix}_{int(normalized):02d}"


def _display_recording_id(value: object) -> str:
    normalized = str(value).strip().rsplit("_", maxsplit=1)[-1]
    return f"{int(normalized):02d}" if normalized.isdigit() else normalized


def _anchor_pair_keys(anchor_pair: int) -> tuple[str, str]:
    try:
        anchor_id = int(anchor_pair)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid anchor pair: {anchor_pair!r}.") from error
    if not 1 <= anchor_id <= 10:
        raise ValueError("anchor_pair must identify a 2.4 GHz ESP from 1 through 10.")
    return f"esp_{anchor_id:02d}", f"esp_{anchor_id + HIGH_FREQUENCY_ESP_OFFSET:02d}"


def _unique_recording_scenario(
    magnitudes: CsiMap,
    *,
    location_key: str,
    user_key: str,
    trial_key: str,
    esp_keys: tuple[str, str],
) -> str:
    matching_scenarios = []
    for scenario_key, locations_map in magnitudes.items():
        esps_map = locations_map.get(location_key, {}).get(user_key, {})
        if all(trial_key in esps_map.get(esp_key, {}) for esp_key in esp_keys):
            matching_scenarios.append(scenario_key)
    if not matching_scenarios:
        identity = f"{location_key}/{user_key}/{trial_key}/{esp_keys[0]}+{esp_keys[1]}"
        raise ValueError(f"No paired CSI recording was found for {identity}.")
    if len(matching_scenarios) > 1:
        msg = (
            "The selected recording exists in multiple scenarios: "
            f"{', '.join(matching_scenarios)}."
        )
        raise ValueError(msg)
    return matching_scenarios[0]


def _recording_magnitude(
    magnitudes: CsiMap,
    *,
    scenario_key: str,
    location_key: str,
    user_key: str,
    esp_key: str,
    trial_key: str,
) -> np.ndarray:
    try:
        magnitude = magnitudes[scenario_key][location_key][user_key][esp_key][trial_key]
    except KeyError as error:
        identity = f"{scenario_key}/{location_key}/{user_key}/{esp_key}/{trial_key}"
        raise ValueError(f"CSI recording stage is missing for {identity}.") from error
    magnitude_array = np.asarray(magnitude)
    if magnitude_array.ndim != MAGNITUDE_DIMS:
        raise ValueError(f"CSI magnitude for {esp_key} must be a 2D array.")
    return magnitude_array


def _packet_interval_start(available: int, *, packet_count: int, method: str) -> int:
    method_key = str(method).strip().lower()
    if method_key == "start":
        return 0
    if method_key == "middle":
        return (available - packet_count) // 2
    if method_key == "end":
        return available - packet_count
    raise ValueError("packet_selection must be 'start', 'middle', or 'end'.")


def _magnitude_color_mapping(
    values: tuple[np.ndarray, np.ndarray],
    *,
    normalized: bool,
    normalized_limit_percentile: float,
) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite_values.size == 0:
        raise ValueError("The selected CSI interval contains no finite values to plot.")
    if normalized:
        absolute_values = np.abs(finite_values)
        limit = float(np.percentile(absolute_values, normalized_limit_percentile))
        if not np.isfinite(limit) or limit <= 0:
            limit = float(np.max(absolute_values))
        if not np.isfinite(limit) or limit <= 0:
            limit = float(np.finfo(float).eps)
        colors = plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, 257))
        colors[128] = (1.0, 1.0, 1.0, 1.0)
        cmap = mcolors.ListedColormap(colors, name="coolwarm_zero_white")
        return cmap, mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if lower == upper:
        padding = max(abs(lower), 1.0) * 1e-6
        lower -= padding
        upper += padding
    return plt.get_cmap("viridis"), mcolors.Normalize(vmin=lower, vmax=upper)


def _plot_magnitude_heatmap_pair(  # noqa: PLR0913
    values: tuple[np.ndarray, np.ndarray],
    *,
    panel_titles: tuple[str, str],
    figure_title: str,
    subtitle: str,
    value_label: str,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    images = []
    for ax, panel_values, panel_title in zip(axes, values, panel_titles):
        packet_total, subcarrier_total = panel_values.shape
        image = ax.imshow(
            panel_values,
            origin="lower",
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            norm=norm,
            extent=(-0.5, subcarrier_total - 0.5, 0.5, packet_total + 0.5),
        )
        images.append(image)
        ax.set_title(panel_title, fontsize=11, pad=8)
        ax.set_xlabel("Subcarrier index")
        ax.set_ylabel("Packet within selected interval")
        _set_magnitude_axis_ticks(ax, packet_total, subcarrier_total)
        _style_magnitude_axis(ax)
    fig.colorbar(images[0], ax=axes, label=value_label, shrink=0.86, pad=0.025)
    fig.suptitle(f"{figure_title}\n{subtitle}", fontsize=14, linespacing=1.45)
    fig.patch.set_facecolor("white")
    return fig


def _plot_magnitude_surface_pair(  # noqa: PLR0913
    values: tuple[np.ndarray, np.ndarray],
    *,
    panel_titles: tuple[str, str],
    figure_title: str,
    subtitle: str,
    value_label: str,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    elevation: float,
    azimuth: float,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.2),
        constrained_layout=True,
        subplot_kw={"projection": "3d"},
    )
    surfaces = []
    for ax, panel_values, panel_title in zip(axes, values, panel_titles):
        packet_total, subcarrier_total = panel_values.shape
        subcarrier_index, packet_index = np.meshgrid(
            np.arange(subcarrier_total),
            np.arange(1, packet_total + 1),
        )
        surface = ax.plot_surface(
            subcarrier_index,
            packet_index,
            panel_values,
            rstride=1,
            cstride=1,
            linewidth=0,
            cmap=cmap,
            norm=norm,
            antialiased=True,
        )
        surfaces.append(surface)
        ax.set_title(panel_title, fontsize=11, pad=8)
        ax.set_xlabel("Subcarrier index", labelpad=8)
        ax.set_ylabel("Packet within selected interval", labelpad=8)
        ax.set_zlabel(value_label, labelpad=8)
        ax.set_zlim(norm.vmin, norm.vmax)
        ax.view_init(elev=elevation, azim=azimuth)
        _set_magnitude_axis_ticks(ax, packet_total, subcarrier_total)
        _style_magnitude_axis(ax, is_3d=True)
    fig.colorbar(surfaces[0], ax=axes, label=value_label, shrink=0.68, pad=0.035)
    fig.suptitle(f"{figure_title}\n{subtitle}", fontsize=14, linespacing=1.45)
    fig.patch.set_facecolor("white")
    return fig


def _set_magnitude_axis_ticks(ax: Axes, packet_total: int, subcarrier_total: int) -> None:
    x_ticks = np.unique(np.linspace(0, subcarrier_total - 1, min(8, subcarrier_total), dtype=int))
    y_ticks = np.unique(np.linspace(1, packet_total, min(7, packet_total), dtype=int))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)


def _style_magnitude_axis(ax: Axes, *, is_3d: bool = False) -> None:
    ax.set_facecolor("white")
    ax.tick_params(colors="#262626", labelsize=8)
    ax.xaxis.label.set_color("#262626")
    ax.yaxis.label.set_color("#262626")
    if is_3d:
        ax.zaxis.label.set_color("#262626")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor("#d9d9d9")
            axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.55)  # noqa: SLF001
        return
    ax.grid(color="#d0d0d0", linewidth=0.45, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8a8a8a")
    ax.spines["bottom"].set_color("#8a8a8a")


def _finish_magnitude_figure(
    fig,
    *,
    save: bool,
    output_directory: str | Path,
    filename: str,
) -> None:
    if save:
        output_path = Path(output_directory) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[plots] saved {output_path} at dpi=300.")
    plt.show()


def visualization_magnitude_db(
    magnitude: np.ndarray,
    min_db: float = VISUALIZATION_MIN_DB,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert magnitude to dB and retain packets above the display threshold."""
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
    """Render a labelled placeholder when a magnitude plot has no valid values."""
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
    """Hide an unused magnitude subplot."""
    ax.set_axis_off()


def normalize_location_input(value: str) -> str | None:
    """Normalize interactive location input to a nested-map key."""
    normalized_value = value.strip().replace(" ", "").upper()
    normalized_value = normalized_value.removeprefix("LOCATION_")
    match = CSI_LOCATION_PATTERN.fullmatch(normalized_value)
    if match is None:
        return None
    if normalized_value.replace("-", "") == "Z0":
        return EMPTY_ROOM_LOCATION_KEY
    return f"location_{match.group('letter')}-{match.group('number')}"


def normalize_esp_input(value: str) -> str | None:
    """Normalize interactive ESP input to a nested-map key."""
    normalized_value = value.strip().lower().removeprefix("esp_")
    if not normalized_value.isdigit():
        return None
    return f"esp_{int(normalized_value):02d}"


def format_location_key(location_key: str) -> str:
    """Format a location map key for display."""
    return location_key.removeprefix("location_")


def format_esp_key(esp_key: str) -> str:
    """Format an ESP map key for display."""
    return esp_key.removeprefix("esp_")


def sorted_location_keys(location_keys: set[str]) -> list[str]:
    """Sort nested-map location keys in physical grid order."""
    def sort_key(location_key: str) -> tuple[str, int]:
        """Return the row and numeric column used for location ordering."""
        location = format_location_key(location_key)
        if location == "Z-0":
            return "Z", 0
        letter, number = location.split("-", maxsplit=1)
        return letter, int(number)

    return sorted(location_keys, key=sort_key)


def sorted_esp_keys(esp_keys: set[str]) -> list[str]:
    """Sort ESP keys by their numeric identifier."""
    return sorted(esp_keys, key=lambda esp_key: int(format_esp_key(esp_key)))


def paired_esp_keys(esp_keys: list[str]) -> list[str]:
    """Order 2.4 GHz and 5 GHz ESP anchors in corresponding pairs."""
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
    ordered_esp_keys.extend(esp_by_id[esp_id] for esp_id in sorted(set(esp_by_id) - used_ids))
    ordered_esp_keys.extend(unordered_esp_keys)
    return ordered_esp_keys


def get_available_location_keys(magnitudes: CsiMap) -> list[str]:
    """Return locations that contain at least one magnitude recording."""
    return sorted_location_keys(
        {
            location_key
            for locations_map in magnitudes.values()
            for location_key in locations_map
        }
    )


def get_available_esp_keys_for_location(magnitudes: CsiMap, location_key: str) -> list[str]:
    """Return ESP keys with data for a selected location."""
    esp_keys: set[str] = set()
    for locations_map in magnitudes.values():
        users_map = locations_map.get(location_key)
        if users_map is None:
            continue
        for esps_map in users_map.values():
            esp_keys.update(esps_map)
    return sorted_esp_keys(esp_keys)


def prompt_yes_no(prompt: str) -> bool:
    """Prompt interactively until the user supplies a yes-or-no answer."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes", "s", "sim"}:
            return True
        if answer in {"n", "no", "nao", "não"}:
            return False
        print("Please answer yes or no.")


def prompt_location_key(magnitudes: CsiMap) -> str | None:
    """Prompt for one available location, allowing cancellation."""
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
    """Prompt for one or more ESPs available at a location."""
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
    """Yield matching magnitude recordings for selected locations and ESPs."""
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
                }
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
    """Return a compact row/column layout for a number of plots."""
    row_count = max(1, (entry_count + column_count - 1) // column_count)
    return row_count, column_count


def hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    """Hide subplot axes that were not populated."""
    for ax in axes.ravel()[used_count:]:
        ax.set_visible(False)


def plot_selected_magnitude_profiles(
    magnitudes: CsiMap,
    location_key: str,
    esp_keys: list[str],
    column_count: int = 2,
) -> None:
    """Plot packet-averaged magnitude profiles for selected recordings."""
    plot_count = 0
    # Keep each recording group in its own figure so trial and user identity stay
    # visible while ESP profiles are compared.
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
            ax.set_title(f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown")
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
    """Plot packet-by-subcarrier magnitude heatmaps for selected recordings."""
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
            ax.set_title(f"{esp_key} | {magnitude_db.shape[0]}/{magnitude.shape[0]} packets shown")
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
    """Stack profiles with the most common subcarrier width."""
    if not profiles:
        return np.empty((0, 0), dtype=float)
    width_counts: dict[int, int] = {}
    for profile in profiles:
        width_counts[profile.shape[0]] = width_counts.get(profile.shape[0], 0) + 1
    target_width = max(width_counts, key=width_counts.get)
    return np.vstack([profile for profile in profiles if profile.shape[0] == target_width])


def collect_user_mean_profiles_db(
    magnitudes: CsiMap,
    scenario_key: str,
    location_key: str,
    esp_key: str,
) -> dict[str, np.ndarray]:
    """Collect each user's mean dB profile for one location and ESP."""
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
            if magnitude_db.size:
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
    """Return the aggregate empty-room dB profile for one ESP when available."""
    empty_user_profiles = collect_user_mean_profiles_db(
        magnitudes,
        scenario_key,
        empty_room_location_key,
        esp_key,
    )
    matching_profiles = [
        profile for profile in empty_user_profiles.values() if profile.shape == target_shape
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
    """Compare average user profiles and optional empty-room baselines by ESP."""
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
            # Resolve an empty-room reference only after the occupied profiles
            # establish the compatible subcarrier shape.
            empty_profile = empty_room_mean_profile_db(
                magnitudes,
                scenario_key,
                esp_key,
                mean_profile.shape,
                empty_room_location_key=empty_room_location_key,
            )
            entries.append(
                (esp_key, stacked_users.shape[0], mean_profile, std_profile, empty_profile)
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
            f"{scenario_key} / {location_key}"
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
    """Interactively select and display raw or normalized CSI magnitude data."""
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
