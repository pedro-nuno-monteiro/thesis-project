from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            output = Path(save_path) / f"cdf_{_slugify(model_label)}_all-bands_{split}.pdf"
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
        output = output_dir / f"confusion_room_{room}.pdf" if output_dir is not None else None
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
    locations = {
        str(location)
        for series in location_series
        for location in series.dropna().unique()
    }
    return sorted(locations, key=_location_sort_key)


def _location_sort_key(location: str) -> tuple[int, str, int | str]:
    normalized = _normalize_location_label(location)
    if normalized == EMPTY_ROOM_LOCATION:
        return (1, "Z", 0)
    match = LOCATION_PATTERN.fullmatch(normalized)
    if match is None:
        return (2, normalized, normalized)
    return (0, match.group("row"), int(match.group("column")))


def _normalize_location_label(location: object) -> str:
    return str(location).strip().upper().removeprefix("LOCATION_")


def _numeric_distance_errors(predictions: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(predictions["distance_error"], errors="coerce").dropna()


def _slugify(value: str) -> str:
    return value.lower().replace(".", "-").replace(" ", "-").strip("-")


def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        msg = f"Missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def _save_and_show(fig, save_path: str | Path | None) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
