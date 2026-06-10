from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

METADATA_COLUMNS = {
    "frequency_scenario",
    "scenario",
    "location",
    "user",
    "trial",
    "group_id",
    "window_idx",
    "label",
}

EMPTY_ROOM_LABEL = 0
EMPTY_ROOM_LOCATION = "Z-0"
MIN_GROUP_SPLIT_COUNT = 2
DEFAULT_ROW_SPACING = 1.0
DEFAULT_COLUMN_SPACING = 1.0
GRID_ROW_ORIGIN = ord("A")
GRID_COLUMN_ORIGIN = 1
LOCATION_PATTERN = re.compile(r"^(?P<row>[A-Z])[-_ ]?(?P<column>\d+)$")
ROOM_1_COLUMNS = range(1, 10)
ROOM_2_A_COLUMNS = {13, 14}
ROOM_2_BC_COLUMNS = range(10, 15)
ROOM_3_EF_COLUMNS = range(10, 14)
HIERARCHICAL_MODEL_NAME = "hierarchical_rf"
GLOBAL_POSITION_MODEL_NAME = "global_position_rf"

PREDICTION_COLUMNS = [
    "dataset",
    "true_room",
    "pred_room",
    "true_location",
    "pred_location",
    "distance_error",
    "scenario",
    "user",
    "trial",
    "group_id",
    "window_idx",
]

DISTANCE_SUMMARY_COLUMNS = [
    "dataset",
    "samples",
    "mean_error",
    "median_error",
    "rmse_error",
    "p75_error",
    "p90_error",
    "max_error",
]

PER_ROOM_SUMMARY_COLUMNS = [
    "dataset",
    "true_room",
    "samples",
    "room_accuracy",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
]

GLOBAL_PREDICTION_COLUMNS = [
    "dataset",
    "model",
    "true_room",
    "pred_room",
    "true_location",
    "pred_location",
    "distance_error",
    "scenario",
    "user",
    "trial",
    "group_id",
    "window_idx",
]

LOCALIZATION_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "samples",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
]


@dataclass(frozen=True)
class GridDistanceOptions:
    """Spacing used to convert reference-point labels into physical coordinates."""

    row_spacing: float = DEFAULT_ROW_SPACING
    column_spacing: float = DEFAULT_COLUMN_SPACING


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def location_grid_coordinates(
    location: object,
    *,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[float, float] | None:
    """Convert reference-point labels such as A-1 into physical grid coordinates."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    location_label = _normalize_location_label(location)
    if location_label == EMPTY_ROOM_LOCATION:
        return None

    match = LOCATION_PATTERN.fullmatch(location_label)
    if match is None:
        return None

    row_index = ord(match.group("row")) - GRID_ROW_ORIGIN
    column_index = int(match.group("column")) - GRID_COLUMN_ORIGIN
    return row_index * row_spacing, column_index * column_spacing


def location_distance_error(
    true_location: object,
    pred_location: object,
    *,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> float:
    """Euclidean distance between two non-empty reference-point labels."""
    true_coordinates = location_grid_coordinates(
        true_location,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    pred_coordinates = location_grid_coordinates(
        pred_location,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    if true_coordinates is None or pred_coordinates is None:
        return np.nan

    row_error = true_coordinates[0] - pred_coordinates[0]
    column_error = true_coordinates[1] - pred_coordinates[1]
    return float(np.hypot(row_error, column_error))


def split_dataframe(
    df: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by group_id so windows from the same trial do not leak across sets."""
    _validate_required_columns(df, {"group_id", "label"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)
    if df["group_id"].nunique() < MIN_GROUP_SPLIT_COUNT:
        msg = "Group split requires at least two unique group_id values."
        raise ValueError(msg)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["group_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


@dataclass
class HierarchicalPositionClassifier:
    """Two-stage CSI classifier: room first, then room-specific reference point."""

    room_model: object
    position_models: dict[int, object]
    feature_columns: list[str]
    fallback_locations: dict[int, str]

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict room labels and reference point labels for a feature dataframe."""
        _validate_required_columns(df, set(self.feature_columns))
        features = df[self.feature_columns]
        pred_rooms = np.asarray(self.room_model.predict(features), dtype=int)
        pred_locations = np.empty(len(df), dtype=object)

        for raw_room_label in np.unique(pred_rooms):
            room_label = int(raw_room_label)
            room_mask = pred_rooms == room_label
            if room_label == EMPTY_ROOM_LABEL:
                pred_locations[room_mask] = EMPTY_ROOM_LOCATION
                continue

            position_model = self.position_models.get(room_label)
            if position_model is None:
                pred_locations[room_mask] = self.fallback_locations.get(
                    room_label,
                    EMPTY_ROOM_LOCATION,
                )
                continue

            room_indices = np.flatnonzero(room_mask)
            pred_locations[room_mask] = position_model.predict(features.iloc[room_indices])

        return pred_rooms, pred_locations


def train_hierarchical_position_classifier(
    train_df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> HierarchicalPositionClassifier:
    """Train one room classifier and one position classifier per non-empty room."""
    _validate_training_dataframe(train_df)
    columns = feature_columns(train_df)
    room_labels = train_df["label"].astype(int)

    room_model = _random_forest_classifier(random_state=random_state)
    room_model.fit(train_df[columns], room_labels)

    fallback_locations = _fallback_locations(train_df, room_labels)
    position_models: dict[int, object] = {}

    for raw_room_label in sorted(room_labels.unique()):
        room_label = int(raw_room_label)
        if room_label == EMPTY_ROOM_LABEL:
            continue

        room_df = train_df.loc[room_labels == room_label]
        position_model = _random_forest_classifier(random_state=random_state)
        position_model.fit(room_df[columns], room_df["location"].astype(str))
        position_models[room_label] = position_model

    return HierarchicalPositionClassifier(
        room_model=room_model,
        position_models=position_models,
        feature_columns=columns,
        fallback_locations=fallback_locations,
    )


def run_hierarchical_position_experiment(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    test_size: float = 0.3,
    random_state: int = 42,
    distance_options: GridDistanceOptions | None = None,
) -> tuple[HierarchicalPositionClassifier, pd.DataFrame, dict[str, float]]:
    """Train and evaluate the hierarchical classifier for one feature dataframe."""
    distance_options = _resolve_distance_options(distance_options)
    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    model = train_hierarchical_position_classifier(train_df, random_state=random_state)
    pred_rooms, pred_locations = model.predict(test_df)
    distance_errors = _distance_errors(
        test_df["location"],
        pred_locations,
        row_spacing=distance_options.row_spacing,
        column_spacing=distance_options.column_spacing,
    )

    predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "true_room": test_df["label"].astype(int).to_numpy(),
            "pred_room": pred_rooms,
            "true_location": test_df["location"].astype(str).to_numpy(),
            "pred_location": pred_locations,
            "distance_error": distance_errors,
            "scenario": test_df["scenario"].to_numpy(),
            "user": test_df["user"].to_numpy(),
            "trial": test_df["trial"].to_numpy(),
            "group_id": test_df["group_id"].to_numpy(),
            "window_idx": test_df["window_idx"].to_numpy(),
        },
        columns=PREDICTION_COLUMNS,
    )
    metrics = _metrics(predictions)

    return model, predictions, metrics


def run_hierarchical_position_experiments(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, HierarchicalPositionClassifier]]:
    """Run the hierarchical classifier for every named feature dataframe."""
    distance_options = GridDistanceOptions(
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    _validate_grid_spacing(
        row_spacing=distance_options.row_spacing,
        column_spacing=distance_options.column_spacing,
    )
    summary_rows: list[dict[str, float | str]] = []
    predictions_by_dataset: dict[str, pd.DataFrame] = {}
    models: dict[str, HierarchicalPositionClassifier] = {}

    for dataset_name, dataframe in feature_dataframes.items():
        if dataframe.empty:
            predictions_by_dataset[dataset_name] = pd.DataFrame(columns=PREDICTION_COLUMNS)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "room_accuracy": np.nan,
                    "position_accuracy": np.nan,
                    "mean_distance_error": np.nan,
                    "median_distance_error": np.nan,
                    "rmse_distance_error": np.nan,
                    "distance_error_samples": 0.0,
                    "localization_samples": 0.0,
                    "samples": 0.0,
                },
            )
            continue

        model, predictions, metrics = run_hierarchical_position_experiment(
            dataframe,
            dataset_name=dataset_name,
            test_size=test_size,
            random_state=random_state,
            distance_options=distance_options,
        )
        models[dataset_name] = model
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append({"dataset": dataset_name, **metrics})

    return pd.DataFrame(summary_rows), predictions_by_dataset, models


def run_global_position_experiment(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    test_size: float = 0.3,
    random_state: int = 42,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict[str, float]]:
    """Train and evaluate a direct CSI-to-position Random Forest baseline."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label"})

    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    columns = feature_columns(train_df)
    model = _random_forest_classifier(random_state=random_state)
    model.fit(train_df[columns], train_df["location"].astype(str))

    pred_locations = model.predict(test_df[columns])
    pred_rooms = np.asarray(
        [room_label_for_location(location) for location in pred_locations],
        dtype=object,
    )
    distance_errors = _distance_errors(
        test_df["location"],
        pred_locations,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )

    predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": GLOBAL_POSITION_MODEL_NAME,
            "true_room": test_df["label"].astype(int).to_numpy(),
            "pred_room": pred_rooms,
            "true_location": test_df["location"].astype(str).to_numpy(),
            "pred_location": pred_locations,
            "distance_error": distance_errors,
            "scenario": test_df["scenario"].to_numpy(),
            "user": test_df["user"].to_numpy(),
            "trial": test_df["trial"].to_numpy(),
            "group_id": test_df["group_id"].to_numpy(),
            "window_idx": test_df["window_idx"].to_numpy(),
        },
        columns=GLOBAL_PREDICTION_COLUMNS,
    )
    metrics = _localization_metrics(predictions)

    return model, predictions, metrics


def run_global_position_experiments(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, RandomForestClassifier]]:
    """Run the direct global position baseline for every named feature dataframe."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    summary_rows: list[dict[str, float | str]] = []
    predictions_by_dataset: dict[str, pd.DataFrame] = {}
    models: dict[str, RandomForestClassifier] = {}

    for dataset_name, dataframe in feature_dataframes.items():
        if dataframe.empty:
            predictions_by_dataset[dataset_name] = pd.DataFrame(
                columns=GLOBAL_PREDICTION_COLUMNS,
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "position_accuracy": np.nan,
                    "mean_distance_error": np.nan,
                    "median_distance_error": np.nan,
                    "rmse_distance_error": np.nan,
                    "samples": 0.0,
                },
            )
            continue

        model, predictions, metrics = run_global_position_experiment(
            dataframe,
            dataset_name=dataset_name,
            test_size=test_size,
            random_state=random_state,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        models[dataset_name] = model
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append({"dataset": dataset_name, **metrics})

    return pd.DataFrame(summary_rows), predictions_by_dataset, models


def combine_localization_predictions(
    hierarchical_predictions: dict[str, pd.DataFrame],
    global_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine hierarchical and direct global localization predictions."""
    combined_predictions: list[pd.DataFrame] = []

    for predictions in hierarchical_predictions.values():
        hierarchical_df = predictions.copy()
        if "model" not in hierarchical_df.columns:
            hierarchical_df.insert(1, "model", HIERARCHICAL_MODEL_NAME)
        else:
            hierarchical_df["model"] = hierarchical_df["model"].fillna(HIERARCHICAL_MODEL_NAME)
        combined_predictions.append(_align_localization_prediction_columns(hierarchical_df))

    for predictions in global_predictions.values():
        global_df = predictions.copy()
        if "model" not in global_df.columns:
            global_df.insert(1, "model", GLOBAL_POSITION_MODEL_NAME)
        else:
            global_df["model"] = global_df["model"].fillna(GLOBAL_POSITION_MODEL_NAME)
        combined_predictions.append(_align_localization_prediction_columns(global_df))

    if not combined_predictions:
        return pd.DataFrame(columns=GLOBAL_PREDICTION_COLUMNS)

    return pd.concat(combined_predictions, ignore_index=True)


def localization_model_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize localization quality by dataset and model."""
    _validate_required_columns(
        predictions,
        {"dataset", "model", "true_location", "pred_location", "distance_error"},
    )
    summary_rows: list[dict[str, object]] = []

    for (dataset_name, model_name), model_predictions in predictions.groupby(
        ["dataset", "model"],
        sort=False,
    ):
        summary_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                **_localization_metrics(model_predictions),
            },
        )

    return pd.DataFrame(summary_rows, columns=LOCALIZATION_SUMMARY_COLUMNS)


def summarize_distance_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-NaN localization distance errors for each dataset."""
    summary_rows: list[dict[str, object]] = []

    for dataset_name, errors in _distance_error_groups(predictions):
        if errors.empty:
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "samples": 0,
                    "mean_error": np.nan,
                    "median_error": np.nan,
                    "rmse_error": np.nan,
                    "p75_error": np.nan,
                    "p90_error": np.nan,
                    "max_error": np.nan,
                },
            )
            continue

        error_values = errors.to_numpy(dtype=float)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "samples": int(error_values.size),
                "mean_error": float(np.mean(error_values)),
                "median_error": float(np.median(error_values)),
                "rmse_error": float(np.sqrt(np.mean(np.square(error_values)))),
                "p75_error": float(np.percentile(error_values, 75)),
                "p90_error": float(np.percentile(error_values, 90)),
                "max_error": float(np.max(error_values)),
            },
        )

    return pd.DataFrame(summary_rows, columns=DISTANCE_SUMMARY_COLUMNS)


def plot_distance_error_cdf(predictions: pd.DataFrame) -> None:
    """Plot one empirical CDF curve per dataset using non-NaN distance errors."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted_curve = False

    for dataset_name, errors in _distance_error_groups(predictions):
        if errors.empty:
            continue

        error_values = np.sort(errors.to_numpy(dtype=float))
        cdf = np.arange(1, error_values.size + 1) / error_values.size
        ax.plot(error_values, cdf, linewidth=2, label=str(dataset_name))
        plotted_curve = True

    if plotted_curve:
        ax.legend(title="Dataset")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid distance errors to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Hierarchical localization error CDF")
    ax.set_xlabel("Distance error (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_distance_error_boxplot(predictions: pd.DataFrame) -> None:
    """Plot localization distance-error distributions by dataset."""
    groups = [
        (dataset_name, errors)
        for dataset_name, errors in _distance_error_groups(predictions)
        if not errors.empty
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if groups:
        dataset_names = [str(dataset_name) for dataset_name, _ in groups]
        error_values = [errors.to_numpy(dtype=float) for _, errors in groups]
        ax.boxplot(error_values, labels=dataset_names, showmeans=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid distance errors to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Hierarchical localization distance error")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Distance error (m)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_global_position_confusion_matrix(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    normalize: str | None = None,
) -> None:
    """Plot true vs predicted position labels for the direct global baseline."""
    dataset_predictions = _dataset_predictions(predictions, dataset=dataset)
    if dataset_predictions.empty:
        return

    _plot_position_confusion_matrix(
        dataset_predictions,
        title=f"{dataset} - global position confusion matrix",
        normalize=normalize,
    )


def plot_localization_error_cdf_by_model(
    predictions: pd.DataFrame,
    *,
    dataset: str,
) -> None:
    """Plot localization distance-error CDF curves by model for one dataset."""
    _validate_required_columns(predictions, {"dataset", "model", "distance_error"})
    dataset_predictions = predictions.loc[predictions["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted_curve = False

    for model_name, model_predictions in dataset_predictions.groupby("model", sort=False):
        errors = _numeric_distance_errors(model_predictions)
        if errors.empty:
            continue

        error_values = np.sort(errors.to_numpy(dtype=float))
        cdf = np.arange(1, error_values.size + 1) / error_values.size
        ax.plot(error_values, cdf, linewidth=2, label=str(model_name))
        plotted_curve = True

    if plotted_curve:
        ax.legend(title="Model")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid distance errors to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(f"{dataset} - localization error CDF by model")
    ax.set_xlabel("Distance error (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def per_room_hierarchical_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize hierarchical localization quality per dataset and true room."""
    _validate_required_columns(
        predictions,
        {
            "dataset",
            "true_room",
            "pred_room",
            "true_location",
            "pred_location",
            "distance_error",
        },
    )
    summary_rows: list[dict[str, object]] = []

    for (dataset_name, true_room), room_predictions in predictions.groupby(
        ["dataset", "true_room"],
        sort=False,
    ):
        distance_errors = _numeric_distance_errors(room_predictions)
        error_values = distance_errors.to_numpy(dtype=float)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "true_room": true_room,
                "samples": len(room_predictions),
                "room_accuracy": _accuracy(
                    room_predictions["true_room"],
                    room_predictions["pred_room"],
                ),
                "position_accuracy": _accuracy(
                    room_predictions["true_location"],
                    room_predictions["pred_location"],
                ),
                "mean_distance_error": float(np.mean(error_values))
                if error_values.size
                else np.nan,
                "median_distance_error": float(np.median(error_values))
                if error_values.size
                else np.nan,
                "rmse_distance_error": float(np.sqrt(np.mean(np.square(error_values))))
                if error_values.size
                else np.nan,
            },
        )

    return pd.DataFrame(summary_rows, columns=PER_ROOM_SUMMARY_COLUMNS)


def plot_position_confusion_by_true_room(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    normalize: str | None = None,
) -> None:
    """Plot end-to-end position confusion matrices for each true room."""
    dataset_predictions = _dataset_predictions(predictions, dataset=dataset)

    for room in _room_values(dataset_predictions["true_room"]):
        room_predictions = dataset_predictions.loc[dataset_predictions["true_room"] == room]
        if room_predictions.empty:
            continue

        _plot_position_confusion_matrix(
            room_predictions,
            title=f"{dataset} - position confusion | true room {room}",
            normalize=normalize,
        )


def plot_position_confusion_when_room_correct(
    predictions: pd.DataFrame,
    *,
    dataset: str,
    normalize: str | None = None,
) -> None:
    """Plot second-stage position confusion matrices after correct room routing."""
    dataset_predictions = _dataset_predictions(predictions, dataset=dataset)

    for room in _room_values(dataset_predictions["true_room"]):
        room_predictions = dataset_predictions.loc[
            (dataset_predictions["true_room"] == room)
            & (dataset_predictions["pred_room"] == room)
        ]
        if room_predictions.empty:
            continue

        _plot_position_confusion_matrix(
            room_predictions,
            title=f"{dataset} - position confusion | correctly routed room {room}",
            normalize=normalize,
        )


def _random_forest_classifier(*, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def _fallback_locations(df: pd.DataFrame, room_labels: pd.Series) -> dict[int, str]:
    fallback_locations: dict[int, str] = {}

    for raw_room_label in sorted(room_labels.unique()):
        room_label = int(raw_room_label)
        if room_label == EMPTY_ROOM_LABEL:
            fallback_locations[room_label] = EMPTY_ROOM_LOCATION
            continue

        locations = df.loc[room_labels == room_label, "location"].astype(str)
        fallback_locations[room_label] = str(locations.value_counts().idxmax())

    return fallback_locations


def _distance_errors(
    true_locations: pd.Series,
    pred_locations: np.ndarray,
    *,
    row_spacing: float,
    column_spacing: float,
) -> np.ndarray:
    return np.asarray(
        [
            location_distance_error(
                true_location,
                pred_location,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            for true_location, pred_location in zip(true_locations, pred_locations)
        ],
        dtype=float,
    )


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    samples = float(len(predictions))
    localization_samples = float(
        (predictions["true_location"].map(_normalize_location_label) != EMPTY_ROOM_LOCATION).sum(),
    )
    if predictions.empty:
        return {
            "room_accuracy": np.nan,
            "position_accuracy": np.nan,
            "mean_distance_error": np.nan,
            "median_distance_error": np.nan,
            "rmse_distance_error": np.nan,
            "distance_error_samples": 0.0,
            "localization_samples": localization_samples,
            "samples": samples,
        }

    distance_errors = predictions["distance_error"].dropna().astype(float)
    distance_error_samples = float(len(distance_errors))
    mean_distance_error = float(distance_errors.mean()) if not distance_errors.empty else np.nan
    median_distance_error = (
        float(distance_errors.median()) if not distance_errors.empty else np.nan
    )
    rmse_distance_error = (
        float(np.sqrt(np.mean(np.square(distance_errors)))) if not distance_errors.empty else np.nan
    )

    return {
        "room_accuracy": float((predictions["true_room"] == predictions["pred_room"]).mean()),
        "position_accuracy": float(
            (predictions["true_location"] == predictions["pred_location"]).mean(),
        ),
        "mean_distance_error": mean_distance_error,
        "median_distance_error": median_distance_error,
        "rmse_distance_error": rmse_distance_error,
        "distance_error_samples": distance_error_samples,
        "localization_samples": localization_samples,
        "samples": samples,
    }


def _localization_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {
            "position_accuracy": np.nan,
            "mean_distance_error": np.nan,
            "median_distance_error": np.nan,
            "rmse_distance_error": np.nan,
            "samples": 0.0,
        }

    distance_errors = _numeric_distance_errors(predictions)
    error_values = distance_errors.to_numpy(dtype=float)
    return {
        "position_accuracy": float(
            (predictions["true_location"] == predictions["pred_location"]).mean(),
        ),
        "mean_distance_error": float(np.mean(error_values)) if error_values.size else np.nan,
        "median_distance_error": float(np.median(error_values)) if error_values.size else np.nan,
        "rmse_distance_error": float(np.sqrt(np.mean(np.square(error_values))))
        if error_values.size
        else np.nan,
        "samples": float(len(predictions)),
    }


def _distance_error_groups(predictions: pd.DataFrame) -> list[tuple[object, pd.Series]]:
    _validate_required_columns(predictions, {"dataset", "distance_error"})
    return [
        (dataset_name, _numeric_distance_errors(dataset_predictions))
        for dataset_name, dataset_predictions in predictions.groupby("dataset", sort=False)
    ]


def _numeric_distance_errors(predictions: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(predictions["distance_error"], errors="coerce").dropna()


def _dataset_predictions(predictions: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    _validate_required_columns(
        predictions,
        {"dataset", "true_room", "pred_room", "true_location", "pred_location"},
    )
    dataset_predictions = predictions.loc[predictions["dataset"] == dataset].copy()
    if dataset_predictions.empty:
        print(f"No localization predictions found for dataset {dataset!r}.")
    return dataset_predictions


def _plot_position_confusion_matrix(
    predictions: pd.DataFrame,
    *,
    title: str,
    normalize: str | None,
) -> None:
    labels = _location_values(predictions["true_location"], predictions["pred_location"])
    if not labels:
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found.*",
            category=UserWarning,
        )
        matrix = confusion_matrix(
            predictions["true_location"],
            predictions["pred_location"],
            labels=labels,
            normalize=normalize,
        )
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    width = max(5, min(18, 0.45 * len(labels) + 3))
    height = max(4, min(16, 0.45 * len(labels) + 2))
    _, ax = plt.subplots(figsize=(width, height))
    values_format = ".2f" if normalize is not None else "d"
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format=values_format)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    plt.show()


def _accuracy(left: pd.Series, right: pd.Series) -> float:
    if left.empty:
        return np.nan
    return float((left.to_numpy() == right.to_numpy()).mean())


def _align_localization_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    aligned_predictions = predictions.copy()
    for column in GLOBAL_PREDICTION_COLUMNS:
        if column not in aligned_predictions.columns:
            aligned_predictions[column] = np.nan
    return aligned_predictions[GLOBAL_PREDICTION_COLUMNS]


def room_label_for_location(location: object) -> int | None:
    """Map reference-point labels to room labels using the feature-pipeline room layout."""
    location_label = _normalize_location_label(location)
    if location_label == EMPTY_ROOM_LOCATION:
        return EMPTY_ROOM_LABEL

    match = LOCATION_PATTERN.fullmatch(location_label)
    if match is None:
        return None

    row = match.group("row")
    column = int(match.group("column"))
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


def _room_values(rooms: pd.Series) -> list[object]:
    return sorted(rooms.dropna().unique(), key=lambda value: (str(type(value)), value))


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


def _resolve_distance_options(
    distance_options: GridDistanceOptions | None,
) -> GridDistanceOptions:
    if distance_options is None:
        distance_options = GridDistanceOptions()
    _validate_grid_spacing(
        row_spacing=distance_options.row_spacing,
        column_spacing=distance_options.column_spacing,
    )
    return distance_options


def _validate_grid_spacing(*, row_spacing: float, column_spacing: float) -> None:
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        msg = "row_spacing must be a finite positive value."
        raise ValueError(msg)
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        msg = "column_spacing must be a finite positive value."
        raise ValueError(msg)


def _validate_training_dataframe(df: pd.DataFrame) -> None:
    _validate_required_columns(df, METADATA_COLUMNS)
    if df.empty:
        msg = "Cannot train on an empty dataframe."
        raise ValueError(msg)
    if not feature_columns(df):
        msg = "No CSI feature columns found."
        raise ValueError(msg)


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        msg = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)
