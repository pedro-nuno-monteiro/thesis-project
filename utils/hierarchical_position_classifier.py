from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
DEFAULT_BLOCK_COUNT = 10
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
ROOM_SPECIFIC_MODEL_NAME = "room_specific_position_rf"
GLOBAL_KNN_MODEL_NAME = "global_position_knn"
HIERARCHICAL_KNN_MODEL_NAME = "hierarchical_knn"

ROOM_LOCAL_ESPS: dict[int, tuple[str, ...]] = {
    1: (
        "esp_06", "esp_07", "esp_08", "esp_09", "esp_10",
        "esp_16", "esp_17", "esp_18", "esp_19", "esp_20",
    ),
    2: ("esp_01", "esp_02", "esp_03", "esp_11", "esp_12", "esp_13"),
    3: ("esp_04", "esp_05", "esp_14", "esp_15"),
}

PREDICTION_COLUMNS = [
    "dataset",
    "model",
    "split",
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

POSITION_SPLIT_DIAGNOSTIC_COLUMNS = [
    "dataset",
    "location",
    "room_label",
    "train_groups",
    "test_groups",
    "train_samples",
    "test_samples",
    "in_train",
    "in_test",
    "test_location_missing_from_train",
]

POSITION_SPLIT_SUMMARY_COLUMNS = [
    "dataset",
    "locations_total",
    "locations_in_train",
    "locations_in_test",
    "test_locations_missing_from_train",
    "test_samples_with_unseen_location",
    "total_test_samples",
    "fraction_test_samples_with_unseen_location",
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
    "split",
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

HIERARCHICAL_SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "esp_mode",
    "room_accuracy",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "distance_error_samples",
    "localization_samples",
    "samples",
]

GLOBAL_SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "samples",
]

ROOM_SPECIFIC_PREDICTION_COLUMNS = [
    "dataset",
    "model",
    "split",
    "room",
    "esp_mode",
    "true_location",
    "pred_location",
    "distance_error",
    "scenario",
    "user",
    "trial",
    "group_id",
    "window_idx",
]

ROOM_SPECIFIC_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "split",
    "room",
    "esp_mode",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "samples",
]

COMBINED_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "split",
    "room",
    "esp_mode",
    "position_accuracy",
    "room_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "samples",
    "k",
]

LOCALIZATION_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "split",
    "samples",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
]

GLOBAL_KNN_SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "k",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "samples",
]

HIERARCHICAL_KNN_SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "esp_mode",
    "k",
    "room_accuracy",
    "position_accuracy",
    "mean_distance_error",
    "median_distance_error",
    "rmse_distance_error",
    "distance_error_samples",
    "localization_samples",
    "samples",
]


@dataclass(frozen=True)
class GridDistanceOptions:
    """Spacing used to convert reference-point labels into physical coordinates."""

    row_spacing: float = DEFAULT_ROW_SPACING
    column_spacing: float = DEFAULT_COLUMN_SPACING


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names by excluding metadata columns."""
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
    split_mode: Literal["group", "random", "block"] = "group",
    stratify_column: str | None = None,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe for train/test evaluation.

    split_mode='group' (default, realistic): keeps all sliding windows from the
    same acquisition group together via GroupShuffleSplit. Strong cross-session
    generalization but no packet sharing across splits.

    split_mode='random' (optimistic sanity-check): random row-level split that
    may leak similar windows from the same acquisition across boundaries due to
    the 50% sliding-window overlap.

    split_mode='block' (leak-free random): splits each session into n_blocks
    contiguous time blocks, randomly assigns blocks to test, and drops windows
    that straddle a train/test boundary. Eliminates packet-level leakage while
    exposing every user, position, and trial to training.
    """
    _validate_required_columns(df, {"group_id"})
    if df.empty:
        msg = "Cannot split an empty dataframe."
        raise ValueError(msg)
    if not 0 < test_size < 1:
        msg = "test_size must be between 0 and 1."
        raise ValueError(msg)

    if split_mode == "group":
        _validate_required_columns(df, {"label"})
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

    if split_mode == "random":
        stratify = None
        if stratify_column is not None and stratify_column in df.columns:
            col = df[stratify_column]
            if col.nunique() >= 2 and col.value_counts().min() >= 2:
                stratify = col
        row_indices = list(range(len(df)))
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    if split_mode == "block":
        if n_blocks < 1:
            msg = "n_blocks must be at least 1 for block split mode."
            raise ValueError(msg)
        _validate_required_columns(df, {"window_idx"})
        group_ids = df["group_id"].to_numpy()
        window_idxs = df["window_idx"].to_numpy()
        n_test_blocks = min(n_blocks, max(1, round(test_size * n_blocks)))

        train_locs: list[int] = []
        test_locs: list[int] = []

        for group_id in df["group_id"].unique():
            session_pos = np.flatnonzero(group_ids == group_id)
            # Sort windows by temporal order within this session.
            sort_order = np.argsort(window_idxs[session_pos], kind="stable")
            session_pos_sorted = session_pos[sort_order]
            n_windows = len(session_pos_sorted)

            if n_windows < n_blocks:
                # Too few windows to split into blocks; keep whole session in train.
                train_locs.extend(session_pos_sorted.tolist())
                continue

            # Assign windows to contiguous time blocks.
            block_assignment = (np.arange(n_windows) * n_blocks) // n_windows

            # Reproducible per-session seed: stable across Python runs (no hash()).
            gid_int = int.from_bytes(
                hashlib.md5(str(group_id).encode()).digest()[:4], "big"
            )
            rng = np.random.RandomState((gid_int + random_state) % (2**31))
            test_block_set = set(
                rng.choice(n_blocks, size=n_test_blocks, replace=False).tolist()
            )

            is_test = np.array([ba in test_block_set for ba in block_assignment])

            # Drop boundary windows (adjacent to a window in the opposite set).
            is_boundary = np.zeros(n_windows, dtype=bool)
            if n_windows > 1:
                is_boundary[:-1] |= is_test[:-1] != is_test[1:]
                is_boundary[1:] |= is_test[1:] != is_test[:-1]

            for pos, tb, bd in zip(session_pos_sorted, is_test, is_boundary):
                if bd:
                    continue
                (test_locs if tb else train_locs).append(int(pos))

        if not train_locs or not test_locs:
            msg = "Block split produced an empty train or test set."
            raise ValueError(msg)

        return df.iloc[train_locs].copy(), df.iloc[test_locs].copy()

    msg = f"Unknown split_mode {split_mode!r}. Must be 'group', 'random', or 'block'."
    raise ValueError(msg)


def position_split_diagnostics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    dataset_name: str = "",
) -> pd.DataFrame:
    """Report whether each test location was also present in the training split."""
    _validate_required_columns(train_df, {"group_id", "location", "label"})
    _validate_required_columns(test_df, {"group_id", "location", "label"})

    train_locations = train_df.loc[
        train_df["location"].notna(), ["location", "group_id", "label"]
    ].copy()
    test_locations = test_df.loc[
        test_df["location"].notna(), ["location", "group_id", "label"]
    ].copy()

    train_locations["location"] = train_locations["location"].map(_normalize_location_label)
    test_locations["location"] = test_locations["location"].map(_normalize_location_label)

    all_locations = sorted(
        set(train_locations["location"].unique()).union(test_locations["location"].unique()),
        key=_location_sort_key,
    )

    def _room_label_for_location(location: str) -> object:
        combined = pd.concat(
            [
                train_locations.loc[train_locations["location"] == location, "label"],
                test_locations.loc[test_locations["location"] == location, "label"],
            ],
            ignore_index=True,
        )
        numeric_labels = pd.to_numeric(combined, errors="coerce").dropna()
        if numeric_labels.empty:
            return np.nan
        return int(numeric_labels.mode().iat[0])

    diagnostics_rows: list[dict[str, object]] = []
    for location in all_locations:
        train_rows = train_locations.loc[train_locations["location"] == location]
        test_rows = test_locations.loc[test_locations["location"] == location]
        train_samples = int(len(train_rows))
        test_samples = int(len(test_rows))
        diagnostics_rows.append(
            {
                "dataset": dataset_name,
                "location": location,
                "room_label": _room_label_for_location(location),
                "train_groups": int(train_rows["group_id"].nunique()),
                "test_groups": int(test_rows["group_id"].nunique()),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "in_train": train_samples > 0,
                "in_test": test_samples > 0,
                "test_location_missing_from_train": test_samples > 0 and train_samples == 0,
            },
        )

    return pd.DataFrame(diagnostics_rows, columns=POSITION_SPLIT_DIAGNOSTIC_COLUMNS)


def diagnose_position_split(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe and return the train/test partitions plus diagnostics."""
    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    diagnostics = position_split_diagnostics(train_df, test_df, dataset_name=dataset_name)
    return train_df, test_df, diagnostics


def diagnose_position_splits(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run split diagnostics for every named feature dataframe."""
    diagnostics_frames: list[pd.DataFrame] = []

    for dataset_name, dataframe in feature_dataframes.items():
        if dataframe.empty:
            diagnostics_frames.append(
                pd.DataFrame(columns=POSITION_SPLIT_DIAGNOSTIC_COLUMNS),
            )
            continue

        _, _, diagnostics = diagnose_position_split(
            dataframe,
            dataset_name=dataset_name,
            test_size=test_size,
            random_state=random_state,
        )
        diagnostics_frames.append(diagnostics)

    if not diagnostics_frames:
        return pd.DataFrame(columns=POSITION_SPLIT_DIAGNOSTIC_COLUMNS)

    return pd.concat(diagnostics_frames, ignore_index=True)


def summarize_position_split_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize which datasets and samples contain unseen test locations."""
    _validate_required_columns(diagnostics, set(POSITION_SPLIT_DIAGNOSTIC_COLUMNS))
    summary_rows: list[dict[str, object]] = []

    for dataset_name, dataset_diagnostics in diagnostics.groupby("dataset", sort=False):
        unseen_mask = dataset_diagnostics["test_location_missing_from_train"].astype(bool)
        test_samples_with_unseen_location = int(
            dataset_diagnostics.loc[unseen_mask, "test_samples"].sum()
        )
        total_test_samples = int(dataset_diagnostics["test_samples"].sum())
        summary_rows.append(
            {
                "dataset": dataset_name,
                "locations_total": int(len(dataset_diagnostics)),
                "locations_in_train": int(dataset_diagnostics["in_train"].astype(bool).sum()),
                "locations_in_test": int(dataset_diagnostics["in_test"].astype(bool).sum()),
                "test_locations_missing_from_train": int(unseen_mask.sum()),
                "test_samples_with_unseen_location": test_samples_with_unseen_location,
                "total_test_samples": total_test_samples,
                "fraction_test_samples_with_unseen_location": (
                    float(test_samples_with_unseen_location / total_test_samples)
                    if total_test_samples
                    else np.nan
                ),
            },
        )

    return pd.DataFrame(summary_rows, columns=POSITION_SPLIT_SUMMARY_COLUMNS)


@dataclass
class HierarchicalPositionClassifier:
    """Two-stage CSI classifier: room first, then room-specific reference point."""

    room_model: object
    position_models: dict[int, object]
    feature_columns: list[str]
    fallback_locations: dict[int, str]
    esp_mode: Literal["all", "local"] = "all"
    # Maps room label → feature columns used by that room's position model.
    room_feature_columns: dict[int, list[str]] = field(default_factory=dict)

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

            # In local mode, restrict to this room's ESP feature columns.
            room_feat_cols = self.room_feature_columns.get(room_label, self.feature_columns)
            room_indices = np.flatnonzero(room_mask)
            pred_locations[room_mask] = position_model.predict(
                df[room_feat_cols].iloc[room_indices]
            )

        return pred_rooms, pred_locations


@dataclass
class HierarchicalKNNClassifier:
    """Two-stage KNN classifier: room first, then room-specific reference point.

    Each stage has its own StandardScaler fit on that stage's training data.
    """

    room_model: KNeighborsClassifier
    room_scaler: StandardScaler
    position_models: dict[int, KNeighborsClassifier]
    position_scalers: dict[int, StandardScaler]
    feature_columns: list[str]
    fallback_locations: dict[int, str]
    esp_mode: Literal["all", "local"] = "all"
    room_feature_columns: dict[int, list[str]] = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict room labels and reference point labels for a feature dataframe."""
        _validate_required_columns(df, set(self.feature_columns))
        X_room = self.room_scaler.transform(df[self.feature_columns])
        pred_rooms = np.asarray(self.room_model.predict(X_room), dtype=int)
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

            room_feat_cols = self.room_feature_columns.get(room_label, self.feature_columns)
            pos_scaler = self.position_scalers[room_label]
            room_indices = np.flatnonzero(room_mask)
            X_pos = pos_scaler.transform(df[room_feat_cols].iloc[room_indices])
            pred_locations[room_mask] = position_model.predict(X_pos)

        return pred_rooms, pred_locations


def train_hierarchical_position_classifier(
    train_df: pd.DataFrame,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> HierarchicalPositionClassifier:
    """Train one room classifier and one position classifier per non-empty room.

    When esp_mode='local', each room's position model trains only on feature columns
    for the ESPs assigned to that room in room_local_esps (defaults to ROOM_LOCAL_ESPS).
    The room classifier always uses all feature columns regardless of esp_mode.
    """
    _validate_training_dataframe(train_df)
    columns = feature_columns(train_df)
    room_labels = train_df["label"].astype(int)

    # Stage 1: room classifier always sees all ESP features.
    room_model = _random_forest_classifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    room_model.fit(train_df[columns], room_labels)

    fallback_locations = _fallback_locations(train_df, room_labels)
    position_models: dict[int, object] = {}
    room_feature_columns: dict[int, list[str]] = {}
    resolved_esps = room_local_esps if room_local_esps is not None else ROOM_LOCAL_ESPS

    for raw_room_label in sorted(room_labels.unique()):
        room_label = int(raw_room_label)
        if room_label == EMPTY_ROOM_LABEL:
            continue

        room_df = train_df.loc[room_labels == room_label]

        # Stage 2: position classifier uses local ESPs when esp_mode="local".
        if esp_mode == "local":
            esp_keys = resolved_esps.get(room_label, ())
            room_feat_cols = (
                feature_columns_for_esps(room_df, esp_keys) if esp_keys else columns
            )
        else:
            room_feat_cols = columns

        room_feature_columns[room_label] = room_feat_cols
        position_model = _random_forest_classifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        position_model.fit(room_df[room_feat_cols], room_df["location"].astype(str))
        position_models[room_label] = position_model

    return HierarchicalPositionClassifier(
        room_model=room_model,
        position_models=position_models,
        feature_columns=columns,
        fallback_locations=fallback_locations,
        esp_mode=esp_mode,
        room_feature_columns=room_feature_columns,
    )


def _knn_classifier(*, k: int) -> KNeighborsClassifier:
    """Return a KNN classifier with Euclidean distance, uniform weights, and all CPU cores."""
    return KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform", n_jobs=-1)


def train_hierarchical_knn_classifier(
    train_df: pd.DataFrame,
    *,
    k: int,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> HierarchicalKNNClassifier:
    """Train a two-stage KNN room+position classifier with per-stage StandardScaler.

    The room classifier uses all ESP features. Each room's position classifier uses
    only that room's local ESP features when esp_mode='local'. Every stage gets its
    own scaler fit solely on training data to prevent leakage.
    """
    _validate_training_dataframe(train_df)
    columns = feature_columns(train_df)
    room_labels = train_df["label"].astype(int)

    room_scaler = StandardScaler()
    X_room = room_scaler.fit_transform(train_df[columns])
    room_model = _knn_classifier(k=k)
    room_model.fit(X_room, room_labels)

    fallback_locations = _fallback_locations(train_df, room_labels)
    position_models: dict[int, KNeighborsClassifier] = {}
    position_scalers: dict[int, StandardScaler] = {}
    room_feature_columns_map: dict[int, list[str]] = {}
    resolved_esps = room_local_esps if room_local_esps is not None else ROOM_LOCAL_ESPS

    for raw_room_label in sorted(room_labels.unique()):
        room_label = int(raw_room_label)
        if room_label == EMPTY_ROOM_LABEL:
            continue

        room_df = train_df.loc[room_labels == room_label]

        if esp_mode == "local":
            esp_keys = resolved_esps.get(room_label, ())
            room_feat_cols = (
                feature_columns_for_esps(room_df, esp_keys) if esp_keys else columns
            )
        else:
            room_feat_cols = columns

        room_feature_columns_map[room_label] = room_feat_cols
        pos_scaler = StandardScaler()
        X_pos = pos_scaler.fit_transform(room_df[room_feat_cols])
        pos_model = _knn_classifier(k=k)
        pos_model.fit(X_pos, room_df["location"].astype(str))
        position_models[room_label] = pos_model
        position_scalers[room_label] = pos_scaler

    return HierarchicalKNNClassifier(
        room_model=room_model,
        room_scaler=room_scaler,
        position_models=position_models,
        position_scalers=position_scalers,
        feature_columns=columns,
        fallback_locations=fallback_locations,
        esp_mode=esp_mode,
        room_feature_columns=room_feature_columns_map,
    )


def run_hierarchical_position_experiment(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    distance_options: GridDistanceOptions | None = None,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> tuple[HierarchicalPositionClassifier, pd.DataFrame, dict[str, float]]:
    """Train and evaluate the hierarchical classifier for one feature dataframe."""
    if distance_options is None:
        distance_options = GridDistanceOptions(
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
    distance_options = _resolve_distance_options(distance_options)
    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    model = train_hierarchical_position_classifier(
        train_df,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        esp_mode=esp_mode,
        room_local_esps=room_local_esps,
    )
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
            "model": HIERARCHICAL_MODEL_NAME,
            "split": split_mode,
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
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
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
                    "split": split_mode,
                    "esp_mode": esp_mode,
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
            split_mode=split_mode,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            distance_options=distance_options,
            esp_mode=esp_mode,
            room_local_esps=room_local_esps,
        )
        models[dataset_name] = model
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append(
            {"dataset": dataset_name, "split": split_mode, "esp_mode": esp_mode, **metrics}
        )

    return (
        pd.DataFrame(summary_rows, columns=HIERARCHICAL_SUMMARY_COLUMNS),
        predictions_by_dataset,
        models,
    )


def run_global_position_experiment(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
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
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    columns = feature_columns(train_df)
    model = _random_forest_classifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
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
            "split": split_mode,
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
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
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
                    "split": split_mode,
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
            split_mode=split_mode,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        models[dataset_name] = model
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append({"dataset": dataset_name, "split": split_mode, **metrics})

    return (
        pd.DataFrame(summary_rows, columns=GLOBAL_SUMMARY_COLUMNS),
        predictions_by_dataset,
        models,
    )


def run_hierarchical_position_experiments_by_split(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    split_modes: tuple[str, ...] = ("group", "random"),
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], HierarchicalPositionClassifier],
]:
    """Run hierarchical experiments for every dataset × split mode combination."""
    all_summary_frames: list[pd.DataFrame] = []
    all_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    all_models: dict[tuple[str, str], HierarchicalPositionClassifier] = {}

    for split_mode in split_modes:
        summary, predictions, models = run_hierarchical_position_experiments(
            feature_dataframes,
            split_mode=split_mode,  # type: ignore[arg-type]
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            esp_mode=esp_mode,
            room_local_esps=room_local_esps,
        )
        all_summary_frames.append(summary)
        for dataset_name, preds in predictions.items():
            all_predictions[(dataset_name, split_mode)] = preds
        for dataset_name, model in models.items():
            all_models[(dataset_name, split_mode)] = model

    combined_summary = (
        pd.concat(all_summary_frames, ignore_index=True)
        if all_summary_frames
        else pd.DataFrame(columns=HIERARCHICAL_SUMMARY_COLUMNS)
    )
    return combined_summary, all_predictions, all_models


def run_global_position_experiments_by_split(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    split_modes: tuple[str, ...] = ("group", "random"),
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], pd.DataFrame],
    dict[tuple[str, str], RandomForestClassifier],
]:
    """Run global position experiments for every dataset × split mode combination."""
    all_summary_frames: list[pd.DataFrame] = []
    all_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    all_models: dict[tuple[str, str], RandomForestClassifier] = {}

    for split_mode in split_modes:
        summary, predictions, models = run_global_position_experiments(
            feature_dataframes,
            split_mode=split_mode,  # type: ignore[arg-type]
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        all_summary_frames.append(summary)
        for dataset_name, preds in predictions.items():
            all_predictions[(dataset_name, split_mode)] = preds
        for dataset_name, model in models.items():
            all_models[(dataset_name, split_mode)] = model

    combined_summary = (
        pd.concat(all_summary_frames, ignore_index=True)
        if all_summary_frames
        else pd.DataFrame(columns=GLOBAL_SUMMARY_COLUMNS)
    )
    return combined_summary, all_predictions, all_models


# ── KNN experiment runners ────────────────────────────────────────────────────


def run_global_position_experiment_knn(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    k: int,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[tuple[StandardScaler, KNeighborsClassifier], pd.DataFrame, dict[str, float]]:
    """Train and evaluate a direct CSI-to-position KNN baseline."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_training_dataframe(df)
    _validate_required_columns(df, {"location", "group_id", "label"})

    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    columns = feature_columns(train_df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[columns])
    model = _knn_classifier(k=k)
    model.fit(X_train, train_df["location"].astype(str))

    X_test = scaler.transform(test_df[columns])
    pred_locations = model.predict(X_test)
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
            "model": GLOBAL_KNN_MODEL_NAME,
            "split": split_mode,
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
    metrics["k"] = float(k)

    return (scaler, model), predictions, metrics


def run_global_position_experiments_knn(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    k: int,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, tuple[StandardScaler, KNeighborsClassifier]],
]:
    """Run the global KNN baseline for every named feature dataframe."""
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)
    summary_rows: list[dict[str, float | str]] = []
    predictions_by_dataset: dict[str, pd.DataFrame] = {}
    models: dict[str, tuple[StandardScaler, KNeighborsClassifier]] = {}

    for dataset_name, dataframe in feature_dataframes.items():
        if dataframe.empty:
            predictions_by_dataset[dataset_name] = pd.DataFrame(
                columns=GLOBAL_PREDICTION_COLUMNS,
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_mode,
                    "k": float(k),
                    "position_accuracy": np.nan,
                    "mean_distance_error": np.nan,
                    "median_distance_error": np.nan,
                    "rmse_distance_error": np.nan,
                    "samples": 0.0,
                },
            )
            continue

        model_pair, predictions, metrics = run_global_position_experiment_knn(
            dataframe,
            dataset_name=dataset_name,
            k=k,
            split_mode=split_mode,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        models[dataset_name] = model_pair
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append({"dataset": dataset_name, "split": split_mode, **metrics})

    return (
        pd.DataFrame(summary_rows, columns=GLOBAL_KNN_SUMMARY_COLUMNS),
        predictions_by_dataset,
        models,
    )


def run_global_position_experiments_by_split_knn(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    split_modes: tuple[str, ...] = ("group", "random"),
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str, int], pd.DataFrame],
    dict[tuple[str, str, int], tuple[StandardScaler, KNeighborsClassifier]],
]:
    """Run global KNN experiments for every dataset × split mode × k combination."""
    all_summary_frames: list[pd.DataFrame] = []
    all_predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    all_models: dict[tuple[str, str, int], tuple[StandardScaler, KNeighborsClassifier]] = {}

    for k in k_values:
        for split_mode in split_modes:
            summary, predictions, models = run_global_position_experiments_knn(
                feature_dataframes,
                k=k,
                split_mode=split_mode,  # type: ignore[arg-type]
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
            )
            all_summary_frames.append(summary)
            for dataset_name, preds in predictions.items():
                all_predictions[(dataset_name, split_mode, k)] = preds
            for dataset_name, model in models.items():
                all_models[(dataset_name, split_mode, k)] = model

    combined_summary = (
        pd.concat(all_summary_frames, ignore_index=True)
        if all_summary_frames
        else pd.DataFrame(columns=GLOBAL_KNN_SUMMARY_COLUMNS)
    )
    return combined_summary, all_predictions, all_models


def run_hierarchical_position_experiment_knn(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    k: int,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    distance_options: GridDistanceOptions | None = None,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> tuple[HierarchicalKNNClassifier, pd.DataFrame, dict[str, float]]:
    """Train and evaluate the hierarchical KNN classifier for one feature dataframe."""
    distance_options = _resolve_distance_options(distance_options)
    train_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    model = train_hierarchical_knn_classifier(
        train_df,
        k=k,
        esp_mode=esp_mode,
        room_local_esps=room_local_esps,
    )
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
            "model": HIERARCHICAL_KNN_MODEL_NAME,
            "split": split_mode,
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
    metrics["k"] = float(k)

    return model, predictions, metrics


def run_hierarchical_position_experiments_knn(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    k: int,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, HierarchicalKNNClassifier]]:
    """Run the hierarchical KNN classifier for every named feature dataframe."""
    distance_options = GridDistanceOptions(row_spacing=row_spacing, column_spacing=column_spacing)
    _validate_grid_spacing(
        row_spacing=distance_options.row_spacing,
        column_spacing=distance_options.column_spacing,
    )
    summary_rows: list[dict[str, float | str]] = []
    predictions_by_dataset: dict[str, pd.DataFrame] = {}
    models: dict[str, HierarchicalKNNClassifier] = {}

    for dataset_name, dataframe in feature_dataframes.items():
        if dataframe.empty:
            predictions_by_dataset[dataset_name] = pd.DataFrame(columns=PREDICTION_COLUMNS)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_mode,
                    "esp_mode": esp_mode,
                    "k": float(k),
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

        model, predictions, metrics = run_hierarchical_position_experiment_knn(
            dataframe,
            dataset_name=dataset_name,
            k=k,
            split_mode=split_mode,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
            distance_options=distance_options,
            esp_mode=esp_mode,
            room_local_esps=room_local_esps,
        )
        models[dataset_name] = model
        predictions_by_dataset[dataset_name] = predictions
        summary_rows.append(
            {"dataset": dataset_name, "split": split_mode, "esp_mode": esp_mode, **metrics}
        )

    return (
        pd.DataFrame(summary_rows, columns=HIERARCHICAL_KNN_SUMMARY_COLUMNS),
        predictions_by_dataset,
        models,
    )


def run_hierarchical_position_experiments_by_split_knn(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    split_modes: tuple[str, ...] = ("group", "random"),
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str, int], pd.DataFrame],
    dict[tuple[str, str, int], HierarchicalKNNClassifier],
]:
    """Run hierarchical KNN experiments for every dataset × split mode × k combination."""
    all_summary_frames: list[pd.DataFrame] = []
    all_predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    all_models: dict[tuple[str, str, int], HierarchicalKNNClassifier] = {}

    for k in k_values:
        for split_mode in split_modes:
            summary, predictions, models = run_hierarchical_position_experiments_knn(
                feature_dataframes,
                k=k,
                split_mode=split_mode,  # type: ignore[arg-type]
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
                row_spacing=row_spacing,
                column_spacing=column_spacing,
                esp_mode=esp_mode,
                room_local_esps=room_local_esps,
            )
            all_summary_frames.append(summary)
            for dataset_name, preds in predictions.items():
                all_predictions[(dataset_name, split_mode, k)] = preds
            for dataset_name, model in models.items():
                all_models[(dataset_name, split_mode, k)] = model

    combined_summary = (
        pd.concat(all_summary_frames, ignore_index=True)
        if all_summary_frames
        else pd.DataFrame(columns=HIERARCHICAL_KNN_SUMMARY_COLUMNS)
    )
    return combined_summary, all_predictions, all_models


def feature_columns_for_esps(
    df: pd.DataFrame,
    esp_keys: tuple[str, ...] | list[str],
) -> list[str]:
    """Return feature columns whose names start with one of the given ESP prefixes."""
    prefixes = tuple(f"{esp}_" for esp in esp_keys)
    return [col for col in df.columns if col not in METADATA_COLUMNS and col.startswith(prefixes)]


def room_specific_feature_dataframe(
    df: pd.DataFrame,
    *,
    room_label: int,
    esp_keys: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Filter to one room and optionally restrict to the given ESP feature columns."""
    room_df = df.loc[df["label"] == room_label]
    if esp_keys is None:
        return room_df.copy()
    metadata_cols = [col for col in df.columns if col in METADATA_COLUMNS]
    selected_feature_cols = feature_columns_for_esps(room_df, esp_keys)
    return room_df[metadata_cols + selected_feature_cols].copy()


def run_room_specific_position_experiment(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dataset_name: str,
    room_label: int,
    esp_mode: Literal["all", "local"] = "all",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
    split_mode: Literal["group", "random", "block"] = "group",
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict[str, object]]:
    """Train a position classifier for samples inside one room.

    esp_mode='all': uses every ESP feature column present in the dataframe.
    esp_mode='local': uses only the ESPs listed in room_local_esps[room_label].

    This experiment assumes the room is already known (oracle-room scenario).
    """
    _validate_grid_spacing(row_spacing=row_spacing, column_spacing=column_spacing)

    esp_keys: tuple[str, ...] | None = None
    if esp_mode == "local":
        esp_keys = (room_local_esps if room_local_esps is not None else ROOM_LOCAL_ESPS).get(
            room_label,
        )

    room_df = room_specific_feature_dataframe(df, room_label=room_label, esp_keys=esp_keys)
    _validate_training_dataframe(room_df)
    _validate_required_columns(room_df, {"location", "group_id"})

    train_df, test_df = split_dataframe(
        room_df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )

    columns = feature_columns(train_df)
    model = _random_forest_classifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(train_df[columns], train_df["location"].astype(str))

    pred_locations = model.predict(test_df[columns])
    distance_errors = _distance_errors(
        test_df["location"],
        pred_locations,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )

    predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "model": ROOM_SPECIFIC_MODEL_NAME,
            "split": split_mode,
            "room": room_label,
            "esp_mode": esp_mode,
            "true_location": test_df["location"].astype(str).to_numpy(),
            "pred_location": pred_locations,
            "distance_error": distance_errors,
            "scenario": test_df["scenario"].to_numpy(),
            "user": test_df["user"].to_numpy(),
            "trial": test_df["trial"].to_numpy(),
            "group_id": test_df["group_id"].to_numpy(),
            "window_idx": test_df["window_idx"].to_numpy(),
        },
        columns=ROOM_SPECIFIC_PREDICTION_COLUMNS,
    )
    metrics = _room_specific_metrics(predictions, dataset_name, room_label, esp_mode, split_mode)
    return model, predictions, metrics


def run_room_specific_position_experiments(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    rooms: tuple[int, ...] = (1, 2, 3),
    esp_modes: tuple[str, ...] = ("all", "local"),
    split_modes: tuple[str, ...] = ("group", "random"),
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    n_blocks: int = DEFAULT_BLOCK_COUNT,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    row_spacing: float = DEFAULT_ROW_SPACING,
    column_spacing: float = DEFAULT_COLUMN_SPACING,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, int, str, str], pd.DataFrame],
    dict[tuple[str, int, str, str], RandomForestClassifier],
]:
    """Run room-specific position classifiers over all combinations of dataset/room/ESP/split."""
    all_summary_rows: list[dict[str, object]] = []
    all_predictions: dict[tuple[str, int, str, str], pd.DataFrame] = {}
    all_models: dict[tuple[str, int, str, str], RandomForestClassifier] = {}
    resolved_esps = room_local_esps if room_local_esps is not None else ROOM_LOCAL_ESPS

    for dataset_name, dataframe in feature_dataframes.items():
        for room_label in rooms:
            for esp_mode in esp_modes:
                esp_keys: tuple[str, ...] | None = None
                if esp_mode == "local":
                    esp_keys = resolved_esps.get(room_label)

                room_df = room_specific_feature_dataframe(
                    dataframe,
                    room_label=room_label,
                    esp_keys=esp_keys,
                )

                if room_df.empty:
                    print(
                        f"[room-specific] Skipping {dataset_name} room {room_label} "
                        f"esp_mode={esp_mode!r}: empty dataframe.",
                    )
                    continue

                if room_df["location"].nunique() < 2:
                    print(
                        f"[room-specific] Skipping {dataset_name} room {room_label} "
                        f"esp_mode={esp_mode!r}: fewer than 2 unique locations.",
                    )
                    continue

                for split_mode in split_modes:
                    if (
                        split_mode == "group"
                        and room_df["group_id"].nunique() < MIN_GROUP_SPLIT_COUNT
                    ):
                        print(
                            f"[room-specific] Skipping {dataset_name} room {room_label} "
                            f"esp_mode={esp_mode!r} split={split_mode!r}: "
                            f"too few groups ({room_df['group_id'].nunique()}).",
                        )
                        continue

                    try:
                        model, predictions, metrics = run_room_specific_position_experiment(
                            dataframe,
                            dataset_name=dataset_name,
                            room_label=room_label,
                            esp_mode=esp_mode,  # type: ignore[arg-type]
                            room_local_esps=room_local_esps,
                            split_mode=split_mode,  # type: ignore[arg-type]
                            test_size=test_size,
                            random_state=random_state,
                            n_blocks=n_blocks,
                            n_estimators=n_estimators,
                            max_features=max_features,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            row_spacing=row_spacing,
                            column_spacing=column_spacing,
                        )
                    except Exception as exc:
                        print(
                            f"[room-specific] Skipping {dataset_name} room {room_label} "
                            f"esp_mode={esp_mode!r} split={split_mode!r}: {exc}",
                        )
                        continue

                    key: tuple[str, int, str, str] = (
                        dataset_name,
                        room_label,
                        esp_mode,
                        split_mode,
                    )
                    all_models[key] = model
                    all_predictions[key] = predictions
                    all_summary_rows.append(metrics)

    if not all_summary_rows:
        return (
            pd.DataFrame(columns=ROOM_SPECIFIC_SUMMARY_COLUMNS),
            all_predictions,
            all_models,
        )

    return (
        pd.DataFrame(all_summary_rows, columns=ROOM_SPECIFIC_SUMMARY_COLUMNS),
        all_predictions,
        all_models,
    )


def combine_position_experiment_summaries(
    global_summary: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    room_specific_summary: pd.DataFrame,
    global_knn_summary: pd.DataFrame | None = None,
    hierarchical_knn_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge global, hierarchical, room-specific, and optional KNN summaries into one table.

    Global and hierarchical rows get room='all'. Only the hierarchical summary has
    a meaningful room_accuracy; global gets NaN. Hierarchical rows preserve their
    esp_mode column values. RF rows get k=NaN; KNN rows carry their k value.
    """
    frames: list[pd.DataFrame] = []

    if not global_summary.empty:
        global_frame = global_summary.copy()
        if "model" not in global_frame.columns:
            global_frame["model"] = GLOBAL_POSITION_MODEL_NAME
        global_frame["room"] = "all"
        global_frame["esp_mode"] = "all"
        global_frame["room_accuracy"] = np.nan
        global_frame["k"] = np.nan
        frames.append(global_frame)

    if not hierarchical_summary.empty:
        hier_frame = hierarchical_summary.copy()
        if "model" not in hier_frame.columns:
            hier_frame["model"] = HIERARCHICAL_MODEL_NAME
        hier_frame["room"] = "all"
        # Preserve the esp_mode column already set by the experiment runner.
        if "esp_mode" not in hier_frame.columns:
            hier_frame["esp_mode"] = "all"
        if "room_accuracy" not in hier_frame.columns:
            hier_frame["room_accuracy"] = np.nan
        hier_frame["k"] = np.nan
        frames.append(hier_frame)

    if not room_specific_summary.empty:
        room_frame = room_specific_summary.copy()
        if "room_accuracy" not in room_frame.columns:
            room_frame["room_accuracy"] = np.nan
        room_frame["k"] = np.nan
        frames.append(room_frame)

    if global_knn_summary is not None and not global_knn_summary.empty:
        knn_global_frame = global_knn_summary.copy()
        if "model" not in knn_global_frame.columns:
            knn_global_frame["model"] = GLOBAL_KNN_MODEL_NAME
        knn_global_frame["room"] = "all"
        knn_global_frame["esp_mode"] = "all"
        if "room_accuracy" not in knn_global_frame.columns:
            knn_global_frame["room_accuracy"] = np.nan
        frames.append(knn_global_frame)

    if hierarchical_knn_summary is not None and not hierarchical_knn_summary.empty:
        knn_hier_frame = hierarchical_knn_summary.copy()
        if "model" not in knn_hier_frame.columns:
            knn_hier_frame["model"] = HIERARCHICAL_KNN_MODEL_NAME
        knn_hier_frame["room"] = "all"
        if "esp_mode" not in knn_hier_frame.columns:
            knn_hier_frame["esp_mode"] = "all"
        if "room_accuracy" not in knn_hier_frame.columns:
            knn_hier_frame["room_accuracy"] = np.nan
        frames.append(knn_hier_frame)

    if not frames:
        return pd.DataFrame(columns=COMBINED_SUMMARY_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    for col in COMBINED_SUMMARY_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    return combined[COMBINED_SUMMARY_COLUMNS]


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
    """Summarize localization quality by dataset, model, and split (if present)."""
    _validate_required_columns(
        predictions,
        {"dataset", "model", "true_location", "pred_location", "distance_error"},
    )
    group_cols = ["dataset", "model"]
    if "split" in predictions.columns:
        group_cols.append("split")

    summary_rows: list[dict[str, object]] = []

    for group_keys, model_predictions in predictions.groupby(group_cols, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row_base = dict(zip(group_cols, group_keys))
        summary_rows.append({**row_base, **_localization_metrics(model_predictions)})

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


def _random_forest_classifier(
    *,
    random_state: int,
    n_estimators: int = 300,
    max_features: str | float = "sqrt",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> RandomForestClassifier:
    """Return a balanced Random Forest with configurable hyperparameters."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def _fallback_locations(df: pd.DataFrame, room_labels: pd.Series) -> dict[int, str]:
    """Build a per-room fallback location (most-frequent) used when a position model is missing."""
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
    """Compute per-sample Euclidean distance errors between true and predicted locations."""
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
    """Compute hierarchical classifier metrics from a predictions dataframe."""
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
    """Compute position accuracy and distance error statistics from predictions."""
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


def _room_specific_metrics(
    predictions: pd.DataFrame,
    dataset_name: str,
    room_label: int,
    esp_mode: str,
    split_mode: str,
) -> dict[str, object]:
    """Compute room-specific position metrics and return as a summary dict."""
    base: dict[str, object] = {
        "dataset": dataset_name,
        "model": ROOM_SPECIFIC_MODEL_NAME,
        "split": split_mode,
        "room": room_label,
        "esp_mode": esp_mode,
    }
    if predictions.empty:
        return {
            **base,
            "position_accuracy": np.nan,
            "mean_distance_error": np.nan,
            "median_distance_error": np.nan,
            "rmse_distance_error": np.nan,
            "samples": 0.0,
        }
    distance_errors = _numeric_distance_errors(predictions)
    error_values = distance_errors.to_numpy(dtype=float)
    return {
        **base,
        "position_accuracy": float(
            (predictions["true_location"] == predictions["pred_location"]).mean(),
        ),
        "mean_distance_error": float(np.mean(error_values)) if error_values.size else np.nan,
        "median_distance_error": float(np.median(error_values)) if error_values.size else np.nan,
        "rmse_distance_error": (
            float(np.sqrt(np.mean(np.square(error_values)))) if error_values.size else np.nan
        ),
        "samples": float(len(predictions)),
    }


def _distance_error_groups(predictions: pd.DataFrame) -> list[tuple[object, pd.Series]]:
    """Group predictions by dataset and return (dataset_name, numeric_errors) pairs."""
    _validate_required_columns(predictions, {"dataset", "distance_error"})
    return [
        (dataset_name, _numeric_distance_errors(dataset_predictions))
        for dataset_name, dataset_predictions in predictions.groupby("dataset", sort=False)
    ]


def _numeric_distance_errors(predictions: pd.DataFrame) -> pd.Series:
    """Extract non-NaN numeric distance errors from a predictions dataframe."""
    return pd.to_numeric(predictions["distance_error"], errors="coerce").dropna()


def _align_localization_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Ensure predictions has exactly the GLOBAL_PREDICTION_COLUMNS, adding NaN for any missing."""
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


def _accuracy(left: pd.Series, right: pd.Series) -> float:
    """Compute element-wise accuracy between two series."""
    if left.empty:
        return np.nan
    return float((left.to_numpy() == right.to_numpy()).mean())


def _location_sort_key(location: str) -> tuple[int, str, int | str]:
    """Sort key that orders locations by row letter then column number."""
    normalized = _normalize_location_label(location)
    if normalized == EMPTY_ROOM_LOCATION:
        return (1, "Z", 0)

    match = LOCATION_PATTERN.fullmatch(normalized)
    if match is None:
        return (2, normalized, normalized)

    return (0, match.group("row"), int(match.group("column")))


def _normalize_location_label(location: object) -> str:
    """Normalize a location label to uppercase without the LOCATION_ prefix."""
    return str(location).strip().upper().removeprefix("LOCATION_")


def _resolve_distance_options(
    distance_options: GridDistanceOptions | None,
) -> GridDistanceOptions:
    """Return distance_options, defaulting to GridDistanceOptions() if None."""
    if distance_options is None:
        distance_options = GridDistanceOptions()
    _validate_grid_spacing(
        row_spacing=distance_options.row_spacing,
        column_spacing=distance_options.column_spacing,
    )
    return distance_options


def _validate_grid_spacing(*, row_spacing: float, column_spacing: float) -> None:
    """Raise ValueError if either spacing is not a finite positive number."""
    if not np.isfinite(row_spacing) or row_spacing <= 0:
        msg = "row_spacing must be a finite positive value."
        raise ValueError(msg)
    if not np.isfinite(column_spacing) or column_spacing <= 0:
        msg = "column_spacing must be a finite positive value."
        raise ValueError(msg)


def _validate_training_dataframe(df: pd.DataFrame) -> None:
    """Raise ValueError if df is empty or has no feature columns."""
    _validate_required_columns(df, METADATA_COLUMNS)
    if df.empty:
        msg = "Cannot train on an empty dataframe."
        raise ValueError(msg)
    if not feature_columns(df):
        msg = "No CSI feature columns found."
        raise ValueError(msg)


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise ValueError listing any columns missing from df."""
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        msg = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)
