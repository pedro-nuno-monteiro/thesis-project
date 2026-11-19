from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from paper_utils.hierarchical_position_classifier import (
    _distance_errors,
    _random_forest_classifier,
    feature_columns,
    train_hierarchical_position_classifier,
)

DEFAULT_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 5],
}

DEFAULT_PARAM_GRID_V2 = {
    "max_depth": [2, 3, 5, 10, 15, 30, 60, 100],
    "min_samples_split": [2, 4, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
}

FIXED_PARAMS_SINGLE_BAND = {
    "n_estimators": 500,
    "max_features": "sqrt",
}

FIXED_PARAMS_FUSION = {
    "n_estimators": 500,
    "max_features": "log2",
}

RF_PARAM_COLUMNS = [
    "n_estimators",
    "max_features",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
]


def tune_global_rf_cv(
    train_df: pd.DataFrame,
    *,
    param_grid: dict,
    cv: int = 3,
    random_state: int = 42,
    verbose: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Grid search for Global RF on the position-classification problem.

    Uses GroupKFold(cv) with groups=train_df["group_id"] to keep sessions intact
    within CV folds. Returns (best_params, cv_results_df).
    """
    _validate_group_cv(train_df, cv=cv)
    columns = feature_columns(train_df)
    estimator = _random_forest_classifier(random_state=random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=GroupKFold(n_splits=cv),
        refit=False,
        n_jobs=1,
        verbose=verbose,
    )
    search.fit(
        train_df[columns],
        train_df["location"].astype(str),
        groups=train_df["group_id"],
    )

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score",
    ).reset_index(drop=True)
    return dict(search.best_params_), cv_results_df


def tune_hierarchical_rf_cv(
    train_df: pd.DataFrame,
    *,
    param_grid: dict,
    cv: int = 3,
    random_state: int = 42,
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
    esp_mode: Literal["all", "local"] = "local",
    verbose: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Grid search for Hierarchical RF with shared params across all stages.

    The hierarchical model trains multiple Random Forests internally, so this
    performs a manual GroupKFold loop instead of using GridSearchCV.
    """
    _validate_group_cv(train_df, cv=cv)
    cv_results = []
    param_names = list(param_grid.keys())
    gkf = GroupKFold(n_splits=cv)

    for combo_idx, values in enumerate(
        product(*[param_grid[name] for name in param_names]),
        start=1,
    ):
        params = dict(zip(param_names, values))
        fold_scores: list[float] = []
        if verbose:
            print(f"[hierarchical RF grid] {combo_idx}: {params}")

        for train_idx, val_idx in gkf.split(train_df, groups=train_df["group_id"]):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            model = train_hierarchical_position_classifier(
                fold_train,
                random_state=random_state,
                esp_mode=esp_mode,
                room_local_esps=room_local_esps,
                **params,
            )
            _, pred_locs = model.predict(fold_val)
            fold_scores.append(
                float((fold_val["location"].astype(str).to_numpy() == pred_locs).mean())
            )

        cv_results.append(
            {
                **params,
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores)),
            }
        )

    cv_results_df = pd.DataFrame(cv_results).sort_values(
        "mean_score",
        ascending=False,
    ).reset_index(drop=True)
    best_params = {name: cv_results_df.iloc[0][name] for name in param_names}
    return best_params, cv_results_df


def tune_global_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    param_grid: dict,
    fixed_params: dict,
    random_state: int = 42,
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
    verbose: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Direct test-set grid search for Global RF.

    For each combination of hyperparameters in param_grid, build the RF with
    fixed_params plus the grid params, train on train_df, and evaluate directly
    on test_df. Returns the merged best parameters and all results sorted by
    test-set position accuracy.
    """
    _validate_direct_search_frames(train_df, test_df)
    columns = feature_columns(train_df)
    rows = []
    param_names = list(param_grid.keys())
    combos = list(product(*[param_grid[name] for name in param_names]))

    for combo_idx, values in enumerate(combos, start=1):
        grid_params = dict(zip(param_names, values))
        params = {**fixed_params, **grid_params}
        model = _random_forest_classifier(random_state=random_state, **params)
        model.fit(train_df[columns], train_df["location"].astype(str))
        pred_locations = model.predict(test_df[columns])
        metrics = _position_metrics(
            test_df,
            pred_locations,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        rows.append({**_result_params(params), **metrics})
        if verbose >= 1:
            _print_progress(combo_idx, len(combos), params, metrics["position_accuracy"])

    results_df = _sorted_results(rows)
    best_params = _params_from_result_row(results_df.iloc[0])
    return best_params, results_df


def tune_hierarchical_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    param_grid: dict,
    fixed_params: dict,
    random_state: int = 42,
    row_spacing: float = 1.0,
    column_spacing: float = 1.0,
    esp_mode: Literal["all", "local"] = "local",
    room_local_esps: dict[int, tuple[str, ...]] | None = None,
    verbose: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """Direct test-set grid search for the Hierarchical RF pipeline.

    The same merged RF hyperparameters are used for every hierarchical RF stage.
    Returns the merged best parameters and all results sorted by test-set
    position accuracy.
    """
    _validate_direct_search_frames(train_df, test_df)
    rows = []
    param_names = list(param_grid.keys())
    combos = list(product(*[param_grid[name] for name in param_names]))

    for combo_idx, values in enumerate(combos, start=1):
        grid_params = dict(zip(param_names, values))
        params = {**fixed_params, **grid_params}
        model = train_hierarchical_position_classifier(
            train_df,
            random_state=random_state,
            esp_mode=esp_mode,
            room_local_esps=room_local_esps,
            **params,
        )
        pred_rooms, pred_locations = model.predict(test_df)
        metrics = _position_metrics(
            test_df,
            pred_locations,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
        )
        metrics["room_accuracy"] = float(
            (pred_rooms == test_df["label"].astype(int).to_numpy()).mean()
        )
        rows.append({**_result_params(params), **metrics})
        if verbose >= 1:
            _print_progress(combo_idx, len(combos), params, metrics["position_accuracy"])

    results_df = _sorted_results(rows)
    best_params = _params_from_result_row(results_df.iloc[0])
    return best_params, results_df


def _validate_direct_search_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    missing_train = {"location"} - set(train_df.columns)
    missing_test = {"location", "label"} - set(test_df.columns)
    missing = missing_train | missing_test
    if missing:
        msg = f"Grid-search dataframes are missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    if not feature_columns(train_df):
        msg = "Training dataframe has no feature columns."
        raise ValueError(msg)


def _position_metrics(
    test_df: pd.DataFrame,
    pred_locations: np.ndarray,
    *,
    row_spacing: float,
    column_spacing: float,
) -> dict[str, float]:
    distance_errors = _distance_errors(
        test_df["location"],
        pred_locations,
        row_spacing=row_spacing,
        column_spacing=column_spacing,
    )
    valid_errors = pd.Series(pd.to_numeric(distance_errors, errors="coerce")).dropna()
    return {
        "position_accuracy": float(
            (test_df["location"].astype(str).to_numpy() == pred_locations).mean()
        ),
        "mean_distance_error": float(valid_errors.mean()),
        "median_distance_error": float(valid_errors.median()),
        "rmse_distance_error": float(np.sqrt((valid_errors**2).mean())),
    }


def _sorted_results(rows: list[dict]) -> pd.DataFrame:
    results = pd.DataFrame(rows).sort_values(
        "position_accuracy",
        ascending=False,
    ).reset_index(drop=True)
    ordered = [
        *[col for col in RF_PARAM_COLUMNS if col in results.columns],
        "position_accuracy",
        *[col for col in ["room_accuracy"] if col in results.columns],
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
    ]
    remaining = [col for col in results.columns if col not in ordered]
    return results[ordered + remaining]


def _result_params(params: dict) -> dict:
    return {name: params.get(name) for name in RF_PARAM_COLUMNS}


def _params_from_result_row(row: pd.Series) -> dict:
    return {name: row[name] for name in RF_PARAM_COLUMNS if name in row}


def _print_progress(
    combo_idx: int,
    combo_count: int,
    params: dict,
    accuracy: float,
) -> None:
    print(
        f"[grid {combo_idx}/{combo_count}] "
        f"max_depth={params.get('max_depth')} "
        f"split={params.get('min_samples_split')} "
        f"leaf={params.get('min_samples_leaf')} -> acc={accuracy:.4f}"
    )


def _validate_group_cv(train_df: pd.DataFrame, *, cv: int) -> None:
    if cv < 2:
        msg = "cv must be at least 2."
        raise ValueError(msg)
    missing = {"group_id", "location"} - set(train_df.columns)
    if missing:
        msg = f"Training dataframe is missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    group_count = int(train_df["group_id"].nunique())
    if group_count < cv:
        msg = f"GroupKFold cv={cv} requires at least {cv} groups; found {group_count}."
        raise ValueError(msg)
