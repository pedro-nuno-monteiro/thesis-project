from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from hierarchical_position_classifier import (
    _random_forest_classifier,
    feature_columns,
    train_hierarchical_position_classifier,
)

DEFAULT_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 5],
}


def tune_global_rf(
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


def tune_hierarchical_rf(
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
