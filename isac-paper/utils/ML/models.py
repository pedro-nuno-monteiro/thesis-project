from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

DEFAULT_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "RF": {
        "2.4 GHz": {
            "n_estimators": 500,
            "max_features": "log2",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "5 GHz": {
            "n_estimators": 500,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "Fusion": {
            "n_estimators": 500,
            "max_features": "log2",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
    },
    "KNN": {
        "2.4 GHz": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        "5 GHz": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
        "Fusion": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"},
    },
    "SVM": {
        "2.4 GHz": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        "5 GHz": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        "Fusion": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    },
}

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "RF": {
        "n_estimators": [200, 300, 500, 800],
        "max_features": ["sqrt", "log2"],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    },
    "KNN": {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
    "SVM": {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 1e-3, 1e-4],
    },
}


def default_params_for(model: str, band: str) -> dict[str, Any]:
    """Return a mutable copy of the configured default parameters."""
    model_key = model.upper()
    try:
        return deepcopy(DEFAULT_PARAMS[model_key][band])
    except KeyError as exc:
        msg = f"No default parameters configured for model={model!r}, band={band!r}."
        raise ValueError(msg) from exc


def build_estimator(
    name: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    """Build a classical baseline estimator pipeline.

    RF intentionally has no scaler, matching the published baseline. KNN and SVM
    scale inside the pipeline so scaling is fit on training data only.
    """
    model_name = name.upper()
    model_params = dict(params)

    # Tree models operate directly on the statistical features, matching the
    # established RF baseline without introducing scaling.
    if model_name == "RF":
        estimator = RandomForestClassifier(
            n_estimators=int(model_params.pop("n_estimators", 300)),
            max_features=model_params.pop("max_features", "sqrt"),
            max_depth=model_params.pop("max_depth", None),
            min_samples_split=int(model_params.pop("min_samples_split", 2)),
            min_samples_leaf=int(model_params.pop("min_samples_leaf", 1)),
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=model_params.pop("class_weight", "balanced"),
            **model_params,
        )
        return Pipeline([("classifier", estimator)])

    # Distance- and margin-based models fit their scaler inside the pipeline so
    # test data never influences feature scaling.
    if model_name == "KNN":
        estimator = KNeighborsClassifier(
            n_neighbors=int(model_params.pop("n_neighbors", 5)),
            weights=model_params.pop("weights", "uniform"),
            metric=model_params.pop("metric", "euclidean"),
            n_jobs=n_jobs,
            **model_params,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler(copy=False)),
                ("classifier", estimator),
            ]
        )

    if model_name == "SVM":
        kernel = model_params.pop("kernel", "rbf")
        if kernel == "linear_svc":
            estimator = LinearSVC(
                C=float(model_params.pop("C", 1.0)),
                random_state=random_state,
                dual=model_params.pop("dual", "auto"),
                max_iter=int(model_params.pop("max_iter", 10000)),
                **model_params,
            )
        else:
            estimator = SVC(
                kernel=kernel,
                C=float(model_params.pop("C", 1.0)),
                gamma=model_params.pop("gamma", "scale"),
                probability=False,
                cache_size=int(model_params.pop("cache_size", 1000)),
                random_state=random_state,
                **model_params,
            )
        return Pipeline(
            [
                ("scaler", StandardScaler(copy=False)),
                ("classifier", estimator),
            ]
        )

    msg = f"Unknown model {name!r}. Expected one of RF, KNN, SVM."
    raise ValueError(msg)
