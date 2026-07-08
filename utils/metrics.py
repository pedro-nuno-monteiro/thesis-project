from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def compute_localization_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    """Compute model-agnostic localization metrics from stored predictions."""
    true_position = _column(predictions_df, "true_position", "true_location")
    pred_position = _column(predictions_df, "pred_position", "pred_location")
    true_room = _column(predictions_df, "true_room")
    pred_room = _column(predictions_df, "pred_room")

    if predictions_df.empty:
        return {
            "position_accuracy": np.nan,
            "macro_f1": np.nan,
            "room_accuracy": np.nan,
            "mean_distance_error": np.nan,
            "median_distance_error": np.nan,
            "rmse_distance_error": np.nan,
            "p90_distance_error": np.nan,
            "samples": 0.0,
        }

    distance_errors = pd.to_numeric(
        predictions_df["distance_error"],
        errors="coerce",
    ).dropna()
    error_values = distance_errors.to_numpy(dtype=float)

    return {
        "position_accuracy": _accuracy(true_position, pred_position),
        "macro_f1": float(
            f1_score(
                true_position.astype(str),
                pred_position.astype(str),
                average="macro",
                zero_division=0,
            )
        ),
        "room_accuracy": _accuracy(true_room, pred_room),
        "mean_distance_error": float(np.mean(error_values)) if error_values.size else np.nan,
        "median_distance_error": float(np.median(error_values)) if error_values.size else np.nan,
        "rmse_distance_error": (
            float(np.sqrt(np.mean(np.square(error_values)))) if error_values.size else np.nan
        ),
        "p90_distance_error": (
            float(np.percentile(error_values, 90)) if error_values.size else np.nan
        ),
        "samples": float(len(predictions_df)),
    }


def _column(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first available column from names."""
    for name in names:
        if name in df.columns:
            return df[name]
    msg = f"Missing required column. Expected one of: {', '.join(names)}"
    raise ValueError(msg)


def _accuracy(left: pd.Series, right: pd.Series) -> float:
    if left.empty:
        return np.nan
    return float((left.to_numpy() == right.to_numpy()).mean())
