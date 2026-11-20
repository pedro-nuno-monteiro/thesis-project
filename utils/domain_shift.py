"""Cross-session fingerprint shift measures and their link to localization performance.

All measures are computed on the feature DataFrames from ``feature_pipeline`` without any
extra scaling. A *profile* is the mean of one feature family (default ``mean``) over all
windows of a recording, for one ESP, across its retained subcarriers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.config import ANCHOR_GROUPS
from utils.fingerprint import (
    FEATURE_COLUMN_PATTERN,
    normalize_position,
    normalize_two_digit_id,
)
from utils.results import location_grid_coordinates, room_label_for_location

if TYPE_CHECKING:
    from collections.abc import Sequence

ESP_BANDS = {esp: band for band, esps in ANCHOR_GROUPS.items() for esp in esps}
SHIFT_METRICS = ("pearson_r", "rmse", "mean_offset", "scale_ratio", "smd", "shift_ratio")
SHIFT_METRIC_LABELS = {
    "pearson_r": "Pearson r (profile shape)",
    "rmse": "RMSE (magnitude-sensitive)",
    "mean_offset": "Mean offset (trial B - trial A)",
    "scale_ratio": "Scale ratio (SD B / SD A)",
    "smd": "Standardized mean difference",
    "shift_ratio": "Shift ratio (RMSE / neighbour RMSE)",
    "neighbour_rmse": "Neighbour RMSE (same session)",
    "within_sd": "Within-recording SD",
}
SHIFT_METRIC_REFERENCES = {
    "pearson_r": 0.0,
    "mean_offset": 0.0,
    "scale_ratio": 1.0,
    "smd": 1.0,
    "shift_ratio": 1.0,
}
SHIFT_METRIC_CMAPS = {"pearson_r": "viridis", "scale_ratio": "RdBu_r", "mean_offset": "RdBu_r"}
PERFORMANCE_METRICS = ("position_accuracy", "mean_distance_error", "room_accuracy")
PERFORMANCE_LABELS = {
    "position_accuracy": "Position accuracy",
    "mean_distance_error": "Mean distance error (m)",
    "median_distance_error": "Median distance error (m)",
    "room_accuracy": "Room accuracy",
}
_EPS = 1e-12


def _family_columns(
    df: pd.DataFrame,
    feature_family: str,
    exclude_subcarriers: dict[str, Sequence[int]] | None = None,
) -> dict[str, list[str]]:
    """Map each ESP to its ``feature_family`` columns, in subcarrier order.

    ``exclude_subcarriers`` maps an ESP to zero-based retained indices to leave out.
    """
    exclude = exclude_subcarriers or {}
    columns: dict[str, list[str]] = {}
    for column in df.columns:
        match = FEATURE_COLUMN_PATTERN.fullmatch(str(column))
        if match is None or match.group(2) != feature_family:
            continue
        esp, _, subcarrier = match.groups()
        if int(subcarrier) not in exclude.get(esp, ()):
            columns.setdefault(esp, []).append(column)
    return columns


def _normalized_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical user/trial/location values."""
    return df.assign(
        user=df["user"].map(normalize_two_digit_id),
        trial=df["trial"].map(normalize_two_digit_id),
        location=df["location"].map(normalize_position),
    )


def recording_profiles(
    df: pd.DataFrame,
    *,
    feature_family: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Mean and SD across windows of every recording, plus its window count.

    The returned frames are indexed by (user, location, trial) and hold the
    ``feature_family`` columns of every ESP.
    """
    columns = [column for cols in _family_columns(df, feature_family).values() for column in cols]
    grouped = _normalized_metadata(df).groupby(["user", "location", "trial"])[columns]
    return grouped.mean(), grouped.std(ddof=1), grouped.size()


def within_recording_sd_table(df: pd.DataFrame, *, feature_family: str = "mean") -> pd.DataFrame:
    """Within-recording SD across windows, averaged over subcarriers, per recording and ESP."""
    _, sds, counts = recording_profiles(df, feature_family=feature_family)
    rows = []
    for esp, columns in _family_columns(df, feature_family).items():
        esp_sd = sds[columns].mean(axis=1, skipna=True)
        rows.append(
            pd.DataFrame(
                {"esp": esp, "band": ESP_BANDS[esp], "within_sd": esp_sd, "windows": counts}
            )
        )
    return pd.concat(rows).reset_index()


def grid_neighbours(locations: Sequence[str]) -> dict[str, list[str]]:
    """Map each location to its 4-connected grid neighbours in the same room."""
    coordinates = {location: location_grid_coordinates(location) for location in locations}
    rooms = {location: room_label_for_location(location) for location in locations}
    neighbours = {}
    for location, (row, column) in coordinates.items():
        neighbours[location] = [
            other
            for other, (other_row, other_column) in coordinates.items()
            if rooms[other] == rooms[location]
            and abs(other_row - row) + abs(other_column - column) == 1
        ]
    return neighbours


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two equally shaped arrays (NaN-aware)."""
    a_centered = a - np.nanmean(a, axis=1, keepdims=True)
    b_centered = b - np.nanmean(b, axis=1, keepdims=True)
    numerator = np.nansum(a_centered * b_centered, axis=1)
    denominator = np.sqrt(np.nansum(a_centered**2, axis=1) * np.nansum(b_centered**2, axis=1))
    return numerator / np.maximum(denominator, _EPS)


def _row_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise root-mean-square difference (NaN-aware)."""
    return np.sqrt(np.nanmean((b - a) ** 2, axis=1))


def cross_session_similarity(
    df: pd.DataFrame,
    *,
    feature_family: str = "mean",
    trials: tuple[str, str] = ("01", "02"),
    exclude_subcarriers: dict[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Per (user, location, ESP) similarity between the trial-A and trial-B profiles.

    Only users recorded in both trials are included; ``exclude_subcarriers`` (ESP ->
    zero-based retained indices, e.g. pilot tones) drops subcarriers from every measure.
    Columns:

    - ``pearson_r``: correlation of the two subcarrier profiles (shape only);
    - ``rmse``: root-mean-square profile difference (magnitude-sensitive);
    - ``mean_offset``: mean of (B - A) over subcarriers (level change);
    - ``scale_ratio``: SD of B's profile over SD of A's profile (contrast change);
    - ``smd``: mean over subcarriers of ``|B - A|`` divided by the pooled within-recording
      SD across windows (shift relative to window-to-window variability);
    - ``neighbour_rmse``: mean RMSE between A's profile and the trial-A profiles of the
      same user at the 4-connected neighbouring positions in the same room;
    - ``shift_ratio``: ``rmse / neighbour_rmse`` (above 1, the session shift is larger
      than the difference between adjacent reference points).
    """
    trial_a, trial_b = (normalize_two_digit_id(trial) for trial in trials)
    means, sds, counts = recording_profiles(df, feature_family=feature_family)
    means_a = means.xs(trial_a, level="trial")
    means_b = means.xs(trial_b, level="trial")
    pairs = means_a.index.intersection(means_b.index)
    neighbours = grid_neighbours(sorted(means_a.index.get_level_values("location").unique()))

    tables = []
    for esp, columns in _family_columns(df, feature_family, exclude_subcarriers).items():
        a = means_a.loc[pairs, columns].to_numpy(dtype=float)
        b = means_b.loc[pairs, columns].to_numpy(dtype=float)
        sd_a = sds.xs(trial_a, level="trial").loc[pairs, columns].to_numpy(dtype=float)
        sd_b = sds.xs(trial_b, level="trial").loc[pairs, columns].to_numpy(dtype=float)
        pooled_sd = np.sqrt((sd_a**2 + sd_b**2) / 2)

        reference = means_a[columns]
        neighbour_rmse = []
        for (user, location), profile in zip(pairs, a, strict=True):
            keys = [(user, other) for other in neighbours.get(location, [])]
            keys = [key for key in keys if key in reference.index]
            if not keys:
                neighbour_rmse.append(np.nan)
                continue
            others = reference.loc[keys].to_numpy(dtype=float)
            neighbour_rmse.append(
                float(np.mean(_row_rmse(np.tile(profile, (len(keys), 1)), others)))
            )

        rmse = _row_rmse(a, b)
        table = pd.DataFrame(
            {
                "user": pairs.get_level_values("user"),
                "location": pairs.get_level_values("location"),
                "esp": esp,
                "band": ESP_BANDS[esp],
                "windows_a": counts.xs(trial_a, level="trial").loc[pairs].to_numpy(),
                "windows_b": counts.xs(trial_b, level="trial").loc[pairs].to_numpy(),
                "pearson_r": _row_pearson(a, b),
                "rmse": rmse,
                "mean_offset": np.nanmean(b - a, axis=1),
                "scale_ratio": np.nanstd(b, axis=1) / np.maximum(np.nanstd(a, axis=1), _EPS),
                "smd": np.nanmean(np.abs(b - a) / np.maximum(pooled_sd, _EPS), axis=1),
                "neighbour_rmse": neighbour_rmse,
            }
        )
        table["shift_ratio"] = table["rmse"] / table["neighbour_rmse"]
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.insert(2, "room", result["location"].map(room_label_for_location))
    return result


def shift_summary_by_esp(
    similarity: pd.DataFrame,
    *,
    by: Sequence[str] = ("esp",),
    metrics: Sequence[str] = SHIFT_METRICS,
) -> pd.DataFrame:
    """Median of every shift metric per ESP (or another grouping)."""
    keys = list(by)
    if "band" not in keys and "esp" in keys:
        keys = [*keys, "band"]
    summary = similarity.groupby(keys)[list(metrics)].median()
    summary.insert(0, "pairs", similarity.groupby(keys).size())
    return summary.reset_index()


def location_shift_table(
    similarity: pd.DataFrame,
    *,
    metrics: Sequence[str] = SHIFT_METRICS,
    esps: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Median of every shift metric across ESPs, per (user, location)."""
    frame = similarity if esps is None else similarity.loc[similarity["esp"].isin(esps)]
    return frame.groupby(["user", "location", "room"])[list(metrics)].median().reset_index()


def localization_by_location(predictions: pd.DataFrame) -> pd.DataFrame:
    """Position/room accuracy and distance errors per (user, true location)."""
    frame = predictions.assign(
        user=predictions["user"].map(normalize_two_digit_id),
        location=predictions["true_location"].map(normalize_position),
        correct=predictions["true_location"] == predictions["pred_location"],
        room_correct=predictions["true_room"] == predictions["pred_room"],
    )
    return (
        frame.groupby(["user", "location"])
        .agg(
            windows=("correct", "size"),
            position_accuracy=("correct", "mean"),
            room_accuracy=("room_correct", "mean"),
            mean_distance_error=("distance_error", "mean"),
            median_distance_error=("distance_error", "median"),
        )
        .reset_index()
    )


def shift_localization_table(
    location_shift: pd.DataFrame,
    localization: pd.DataFrame,
) -> pd.DataFrame:
    """Join position/volunteer-level shift measures with localization performance."""
    return location_shift.merge(localization, on=["user", "location"], how="inner")


def _spearman_row(x: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:  # noqa: PLR2004
        return {"spearman_rho": np.nan, "p_value": np.nan, "n": int(valid.sum())}
    result = spearmanr(x[valid], y[valid])
    return {
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": int(valid.sum()),
    }


def shift_localization_correlations(
    merged: pd.DataFrame,
    *,
    shift_metrics: Sequence[str] = SHIFT_METRICS,
    performance_metrics: Sequence[str] = PERFORMANCE_METRICS,
) -> pd.DataFrame:
    """Spearman correlations between shift and performance, pooled and per user."""
    groups = [("all", merged), *merged.groupby("user")]
    rows = [
        {
            "user": user,
            "shift_metric": shift_metric,
            "performance_metric": performance_metric,
            **_spearman_row(frame[shift_metric], frame[performance_metric]),
        }
        for user, frame in groups
        for shift_metric in shift_metrics
        for performance_metric in performance_metrics
    ]
    return pd.DataFrame(rows)


def esp_shift_localization_correlations(
    similarity: pd.DataFrame,
    localization: pd.DataFrame,
    *,
    shift_metric: str = "rmse",
    performance_metric: str = "mean_distance_error",
) -> pd.DataFrame:
    """Spearman correlation per ESP between its shift metric and location-level performance."""
    merged = similarity.merge(localization, on=["user", "location"], how="inner")
    rows = [
        {
            "esp": esp,
            "band": frame["band"].iloc[0],
            "shift_metric": shift_metric,
            "performance_metric": performance_metric,
            **_spearman_row(frame[shift_metric], frame[performance_metric]),
        }
        for esp, frame in merged.groupby("esp")
    ]
    return pd.DataFrame(rows)
