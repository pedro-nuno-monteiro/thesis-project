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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.config import ANCHOR_GROUPS
from utils.fingerprint import (
    FEATURE_COLUMN_PATTERN,
    fingerprint_feature_columns,
    fingerprints_by_trial,
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
EMBEDDING_METHODS = ("lda", "pca")
EMBEDDING_COLUMNS = ("dim1", "dim2")
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


def filter_users(df: pd.DataFrame, users: Sequence[str]) -> pd.DataFrame:
    """Rows of ``df`` whose user is in ``users`` (any id format, e.g. ``3`` or ``"03"``)."""
    wanted = {normalize_two_digit_id(user) for user in users}
    return df.loc[df["user"].map(normalize_two_digit_id).isin(wanted)].copy()


def profiles_by_user_trial(
    df: pd.DataFrame,
    *,
    position: str,
    users: Sequence[str],
    trials: Sequence[str] = ("01", "02"),
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Organized mean and SD fingerprints of one position for every (user, trial).

    Keys are ``"User XX / TYY"``, ready for ``plot_fingerprint_facets``. Pairs without
    recordings are left out.
    """
    means: dict[str, pd.DataFrame] = {}
    sds: dict[str, pd.DataFrame] = {}
    for user in users:
        user_id = normalize_two_digit_id(user)
        result = fingerprints_by_trial(df, position=position, user=user_id, trials=trials)
        for (label, mean), sd in zip(result.means.items(), result.sds.values(), strict=True):
            trial_id = label.split()[1]
            key = f"User {user_id} / T{trial_id}"
            means[key] = mean
            sds[key] = sd
    return means, sds


def profiles_by_position(
    df: pd.DataFrame,
    *,
    positions: Sequence[str],
    user: str,
    trial: str = "01",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Organized mean and SD fingerprints of several positions of one user and trial.

    Keys are the position labels; positions without recordings are left out.
    """
    means: dict[str, pd.DataFrame] = {}
    sds: dict[str, pd.DataFrame] = {}
    for position in positions:
        result = fingerprints_by_trial(df, position=position, user=user, trials=(trial,))
        if result.means:
            means[normalize_position(position)] = next(iter(result.means.values()))
            sds[normalize_position(position)] = next(iter(result.sds.values()))
    return means, sds


def fit_fingerprint_embedding(
    df: pd.DataFrame,
    *,
    fit_mask: pd.Series,
    method: str = "lda",
    pca_components: int = 50,
    random_state: int = 42,
) -> Pipeline:
    """Fit a 2-D projection of the full fingerprint on the ``fit_mask`` rows only.

    Every feature column is mean-imputed and standardized with statistics of the fit rows.
    ``"pca"`` keeps the two leading principal components. ``"lda"`` first reduces to
    ``pca_components`` principal components (regularizes LDA, which would otherwise
    separate the fit windows almost perfectly in thousands of dimensions) and then keeps
    the two discriminants that best separate the fit rows' locations.
    """
    method = str(method).lower()
    if method not in EMBEDDING_METHODS:
        msg = f"method must be one of {EMBEDDING_METHODS}, got {method!r}."
        raise ValueError(msg)
    columns = fingerprint_feature_columns(df)
    fit_rows = df.loc[fit_mask.reindex(df.index, fill_value=False).astype(bool)]
    if fit_rows.empty:
        msg = "fit_mask selects no rows."
        raise ValueError(msg)
    x = fit_rows[columns].to_numpy(dtype=float)
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]
    if method == "pca":
        steps.append(("pca", PCA(n_components=2, random_state=random_state)))
        embedding = Pipeline(steps).fit(x)
    else:
        n_classes = fit_rows["location"].nunique()
        n_pca = max(2, min(pca_components, len(fit_rows) - n_classes, len(columns)))
        steps.append(("pca", PCA(n_components=n_pca, random_state=random_state)))
        steps.append(("lda", LinearDiscriminantAnalysis(n_components=2)))
        embedding = Pipeline(steps).fit(x, fit_rows["location"].map(normalize_position))
    embedding.feature_columns_ = columns
    return embedding


def embed_windows(df: pd.DataFrame, embedding: Pipeline) -> pd.DataFrame:
    """Project every window of ``df`` with a fitted embedding.

    Returns user, trial, location, room and ``dim1``/``dim2``.
    """
    x = df.reindex(columns=embedding.feature_columns_).to_numpy(dtype=float)
    projected = embedding.transform(x)
    meta = _normalized_metadata(df)[["user", "trial", "location"]].reset_index(drop=True)
    meta["room"] = meta["location"].map(room_label_for_location)
    meta[list(EMBEDDING_COLUMNS)] = projected[:, :2]
    return meta


def centroid_table(
    embedded: pd.DataFrame,
    *,
    by: Sequence[str] = ("user", "trial", "location"),
) -> pd.DataFrame:
    """Centroid, 2x2 covariance (``var1``, ``var2``, ``cov12``) and size of every group."""
    grouped = embedded.groupby(list(by))
    table = grouped[list(EMBEDDING_COLUMNS)].mean()
    table["var1"] = grouped["dim1"].var(ddof=1)
    table["var2"] = grouped["dim2"].var(ddof=1)
    table["cov12"] = grouped[list(EMBEDDING_COLUMNS)].apply(
        lambda group: group["dim1"].cov(group["dim2"])
    )
    table["windows"] = grouped.size()
    return table.reset_index()


def centroid_shift_table(
    centroids: pd.DataFrame,
    *,
    trials: tuple[str, str] = ("01", "02"),
) -> pd.DataFrame:
    """How far each (user, location) centroid moves between trials, in embedding units.

    - ``shift``: distance between the trial-A and trial-B centroids;
    - ``nearest_other``: distance from the trial-A centroid to the closest trial-A centroid
      of another location of the same user;
    - ``embedding_shift_ratio``: ``shift / nearest_other`` (above 1, the session moves the
      fingerprint further than the gap to the closest other position);
    - ``nearest_a_location``: the trial-A location whose centroid is closest to the
      trial-B centroid, and ``nearest_is_correct`` whether that is the true location.

    Ratios are dimensionless, so they can be compared across normalizations.
    """
    trial_a, trial_b = (normalize_two_digit_id(trial) for trial in trials)
    rows = []
    for user, group in centroids.groupby("user"):
        a = group.loc[group["trial"] == trial_a].set_index("location")[list(EMBEDDING_COLUMNS)]
        b = group.loc[group["trial"] == trial_b].set_index("location")[list(EMBEDDING_COLUMNS)]
        a_points = a.to_numpy(dtype=float)
        for location in a.index.intersection(b.index):
            point_a = a.loc[location].to_numpy(dtype=float)
            point_b = b.loc[location].to_numpy(dtype=float)
            to_a = np.linalg.norm(a_points - point_a, axis=1)
            to_b = np.linalg.norm(a_points - point_b, axis=1)
            others = a.index != location
            shift = float(np.linalg.norm(point_b - point_a))
            nearest_other = float(to_a[others].min()) if others.any() else np.nan
            nearest = str(a.index[int(np.argmin(to_b))])
            rows.append(
                {
                    "user": user,
                    "location": location,
                    "room": room_label_for_location(location),
                    "shift": shift,
                    "nearest_other": nearest_other,
                    "embedding_shift_ratio": shift / max(nearest_other, _EPS),
                    "nearest_a_location": nearest,
                    "nearest_is_correct": nearest == location,
                }
            )
    return pd.DataFrame(rows)


def compare_normalizations(
    similarities: dict[str, pd.DataFrame],
    centroid_shifts: dict[str, pd.DataFrame] | None = None,
    *,
    by: Sequence[str] = ("user",),
) -> pd.DataFrame:
    """Per-group medians of the scale-free shift metrics for several normalizations.

    ``similarities`` maps a normalization label to a ``cross_session_similarity`` table;
    ``centroid_shifts`` (same labels) adds the embedding ratio and the share of positions
    whose trial-B centroid lands nearest its own trial-A centroid.
    """
    keys = list(by)
    frames = []
    for label, similarity in similarities.items():
        location = location_shift_table(similarity, metrics=("pearson_r", "shift_ratio"))
        summary = location.groupby(keys)[["pearson_r", "shift_ratio"]].median()
        summary["shift_ratio_above_1"] = location.groupby(keys)["shift_ratio"].apply(
            lambda values: float((values > 1).mean())
        )
        if centroid_shifts is not None and label in centroid_shifts:
            shifts = centroid_shifts[label].groupby(keys)
            summary["embedding_shift_ratio"] = shifts["embedding_shift_ratio"].median()
            summary["nearest_centroid_correct"] = shifts["nearest_is_correct"].mean()
        summary.insert(0, "normalization", label)
        frames.append(summary.reset_index())
    return pd.concat(frames, ignore_index=True)


def neighbour_centroids(
    embedded: pd.DataFrame,
    *,
    position: str,
    trial: str = "01",
) -> pd.DataFrame:
    """Centroids (pooled over users) of ``position``'s grid neighbours in ``trial``.

    Used as the "one grid cell" scale reference in a cross-session bubble map.
    """
    location = normalize_position(position)
    neighbours = grid_neighbours(sorted(embedded["location"].unique())).get(location, [])
    mask = embedded["location"].isin(neighbours) & embedded["trial"].eq(
        normalize_two_digit_id(trial)
    )
    return centroid_table(embedded.loc[mask], by=("location",))


def ideal_reference_features(
    df: pd.DataFrame,
    *,
    template_trial: str = "01",
    noise_scale: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Synthetic "ideal system" counterpart of ``df``, for reference plots.

    In the ideal system a fingerprint depends on the position only: every window of every
    (user, trial) at a position is the position template plus small noise. That means no
    session shift, no volunteer effect, and positions separated by the real gap between
    their templates.

    - template: mean of the ``template_trial`` windows at each location, pooled over the
      users in ``df``;
    - noise: Gaussian with ``noise_scale`` x the median within-recording SD of each feature
      column (the real window-to-window spread, shrunk).

    The rows, metadata and window counts are the same as in ``df``, so every analysis and
    plot runs on it unchanged. The feature families are perturbed independently (e.g.
    ``variance`` is not kept equal to ``std**2``); the values are a visual reference, not
    a physical simulation.
    """
    columns = fingerprint_feature_columns(df)
    meta = _normalized_metadata(df)
    template_rows = meta.loc[meta["trial"].eq(normalize_two_digit_id(template_trial))]
    if template_rows.empty:
        msg = f"No rows in template trial {template_trial!r}."
        raise ValueError(msg)
    templates = template_rows.groupby("location")[columns].mean()
    noise_sd = template_rows.groupby(["user", "location"])[columns].std(ddof=1).median().fillna(0.0)
    missing = sorted(set(meta["location"]) - set(templates.index))
    if missing:
        msg = f"Locations without a template-trial recording: {missing}."
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)
    values = templates.loc[meta["location"]].to_numpy(dtype=float, copy=True)
    values += rng.normal(size=values.shape) * (noise_scale * noise_sd.to_numpy(dtype=float))
    ideal = df.copy()
    ideal[columns] = values
    return ideal


def ideal_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Perfect-classifier counterpart of a predictions table (every window correct)."""
    ideal = predictions.copy()
    for target, source in (
        ("pred_position", "true_position"),
        ("pred_location", "true_location"),
        ("pred_room", "true_room"),
        ("pred_x", "true_x"),
        ("pred_y", "true_y"),
    ):
        if target in ideal.columns and source in ideal.columns:
            ideal[target] = ideal[source]
    ideal["distance_error"] = 0.0
    return ideal


def build_fingerprint_space(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    trials: tuple[str, str] = ("01", "02"),
    method: str = "lda",
    pca_components: int = 50,
    feature_family: str = "mean",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Embedding (fit on ``trials[0]``), centroids, centroid shifts and similarity of ``df``.

    Returns ``windows``, ``centroids``, ``shifts`` (``centroid_shift_table``) and
    ``similarity`` (``cross_session_similarity``), so measured and ideal-reference data
    go through exactly the same steps.
    """
    fit_mask = df["trial"].map(normalize_two_digit_id).eq(normalize_two_digit_id(trials[0]))
    embedding = fit_fingerprint_embedding(
        df,
        fit_mask=fit_mask,
        method=method,
        pca_components=pca_components,
        random_state=random_state,
    )
    windows = embed_windows(df, embedding)
    centroids = centroid_table(windows)
    return {
        "windows": windows,
        "centroids": centroids,
        "shifts": centroid_shift_table(centroids, trials=trials),
        "similarity": cross_session_similarity(df, feature_family=feature_family, trials=trials),
    }
