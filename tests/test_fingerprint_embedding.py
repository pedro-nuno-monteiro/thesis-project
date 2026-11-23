from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from utils.domain_shift import (
    centroid_shift_table,
    centroid_table,
    compare_normalizations,
    cross_session_similarity,
    embed_windows,
    filter_users,
    fit_fingerprint_embedding,
    ideal_predictions,
    ideal_reference_features,
    profiles_by_user_trial,
)
from utils.plots import (
    plot_centroid_shift_arrows,
    plot_fingerprint_embedding,
    plot_fingerprint_facets,
    user_trial_styles,
)

if TYPE_CHECKING:
    from pathlib import Path

LOCATIONS = ("A-1", "A-2", "B-1", "B-2")
N_SUBCARRIERS = 6


def _feature_df(trial_02_offset: float = 0.0) -> pd.DataFrame:
    """Three users, four positions, five windows; trial 02 = trial 01 + offset."""
    rng = np.random.default_rng(0)
    rows = []
    for user in ("01", "03", "05"):
        for loc_idx, location in enumerate(LOCATIONS):
            base = np.sin(np.arange(N_SUBCARRIERS) + loc_idx) * 3 + loc_idx
            for trial in ("01", "02"):
                for window_idx in range(5):
                    profile = base + rng.normal(0, 0.05, N_SUBCARRIERS)
                    if trial == "02":
                        profile = profile + trial_02_offset
                    row = {
                        "user": user,
                        "location": location,
                        "trial": trial,
                        "scenario": "1",
                        "group_id": f"{user}_{location}_{trial}",
                        "window_idx": window_idx,
                    }
                    for sc, value in enumerate(profile):
                        row[f"esp_11_mean_sc_{sc:02d}"] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def test_filter_users_accepts_any_id_format() -> None:
    df = _feature_df()

    assert set(filter_users(df, [1, "user_03"])["user"]) == {"01", "03"}


@pytest.mark.parametrize("method", ["lda", "pca"])
def test_embedding_is_fit_on_trial_01_only(method: str) -> None:
    clean = _feature_df()
    shifted = _feature_df(trial_02_offset=100.0)
    fit_mask = clean["trial"].eq("01")

    emb_clean = fit_fingerprint_embedding(clean, fit_mask=fit_mask, method=method)
    emb_shifted = fit_fingerprint_embedding(shifted, fit_mask=fit_mask, method=method)

    # Trial-02 rows never reach the fit, so the scaler statistics are identical.
    np.testing.assert_allclose(emb_clean["scale"].mean_, emb_shifted["scale"].mean_)
    embedded = embed_windows(shifted, emb_shifted)
    assert list(embedded.columns) == ["user", "trial", "location", "room", "dim1", "dim2"]
    assert len(embedded) == len(shifted)


def test_unshifted_session_lands_on_its_own_position() -> None:
    df = _feature_df()
    embedding = fit_fingerprint_embedding(df, fit_mask=df["trial"].eq("01"))
    centroids = centroid_table(embed_windows(df, embedding))
    shifts = centroid_shift_table(centroids)

    assert len(centroids) == 3 * 2 * len(LOCATIONS)
    assert {"var1", "var2", "cov12", "windows"} <= set(centroids.columns)
    assert len(shifts) == 3 * len(LOCATIONS)
    assert shifts["nearest_is_correct"].all()
    assert (shifts["embedding_shift_ratio"] < 1).all()


def test_compare_normalizations_stacks_labels() -> None:
    df = _feature_df()
    embedding = fit_fingerprint_embedding(df, fit_mask=df["trial"].eq("01"))
    shifts = centroid_shift_table(centroid_table(embed_windows(df, embedding)))
    similarity = cross_session_similarity(df)

    table = compare_normalizations(
        {"none": similarity, "ebl": similarity}, {"none": shifts, "ebl": shifts}
    )

    assert list(table["normalization"]) == ["none"] * 3 + ["ebl"] * 3
    assert {"shift_ratio", "embedding_shift_ratio", "nearest_centroid_correct"} <= set(
        table.columns
    )


def test_profiles_and_embedding_plots_save_pdf(tmp_path: Path) -> None:
    df = _feature_df(trial_02_offset=0.5)
    means, sds = profiles_by_user_trial(df, position="A-1", users=("01", "03", "05"))
    assert list(means) == [f"User {u} / T{t}" for u in ("01", "03", "05") for t in ("01", "02")]

    plot_fingerprint_facets(
        means,
        spreads=sds,
        feature_family="mean",
        styles=user_trial_styles(list(means)),
        save_path=tmp_path / "profiles.pdf",
        show=False,
    )
    embedding = fit_fingerprint_embedding(df, fit_mask=df["trial"].eq("01"))
    windows = embed_windows(df, embedding)
    centroids = centroid_table(windows)
    focus = centroids.loc[centroids["location"].eq("A-1")]
    plot_fingerprint_embedding(
        {"panel": (windows, focus)},
        color_by="user",
        reference={"panel": focus.loc[focus["trial"].eq("01")]},
        save_path=tmp_path / "bubbles.pdf",
        show=False,
    )
    plot_centroid_shift_arrows(
        {"User 01": centroids.loc[centroids["user"].eq("01")]},
        save_path=tmp_path / "arrows.pdf",
        show=False,
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "profiles.pdf",
        "bubbles.pdf",
        "arrows.pdf",
    }


def test_ideal_reference_has_no_session_or_user_effect() -> None:
    real = _feature_df(trial_02_offset=5.0)
    ideal = ideal_reference_features(real, noise_scale=0.0)

    columns = [column for column in real.columns if column.startswith("esp_")]
    assert ideal.drop(columns=columns).equals(real.drop(columns=columns))
    per_location = ideal.groupby("location")[columns].nunique()
    assert (per_location == 1).all().all()
    similarity = cross_session_similarity(ideal_reference_features(real))
    assert (similarity["shift_ratio"] < 0.1).all()


def test_ideal_predictions_are_perfect() -> None:
    predictions = pd.DataFrame(
        {
            "true_position": ["A-1", "B-2"],
            "pred_position": ["A-2", "B-2"],
            "true_room": [1, 1],
            "pred_room": [2, 1],
            "true_x": [0.0, 1.0],
            "pred_x": [1.0, 1.0],
            "distance_error": [1.0, 0.0],
        }
    )
    ideal = ideal_predictions(predictions)

    assert (ideal["pred_position"] == ideal["true_position"]).all()
    assert (ideal["distance_error"] == 0).all()
    assert predictions["distance_error"].sum() == 1.0
