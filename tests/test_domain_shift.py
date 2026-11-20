from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from utils.domain_shift import (
    cross_session_similarity,
    grid_neighbours,
    localization_by_location,
    location_shift_table,
    shift_localization_correlations,
    shift_localization_table,
    shift_summary_by_esp,
    within_recording_sd_table,
)
from utils.plots import plot_floor_plan_metric, plot_metric_by_esp, plot_shift_vs_localization

if TYPE_CHECKING:
    from pathlib import Path

LOCATIONS = ("A-1", "A-2", "B-1", "B-2")
PROFILE = np.array([0.0, 1.0, 3.0, 2.0])


def _feature_df() -> pd.DataFrame:
    """Two users, four Room-1 positions; trial 02 = 2 x trial 01 + 0.5."""
    rows = []
    for user in ("01", "03"):
        for loc_idx, location in enumerate(LOCATIONS):
            base = PROFILE + loc_idx
            for trial in ("01", "02"):
                for window_idx, noise in enumerate((-0.1, 0.1)):
                    profile = base if trial == "01" else 2 * base + 0.5
                    row = {
                        "user": user,
                        "location": location,
                        "trial": trial,
                        "group_id": f"{user}_{location}_{trial}",
                        "window_idx": window_idx,
                    }
                    for sc, value in enumerate(profile + noise):
                        row[f"esp_11_mean_sc_{sc:02d}"] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def test_grid_neighbours_are_four_connected() -> None:
    neighbours = grid_neighbours(list(LOCATIONS))

    assert sorted(neighbours["A-1"]) == ["A-2", "B-1"]
    assert sorted(neighbours["B-2"]) == ["A-2", "B-1"]


def test_cross_session_similarity_separates_shape_and_magnitude() -> None:
    similarity = cross_session_similarity(_feature_df())

    assert len(similarity) == len(LOCATIONS) * 2
    assert similarity["pearson_r"].to_numpy() == pytest.approx(1.0)
    assert similarity["scale_ratio"].to_numpy() == pytest.approx(2.0)
    a1 = similarity.loc[(similarity["user"] == "01") & (similarity["location"] == "A-1")].iloc[0]
    assert a1["mean_offset"] == pytest.approx(PROFILE.mean() + 0.5)
    # A-1's neighbours A-2 and B-1 differ by a constant +1 and +2 -> RMSE 1 and 2.
    assert a1["neighbour_rmse"] == pytest.approx(1.5)
    assert a1["shift_ratio"] == pytest.approx(a1["rmse"] / 1.5)
    assert a1["room"] == 1


def test_exclude_subcarriers_changes_profile_length() -> None:
    similarity = cross_session_similarity(_feature_df(), exclude_subcarriers={"esp_11": [2]})
    a1 = similarity.loc[(similarity["user"] == "01") & (similarity["location"] == "A-1")].iloc[0]

    assert a1["mean_offset"] == pytest.approx(np.delete(PROFILE, 2).mean() + 0.5)


def test_within_recording_sd_and_summaries() -> None:
    similarity = cross_session_similarity(_feature_df())

    sd = within_recording_sd_table(_feature_df())
    assert sd["within_sd"].to_numpy() == pytest.approx(np.sqrt(0.02))
    assert shift_summary_by_esp(similarity)["pairs"].item() == len(similarity)
    assert len(location_shift_table(similarity)) == len(similarity)


def _predictions() -> pd.DataFrame:
    rows = []
    for user in ("01", "03"):
        for loc_idx, location in enumerate(LOCATIONS):
            for window in range(4):
                correct = window < 4 - loc_idx
                rows.append(
                    {
                        "user": user,
                        "true_location": location,
                        "pred_location": location if correct else "A-1",
                        "true_room": 1,
                        "pred_room": 1,
                        "distance_error": 0.0 if correct else float(loc_idx),
                    }
                )
    return pd.DataFrame(rows)


def test_localization_link_and_correlations() -> None:
    localization = localization_by_location(_predictions())
    merged = shift_localization_table(
        location_shift_table(cross_session_similarity(_feature_df())), localization
    )
    correlations = shift_localization_correlations(
        merged, shift_metrics=("rmse",), performance_metrics=("position_accuracy",)
    )

    a2 = localization.query("user == '01' and location == 'A-2'").iloc[0]
    assert a2["position_accuracy"] == pytest.approx(0.75)
    assert len(merged) == len(LOCATIONS) * 2
    # RMSE grows with loc_idx while accuracy falls -> perfect negative rank correlation.
    assert correlations.loc[correlations["user"] == "all", "spearman_rho"].item() == pytest.approx(
        -1.0
    )


def test_shift_plots_save_pdf(tmp_path: Path) -> None:
    similarity = cross_session_similarity(_feature_df())
    merged = shift_localization_table(
        location_shift_table(similarity), localization_by_location(_predictions())
    )

    plot_metric_by_esp(
        similarity, metric="rmse", hue="user", save_path=tmp_path / "box.pdf", show=False
    )
    plot_floor_plan_metric(
        merged.groupby("location")["rmse"].mean(),
        label="RMSE",
        save_path=tmp_path / "floor.pdf",
        show=False,
    )
    plot_shift_vs_localization(
        merged,
        shift_metrics=("rmse", "pearson_r"),
        performance_metrics=("position_accuracy",),
        save_path=tmp_path / "scatter.pdf",
        show=False,
    )

    assert {path.name for path in tmp_path.iterdir()} == {"box.pdf", "floor.pdf", "scatter.pdf"}
