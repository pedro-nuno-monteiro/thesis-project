from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from utils.feature_pipeline import FEATURE_NAMES
from utils.fingerprint import (
    feature_family_table,
    feature_inventory_table,
    fingerprint_table_preview,
    fingerprints_by_trial,
    select_fingerprint,
)
from utils.plots import plot_fingerprint_esp_overlay, plot_fingerprint_facets

if TYPE_CHECKING:
    from pathlib import Path

ESP_SUBCARRIERS = {"esp_01": 3, "esp_11": 4}


def _feature_df(trials: tuple[str, ...] = ("01", "02"), windows: int = 3) -> pd.DataFrame:
    rows = []
    for trial in trials:
        for window_idx in range(windows):
            row = {
                "frequency_scenario": "Fusion",
                "scenario": "lab",
                "location": "B-8",
                "user": "03",
                "trial": trial,
                "group_id": f"lab_B-8_03_{trial}",
                "window_idx": window_idx,
                "label": 0,
            }
            for esp, subcarriers in ESP_SUBCARRIERS.items():
                for family_idx, family in enumerate(FEATURE_NAMES):
                    for sc in range(subcarriers):
                        row[f"{esp}_{family}_sc_{sc:02d}"] = (
                            100 * int(trial) + 10 * window_idx + family_idx + sc / 10
                        )
            rows.append(row)
    return pd.DataFrame(rows)


def test_individual_selection_organizes_every_esp_and_family() -> None:
    selection = select_fingerprint(
        {"Fusion": _feature_df()},
        band="Fusion",
        position="location_b-8",
        user=3,
        trial="trial_01",
        mode="individual",
        window_index=2,
    )

    assert selection.position == "B-8"
    assert selection.user == "03"
    assert selection.n_fingerprints == 1
    assert selection.esps == ["esp_01", "esp_11"]
    assert list(selection.table.columns) == list(FEATURE_NAMES)
    assert len(selection.table) == sum(ESP_SUBCARRIERS.values())
    assert selection.table.loc[("esp_11", 3, 4), "mean"] == pytest.approx(100 + 20 + 0.3)
    assert selection.summary["windows_in_recording"] == 3
    assert selection.slug == "B-8-u03-fusion-t01-individual-w2"


def test_mean_selection_averages_all_windows() -> None:
    selection = select_fingerprint(
        {"Fusion": _feature_df()},
        band="Fusion",
        position="B-8",
        user="03",
        trial="02",
        mode="mean",
        esps=("esp_11",),
    )

    assert selection.n_fingerprints == 3
    assert selection.window_index is None
    assert selection.esps == ["esp_11"]
    assert selection.table.loc[("esp_01", 0, 1), "std"] == pytest.approx(200 + 10 + 1)


def test_unknown_window_and_recording_raise() -> None:
    dataframes = {"Fusion": _feature_df()}
    common = {"band": "Fusion", "position": "B-8", "user": "03"}
    with pytest.raises(ValueError, match="window_idx=9"):
        select_fingerprint(dataframes, **common, trial="01", window_index=9)
    with pytest.raises(LookupError, match="No fingerprints"):
        select_fingerprint(dataframes, **common, trial="07")


def test_family_table_and_preview_shapes() -> None:
    selection = select_fingerprint(
        {"Fusion": _feature_df()}, band="Fusion", position="B-8", user="03", trial="01"
    )

    wide = feature_family_table(selection.table, feature_family="maximum", esps=selection.esps)
    assert list(wide.columns) == ["esp_01", "esp_11"]
    assert list(wide.index) == [1, 2, 3, 4]
    assert np.isnan(wide.loc[4, "esp_01"])

    assert len(fingerprint_table_preview(selection.table, 2)) == 4
    assert len(fingerprint_table_preview(selection.table, None)) == len(selection.table)


def test_mean_fingerprints_by_trial_reports_missing_trials() -> None:
    result = fingerprints_by_trial(
        _feature_df(trials=("01",)),
        position="B-8",
        user="03",
        trials=("01", "02"),
    )

    assert list(result.means) == ["Trial 01 (n=3)"]
    assert result.sds["Trial 01 (n=3)"].loc[("esp_01", 0, 1), "mean"] == pytest.approx(10.0)
    assert result.inventory.to_dict("records") == [
        {"trial": "01", "windows_used": 3},
        {"trial": "02", "windows_used": 0},
    ]


def test_feature_inventory_table_counts() -> None:
    inventory = feature_inventory_table({"Fusion": _feature_df()})

    row = inventory.iloc[0]
    assert row["fingerprints"] == 6
    assert row["recordings"] == 2
    assert row["trials"] == "01, 02"
    assert row["ESPs"] == 2
    assert row["feature_columns"] == len(FEATURE_NAMES) * sum(ESP_SUBCARRIERS.values())


def test_fingerprint_plots_save_pdf(tmp_path: Path) -> None:
    result = fingerprints_by_trial(_feature_df(), position="B-8", user="03", trials=("01", "02"))
    table = next(iter(result.means.values()))

    plot_fingerprint_facets(
        result.means,
        spreads=result.sds,
        feature_family="mean",
        save_path=tmp_path / "facets.pdf",
        show=False,
    )
    plot_fingerprint_esp_overlay(
        table, feature_family="energy", save_path=tmp_path / "overlay.pdf", show=False
    )

    assert (tmp_path / "facets.pdf").is_file()
    assert (tmp_path / "overlay.pdf").is_file()
