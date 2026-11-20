from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from utils.data_quality import (
    aligned_window_table,
    alignment_loss_table,
    duplicate_recordings_table,
    duplicate_summary,
    missing_recordings_table,
    pilot_excess_table,
    pilot_feature_ratio_table,
    pilot_subcarrier_numbers_by_esp,
    session_dates_table,
    subcarrier_tone_map,
    z0_reference_table,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_five_ghz_pilots_map_to_retained_numbers_8_22_35_49() -> None:
    tone_map = subcarrier_tone_map("5 GHz")

    pilots = tone_map.loc[tone_map["is_pilot"]]
    assert pilots["tone"].tolist() == [-21, -7, 7, 21]
    assert pilots["retained_number"].tolist() == [8, 22, 35, 49]
    assert tone_map.loc[tone_map["retained_index"].isna(), "tone"].tolist() == [0]
    assert int(tone_map["retained_index"].max()) + 1 == 56  # noqa: PLR2004


def test_two_ghz_selection_keeps_50_tones_and_drops_dc_and_plus_one() -> None:
    tone_map = subcarrier_tone_map("2.4 GHz")
    kept = tone_map.dropna(subset=["retained_index"]).sort_values("retained_index")

    assert len(kept) == 50  # noqa: PLR2004
    assert kept["tone"].tolist() == [*range(-26, 0), *range(2, 26)]
    assert pilot_subcarrier_numbers_by_esp()["esp_01"] == [32, 46, 6, 20]


def _magnitude_map(spike: float) -> dict:
    five = np.ones((10, 56))
    five[:, [7, 21, 34, 48]] = spike
    return {
        "scenario_1": {
            "location_A-1": {
                "user_01": {
                    "esp_11": {"trial_01": five},
                    "esp_01": {"trial_01": np.ones((10, 50))},
                }
            }
        }
    }


def test_pilot_excess_detects_spiking_recordings() -> None:
    five = pilot_excess_table(_magnitude_map(3.0), bands=("5 GHz",))
    two = pilot_excess_table(_magnitude_map(3.0), bands=("2.4 GHz",))

    assert five["pilot_ratio_median"].item() == pytest.approx(3.0)
    assert five["spike_packet_fraction"].item() == 1.0
    assert five["pilot_power_fraction"].item() == pytest.approx(36 / (52 + 36))
    assert two["pilot_ratio_median"].item() == pytest.approx(1.0)


def test_pilot_feature_ratio_table() -> None:
    columns = {f"esp_11_std_sc_{sc:02d}": [1.0, 1.0] for sc in range(56)}
    for sc in (7, 21, 34, 48):
        columns[f"esp_11_std_sc_{sc:02d}"] = [2.0, 4.0]
    df = pd.DataFrame({"trial": ["01", "02"], **columns})

    table = pilot_feature_ratio_table(df, feature_family="std")

    assert table.loc[0, ["01", "02", "all"]].tolist() == pytest.approx([2.0, 4.0, 3.0])


def _diagnostics() -> pd.DataFrame:
    rows = []
    for user, trial, date in (("01", "01", "10-06"), ("01", "02", "16-07")):
        for location in ("A-1", "Z-0"):
            for esp in ("esp_01", "esp_11"):
                if (user, trial, location, esp) == ("01", "02", "A-1", "esp_11"):
                    continue
                rows.append(
                    {
                        "user": user,
                        "trial": trial,
                        "location": location,
                        "esp": esp,
                        "band": "2.4 GHz" if esp == "esp_01" else "5 GHz",
                        "file_path": (
                            f"/d/1_{location}_{user}_{esp[-2:]}_{trial}_{date}_10-00-00.csv"
                        ),
                        "valid_magnitude_rows": 130 if esp == "esp_01" else 70,
                    }
                )
    return pd.DataFrame(rows)


def test_inventory_tables_find_missing_recordings_and_dates() -> None:
    diagnostics = _diagnostics()

    summary, detail = missing_recordings_table(diagnostics)
    dates = session_dates_table(diagnostics)
    z0 = z0_reference_table(diagnostics)
    loss = alignment_loss_table(diagnostics, window_size=60)

    recorded = detail.loc[detail["esp"].isin(["esp_01", "esp_11"])]
    assert recorded[["trial", "location", "esp"]].to_numpy().tolist() == [["02", "A-1", "esp_11"]]
    assert summary.loc[0, "expected"] == 9 * 2  # 9 configured 2.4 GHz ESPs x 2 locations
    assert dates["date"].tolist() == ["10-06", "16-07"]
    assert z0["z0_esps"].tolist()[:2] == ["1/9", "1/10"]
    assert loss["recordings"].tolist() == [1, 1, 1]


def test_duplicate_recordings_table_finds_identical_files(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.csv", "b.csv", "c.csv")]
    paths[0].write_text("same")
    paths[1].write_text("same")
    paths[2].write_text("diff")
    diagnostics = pd.DataFrame(
        {
            "user": ["01", "06", "02"],
            "trial": "01",
            "location": "A-1",
            "esp": "esp_08",
            "band": "2.4 GHz",
            "file_path": [str(path) for path in paths],
        }
    )

    duplicates = duplicate_recordings_table(diagnostics)

    assert duplicates["user"].tolist() == ["01", "06"]
    assert duplicate_summary(duplicates)["duplicates_of_users"].tolist() == ["06", "01"]


def test_aligned_window_table_counts_windows_per_recording() -> None:
    df = pd.DataFrame(
        {
            "user": ["01"] * 5,
            "trial": ["01", "01", "01", "02", "02"],
            "group_id": ["g1", "g1", "g2", "g3", "g3"],
        }
    )

    table = aligned_window_table({"Fusion": df})

    assert table["windows"].tolist() == [3, 2]
    assert table["windows_per_recording_min"].tolist() == [1, 2]
