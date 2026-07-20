from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.feature_pipeline import (
    _esp_keys_for_scenario,
    build_frequency_feature_dataframes,
    iter_window_groups,
)
from utils.ml_pipeline import load_feature_dataframes
from utils.window_arrays import build_frequency_window_arrays

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Assertions are the pytest API.
# ruff: noqa: S101

DISPLAY_BANDS = ("2.4 GHz", "5 GHz", "Fusion")
EXPECTED_WINDOW_COUNT = 3


def _tiny_magnitude_data() -> dict[str, object]:
    magnitude_24 = np.arange(8, dtype=np.float32).reshape(4, 2)
    magnitude_5 = np.arange(12, dtype=np.float32).reshape(4, 3)
    return {
        "scenario_1": {
            "location_A-1": {
                "user_01": {
                    "esp_01": {"trial_01": magnitude_24},
                    "esp_11": {"trial_01": magnitude_5},
                },
            },
        },
    }


@pytest.mark.parametrize("band", DISPLAY_BANDS)
def test_iter_window_groups_accepts_display_band_names(band: str) -> None:
    groups = list(
        iter_window_groups(
            _tiny_magnitude_data(),
            band,
            window_size=2,
            overlap_size=1,
            require_all_esps=False,
        ),
    )

    assert len(groups) == 1
    assert groups[0].min_windows == EXPECTED_WINDOW_COUNT


def test_feature_dataframe_build_preserves_window_counts_for_display_names() -> None:
    frames = build_frequency_feature_dataframes(
        _tiny_magnitude_data(),
        window_size=2,
        overlap_size=1,
        require_all_esps=False,
    )

    assert [len(frame) for frame in frames] == [EXPECTED_WINDOW_COUNT] * len(DISPLAY_BANDS)
    assert [frame["frequency_scenario"].unique().tolist() for frame in frames] == [
        ["2.4 GHz"],
        ["5 GHz"],
        ["Fusion"],
    ]


def test_load_feature_dataframes_builds_inventory_with_display_names() -> None:
    observed_scenarios: list[str] = []

    def capture_scenario(*args: object, **_kwargs: object) -> Iterator[object]:
        observed_scenarios.append(str(args[1]))
        return iter(())

    empty_frames = {band: pd.DataFrame() for band in DISPLAY_BANDS}
    with (
        patch("utils.ml_pipeline.iter_window_groups", side_effect=capture_scenario),
        patch("utils.ml_pipeline.get_all_dataframes", return_value=empty_frames),
    ):
        load_feature_dataframes(
            _tiny_magnitude_data(),
            preproc_opts={"normalization": "none"},
            feat_opts={"window_size": 2, "overlap_size": 1, "require_all_esps": False},
            bands_to_run=DISPLAY_BANDS,
        )

    assert observed_scenarios == list(DISPLAY_BANDS)


def test_window_array_builder_delegates_with_display_name(tmp_path: Path) -> None:
    with (
        patch("utils.window_arrays.get_results_path", return_value=tmp_path),
        patch("utils.window_arrays.iter_window_groups", return_value=iter(())) as groups,
        pytest.raises(ValueError, match=r"No windows found for 2\.4 GHz"),
    ):
        build_frequency_window_arrays({}, "2.4 GHz")

    assert groups.call_args.args[1] == "2.4 GHz"


def test_unknown_frequency_scenario_has_descriptive_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown frequency_scenario '2_4ghz'\.",
    ) as error:
        _esp_keys_for_scenario("2_4ghz")

    message = str(error.value)
    assert "Unknown frequency_scenario '2_4ghz'" in message
    assert "2.4 GHz" in message
    assert "5 GHz" in message
    assert "Fusion" in message
    assert "slugs ('2_4ghz') are for cache paths only" in message
