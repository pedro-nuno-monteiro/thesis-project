# ruff: noqa: S101

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from utils.csi_processing import (
    _calibrate_complex_csi_with_rssi,
    _normalize_complex_csi_per_packet,
)
from utils.load_csi import (
    FIVE_GHZ_RAW_LENGTH,
    FIVE_GHZ_RAW_LENGTH_228,
    TWO_GHZ_RAW_LENGTH,
    _processor_identity,
    _RunOptions,
    process_csv_files,
)

EXPECTED_FILTERED_PACKETS = 2
EXPECTED_CALIBRATED_FILES = 2
EXPECTED_ACCEPTED_FIVE_GHZ_PACKET_ROWS = 2
VISUALIZATION_FLOOR_MAGNITUDE = 1e-4
LOW_VALID_RSSI_DBM = -90.0


def _csi_payload(raw_length: int, *, zero: bool = False) -> str:
    values = ["0" if zero else value for value in ("1", "2") * (raw_length // 2)]
    return f"[{' '.join(values)}]"


def _csi_payload_from_values(values: list[int]) -> str:
    return f"[{' '.join(str(value) for value in values)}]"


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as file:
        csv.writer(file).writerows(rows)


def _two_ghz_row(rssi_dbm: float, csi_payload: str) -> list[str]:
    row = ["0"] * 25
    row[3] = str(rssi_dbm)
    row[24] = csi_payload
    return row


def _five_ghz_row(rssi_dbm: float, agc_gain: int, csi_payload: str) -> list[str]:
    row = ["0"] * 15
    row[3] = str(rssi_dbm)
    row[7] = str(agc_gain)
    row[14] = csi_payload
    return row


def test_rssi_calibration_preserves_shape_and_matches_packet_power() -> None:
    csi_complex = np.array(
        [
            [1 + 2j, 3 + 4j, 5 + 6j],
            [2 + 0j, 0 + 2j, 2 + 2j],
        ],
        dtype=complex,
    )
    rssi_dbm = np.array([-40.0, -55.0])

    csi_calibrated = _calibrate_complex_csi_with_rssi(csi_complex, rssi_dbm)

    assert csi_calibrated.shape == csi_complex.shape
    assert np.isfinite(csi_calibrated).all()
    calibrated_power = np.sum(np.abs(csi_calibrated) ** 2, axis=1)
    rssi_mw = 10 ** (rssi_dbm / 10.0)
    assert np.allclose(calibrated_power, rssi_mw, rtol=1e-5, atol=1e-12)


def test_packet_norm_preserves_shape_and_unit_power() -> None:
    csi_complex = np.array([[3 + 4j, 0 + 0j], [1 + 1j, 1 - 1j]], dtype=complex)

    normalized = _normalize_complex_csi_per_packet(csi_complex)

    assert normalized.shape == csi_complex.shape
    assert np.isfinite(normalized).all()
    assert np.allclose(np.sum(np.abs(normalized) ** 2, axis=1), 1.0)


def test_cache_identity_separates_calibration_modes() -> None:
    none_identity = _processor_identity(_RunOptions(calibration_mode="none"))
    rssi_identity = _processor_identity(_RunOptions(calibration_mode="rssi"))

    assert none_identity != rssi_identity


def test_process_csv_files_rssi_filters_packets_and_keeps_agc_aligned() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        two_file = root / "two.csv"
        five_file = root / "five.csv"
        good_two_csi = _csi_payload(TWO_GHZ_RAW_LENGTH)
        good_five_csi = _csi_payload(FIVE_GHZ_RAW_LENGTH)
        _write_csv(
            two_file,
            [
                _two_ghz_row(-40.0, good_two_csi),
                _two_ghz_row(-100.0, good_two_csi),
            ],
        )
        _write_csv(
            five_file,
            [
                _five_ghz_row(-40.0, 21, good_five_csi),
                _five_ghz_row(-100.0, 22, good_five_csi),
            ],
        )
        data_files = {
            "scenario_1": {
                "location_A-1": {
                    "user_01": {
                        "esp_01": {"trial_01": two_file},
                        "esp_11": {"trial_01": five_file},
                    },
                },
            },
        }

        magnitudes, agc_gain_data, stats = process_csv_files(
            data_files,
            max_workers=1,
            use_cache=False,
            return_stats=True,
            calibration_mode="rssi",
            min_rssi_dbm=-95.0,
        )

    mag_24 = magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_01"]["trial_01"]
    mag_5 = magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_11"]["trial_01"]
    agc_gains = agc_gain_data["scenario_1"]["location_A-1"]["user_01"]["esp_11"]["trial_01"]
    expected_power = 10 ** (-40.0 / 10.0)

    assert mag_24.shape == (1, 50)
    assert mag_5.shape == (1, 56)
    assert agc_gains.tolist() == [21]
    assert np.allclose(np.sum(mag_24**2, axis=1), expected_power)
    assert np.allclose(np.sum(mag_5**2, axis=1), expected_power)
    assert stats.invalid_packets_removed == EXPECTED_FILTERED_PACKETS
    assert stats.calibration_applied_files == EXPECTED_CALIBRATED_FILES


def test_five_ghz_accepts_228_byte_packets_by_keeping_first_114_values() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        two_file = root / "two.csv"
        five_file = root / "five.csv"
        five_114_values = list(range(1, FIVE_GHZ_RAW_LENGTH + 1))
        five_228_values = list(range(201, 201 + FIVE_GHZ_RAW_LENGTH_228))
        invalid_5ghz_values = list(range(1, FIVE_GHZ_RAW_LENGTH + 2))

        _write_csv(
            two_file,
            [
                _two_ghz_row(-40.0, _csi_payload(TWO_GHZ_RAW_LENGTH)),
                _two_ghz_row(-40.0, _csi_payload(FIVE_GHZ_RAW_LENGTH_228)),
            ],
        )
        _write_csv(
            five_file,
            [
                _five_ghz_row(-40.0, 11, _csi_payload_from_values(five_114_values)),
                _five_ghz_row(-41.0, 12, _csi_payload_from_values(five_228_values)),
                _five_ghz_row(-42.0, 13, _csi_payload_from_values(invalid_5ghz_values)),
            ],
        )
        data_files = {
            "scenario_1": {
                "location_A-1": {
                    "user_01": {
                        "esp_01": {"trial_01": two_file},
                        "esp_11": {"trial_01": five_file},
                    },
                },
            },
        }

        magnitudes, agc_gain_data, diagnostics = process_csv_files(
            data_files,
            max_workers=1,
            use_cache=False,
            return_diagnostics=True,
            calibration_mode="none",
        )

    mag_24 = magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_01"]["trial_01"]
    mag_5 = magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_11"]["trial_01"]
    agc_gains = agc_gain_data["scenario_1"]["location_A-1"]["user_01"]["esp_11"]["trial_01"]

    first_114_complex = np.array(five_114_values[1::2]) + 1j * np.array(five_114_values[::2])
    first_228_complex = np.array(five_228_values[1:FIVE_GHZ_RAW_LENGTH:2]) + 1j * np.array(
        five_228_values[:FIVE_GHZ_RAW_LENGTH:2],
    )
    expected_5 = np.abs(np.delete(np.vstack([first_114_complex, first_228_complex]), [28], axis=1))
    five_diagnostics = diagnostics.loc[diagnostics["esp"] == "esp_11"].iloc[0]
    two_diagnostics = diagnostics.loc[diagnostics["esp"] == "esp_01"].iloc[0]

    assert mag_24.shape == (1, 50)
    assert mag_5.shape == (2, 56)
    assert np.array_equal(agc_gains, np.array([11, 12]))
    assert np.array_equal(mag_5, expected_5)
    assert five_diagnostics["dropped_csi_length_count"] == 1
    assert five_diagnostics["valid_magnitude_rows"] == EXPECTED_ACCEPTED_FIVE_GHZ_PACKET_ROWS
    assert two_diagnostics["dropped_csi_length_count"] == 1
    assert two_diagnostics["valid_magnitude_rows"] == 1


def test_rssi_calibration_keeps_valid_packets_below_visualization_floor() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        two_file = root / "two.csv"
        good_two_csi = _csi_payload(TWO_GHZ_RAW_LENGTH)
        _write_csv(
            two_file,
            [
                _two_ghz_row(LOW_VALID_RSSI_DBM, good_two_csi),
                _two_ghz_row(-100.0, good_two_csi),
            ],
        )
        data_files = {
            "scenario_1": {
                "location_A-1": {
                    "user_01": {
                        "esp_01": {"trial_01": two_file},
                    },
                },
            },
        }

        magnitudes, _, stats = process_csv_files(
            data_files,
            max_workers=1,
            use_cache=False,
            return_stats=True,
            calibration_mode="rssi",
            min_rssi_dbm=-95.0,
        )

    mag_24 = magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_01"]["trial_01"]

    assert mag_24.shape == (1, 50)
    assert np.all(mag_24 < VISUALIZATION_FLOOR_MAGNITUDE)
    assert np.allclose(np.sum(mag_24**2, axis=1), 10 ** (LOW_VALID_RSSI_DBM / 10.0))
    assert stats.invalid_packets_removed == 1


def test_none_mode_keeps_zero_power_packets() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        two_file = root / "two.csv"
        _write_csv(
            two_file,
            [
                _two_ghz_row(-40.0, _csi_payload(TWO_GHZ_RAW_LENGTH, zero=True)),
                _two_ghz_row(-40.0, _csi_payload(TWO_GHZ_RAW_LENGTH)),
            ],
        )
        data_files = {
            "scenario_1": {
                "location_A-1": {
                    "user_01": {
                        "esp_01": {"trial_01": two_file},
                    },
                },
            },
        }

        raw_magnitudes, _ = process_csv_files(
            data_files,
            max_workers=1,
            use_cache=False,
            calibration_mode="none",
        )
        normalized_magnitudes, _ = process_csv_files(
            data_files,
            max_workers=1,
            use_cache=False,
            calibration_mode="packet_norm",
        )

    raw_mag = raw_magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_01"]["trial_01"]
    normalized_mag = normalized_magnitudes["scenario_1"]["location_A-1"]["user_01"]["esp_01"][
        "trial_01"
    ]

    assert raw_mag.shape == (2, 50)
    assert normalized_mag.shape == (1, 50)
    assert np.isfinite(raw_mag).all()
    assert np.isfinite(normalized_mag).all()


def main() -> None:
    test_rssi_calibration_preserves_shape_and_matches_packet_power()
    test_packet_norm_preserves_shape_and_unit_power()
    test_cache_identity_separates_calibration_modes()
    test_process_csv_files_rssi_filters_packets_and_keeps_agc_aligned()
    test_five_ghz_accepts_228_byte_packets_by_keeping_first_114_values()
    test_rssi_calibration_keeps_valid_packets_below_visualization_floor()
    test_none_mode_keeps_zero_power_packets()


if __name__ == "__main__":
    main()
