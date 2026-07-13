from __future__ import annotations

import numpy as np
import pytest

from utils.csi_preprocessing import normalize_magnitude, process_magnitude_data


def test_packet_minmax_normalizes_each_row() -> None:
    magnitude = np.array([[2.0, 4.0, 6.0], [10.0, 20.0, 40.0]], dtype=np.float32)

    normalized = normalize_magnitude(magnitude, "packet_minmax")

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized.min(axis=1), [0.0, 0.0])
    np.testing.assert_allclose(normalized.max(axis=1), [1.0, 1.0])


def test_minmax_normalizes_each_column() -> None:
    magnitude = np.array([[2.0, 4.0, 6.0], [10.0, 20.0, 40.0]], dtype=np.float32)

    normalized = normalize_magnitude(magnitude, "minmax")

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(normalized.max(axis=0), [1.0, 1.0, 1.0])


def test_empty_baseline_uses_supplied_arrays_elementwise() -> None:
    magnitude = np.array([[2.0, 5.0], [4.0, 9.0]], dtype=np.float32)
    baseline_mean = np.array([1.0, 3.0], dtype=np.float32)
    baseline_absmax = np.array([2.0, 4.0], dtype=np.float32)
    epsilon = 1e-8

    normalized = normalize_magnitude(
        magnitude,
        "empty_baseline",
        baseline_mean=baseline_mean,
        baseline_absmax=baseline_absmax,
        epsilon=epsilon,
    )

    expected = (magnitude - baseline_mean[None, :]) / (baseline_absmax[None, :] + epsilon)
    np.testing.assert_allclose(normalized, expected)


@pytest.mark.parametrize(
    ("baseline_mean", "baseline_absmax"),
    [
        (None, np.array([1.0, 2.0], dtype=np.float32)),
        (np.array([1.0, 2.0], dtype=np.float32), None),
    ],
)
def test_empty_baseline_requires_both_baseline_arrays(
    baseline_mean: np.ndarray | None,
    baseline_absmax: np.ndarray | None,
) -> None:
    with pytest.raises(ValueError, match="requires baseline_mean and baseline_absmax"):
        normalize_magnitude(
            np.ones((2, 2), dtype=np.float32),
            "empty_baseline",
            baseline_mean=baseline_mean,
            baseline_absmax=baseline_absmax,
        )


def test_empty_baseline_rejects_mismatched_subcarrier_count() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        normalize_magnitude(
            np.ones((2, 3), dtype=np.float32),
            "empty_baseline",
            baseline_mean=np.ones(2, dtype=np.float32),
            baseline_absmax=np.ones(2, dtype=np.float32),
        )


def test_unknown_normalization_lists_valid_methods() -> None:
    with pytest.raises(ValueError, match="none, zscore, minmax, packet_minmax, empty_baseline"):
        normalize_magnitude(np.ones((2, 2), dtype=np.float32), "missing")


def test_empty_baseline_per_user_names_missing_pair() -> None:
    raw = {
        "scenario_1": {
            "location_A-1": {
                "user_01": {
                    "esp_01": {
                        "trial_01": np.ones((2, 2), dtype=np.float32),
                    },
                },
            },
        },
    }

    with pytest.raises(ValueError, match=r"user=01, esp=01"):
        process_magnitude_data(raw, normalization="empty_baseline", baseline_scope="per_user")
