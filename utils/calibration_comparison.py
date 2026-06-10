from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .csi_preprocessing import process_magnitude_data
from .feature_pipeline import build_frequency_feature_dataframes
from .thesis_csv_processing import CacheStats, CalibrationMode, process_csv_files

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from .thesis_csv_processing import AgcGainMap, CsiMap, FileMap

DEFAULT_CALIBRATION_MODES: tuple[CalibrationMode, ...] = ("none", "packet_norm", "rssi")


@dataclass(frozen=True)
class CalibrationComparisonResult:
    """Artifacts produced for one CSI calibration mode."""

    magnitudes: CsiMap
    agc_gain_data: AgcGainMap
    cache_stats: CacheStats
    processed_magnitudes: CsiMap
    preprocessing_summary: pd.DataFrame
    features_24ghz: pd.DataFrame
    features_5ghz: pd.DataFrame
    features_fusion: pd.DataFrame


def build_calibration_comparison(  # noqa: PLR0913
    data_files: FileMap,
    *,
    calibration_modes: tuple[CalibrationMode, ...] = DEFAULT_CALIBRATION_MODES,
    max_workers: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_reprocess: bool = True,
    min_rssi_dbm: float = -95.0,
    calibration_eps: float = 1e-12,
    window_size: int = 60,
    overlap_size: int = 30,
    require_all_esps: bool = True,
) -> dict[CalibrationMode, CalibrationComparisonResult]:
    results: dict[CalibrationMode, CalibrationComparisonResult] = {}

    for mode in calibration_modes:
        magnitudes, agc_gain_data, cache_stats = process_csv_files(
            data_files,
            max_workers=max_workers,
            cache_dir=cache_dir,
            use_cache=use_cache,
            force_reprocess=force_reprocess,
            return_stats=True,
            calibration_mode=mode,
            min_rssi_dbm=min_rssi_dbm,
            calibration_eps=calibration_eps,
        )
        processed_magnitudes, preprocessing_summary = process_magnitude_data(
            magnitudes,
            agc_gain_data,
            apply_agc_compensation=False,
            filter_method="none",
            normalization="none",
        )
        features_24ghz, features_5ghz, features_fusion = build_frequency_feature_dataframes(
            processed_magnitudes,
            window_size=window_size,
            overlap_size=overlap_size,
            calibrate=False,
            require_all_esps=require_all_esps,
        )
        results[mode] = CalibrationComparisonResult(
            magnitudes=magnitudes,
            agc_gain_data=agc_gain_data,
            cache_stats=cache_stats,
            processed_magnitudes=processed_magnitudes,
            preprocessing_summary=preprocessing_summary,
            features_24ghz=features_24ghz,
            features_5ghz=features_5ghz,
            features_fusion=features_fusion,
        )

    return results
