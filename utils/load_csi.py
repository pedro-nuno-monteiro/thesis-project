"""Load raw CSI CSV rows and assemble structured CSI magnitude maps."""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.cache import csi_cache_path, load_csi_cache, save_csi_cache
from utils.csi_processing import (
    VALID_CALIBRATION_MODES,
    CalibrationMode,
    calibration_mode_error,
    process_complex_csi,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, Path]]]]]
CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]

PROCESSOR_VERSION = "standard-v4"
FIVE_GHZ_MIN_ESP_ID = 11
FIVE_GHZ_MAX_ESP_ID = 20
TWO_GHZ_RAW_LENGTH = 128
FIVE_GHZ_RAW_LENGTH = 114
FIVE_GHZ_RAW_LENGTH_228 = 228
TWO_GHZ_CSI_COLUMN = 24
TWO_GHZ_RSSI_COLUMN = 3
FIVE_GHZ_CSI_COLUMN = 14
FIVE_GHZ_RSSI_COLUMN = 3

PROCESSING_DIAGNOSTIC_COLUMNS = [
    "scenario",
    "location",
    "user",
    "esp",
    "trial",
    "band",
    "file_path",
    "raw_rows",
    "no_match_count",
    "no_complete_count",
    "dropped_csi_length_count",
    "valid_csi_rows_before_filter",
    "invalid_packets_removed",
    "valid_magnitude_rows",
    "subcarriers",
    "calibration_mode",
    "calibration_applied",
    "from_cache",
]

PROCESSING_SUMMARY_COLUMNS = [
    "band",
    "files",
    "raw_rows_mean",
    "raw_rows_median",
    "valid_csi_rows_mean",
    "valid_magnitude_rows_mean",
    "valid_magnitude_rows_median",
    "no_match_total",
    "no_complete_total",
    "invalid_packets_removed_total",
    "mean_packet_retention",
]


@dataclass(frozen=True)
class _ProcessingJob:
    scenario_key: str
    location_key: str
    user_key: str
    esp_key: str
    trial_key: str
    file_path: str
    its5ghz: bool


@dataclass(frozen=True)
class CacheStats:
    """Summarize CSI cache use and optional calibration."""

    processed: int = 0
    cached: int = 0
    calibration_mode: CalibrationMode = "none"
    invalid_packets_removed: int = 0
    calibration_applied_files: int = 0

    @property
    def total(self) -> int:
        return self.processed + self.cached


@dataclass(frozen=True)
class _RunOptions:
    max_workers: int | None = None
    cache_dir: str | Path | None = None
    use_cache: bool = True
    force_reprocess: bool = False
    calibration_mode: CalibrationMode = "none"
    min_rssi_dbm: float = -95.0
    calibration_eps: float = 1e-12


@dataclass(frozen=True)
class _StandardFileResult:
    magnitude: np.ndarray
    no_match_count: int
    no_complete_count: int
    total_rows: int
    invalid_packets_removed: int
    calibration_applied: bool


@dataclass(frozen=True)
class _CachedResult:
    job: _ProcessingJob
    payload: _StandardFileResult
    from_cache: bool


def extract_csi_numbers(entry: object) -> list[float] | None:
    """Extract the numeric CSI payload stored inside square brackets."""
    match = re.search(r"\[(.*?)\]", str(entry))
    if not match:
        return None
    return [float(number) for number in re.findall(r"-?\d+", match.group(1))]


def read_csi_columns(job: _ProcessingJob) -> tuple[pd.Series, pd.Series, int]:
    """Read only the CSI and RSSI columns required for one ESP CSV file."""
    if job.its5ghz:
        file_csv = pd.read_csv(
            job.file_path,
            header=None,
            usecols=[FIVE_GHZ_RSSI_COLUMN, FIVE_GHZ_CSI_COLUMN],
        )
        return (
            file_csv[FIVE_GHZ_CSI_COLUMN],
            pd.to_numeric(file_csv[FIVE_GHZ_RSSI_COLUMN], errors="coerce"),
            FIVE_GHZ_RAW_LENGTH,
        )

    file_csv = pd.read_csv(
        job.file_path,
        header=None,
        usecols=[TWO_GHZ_RSSI_COLUMN, TWO_GHZ_CSI_COLUMN],
    )
    return (
        file_csv[TWO_GHZ_CSI_COLUMN],
        pd.to_numeric(file_csv[TWO_GHZ_RSSI_COLUMN], errors="coerce"),
        TWO_GHZ_RAW_LENGTH,
    )


def parse_valid_packet_rows(
    csi_raw: pd.Series,
    rssi_raw: pd.Series,
    expected_length: int,
    *,
    its5ghz: bool,
) -> tuple[list[list[float]], list[float], int, int]:
    """Parse complete CSI packets and retain their matching RSSI values."""
    valid_csi: list[list[float]] = []
    valid_rssi_dbm: list[float] = []
    no_match_count = 0
    no_complete_count = 0

    for row_index, entry in csi_raw.items():
        numbers = extract_csi_numbers(entry)
        if numbers is None:
            no_match_count += 1
            continue
        if its5ghz and len(numbers) == FIVE_GHZ_RAW_LENGTH_228:
            numbers = numbers[:expected_length]
        elif len(numbers) != expected_length:
            no_complete_count += 1
            continue
        valid_csi.append(numbers)
        valid_rssi_dbm.append(float(rssi_raw.loc[row_index]))

    return valid_csi, valid_rssi_dbm, no_match_count, no_complete_count


def iq_values_to_complex(csi_values: np.ndarray, *, its5ghz: bool) -> np.ndarray:
    """Convert interleaved I/Q values to the established complex CSI ordering."""
    if its5ghz:
        imaginary = csi_values[:, ::2]
        real = csi_values[:, 1::2]
    else:
        real = csi_values[:, 1::2]
        imaginary = csi_values[:, ::2]
    return real + 1j * imaginary


def processing_diagnostics_frame(
    results: list[_CachedResult],
    options: _RunOptions,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        file_result = result.payload
        valid_before_filter = int(
            file_result.total_rows - file_result.no_match_count - file_result.no_complete_count
        )
        rows.append(
            {
                "scenario": _normalized_key(result.job.scenario_key),
                "location": _normalized_key(result.job.location_key),
                "user": _normalized_key(result.job.user_key),
                "esp": result.job.esp_key,
                "trial": _normalized_key(result.job.trial_key),
                "band": _esp_band(result.job.esp_key),
                "file_path": result.job.file_path,
                "raw_rows": int(file_result.total_rows),
                "no_match_count": int(file_result.no_match_count),
                "no_complete_count": int(file_result.no_complete_count),
                "dropped_csi_length_count": int(file_result.no_complete_count),
                "valid_csi_rows_before_filter": max(valid_before_filter, 0),
                "invalid_packets_removed": int(file_result.invalid_packets_removed),
                "valid_magnitude_rows": int(file_result.magnitude.shape[0]),
                "subcarriers": (
                    int(file_result.magnitude.shape[1])
                    if file_result.magnitude.ndim >= 2
                    else 0
                ),
                "calibration_mode": options.calibration_mode,
                "calibration_applied": bool(file_result.calibration_applied),
                "from_cache": bool(result.from_cache),
            }
        )
    return pd.DataFrame(rows, columns=PROCESSING_DIAGNOSTIC_COLUMNS)


def summarize_processing_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize packet retention and parsing diagnostics by frequency band."""
    _validate_required_columns(diagnostics, set(PROCESSING_DIAGNOSTIC_COLUMNS))
    summary_rows: list[dict[str, object]] = []
    for band, band_diagnostics in diagnostics.groupby("band", sort=False):
        raw_rows_mean = float(band_diagnostics["raw_rows"].mean())
        summary_rows.append(
            {
                "band": band,
                "files": int(len(band_diagnostics)),
                "raw_rows_mean": raw_rows_mean,
                "raw_rows_median": float(band_diagnostics["raw_rows"].median()),
                "valid_csi_rows_mean": float(
                    band_diagnostics["valid_csi_rows_before_filter"].mean()
                ),
                "valid_magnitude_rows_mean": float(
                    band_diagnostics["valid_magnitude_rows"].mean()
                ),
                "valid_magnitude_rows_median": float(
                    band_diagnostics["valid_magnitude_rows"].median()
                ),
                "no_match_total": int(band_diagnostics["no_match_count"].sum()),
                "no_complete_total": int(band_diagnostics["no_complete_count"].sum()),
                "invalid_packets_removed_total": int(
                    band_diagnostics["invalid_packets_removed"].sum()
                ),
                "mean_packet_retention": (
                    float(band_diagnostics["valid_magnitude_rows"].mean() / raw_rows_mean)
                    if raw_rows_mean
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summary_rows, columns=PROCESSING_SUMMARY_COLUMNS)


def lowest_packet_count_files(
    diagnostics: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the files with the fewest retained CSI packets."""
    _validate_required_columns(diagnostics, set(PROCESSING_DIAGNOSTIC_COLUMNS))
    return (
        diagnostics.sort_values(
            ["valid_magnitude_rows", "raw_rows", "file_path"],
            ascending=[True, True, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def process_csv_files(  # noqa: PLR0913
    data_files: FileMap,
    *,
    max_workers: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_reprocess: bool = False,
    return_stats: bool = False,
    return_diagnostics: bool = False,
    calibration_mode: CalibrationMode = "none",
    min_rssi_dbm: float = -95.0,
    calibration_eps: float = 1e-12,
) -> (
    CsiMap
    | tuple[CsiMap, CacheStats]
    | tuple[CsiMap, pd.DataFrame]
    | tuple[CsiMap, CacheStats, pd.DataFrame]
):
    """Load all mapped CSV files and return their processed magnitude arrays."""
    if calibration_mode not in VALID_CALIBRATION_MODES:
        raise calibration_mode_error()
    if not np.isfinite(min_rssi_dbm):
        raise ValueError("min_rssi_dbm must be finite.")
    if not np.isfinite(calibration_eps) or calibration_eps <= 0:
        raise ValueError("calibration_eps must be a finite positive value.")

    magnitudes = _initialize_maps(data_files)
    jobs = list(_iter_jobs(data_files))
    options = _RunOptions(
        max_workers=max_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
        calibration_mode=calibration_mode,
        min_rssi_dbm=min_rssi_dbm,
        calibration_eps=calibration_eps,
    )
    results, cache_stats = _run_jobs(jobs, _process_standard_cached, options)

    invalid_packets_removed = 0
    calibration_applied_files = 0
    for result in results:
        _set_magnitude_entry(magnitudes, result.job, result.payload.magnitude)
        invalid_packets_removed += result.payload.invalid_packets_removed
        calibration_applied_files += int(result.payload.calibration_applied)

    diagnostics = processing_diagnostics_frame(results, options) if return_diagnostics else None
    stats = CacheStats(
        processed=cache_stats.processed,
        cached=cache_stats.cached,
        calibration_mode=calibration_mode,
        invalid_packets_removed=invalid_packets_removed,
        calibration_applied_files=calibration_applied_files,
    )

    if return_stats and return_diagnostics:
        return magnitudes, stats, diagnostics if diagnostics is not None else pd.DataFrame(
            columns=PROCESSING_DIAGNOSTIC_COLUMNS
        )
    if return_stats:
        return magnitudes, stats
    if return_diagnostics:
        return magnitudes, diagnostics if diagnostics is not None else pd.DataFrame(
            columns=PROCESSING_DIAGNOSTIC_COLUMNS
        )
    return magnitudes


def _process_standard_file(job: _ProcessingJob, options: _RunOptions) -> _StandardFileResult:
    csi_raw, rssi_raw, expected_length = read_csi_columns(job)
    valid_csi, valid_rssi, no_match_count, no_complete_count = parse_valid_packet_rows(
        csi_raw,
        rssi_raw,
        expected_length,
        its5ghz=job.its5ghz,
    )
    if not valid_csi:
        return _empty_standard_result(
            job,
            no_match_count=no_match_count,
            no_complete_count=no_complete_count,
            total_rows=len(csi_raw),
        )

    csi_values = np.asarray(valid_csi, dtype=float)
    rssi_dbm = np.asarray(valid_rssi, dtype=float)
    complex_csi = iq_values_to_complex(csi_values, its5ghz=job.its5ghz)
    magnitude, invalid_packets_removed, calibration_applied = process_complex_csi(
        complex_csi,
        rssi_dbm,
        its5ghz=job.its5ghz,
        calibration_mode=options.calibration_mode,
        min_rssi_dbm=options.min_rssi_dbm,
        calibration_eps=options.calibration_eps,
    )
    return _StandardFileResult(
        magnitude=magnitude,
        no_match_count=no_match_count,
        no_complete_count=no_complete_count,
        total_rows=len(csi_raw),
        invalid_packets_removed=invalid_packets_removed,
        calibration_applied=calibration_applied,
    )


def _processor_identity(options: _RunOptions) -> str:
    return (
        f"{PROCESSOR_VERSION}-calibration-{options.calibration_mode}"
        f"-min-rssi-{options.min_rssi_dbm:g}-eps-{options.calibration_eps:g}"
    )


def _process_standard_cached(job: _ProcessingJob, options: _RunOptions) -> _CachedResult:
    processor_version = _processor_identity(options)
    cache_file = csi_cache_path(job.file_path, processor_version, options.cache_dir)
    if options.use_cache and not options.force_reprocess and cache_file.exists():
        payload = load_csi_cache(cache_file)
        if isinstance(payload, _StandardFileResult):
            return _CachedResult(job=job, payload=payload, from_cache=True)

    payload = _process_standard_file(job, options)
    if options.use_cache:
        save_csi_cache(cache_file, payload)
    return _CachedResult(job=job, payload=payload, from_cache=False)


def _run_jobs(
    jobs: list[_ProcessingJob],
    worker: Callable[[_ProcessingJob, _RunOptions], _CachedResult],
    options: _RunOptions,
) -> tuple[list[_CachedResult], CacheStats]:
    if not jobs:
        return [], CacheStats(calibration_mode=options.calibration_mode)

    worker_count = (
        (os.cpu_count() or 1) if options.max_workers is None else options.max_workers
    )
    worker_count = max(1, min(worker_count, len(jobs)))
    if worker_count == 1:
        results = [worker(job, options) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(worker, jobs, [options] * len(jobs)))

    cached = sum(1 for result in results if result.from_cache)
    return results, CacheStats(
        processed=len(results) - cached,
        cached=cached,
        calibration_mode=options.calibration_mode,
    )


def _iter_jobs(data_files: FileMap) -> Iterable[_ProcessingJob]:
    for scenario_key, locations_map in data_files.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                for esp_key, trials_map in esps_map.items():
                    esp_id = int(esp_key.removeprefix("esp_"))
                    its5ghz = FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID
                    for trial_key, file_path in trials_map.items():
                        if file_path is not None:
                            yield _ProcessingJob(
                                scenario_key=scenario_key,
                                location_key=location_key,
                                user_key=user_key,
                                esp_key=esp_key,
                                trial_key=trial_key,
                                file_path=str(file_path),
                                its5ghz=its5ghz,
                            )


def _initialize_maps(data_files: FileMap) -> CsiMap:
    magnitudes: CsiMap = {}
    for scenario_key, locations_map in data_files.items():
        magnitudes[scenario_key] = {}
        for location_key, users_map in locations_map.items():
            magnitudes[scenario_key][location_key] = {}
            for user_key, esps_map in users_map.items():
                magnitudes[scenario_key][location_key][user_key] = {}
                for esp_key in esps_map:
                    magnitudes[scenario_key][location_key][user_key][esp_key] = {}
    return magnitudes


def _set_magnitude_entry(
    magnitudes: CsiMap,
    job: _ProcessingJob,
    magnitude: np.ndarray,
) -> None:
    magnitudes[job.scenario_key][job.location_key][job.user_key][job.esp_key][
        job.trial_key
    ] = magnitude


def _empty_standard_result(
    job: _ProcessingJob,
    *,
    no_match_count: int,
    no_complete_count: int,
    total_rows: int,
) -> _StandardFileResult:
    subcarrier_count = 56 if job.its5ghz else 50
    return _StandardFileResult(
        magnitude=np.empty((0, subcarrier_count), dtype=float),
        no_match_count=no_match_count,
        no_complete_count=no_complete_count,
        total_rows=total_rows,
        invalid_packets_removed=0,
        calibration_applied=False,
    )


def _normalized_key(value: str) -> str:
    normalized = value
    for prefix in ("scenario_", "location_", "user_", "trial_"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _esp_band(esp_key: str) -> str:
    esp_id = int(esp_key.removeprefix("esp_"))
    return "5 GHz" if FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID else "2.4 GHz"


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
