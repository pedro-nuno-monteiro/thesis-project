from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, Path]]]]]
CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
CalibrationMode = Literal["none", "packet_norm", "rssi"]

REPO_CACHE_DIR = Path(".cache") / "csi_processing"
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
VALID_CALIBRATION_MODES: tuple[CalibrationMode, ...] = ("none", "packet_norm", "rssi")

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
    """Summarizes cache usage and optional complex-CSI calibration."""

    processed: int = 0
    cached: int = 0
    calibration_mode: CalibrationMode = "none"
    invalid_packets_removed: int = 0
    calibration_applied_files: int = 0

    @property
    def total(self) -> int:
        """Total files handled by the processing call."""
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
    payload: object
    from_cache: bool


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is None:
        return REPO_CACHE_DIR
    return Path(cache_dir)


def _cache_key(file_path: str, processor_version: str) -> str:
    path = Path(file_path)
    stat = path.stat()
    identity = "|".join(
        [
            processor_version,
            os.path.normcase(str(path.resolve())),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ],
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cache_path(file_path: str, processor_version: str, cache_dir: str | Path | None) -> Path:
    root = _cache_root(cache_dir) / processor_version
    return root / f"{_cache_key(file_path, processor_version)}.pkl"


def _processor_identity(options: _RunOptions) -> str:
    return (
        f"{PROCESSOR_VERSION}-calibration-{options.calibration_mode}"
        f"-min-rssi-{options.min_rssi_dbm:g}"
        f"-eps-{options.calibration_eps:g}"
    )


def _read_cache(cache_file: Path) -> object | None:
    try:
        with cache_file.open("rb") as file:
            return pickle.load(file)  # noqa: S301
    except (EOFError, OSError, pickle.PickleError):
        return None


def _write_cache(cache_file: Path, payload: object) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{cache_file.stem}.",
        suffix=".tmp",
        dir=cache_file.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_file)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _process_cached(
    job: _ProcessingJob,
    processor: Callable[[_ProcessingJob, _RunOptions], object],
    processor_version: str,
    options: _RunOptions,
) -> _CachedResult:
    cache_file = _cache_path(job.file_path, processor_version, options.cache_dir)

    if options.use_cache and not options.force_reprocess and cache_file.exists():
        payload = _read_cache(cache_file)
        if payload is not None:
            return _CachedResult(job=job, payload=payload, from_cache=True)

    payload = processor(job, options)
    if options.use_cache:
        _write_cache(cache_file, payload)

    return _CachedResult(job=job, payload=payload, from_cache=False)


def _process_standard_cached(
    job: _ProcessingJob,
    options: _RunOptions,
) -> _CachedResult:
    return _process_cached(job, _process_standard_file, _processor_identity(options), options)


def _run_jobs(
    jobs: list[_ProcessingJob],
    worker: Callable[[_ProcessingJob, _RunOptions], _CachedResult],
    options: _RunOptions,
) -> tuple[list[_CachedResult], CacheStats]:
    if not jobs:
        return [], CacheStats(calibration_mode=options.calibration_mode)

    worker_count = _cpu_count() if options.max_workers is None else options.max_workers
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


def _normalized_key(value: str) -> str:
    normalized = value
    for prefix in ("scenario_", "location_", "user_", "trial_"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _esp_band(esp_key: str) -> str:
    esp_id = int(esp_key.removeprefix("esp_"))
    return "5 GHz" if FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID else "2.4 GHz"


def processing_diagnostics_frame(
    results: list[_CachedResult],
    options: _RunOptions,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for result in results:
        file_result = result.payload
        valid_csi_rows_before_filter = int(
            file_result.total_rows - file_result.no_match_count - file_result.no_complete_count,
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
                "valid_csi_rows_before_filter": max(valid_csi_rows_before_filter, 0),
                "invalid_packets_removed": int(file_result.invalid_packets_removed),
                "valid_magnitude_rows": int(file_result.magnitude.shape[0]),
                "subcarriers": int(file_result.magnitude.shape[1]) if file_result.magnitude.ndim >= 2 else 0,
                "calibration_mode": options.calibration_mode,
                "calibration_applied": bool(file_result.calibration_applied),
                "from_cache": bool(result.from_cache),
            },
        )

    return pd.DataFrame(rows, columns=PROCESSING_DIAGNOSTIC_COLUMNS)


def summarize_processing_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
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
                "valid_csi_rows_mean": float(band_diagnostics["valid_csi_rows_before_filter"].mean()),
                "valid_magnitude_rows_mean": float(band_diagnostics["valid_magnitude_rows"].mean()),
                "valid_magnitude_rows_median": float(band_diagnostics["valid_magnitude_rows"].median()),
                "no_match_total": int(band_diagnostics["no_match_count"].sum()),
                "no_complete_total": int(band_diagnostics["no_complete_count"].sum()),
                "invalid_packets_removed_total": int(band_diagnostics["invalid_packets_removed"].sum()),
                "mean_packet_retention": (
                    float(band_diagnostics["valid_magnitude_rows"].mean() / raw_rows_mean)
                    if raw_rows_mean
                    else np.nan
                ),
            },
        )

    return pd.DataFrame(summary_rows, columns=PROCESSING_SUMMARY_COLUMNS)


def lowest_packet_count_files(
    diagnostics: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    _validate_required_columns(diagnostics, set(PROCESSING_DIAGNOSTIC_COLUMNS))
    return (
        diagnostics.sort_values(
            ["valid_magnitude_rows", "raw_rows", "file_path"],
            ascending=[True, True, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def _iter_jobs(data_files: FileMap) -> Iterable[_ProcessingJob]:
    for scenario_key, locations_map in data_files.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                for esp_key, trials_map in esps_map.items():
                    esp_id = int(esp_key.removeprefix("esp_"))
                    its5ghz = FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID

                    for trial_key, file_path in trials_map.items():
                        if file_path is None:
                            continue

                        yield _ProcessingJob(
                            scenario_key=scenario_key,
                            location_key=location_key,
                            user_key=user_key,
                            esp_key=esp_key,
                            trial_key=trial_key,
                            file_path=str(file_path),
                            its5ghz=its5ghz,
                        )


def _set_magnitude_entry(magnitudes: CsiMap, job: _ProcessingJob, magnitude: np.ndarray) -> None:
    magnitudes.setdefault(job.scenario_key, {})
    magnitudes[job.scenario_key].setdefault(job.location_key, {})
    magnitudes[job.scenario_key][job.location_key].setdefault(job.user_key, {})
    magnitudes[job.scenario_key][job.location_key][job.user_key].setdefault(job.esp_key, {})
    magnitudes[job.scenario_key][job.location_key][job.user_key][job.esp_key][
        job.trial_key
    ] = magnitude


def _empty_24ghz_magnitude() -> np.ndarray:
    return np.empty((0, 50), dtype=float)


def _empty_5ghz_magnitude() -> np.ndarray:
    return np.empty((0, 56), dtype=float)


def _extract_csi_numbers(entry: object) -> list[float] | None:
    match = re.search(r"\[(.*?)\]", str(entry))
    if not match:
        return None
    return [float(number) for number in re.findall(r"-?\d+", match.group(1))]


def _calibration_mode_error() -> ValueError:
    msg = "calibration_mode must be one of: 'none', 'packet_norm', 'rssi'"
    return ValueError(msg)


def _complex_csi_power(complex_csi: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(complex_csi) ** 2, axis=1)


def _normalize_complex_csi_per_packet(
    complex_csi: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    norm = np.sqrt(_complex_csi_power(complex_csi) + eps)
    return complex_csi / norm[:, None]


def _calibrate_complex_csi_with_rssi(
    complex_csi: np.ndarray,
    rssi_dbm: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    if complex_csi.shape[0] != rssi_dbm.shape[0]:
        msg = (
            "RSSI calibration requires one RSSI value per CSI packet: "
            f"got {complex_csi.shape[0]} CSI packets and {rssi_dbm.shape[0]} RSSI values."
        )
        raise ValueError(msg)

    rssi_mw = 10 ** (rssi_dbm / 10.0)
    csi_power = _complex_csi_power(complex_csi)
    scale = np.sqrt(rssi_mw / (csi_power + eps))
    return complex_csi * scale[:, None]


def _apply_complex_csi_calibration(
    complex_csi: np.ndarray,
    *,
    calibration_mode: CalibrationMode,
    rssi_dbm: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    if calibration_mode == "none":
        return complex_csi, False
    if calibration_mode == "packet_norm":
        return _normalize_complex_csi_per_packet(complex_csi, eps=eps), True
    if calibration_mode == "rssi":
        if rssi_dbm is None:
            msg = "RSSI values are required when calibration_mode='rssi'."
            raise ValueError(msg)
        return _calibrate_complex_csi_with_rssi(complex_csi, rssi_dbm, eps=eps), True

    raise _calibration_mode_error()


# this function encapsulates the packet filtering logic based on the selected calibration mode
# and thresholds. It is called during file processing so only valid packets are included in
# the final magnitude calculations and feature extraction steps.
def _valid_packet_mask(
    complex_csi: np.ndarray,
    *,
    rssi_dbm: np.ndarray | None,
    calibration_mode: CalibrationMode,
    min_rssi_dbm: float,
    eps: float,
) -> np.ndarray:
    """Return packets suitable for calibration without applying plotting dB thresholds."""
    if calibration_mode not in VALID_CALIBRATION_MODES:
        raise _calibration_mode_error()
    if calibration_mode == "rssi" and rssi_dbm is None:
        msg = "RSSI values are required when calibration_mode='rssi'."
        raise ValueError(msg)
    if rssi_dbm is not None and complex_csi.shape[0] != rssi_dbm.shape[0]:
        msg = (
            "Packet filtering requires one RSSI value per CSI packet: "
            f"got {complex_csi.shape[0]} CSI packets and {rssi_dbm.shape[0]} RSSI values."
        )
        raise ValueError(msg)

    csi_power = _complex_csi_power(complex_csi)
    mask = np.isfinite(complex_csi).all(axis=1)
    mask &= np.isfinite(csi_power)
    mask &= csi_power > eps

    if calibration_mode == "rssi" and rssi_dbm is not None:
        mask &= np.isfinite(rssi_dbm)
        mask &= rssi_dbm >= min_rssi_dbm

    return mask


def _read_standard_columns(job: _ProcessingJob) -> tuple[pd.Series, pd.Series, int]:
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


def _parse_valid_packet_rows(
    csi_raw: pd.Series,
    rssi_raw: pd.Series,
    expected_length: int,
    *,
    its5ghz: bool,
) -> tuple[list[list[float]], list[float], int, int]:
    valid_csi: list[list[float]] = []
    valid_rssi_dbm: list[float] = []
    no_match_count = 0
    no_complete_count = 0

    for row_index, entry in csi_raw.items():
        nums = _extract_csi_numbers(entry)
        if nums is None:
            no_match_count += 1
            continue

        if its5ghz and len(nums) == FIVE_GHZ_RAW_LENGTH_228:
            nums = nums[:expected_length]
        elif len(nums) != expected_length:
            no_complete_count += 1
            continue

        valid_csi.append(nums)
        valid_rssi_dbm.append(float(rssi_raw.loc[row_index]))

    return valid_csi, valid_rssi_dbm, no_match_count, no_complete_count


def _empty_standard_result(
    job: _ProcessingJob,
    *,
    no_match_count: int,
    no_complete_count: int,
    total_rows: int,
) -> _StandardFileResult:
    mag = _empty_5ghz_magnitude() if job.its5ghz else _empty_24ghz_magnitude()
    return _StandardFileResult(
        magnitude=mag,
        no_match_count=no_match_count,
        no_complete_count=no_complete_count,
        total_rows=total_rows,
        invalid_packets_removed=0,
        calibration_applied=False,
    )


def _select_active_subcarrier_csi(csi_values: np.ndarray, *, its5ghz: bool) -> np.ndarray:
    if its5ghz:
        imag = csi_values[:, ::2]
        real = csi_values[:, 1::2]
        complex_csi = real + 1j * imag
        return np.delete(complex_csi, [28], axis=1)

    real = csi_values[:, 1::2]
    imag = csi_values[:, ::2]
    complex_csi = real + 1j * imag
    fft_csi = np.fft.fftshift(complex_csi, axes=1)
    active_sc = fft_csi[:, 6:58]
    return np.delete(active_sc, [26, 27], axis=1)


def _filter_packets_for_calibration(
    selected_csi: np.ndarray,
    rssi_dbm: np.ndarray,
    options: _RunOptions,
) -> tuple[np.ndarray, np.ndarray, int]:
    if options.calibration_mode == "none":
        return selected_csi, rssi_dbm, 0

    # valid packet mask based on calibration mode and thresholds
    valid_packet_mask = _valid_packet_mask(
        selected_csi,
        rssi_dbm=rssi_dbm,
        calibration_mode=options.calibration_mode,
        min_rssi_dbm=options.min_rssi_dbm,
        eps=options.calibration_eps,
    )
    invalid_packets_removed = int(valid_packet_mask.size - np.count_nonzero(valid_packet_mask))
    return (
        selected_csi[valid_packet_mask],
        rssi_dbm[valid_packet_mask],
        invalid_packets_removed,
    )


def _process_standard_file(
    job: _ProcessingJob,
    options: _RunOptions,
) -> _StandardFileResult:

    # recolhe valores do ficheiro CSV
    csi_raw, rssi_raw, expected_length = _read_standard_columns(job)

    # verifica len de cada vetor CSI
    valid_csi, valid_rssi_dbm, no_match_count, no_complete_count = (
        _parse_valid_packet_rows(
            csi_raw,
            rssi_raw,
            expected_length,
            its5ghz=job.its5ghz,
        )
    )

    if not valid_csi:
        return _empty_standard_result(
            job,
            no_match_count=no_match_count,
            no_complete_count=no_complete_count,
            total_rows=len(csi_raw),
        )

    csi_values = np.array(valid_csi, dtype=float)
    rssi_dbm = np.array(valid_rssi_dbm, dtype=float)

    # escolha de subportadoras
    selected_csi = _select_active_subcarrier_csi(csi_values, its5ghz=job.its5ghz)

    # filtrar pacotes inválidos com base no modo de calibração
    selected_csi, rssi_dbm, invalid_packets_removed = _filter_packets_for_calibration(
        selected_csi,
        rssi_dbm,
        options,
    )

    # aplicar calibração escolhida
    selected_csi, calibration_applied = _apply_complex_csi_calibration(
        selected_csi,
        calibration_mode=options.calibration_mode,
        rssi_dbm=rssi_dbm if options.calibration_mode == "rssi" else None,
        eps=options.calibration_eps,
    )

    # calcular magnitude dos CSI pós filter e calibrate
    mag = np.abs(selected_csi)

    return _StandardFileResult(
        magnitude=mag,
        no_match_count=no_match_count,
        no_complete_count=no_complete_count,
        total_rows=len(csi_raw),
        invalid_packets_removed=invalid_packets_removed,
        calibration_applied=calibration_applied and mag.shape[0] > 0,
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


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        msg = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)


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
    if calibration_mode not in VALID_CALIBRATION_MODES:
        raise _calibration_mode_error()
    if not np.isfinite(min_rssi_dbm):
        msg = "min_rssi_dbm must be finite."
        raise ValueError(msg)
    if not np.isfinite(calibration_eps) or calibration_eps <= 0:
        msg = "calibration_eps must be a finite positive value."
        raise ValueError(msg)

    # para não ter aquele for gigante a passar todo o filename
    magnitudes = _initialize_maps(data_files)

    # processing parallel
    jobs = list(_iter_jobs(data_files))

    # process jobs with caching and collect results
    options = _RunOptions(
        max_workers=max_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
        calibration_mode=calibration_mode,
        min_rssi_dbm=min_rssi_dbm,
        calibration_eps=calibration_eps,
    )

    # O processamento paralelo é feito aqui, e os resultados são coletados para atualização
    results, cache_stats = _run_jobs(jobs, _process_standard_cached, options)
    invalid_packets_removed = 0
    calibration_applied_files = 0

    for result in results:
        file_result = result.payload

        # Atualiza o mapa de magnitudes com os resultados processados, independentemente de terem
        # vindo da cache ou não.
        _set_magnitude_entry(magnitudes, result.job, file_result.magnitude)
        invalid_packets_removed += file_result.invalid_packets_removed
        calibration_applied_files += int(file_result.calibration_applied)
    diagnostics = processing_diagnostics_frame(results, options) if return_diagnostics else None

    # guarda em cache
    stats = CacheStats(
        processed=cache_stats.processed,
        cached=cache_stats.cached,
        calibration_mode=calibration_mode,
        invalid_packets_removed=invalid_packets_removed,
        calibration_applied_files=calibration_applied_files,
    )

    if return_stats and return_diagnostics:
        return magnitudes, stats, diagnostics if diagnostics is not None else pd.DataFrame(
            columns=PROCESSING_DIAGNOSTIC_COLUMNS,
        )
    if return_stats:
        return magnitudes, stats
    if return_diagnostics:
        return magnitudes, diagnostics if diagnostics is not None else pd.DataFrame(
            columns=PROCESSING_DIAGNOSTIC_COLUMNS,
        )
    return magnitudes
