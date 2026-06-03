from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, Path]]]]]
CsiMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]
AgcGainMap = dict[str, dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]]

REPO_CACHE_DIR = Path(".cache") / "csi_processing"
STANDARD_PROCESSOR_VERSION = "standard-v1"
FIVE_GHZ_228_PROCESSOR_VERSION = "five-ghz-228-v1"
FIVE_GHZ_MIN_ESP_ID = 11
FIVE_GHZ_MAX_ESP_ID = 20


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
    """Counts how many files came from cache versus fresh processing."""

    processed: int = 0
    cached: int = 0

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
    processor: Callable[[_ProcessingJob], object],
    processor_version: str,
    options: _RunOptions,
) -> _CachedResult:
    cache_file = _cache_path(job.file_path, processor_version, options.cache_dir)

    if options.use_cache and not options.force_reprocess and cache_file.exists():
        payload = _read_cache(cache_file)
        if payload is not None:
            return _CachedResult(job=job, payload=payload, from_cache=True)

    payload = processor(job)
    if options.use_cache:
        _write_cache(cache_file, payload)

    return _CachedResult(job=job, payload=payload, from_cache=False)


def _process_standard_cached(
    job: _ProcessingJob,
    options: _RunOptions,
) -> _CachedResult:
    return _process_cached(
        job,
        _process_standard_file,
        STANDARD_PROCESSOR_VERSION,
        options,
    )


def _process_228_cached(
    job: _ProcessingJob,
    options: _RunOptions,
) -> _CachedResult:
    return _process_cached(
        job,
        _process_228_file,
        FIVE_GHZ_228_PROCESSOR_VERSION,
        options,
    )


def _run_jobs(
    jobs: list[_ProcessingJob],
    worker: Callable[[_ProcessingJob, _RunOptions], _CachedResult],
    options: _RunOptions,
) -> tuple[list[_CachedResult], CacheStats]:
    if not jobs:
        return [], CacheStats()

    worker_count = _cpu_count() if options.max_workers is None else options.max_workers
    worker_count = max(1, min(worker_count, len(jobs)))

    if worker_count == 1:
        results = [
            worker(job, options)
            for job in jobs
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    worker,
                    jobs,
                    [options] * len(jobs),
                ),
            )

    cached = sum(1 for result in results if result.from_cache)
    return results, CacheStats(processed=len(results) - cached, cached=cached)


def _iter_jobs(data_files: FileMap, *, only_5ghz: bool = False) -> Iterable[_ProcessingJob]:
    for scenario_key, locations_map in data_files.items():
        for location_key, users_map in locations_map.items():
            for user_key, esps_map in users_map.items():
                for esp_key, trials_map in esps_map.items():
                    esp_id = int(esp_key.removeprefix("esp_"))
                    its5ghz = FIVE_GHZ_MIN_ESP_ID <= esp_id <= FIVE_GHZ_MAX_ESP_ID
                    if only_5ghz and not its5ghz:
                        continue

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


def _process_standard_file(
    job: _ProcessingJob,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    csi_raw: pd.Series
    agc_raw: pd.Series | None = None
    valid_agc_gains: list[int] = []

    if job.its5ghz:
        file_csv = pd.read_csv(job.file_path, header=None, usecols=[7, 14])
        csi_raw = file_csv.iloc[:, 1]
        agc_raw = file_csv.iloc[:, 0]
        expected_length = 114
    else:
        file_csv = pd.read_csv(job.file_path, header=None, usecols=[24])
        csi_raw = file_csv.iloc[:, 0]
        expected_length = 128

    valid_csi: list[list[float]] = []
    no_match_count = 0
    no_complete_count = 0

    for row_index, entry in csi_raw.items():
        nums = _extract_csi_numbers(entry)
        if nums is None:
            no_match_count += 1
            continue

        if len(nums) == expected_length:
            valid_csi.append(nums)
            if agc_raw is not None:
                valid_agc_gains.append(int(agc_raw.loc[row_index]))
        else:
            no_complete_count += 1

    if not valid_csi:
        mag = _empty_5ghz_magnitude() if job.its5ghz else _empty_24ghz_magnitude()
        agc_gains = np.array(valid_agc_gains, dtype=int)
        return mag, agc_gains, no_match_count, no_complete_count, len(csi_raw)

    csi_values = np.array(valid_csi, dtype=float)
    agc_gains = np.array(valid_agc_gains, dtype=int)

    if not job.its5ghz:
        real = csi_values[:, 1::2]
        imag = csi_values[:, ::2]
        complex_csi = real + 1j * imag
        fft_csi = np.fft.fftshift(complex_csi, axes=1)
        active_sc = fft_csi[:, 6:58]
        active_sc = np.delete(active_sc, [26, 27], axis=1)
        mag = np.abs(active_sc)
    else:
        imag = csi_values[:, ::2]
        real = csi_values[:, 1::2]
        complex_csi = real + 1j * imag
        mag = np.abs(complex_csi)
        mag = np.delete(mag, [28], axis=1)

    return mag, agc_gains, no_match_count, no_complete_count, len(csi_raw)


def _initialize_standard_maps(data_files: FileMap) -> tuple[CsiMap, AgcGainMap]:
    magnitudes: CsiMap = {}
    agc_gain_data: AgcGainMap = {}

    for scenario_key, locations_map in data_files.items():
        magnitudes[scenario_key] = {}
        agc_gain_data[scenario_key] = {}

        for location_key, users_map in locations_map.items():
            magnitudes[scenario_key][location_key] = {}
            agc_gain_data[scenario_key][location_key] = {}

            for user_key, esps_map in users_map.items():
                magnitudes[scenario_key][location_key][user_key] = {}
                agc_gain_data[scenario_key][location_key][user_key] = {}

                for esp_key in esps_map:
                    magnitudes[scenario_key][location_key][user_key][esp_key] = {}

    return magnitudes, agc_gain_data


def process_standard_files(  # noqa: PLR0913
    data_files: FileMap,
    *,
    max_workers: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_reprocess: bool = False,
    return_stats: bool = False,
) -> tuple[CsiMap, AgcGainMap] | tuple[CsiMap, AgcGainMap, CacheStats]:
    magnitudes, agc_gain_data = _initialize_standard_maps(data_files)
    jobs = list(_iter_jobs(data_files))
    options = _RunOptions(
        max_workers=max_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
    )
    results, stats = _run_jobs(
        jobs,
        _process_standard_cached,
        options,
    )

    for result in results:
        mag, agc_gains, _no_match, _no_complete, _total = result.payload
        _set_magnitude_entry(magnitudes, result.job, mag)

        if result.job.its5ghz:
            agc_gain_data[result.job.scenario_key][result.job.location_key][
                result.job.user_key
            ].setdefault(result.job.esp_key, {})
            agc_gain_data[result.job.scenario_key][result.job.location_key][result.job.user_key][
                result.job.esp_key
            ][result.job.trial_key] = agc_gains

    if return_stats:
        return magnitudes, agc_gain_data, stats
    return magnitudes, agc_gain_data


FIVE_GHZ_RAW_LENGTH_114 = 114
FIVE_GHZ_RAW_LENGTH_228 = 228
FIVE_GHZ_ZERO_SUBCARRIER_INDEXES = {
    FIVE_GHZ_RAW_LENGTH_114: [28],
    FIVE_GHZ_RAW_LENGTH_228: [28, 85],
}
FIVE_GHZ_228_SPIKE_THRESHOLD = 50.0


def empty_magnitude(raw_length: int) -> np.ndarray:
    subcarrier_count = raw_length // 2
    removed_subcarriers = len(FIVE_GHZ_ZERO_SUBCARRIER_INDEXES[raw_length])
    return np.empty((0, subcarrier_count - removed_subcarriers), dtype=float)


def csi_values_to_magnitude(csi_values: np.ndarray, raw_length: int) -> np.ndarray:
    if csi_values.size == 0:
        return empty_magnitude(raw_length)

    imag = csi_values[:, ::2]
    real = csi_values[:, 1::2]
    complex_csi = real + 1j * imag
    mag = np.abs(complex_csi)

    return np.delete(mag, FIVE_GHZ_ZERO_SUBCARRIER_INDEXES[raw_length], axis=1)


def remove_228_spike_subcarriers(
    mag_228: np.ndarray,
    mag_228_first_half: np.ndarray,
    mag_228_second_half: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mag_228.shape[0] == 0:
        return mag_228, mag_228_first_half, mag_228_second_half

    spike_subcarriers = np.flatnonzero(
        mag_228.mean(axis=0) > FIVE_GHZ_228_SPIKE_THRESHOLD,
    )
    if spike_subcarriers.size == 0:
        return mag_228, mag_228_first_half, mag_228_second_half

    first_half_width = mag_228_first_half.shape[1]
    first_half_spikes = spike_subcarriers[spike_subcarriers < first_half_width]
    second_half_spikes = spike_subcarriers[spike_subcarriers >= first_half_width] - first_half_width

    mag_228 = np.delete(mag_228, spike_subcarriers, axis=1)
    mag_228_first_half = np.delete(mag_228_first_half, first_half_spikes, axis=1)
    mag_228_second_half = np.delete(mag_228_second_half, second_half_spikes, axis=1)

    return mag_228, mag_228_first_half, mag_228_second_half


def remove_shared_spike_subcarriers(
    first_magnitude: np.ndarray,
    second_magnitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if first_magnitude.shape[0] == 0 and second_magnitude.shape[0] == 0:
        return first_magnitude, second_magnitude

    first_spikes = (
        np.flatnonzero(first_magnitude.mean(axis=0) > FIVE_GHZ_228_SPIKE_THRESHOLD)
        if first_magnitude.shape[0]
        else np.array([], dtype=int)
    )
    second_spikes = (
        np.flatnonzero(second_magnitude.mean(axis=0) > FIVE_GHZ_228_SPIKE_THRESHOLD)
        if second_magnitude.shape[0]
        else np.array([], dtype=int)
    )
    spike_subcarriers = np.union1d(first_spikes, second_spikes).astype(int)

    if spike_subcarriers.size == 0:
        return first_magnitude, second_magnitude

    return (
        np.delete(first_magnitude, spike_subcarriers, axis=1),
        np.delete(second_magnitude, spike_subcarriers, axis=1),
    )


def split_228_interleaved_parts(csi_228: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if csi_228.size == 0:
        empty_csi = np.empty((0, FIVE_GHZ_RAW_LENGTH_114), dtype=float)
        return empty_csi, empty_csi.copy()

    grouped_csi = csi_228.reshape(csi_228.shape[0], -1, 4)
    first_part = grouped_csi[:, :, :2].reshape(csi_228.shape[0], -1)
    second_part = grouped_csi[:, :, 2:].reshape(csi_228.shape[0], -1)

    return first_part, second_part


def _process_228_file(
    job: _ProcessingJob,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    file_csv = pd.read_csv(job.file_path, header=None, usecols=[14])
    csi_raw = file_csv.iloc[:, 0]

    valid_csi_114: list[list[float]] = []
    valid_csi_228: list[list[float]] = []
    no_match_count = 0
    no_complete_count = 0

    for entry in csi_raw:
        nums = _extract_csi_numbers(entry)
        if nums is None:
            no_match_count += 1
            continue

        if len(nums) == FIVE_GHZ_RAW_LENGTH_114:
            valid_csi_114.append(nums)
        elif len(nums) == FIVE_GHZ_RAW_LENGTH_228:
            valid_csi_228.append(nums)
        else:
            no_complete_count += 1

    csi_114 = np.array(valid_csi_114, dtype=float)
    csi_228 = np.array(valid_csi_228, dtype=float)

    mag_114 = csi_values_to_magnitude(csi_114, FIVE_GHZ_RAW_LENGTH_114)
    mag_228 = csi_values_to_magnitude(csi_228, FIVE_GHZ_RAW_LENGTH_228)

    if csi_228.size == 0:
        mag_228_first_half = empty_magnitude(FIVE_GHZ_RAW_LENGTH_114)
        mag_228_second_half = empty_magnitude(FIVE_GHZ_RAW_LENGTH_114)
        mag_228_interleaved_first = empty_magnitude(FIVE_GHZ_RAW_LENGTH_114)
        mag_228_interleaved_second = empty_magnitude(FIVE_GHZ_RAW_LENGTH_114)
    else:
        mag_228_first_half = csi_values_to_magnitude(
            csi_228[:, :FIVE_GHZ_RAW_LENGTH_114],
            FIVE_GHZ_RAW_LENGTH_114,
        )
        mag_228_second_half = csi_values_to_magnitude(
            csi_228[:, FIVE_GHZ_RAW_LENGTH_114:],
            FIVE_GHZ_RAW_LENGTH_114,
        )
        mag_228, mag_228_first_half, mag_228_second_half = remove_228_spike_subcarriers(
            mag_228,
            mag_228_first_half,
            mag_228_second_half,
        )

        csi_228_interleaved_first, csi_228_interleaved_second = split_228_interleaved_parts(csi_228)
        mag_228_interleaved_first = csi_values_to_magnitude(
            csi_228_interleaved_first,
            FIVE_GHZ_RAW_LENGTH_114,
        )
        mag_228_interleaved_second = csi_values_to_magnitude(
            csi_228_interleaved_second,
            FIVE_GHZ_RAW_LENGTH_114,
        )
        mag_228_interleaved_first, mag_228_interleaved_second = remove_shared_spike_subcarriers(
            mag_228_interleaved_first,
            mag_228_interleaved_second,
        )

    return (
        mag_114,
        mag_228,
        mag_228_first_half,
        mag_228_second_half,
        mag_228_interleaved_first,
        mag_228_interleaved_second,
        no_match_count,
        no_complete_count,
        len(csi_raw),
    )


def _empty_magnitude_maps() -> tuple[CsiMap, CsiMap, CsiMap, CsiMap, CsiMap, CsiMap]:
    return {}, {}, {}, {}, {}, {}


def process_228_files(  # noqa: PLR0913
    data_files: FileMap,
    *,
    max_workers: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_reprocess: bool = False,
    return_stats: bool = False,
) -> (
    tuple[CsiMap, CsiMap, CsiMap, CsiMap, CsiMap, CsiMap]
    | tuple[CsiMap, CsiMap, CsiMap, CsiMap, CsiMap, CsiMap, CacheStats]
):
    (
        magnitudes_114,
        magnitudes_228,
        magnitudes_228_first_half,
        magnitudes_228_second_half,
        magnitudes_228_interleaved_first,
        magnitudes_228_interleaved_second,
    ) = _empty_magnitude_maps()

    jobs = list(_iter_jobs(data_files, only_5ghz=True))
    options = _RunOptions(
        max_workers=max_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_reprocess=force_reprocess,
    )
    results, stats = _run_jobs(
        jobs,
        _process_228_cached,
        options,
    )

    for result in results:
        (
            mag_114,
            mag_228,
            mag_228_first_half,
            mag_228_second_half,
            mag_228_interleaved_first,
            mag_228_interleaved_second,
            _no_match,
            _no_complete,
            _total,
        ) = result.payload

        _set_magnitude_entry(magnitudes_114, result.job, mag_114)
        _set_magnitude_entry(magnitudes_228, result.job, mag_228)
        _set_magnitude_entry(magnitudes_228_first_half, result.job, mag_228_first_half)
        _set_magnitude_entry(magnitudes_228_second_half, result.job, mag_228_second_half)
        _set_magnitude_entry(
            magnitudes_228_interleaved_first,
            result.job,
            mag_228_interleaved_first,
        )
        _set_magnitude_entry(
            magnitudes_228_interleaved_second,
            result.job,
            mag_228_interleaved_second,
        )

    if return_stats:
        return (
            magnitudes_114,
            magnitudes_228,
            magnitudes_228_first_half,
            magnitudes_228_second_half,
            magnitudes_228_interleaved_first,
            magnitudes_228_interleaved_second,
            stats,
        )

    return (
        magnitudes_114,
        magnitudes_228,
        magnitudes_228_first_half,
        magnitudes_228_second_half,
        magnitudes_228_interleaved_first,
        magnitudes_228_interleaved_second,
    )
