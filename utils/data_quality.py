"""Data-quality diagnostics: subcarrier/tone mapping, pilot spikes, and recording inventory.

Nothing here changes the processing pipeline. The functions only describe what the
existing parser and subcarrier selection (``csi_processing.select_active_subcarriers``)
produce, so the original thesis selection stays the baseline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.config import ANCHOR_GROUPS, EMPTY_ROOM_LOCATION
from utils.csi_processing import select_active_subcarriers
from utils.fingerprint import FEATURE_COLUMN_PATTERN, normalize_two_digit_id
from utils.import_data import PATTERN
from utils.load_csi import load_raw_complex_csi

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

PILOT_TONES = (-21, -7, 7, 21)
RAW_TONE_COUNTS = {"2.4 GHz": 64, "5 GHz": 57}
SPIKE_RATIO_THRESHOLD = 1.5
ESP_BANDS = {esp: band for band, esps in ANCHOR_GROUPS.items() for esp in esps}


def raw_tone_numbers(band: str) -> np.ndarray:
    """Return the physical tone number of every raw CSI position reported by the ESP.

    2.4 GHz packets carry 64 tones in FFT order (0..31, -32..-1). 5 GHz packets carry 57
    tones in ascending order (-28..+28) with the DC tone at raw position 28.
    """
    if band == "5 GHz":
        return np.arange(RAW_TONE_COUNTS[band]) - RAW_TONE_COUNTS[band] // 2
    raw = np.arange(RAW_TONE_COUNTS[band])
    return np.where(raw < RAW_TONE_COUNTS[band] // 2, raw, raw - RAW_TONE_COUNTS[band])


def subcarrier_tone_map(band: str) -> pd.DataFrame:
    """Map every raw position to its tone and to the retained index of the actual selection.

    The retained indices are obtained by passing raw positions through
    ``select_active_subcarriers`` itself, so the map always follows the parser code.
    Dropped positions have ``retained_index = <NA>``.
    """
    raw_count = RAW_TONE_COUNTS[band]
    tagged = np.arange(raw_count, dtype=complex)[None, :]
    kept_raw = select_active_subcarriers(tagged, its5ghz=band == "5 GHz")[0].real.astype(int)
    retained = pd.Series(range(len(kept_raw)), index=kept_raw, dtype="Int64")
    tones = raw_tone_numbers(band)
    table = pd.DataFrame(
        {
            "band": band,
            "raw_index": np.arange(raw_count),
            "tone": tones,
            "retained_index": retained.reindex(np.arange(raw_count)).to_numpy(),
        }
    )
    table["retained_number"] = table["retained_index"] + 1
    table["is_pilot"] = table["tone"].isin(PILOT_TONES)
    return table


def pilot_retained_indices(band: str) -> list[int]:
    """Zero-based retained indices of the pilot tones under the actual selection."""
    tone_map = subcarrier_tone_map(band)
    return tone_map.loc[tone_map["is_pilot"], "retained_index"].dropna().astype(int).tolist()


def pilot_subcarrier_numbers_by_esp() -> dict[str, list[int]]:
    """1-based retained subcarrier numbers of the pilot tones for every configured ESP."""
    numbers = {
        band: [index + 1 for index in pilot_retained_indices(band)] for band in ANCHOR_GROUPS
    }
    return {esp: numbers[band] for esp, band in ESP_BANDS.items()}


def pilot_retained_indices_by_esp() -> dict[str, list[int]]:
    """Zero-based retained pilot indices for every configured ESP (for exclusion masks)."""
    indices = {band: pilot_retained_indices(band) for band in ANCHOR_GROUPS}
    return {esp: indices[band] for esp, band in ESP_BANDS.items()}


def pilot_neighbour_indices(band: str) -> list[int]:
    """Retained indices adjacent (+/-1) to a pilot that are not pilots themselves."""
    pilots = set(pilot_retained_indices(band))
    retained_count = int(subcarrier_tone_map(band)["retained_index"].max()) + 1
    neighbours = {
        index + step
        for index in pilots
        for step in (-1, 1)
        if 0 <= index + step < retained_count and index + step not in pilots
    }
    return sorted(neighbours)


def diagnostic_file_path(
    csv_diagnostics: pd.DataFrame,
    *,
    position: str,
    user: str,
    esp: str,
    trial: str,
) -> Path:
    """Return the CSV path of one recording listed in ``csv_diagnostics``."""
    rows = csv_diagnostics.loc[
        csv_diagnostics["location"].astype(str).str.upper().eq(position.upper())
        & csv_diagnostics["user"].map(normalize_two_digit_id).eq(normalize_two_digit_id(user))
        & csv_diagnostics["trial"].map(normalize_two_digit_id).eq(normalize_two_digit_id(trial))
        & csv_diagnostics["esp"].eq(esp)
    ]
    if rows.empty:
        msg = f"No recording for position={position}, user={user}, esp={esp}, trial={trial}."
        raise LookupError(msg)
    return Path(rows["file_path"].iloc[0])


def raw_tone_profile(file_path: str | Path, *, band: str) -> pd.DataFrame:
    """Mean magnitude and coefficient of variation of every raw tone in one file."""
    magnitude = np.abs(load_raw_complex_csi(file_path, its5ghz=band == "5 GHz"))
    profile = subcarrier_tone_map(band)
    mean = magnitude.mean(axis=0)
    profile["mean_magnitude"] = mean
    profile["cv"] = np.divide(
        magnitude.std(axis=0),
        mean,
        out=np.zeros_like(mean),
        where=mean > 0,
    )
    profile["packets"] = magnitude.shape[0]
    return profile


def raw_tone_profiles(
    csv_diagnostics: pd.DataFrame,
    *,
    position: str,
    user: str,
    esps: Sequence[str],
    trials: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Raw tone profiles for several ESPs/trials of one position and user, keyed by label."""
    profiles = {}
    for esp in esps:
        for trial in trials:
            path = diagnostic_file_path(
                csv_diagnostics, position=position, user=user, esp=esp, trial=trial
            )
            label = f"{esp.upper()} / trial {normalize_two_digit_id(trial)}"
            profiles[label] = raw_tone_profile(path, band=ESP_BANDS[esp])
    return profiles


def _iter_recordings(
    magnitude_data: dict,
) -> Iterator[tuple[str, str, str, str, str, np.ndarray]]:
    """Yield (scenario, location, user, esp, trial, magnitude) for every loaded recording."""
    for scenario, locations in magnitude_data.items():
        for location, users in locations.items():
            for user, esps in users.items():
                for esp, trials in esps.items():
                    for trial, magnitude in trials.items():
                        if magnitude is not None:
                            yield scenario, location, user, esp, trial, magnitude


def pilot_excess_table(
    magnitude_data: dict,
    *,
    bands: Sequence[str] = ("2.4 GHz", "5 GHz"),
) -> pd.DataFrame:
    """Per-recording pilot-to-neighbour magnitude ratios for every requested band.

    The per-packet ratio is the mean pilot magnitude over the mean magnitude of the
    adjacent non-pilot tones. RSSI calibration scales each packet by one scalar, so the
    ratio is identical on raw and calibrated magnitudes. ``pilot_power_fraction`` is the
    share of packet power carried by the pilots (``4 / retained tones`` for a flat channel);
    because RSSI calibration fixes total packet power, pilot excess lowers every other tone.
    """
    pilots = {band: pilot_retained_indices(band) for band in bands}
    neighbours = {band: pilot_neighbour_indices(band) for band in bands}
    rows = []
    for scenario, location, user, esp, trial, magnitude in _iter_recordings(magnitude_data):
        band = ESP_BANDS.get(esp)
        if band not in pilots or magnitude.shape[0] == 0:
            continue
        neighbour_mean = magnitude[:, neighbours[band]].mean(axis=1)
        ratio = magnitude[:, pilots[band]].mean(axis=1) / np.maximum(neighbour_mean, 1e-12)
        power = magnitude**2
        rows.append(
            {
                "scenario": scenario.removeprefix("scenario_"),
                "location": location.removeprefix("location_"),
                "user": user.removeprefix("user_"),
                "trial": trial.removeprefix("trial_"),
                "esp": esp,
                "band": band,
                "packets": magnitude.shape[0],
                "pilot_ratio_median": float(np.median(ratio)),
                "pilot_ratio_p95": float(np.percentile(ratio, 95)),
                "spike_packet_fraction": float((ratio > SPIKE_RATIO_THRESHOLD).mean()),
                "pilot_power_fraction": float(
                    (
                        power[:, pilots[band]].sum(axis=1) / np.maximum(power.sum(axis=1), 1e-12)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def pilot_excess_summary(
    pilot_table: pd.DataFrame,
    *,
    by: Sequence[str] = ("band", "esp", "trial"),
) -> pd.DataFrame:
    """Aggregate per-recording pilot ratios; a recording "spikes" if its median ratio > 1.5."""
    grouped = pilot_table.groupby(list(by), sort=True)
    summary = grouped.agg(
        recordings=("pilot_ratio_median", "size"),
        pilot_ratio_median=("pilot_ratio_median", "median"),
        pilot_ratio_p95=("pilot_ratio_p95", "median"),
        spike_packet_fraction=("spike_packet_fraction", "mean"),
        pilot_power_fraction=("pilot_power_fraction", "mean"),
    )
    summary["spiking_recordings"] = grouped["pilot_ratio_median"].apply(
        lambda ratios: float((ratios > SPIKE_RATIO_THRESHOLD).mean())
    )
    return summary.reset_index()


def pilot_feature_ratio_table(
    df: pd.DataFrame,
    *,
    feature_family: str = "std",
    by: str | None = "trial",
) -> pd.DataFrame:
    """Mean |feature| at pilot subcarriers divided by the mean at all other subcarriers.

    One row per ESP; one column per ``by`` value plus ``all``. With ``feature_family="std"``
    this is the within-window fluctuation ratio of the pilots.
    """
    parsed = [
        (column, *FEATURE_COLUMN_PATTERN.fullmatch(column).groups())
        for column in df.columns
        if FEATURE_COLUMN_PATTERN.fullmatch(str(column))
    ]
    groups = {"all": df} if by is None else {"all": df, **dict(tuple(df.groupby(by)))}
    rows = []
    for esp in sorted({esp for _, esp, family, _ in parsed if family == feature_family}):
        pilots = set(pilot_retained_indices(ESP_BANDS[esp]))
        columns = [
            (column, int(sc))
            for column, e, family, sc in parsed
            if e == esp and family == feature_family
        ]
        pilot_columns = [column for column, sc in columns if sc in pilots]
        other_columns = [column for column, sc in columns if sc not in pilots]
        row = {"esp": esp, "band": ESP_BANDS[esp]}
        for label, frame in groups.items():
            values = frame[pilot_columns + other_columns].abs()
            row[str(label)] = float(
                np.nanmean(values[pilot_columns].to_numpy())
                / np.nanmean(values[other_columns].to_numpy())
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    ordered = ["esp", "band", *[str(key) for key in groups if key != "all"], "all"]
    return table[ordered]


def session_dates_table(csv_diagnostics: pd.DataFrame, *, max_listed: int = 5) -> pd.DataFrame:
    """Return file counts per user, trial, and recording date (DD-MM, from file names).

    Locations are listed when a date covers at most ``max_listed`` of them, which exposes
    recordings filed under a session but captured on another day.
    """
    frame = csv_diagnostics.assign(
        user=csv_diagnostics["user"].map(normalize_two_digit_id),
        trial=csv_diagnostics["trial"].map(normalize_two_digit_id),
        date=csv_diagnostics["file_path"].map(
            lambda path: PATTERN.match(Path(path).name).group("timestamp").split("_")[0]
        ),
    )
    table = (
        frame.groupby(["user", "trial", "date"])
        .agg(
            files=("file_path", "size"),
            locations=("location", "nunique"),
            esps=("esp", "nunique"),
            location_list=("location", lambda locs: ", ".join(sorted(set(locs)))),
        )
        .reset_index()
    )
    table.loc[table["locations"] > max_listed, "location_list"] = "..."
    return table


def duplicate_recordings_table(csv_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Find byte-identical CSV files filed under different recording keys.

    Files are grouped by size first and only size collisions are hashed. Returns one row
    per file in a duplicate group, with ``duplicate_group`` shared by identical files.
    """
    frame = csv_diagnostics.assign(
        size=csv_diagnostics["file_path"].map(lambda path: Path(path).stat().st_size)
    )
    candidates = frame.loc[frame["size"].duplicated(keep=False)].copy()
    candidates["sha256"] = candidates["file_path"].map(
        lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    )
    duplicates = candidates.loc[candidates["sha256"].duplicated(keep=False)].copy()
    duplicates["duplicate_group"] = duplicates.groupby("sha256").ngroup()
    duplicates["file"] = duplicates["file_path"].map(lambda path: Path(path).name)
    columns = ["duplicate_group", "user", "trial", "location", "esp", "band", "file"]
    return duplicates.sort_values(["duplicate_group", "user"])[columns].reset_index(drop=True)


def duplicate_summary(duplicates: pd.DataFrame) -> pd.DataFrame:
    """Count duplicated files per (user, trial, ESP) and name the users they duplicate."""
    if duplicates.empty:
        return pd.DataFrame(columns=["user", "trial", "esp", "files", "duplicates_of_users"])
    partners = duplicates.groupby("duplicate_group")["user"].agg(set)
    rows = duplicates.assign(
        partners=duplicates.apply(
            lambda row: ", ".join(sorted(partners[row["duplicate_group"]] - {row["user"]})),
            axis=1,
        )
    )
    return (
        rows.groupby(["user", "trial", "esp"])
        .agg(
            files=("file", "size"),
            duplicates_of_users=("partners", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )


def diagnostics_by_trial(csv_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Packet counts, parsing losses, and retention per band and trial."""
    frame = csv_diagnostics.assign(
        trial=csv_diagnostics["trial"].map(normalize_two_digit_id),
        retention=csv_diagnostics["valid_magnitude_rows"]
        / csv_diagnostics["raw_rows"].replace(0, np.nan),
    )
    return (
        frame.groupby(["band", "trial"])
        .agg(
            files=("file_path", "size"),
            empty_files=("valid_magnitude_rows", lambda rows: int((rows == 0).sum())),
            raw_rows_median=("raw_rows", "median"),
            valid_rows_median=("valid_magnitude_rows", "median"),
            valid_rows_min=("valid_magnitude_rows", "min"),
            valid_rows_p05=("valid_magnitude_rows", lambda rows: rows.quantile(0.05)),
            retention_median=("retention", "median"),
            no_match_total=("no_match_count", "sum"),
            wrong_length_total=("dropped_csi_length_count", "sum"),
            invalid_packets_total=("invalid_packets_removed", "sum"),
            subcarriers=("subcarriers", lambda counts: ", ".join(map(str, sorted(set(counts))))),
        )
        .reset_index()
    )


def missing_recordings_table(csv_diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find absent or empty ESP recordings for every recorded session (user, trial).

    Expected recordings are every location seen in the dataset (52 positions plus
    ``Z-0``) times every configured ESP. Returns ``(summary, detail)``: one summary row
    per session and band, and one detail row per missing or empty recording.
    """
    frame = csv_diagnostics.assign(
        user=csv_diagnostics["user"].map(normalize_two_digit_id),
        trial=csv_diagnostics["trial"].map(normalize_two_digit_id),
    )
    locations = sorted(frame["location"].astype(str).unique())
    present = frame.set_index(["user", "trial", "location", "esp"])["valid_magnitude_rows"]
    detail = []
    summary = []
    for user, trial in sorted(set(zip(frame["user"], frame["trial"], strict=True))):
        for band, esps in ANCHOR_GROUPS.items():
            missing = empty = 0
            for location in locations:
                for esp in esps:
                    key = (user, trial, location, esp)
                    if key not in present.index:
                        missing += 1
                        detail.append(
                            {
                                **dict(zip(present.index.names, key, strict=True)),
                                "band": band,
                                "issue": "missing file",
                            }
                        )
                    elif int(present.loc[key]) == 0:
                        empty += 1
                        detail.append(
                            {
                                **dict(zip(present.index.names, key, strict=True)),
                                "band": band,
                                "issue": "no valid packets",
                            }
                        )
            expected = len(locations) * len(esps)
            summary.append(
                {
                    "user": user,
                    "trial": trial,
                    "band": band,
                    "expected": expected,
                    "present": expected - missing,
                    "missing": missing,
                    "empty": empty,
                }
            )
    detail_columns = ["user", "trial", "location", "esp", "band", "issue"]
    return pd.DataFrame(summary), pd.DataFrame(detail, columns=detail_columns)


def z0_reference_table(csv_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Empty-room (Z-0) baseline coverage per session and band."""
    frame = csv_diagnostics.assign(
        user=csv_diagnostics["user"].map(normalize_two_digit_id),
        trial=csv_diagnostics["trial"].map(normalize_two_digit_id),
    )
    z0 = frame.loc[frame["location"].astype(str).str.upper() == EMPTY_ROOM_LOCATION]
    rows = []
    for (user, trial), session in frame.groupby(["user", "trial"]):
        session_z0 = z0.loc[(z0["user"] == user) & (z0["trial"] == trial)]
        for band, esps in ANCHOR_GROUPS.items():
            band_z0 = session_z0.loc[session_z0["esp"].isin(esps)]
            valid = band_z0.loc[band_z0["valid_magnitude_rows"] > 0]
            rows.append(
                {
                    "user": user,
                    "trial": trial,
                    "band": band,
                    "z0_esps": f"{len(valid)}/{len(esps)}",
                    "z0_rows_min": band_z0["valid_magnitude_rows"].min(),
                    "z0_rows_median": band_z0["valid_magnitude_rows"].median(),
                    "missing_esps": ", ".join(sorted(set(esps) - set(valid["esp"]))) or "-",
                    "session_files": len(session),
                }
            )
    return pd.DataFrame(rows)


def aligned_window_table(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    by: Sequence[str] = ("trial",),
) -> pd.DataFrame:
    """Aligned multi-ESP windows (fingerprints) per band and ``by`` group."""
    tables = []
    for band, df in feature_dataframes.items():
        frame = df.assign(
            user=df["user"].map(normalize_two_digit_id),
            trial=df["trial"].map(normalize_two_digit_id),
        )
        per_recording = frame.groupby([*by, "group_id"]).size().rename("windows").reset_index()
        table = (
            per_recording.groupby(list(by))
            .agg(
                recordings=("group_id", "size"),
                windows=("windows", "sum"),
                windows_per_recording_min=("windows", "min"),
                windows_per_recording_median=("windows", "median"),
                windows_per_recording_max=("windows", "max"),
            )
            .reset_index()
        )
        table.insert(0, "band", band)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def alignment_loss_table(csv_diagnostics: pd.DataFrame, *, window_size: int) -> pd.DataFrame:
    """Windows lost because multi-ESP alignment is limited by the shortest ESP recording.

    For every (session, location, band) the aligned window count is
    ``min_rows // window_size``; the loss is relative to ``max_rows // window_size``.
    """
    frame = csv_diagnostics.loc[
        csv_diagnostics["location"].astype(str).str.upper() != EMPTY_ROOM_LOCATION
    ].assign(
        user=csv_diagnostics["user"].map(normalize_two_digit_id),
        trial=csv_diagnostics["trial"].map(normalize_two_digit_id),
    )
    recordings = frame.groupby(["band", "trial", "user", "location"])
    per_recording = recordings["valid_magnitude_rows"].agg(["min", "max"]).reset_index()
    per_recording["limiting_esp"] = recordings.apply(
        lambda group: group.loc[group["valid_magnitude_rows"].idxmin(), "esp"],
        include_groups=False,
    ).to_numpy()
    per_recording["aligned_windows"] = per_recording["min"] // window_size
    per_recording["max_windows"] = per_recording["max"] // window_size
    grouped = per_recording.groupby(["band", "trial"])
    summary = grouped.agg(
        recordings=("location", "size"),
        aligned_windows=("aligned_windows", "sum"),
        max_esp_windows=("max_windows", "sum"),
        most_limiting_esp=("limiting_esp", lambda esps: esps.mode().iloc[0]),
    ).reset_index()
    summary["window_loss_fraction"] = 1 - summary["aligned_windows"] / summary["max_esp_windows"]
    return summary
