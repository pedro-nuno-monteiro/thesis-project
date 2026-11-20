"""Select, organize, and summarize CSI fingerprints from the ML feature DataFrames.

A *fingerprint* is one row of a feature DataFrame built by ``feature_pipeline``: the
statistical features (``FEATURE_NAMES``) of every ESP and retained subcarrier for one
window of one recording. Nothing here rescales the values; they are consumed exactly as
the feature pipeline produced them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from utils.feature_pipeline import FEATURE_NAMES, METADATA_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FEATURE_COLUMN_PATTERN = re.compile(
    rf"^(esp_\d+)_({'|'.join(FEATURE_NAMES)})_sc_(\d+)$",
)
FEATURE_LABELS = {
    "mean": "Mean",
    "std": "Standard deviation",
    "variance": "Variance",
    "max": "Maximum",
    "min": "Minimum",
    "energy": "Energy",
}
FEATURE_FAMILY_ALIASES = {"maximum": "max", "minimum": "min"}
ANALYSIS_MODES = ("individual", "mean")
FINGERPRINT_INDEX = ["ESP", "subcarrier_index", "subcarrier_number"]


@dataclass(frozen=True)
class FingerprintSelection:
    """One selected fingerprint (individual window or recording mean) and its context."""

    band: str
    position: str
    user: str
    trial: str
    mode: str
    window_index: int | None
    recording_rows: pd.DataFrame
    vector: pd.Series
    table: pd.DataFrame
    n_fingerprints: int
    esps: list[str]

    @property
    def label(self) -> str:
        """Short legend label describing how the vector was obtained."""
        if self.mode == "individual":
            return f"window_idx={self.window_index}"
        return f"mean of {self.n_fingerprints} windows"

    @property
    def title(self) -> str:
        """Figure title identifying the recording and the selection mode."""
        return (
            f"{self.position} / User {self.user} / Trial {self.trial} / {self.band}: {self.label}"
        )

    @property
    def recording_slug(self) -> str:
        """Filename-safe position/user/band identifier, shared by every trial."""
        band = self.band.lower().replace(".", "-").replace(" ", "")
        return f"{self.position}-u{self.user}-{band}"

    @property
    def slug(self) -> str:
        """Filename-safe identifier of the selection (without feature family)."""
        window = f"-w{self.window_index}" if self.mode == "individual" else ""
        return f"{self.recording_slug}-t{self.trial}-{self.mode}{window}"

    @property
    def summary(self) -> pd.Series:
        """One-column overview of the selection, for display in notebooks."""
        return pd.Series(
            {
                "position": self.position,
                "user": self.user,
                "trial": self.trial,
                "band": self.band,
                "mode": self.mode,
                "window_idx": self.window_index if self.mode == "individual" else "all",
                "windows_in_recording": len(self.recording_rows),
                "windows_used": self.n_fingerprints,
                "feature_values": int(self.vector.notna().sum()),
                "ESPs": len(self.esps),
            },
            name="selection",
        )


def normalize_two_digit_id(value: object) -> str:
    """Normalize a user/trial value (``3``, ``"user_03"``, ``3.0``) to ``"03"``."""
    text = str(value).strip().removeprefix("user_").removeprefix("trial_").removesuffix(".0")
    return text.zfill(2)


def normalize_position(value: object) -> str:
    """Normalize a location value (``"location_b-8"``) to its grid label (``"B-8"``)."""
    return str(value).strip().upper().removeprefix("LOCATION_")


def canonical_feature_family(feature_family: str) -> str:
    """Return the ``FEATURE_NAMES`` entry for a feature-family name or alias."""
    family = str(feature_family).lower()
    family = FEATURE_FAMILY_ALIASES.get(family, family)
    if family not in FEATURE_NAMES:
        msg = f"Unknown feature family {feature_family!r}; choose one of {FEATURE_NAMES}."
        raise ValueError(msg)
    return family


def fingerprint_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ``esp_XX_<family>_sc_YY`` columns of a feature DataFrame, in order."""
    return [column for column in df.columns if FEATURE_COLUMN_PATTERN.fullmatch(str(column))]


def feature_inventory_table(feature_dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize fingerprints, recordings, and feature width for every band."""
    rows = []
    for band, df in feature_dataframes.items():
        columns = fingerprint_feature_columns(df)
        esps = {FEATURE_COLUMN_PATTERN.fullmatch(column).group(1) for column in columns}
        rows.append(
            {
                "band": band,
                "fingerprints": len(df),
                "recordings": df["group_id"].nunique() if "group_id" in df else pd.NA,
                "positions": df["location"].nunique(),
                "users": df["user"].nunique(),
                "trials": ", ".join(sorted(df["trial"].map(normalize_two_digit_id).unique())),
                "ESPs": len(esps),
                "feature_columns": len(columns),
                "metadata_columns": len(df.columns.intersection(METADATA_COLUMNS)),
            }
        )
    return pd.DataFrame(rows)


def select_recording_rows(
    df: pd.DataFrame,
    *,
    position: str,
    user: str,
    trial: str,
    scenario: str | None = None,
) -> pd.DataFrame:
    """Return every window of one recording, sorted by ``window_idx``."""
    required = {"location", "user", "trial", "window_idx"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Feature DataFrame is missing metadata columns: {sorted(missing)}"
        raise KeyError(msg)

    mask = (
        df["location"].map(normalize_position).eq(normalize_position(position))
        & df["user"].map(normalize_two_digit_id).eq(normalize_two_digit_id(user))
        & df["trial"].map(normalize_two_digit_id).eq(normalize_two_digit_id(trial))
    )
    if scenario is not None:
        if "scenario" not in df.columns:
            msg = "A scenario was requested, but the DataFrame has no 'scenario' column."
            raise KeyError(msg)
        scenario_value = str(scenario).removeprefix("scenario_")
        mask &= df["scenario"].astype(str).str.removeprefix("scenario_").eq(scenario_value)

    rows = df.loc[mask].sort_values("window_idx").copy()
    if rows.empty:
        msg = (
            f"No fingerprints for position={normalize_position(position)}, "
            f"user={normalize_two_digit_id(user)}, trial={normalize_two_digit_id(trial)}."
        )
        raise LookupError(msg)
    if scenario is None and "scenario" in rows.columns and rows["scenario"].nunique() > 1:  # noqa: PD101
        choices = sorted(rows["scenario"].astype(str).unique())
        msg = f"Selection matches multiple scenarios {choices}; set the scenario explicitly."
        raise ValueError(msg)
    if rows["window_idx"].duplicated().any():
        msg = "Selection contains duplicate window_idx values; set the scenario explicitly."
        raise ValueError(msg)
    return rows


def fingerprint_vector(
    recording_rows: pd.DataFrame,
    *,
    mode: str,
    window_index: int = 0,
) -> tuple[pd.Series, int]:
    """Return one window's feature vector, or the column-wise recording mean, and its count."""
    mode = str(mode).lower()
    columns = fingerprint_feature_columns(recording_rows)
    if not columns:
        msg = "No fingerprint feature columns were found."
        raise ValueError(msg)

    if mode == "individual":
        match = recording_rows.loc[recording_rows["window_idx"].eq(window_index)]
        if len(match) != 1:
            available = recording_rows["window_idx"].astype(int).tolist()
            msg = f"window_idx={window_index} is unavailable; choose from {available}."
            raise ValueError(msg)
        return match.iloc[0][columns].astype(float), 1
    if mode == "mean":
        return recording_rows[columns].mean(axis=0).astype(float), len(recording_rows)
    msg = f"mode must be one of {ANALYSIS_MODES}, got {mode!r}."
    raise ValueError(msg)


def organize_fingerprint(vector: pd.Series) -> pd.DataFrame:
    """Reshape a flat feature vector into rows (ESP, subcarrier) x columns (feature family)."""
    records = []
    for column, value in vector.items():
        match = FEATURE_COLUMN_PATTERN.fullmatch(str(column))
        if match is None:
            continue
        esp, family, subcarrier_text = match.groups()
        subcarrier_index = int(subcarrier_text)
        records.append(
            {
                "ESP": esp,
                "subcarrier_index": subcarrier_index,
                "subcarrier_number": subcarrier_index + 1,
                "feature_family": family,
                "value": float(value),
            }
        )
    if not records:
        msg = "The fingerprint contains no recognized feature values."
        raise ValueError(msg)
    table = (
        pd.DataFrame.from_records(records)  # noqa: PD010
        .pivot(index=FINGERPRINT_INDEX, columns="feature_family", values="value")
        .reindex(columns=list(FEATURE_NAMES))
        .sort_index()
    )
    table.columns.name = None
    return table


def resolve_esps(table: pd.DataFrame, esps: str | Iterable[str] = "all") -> list[str]:
    """Return the requested ESPs present in an organized fingerprint, in table order."""
    available = list(dict.fromkeys(table.index.get_level_values("ESP")))
    if esps == "all":
        return available
    requested = [str(esp).lower() for esp in esps]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        msg = f"Requested ESPs are unavailable in this fingerprint: {unknown}"
        raise ValueError(msg)
    return [esp for esp in available if esp in requested]


def select_fingerprint(  # noqa: PLR0913
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    band: str,
    position: str,
    user: str,
    trial: str,
    mode: str = "individual",
    window_index: int = 0,
    esps: str | Iterable[str] = "all",
    scenario: str | None = None,
) -> FingerprintSelection:
    """Select and organize one fingerprint (individual window or recording mean)."""
    if band not in feature_dataframes:
        msg = f"Unknown band {band!r}; choose from {tuple(feature_dataframes)}."
        raise KeyError(msg)
    recording_rows = select_recording_rows(
        feature_dataframes[band],
        position=position,
        user=user,
        trial=trial,
        scenario=scenario,
    )
    vector, n_fingerprints = fingerprint_vector(
        recording_rows,
        mode=mode,
        window_index=window_index,
    )
    table = organize_fingerprint(vector)
    return FingerprintSelection(
        band=band,
        position=normalize_position(position),
        user=normalize_two_digit_id(user),
        trial=normalize_two_digit_id(trial),
        mode=str(mode).lower(),
        window_index=window_index if str(mode).lower() == "individual" else None,
        recording_rows=recording_rows,
        vector=vector,
        table=table,
        n_fingerprints=n_fingerprints,
        esps=resolve_esps(table, esps),
    )


def fingerprint_table_preview(table: pd.DataFrame, rows_per_esp: int | None) -> pd.DataFrame:
    """Return the first ``rows_per_esp`` subcarriers of every ESP (all rows when ``None``)."""
    if rows_per_esp is None:
        return table
    return table.groupby(level="ESP", sort=False).head(rows_per_esp)


def feature_family_table(
    table: pd.DataFrame,
    *,
    feature_family: str,
    esps: Sequence[str],
) -> pd.DataFrame:
    """Return one feature family as a wide table: subcarrier number x ESP."""
    family = canonical_feature_family(feature_family)
    wide = table[family].unstack("ESP")  # noqa: PD010
    wide = wide.droplevel("subcarrier_index").reindex(columns=list(esps))
    wide.index.name = "subcarrier_number"
    wide.columns.name = f"{FEATURE_LABELS[family]} by ESP"
    return wide


@dataclass(frozen=True)
class TrialFingerprints:
    """Mean and SD (across windows) organized fingerprints of several trials of one recording."""

    means: dict[str, pd.DataFrame]
    sds: dict[str, pd.DataFrame]
    inventory: pd.DataFrame


def fingerprints_by_trial(
    df: pd.DataFrame,
    *,
    position: str,
    user: str,
    trials: Iterable[str],
    scenario: str | None = None,
) -> TrialFingerprints:
    """Return the organized mean and SD fingerprints of every available trial.

    ``means`` and ``sds`` are keyed by a ``"Trial XX (n=N)"`` label. Trials without
    recordings are kept in ``inventory`` with ``windows_used=0`` and left out of both.
    """
    means: dict[str, pd.DataFrame] = {}
    sds: dict[str, pd.DataFrame] = {}
    inventory = []
    for trial in trials:
        trial_id = normalize_two_digit_id(trial)
        try:
            rows = select_recording_rows(
                df,
                position=position,
                user=user,
                trial=trial_id,
                scenario=scenario,
            )
        except LookupError:
            inventory.append({"trial": trial_id, "windows_used": 0})
            continue
        columns = fingerprint_feature_columns(rows)
        label = f"Trial {trial_id} (n={len(rows)})"
        means[label] = organize_fingerprint(rows[columns].mean(axis=0))
        sds[label] = organize_fingerprint(rows[columns].std(axis=0, ddof=1))
        inventory.append({"trial": trial_id, "windows_used": len(rows)})
    return TrialFingerprints(means=means, sds=sds, inventory=pd.DataFrame(inventory))
