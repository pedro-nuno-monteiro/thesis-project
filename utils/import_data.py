import re
import warnings
from pathlib import Path

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, Path]]]]]

# scenario_id - location_x-location_y - user_id - esp_id - trial - timestamp.csv
# 1_A-1_00_10_01_15-05_14-59-23.csv
PATTERN = re.compile(
    r"^(?P<scenario>\d+)_(?P<location>(?:[A-G]-(?:[1-9]|1[0-4])|Z-0))_"
    r"(?P<user>\d+)_(?P<esp>\d+)_(?P<trial>\d+)_"
    r"(?P<timestamp>\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)


def sort_meta_info(
    path: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    base: Path = Path(path)
    scenarios_id = set()
    locations_id = set()
    users_id = set()
    esps_id = set()
    trial_id = set()

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            print("skipped on no match, ", file.name)
            continue

        scenario = match.group("scenario")
        location = match.group("location").upper()
        user = match.group("user")
        esp = match.group("esp")
        trial = match.group("trial")

        scenarios_id.add(scenario)
        locations_id.add(location)
        users_id.add(user)
        esps_id.add(esp)
        trial_id.add(trial)

    return (
        sorted(scenarios_id),
        sorted(locations_id),
        sorted(users_id),
        sorted(esps_id),
        sorted(trial_id),
    )


def get_csv_files(path: str) -> FileMap:
    files: FileMap = {}
    base: Path = Path(path)

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            continue

        scenario = match.group("scenario")
        location = match.group("location").upper()
        user = match.group("user")
        esp = match.group("esp")
        trial = match.group("trial")

        scenario_key = f"scenario_{scenario}"
        location_key = f"location_{location}"
        user_key = f"user_{user}"
        esp_key = f"esp_{esp}"
        trial_key = f"trial_{trial}"

        files.setdefault(scenario_key, {})
        files[scenario_key].setdefault(location_key, {})
        files[scenario_key][location_key].setdefault(user_key, {})
        files[scenario_key][location_key][user_key].setdefault(esp_key, {})
        trial_path = files[scenario_key][location_key][user_key][esp_key].get(trial_key)
        if trial_path is not None:
            warning_msg = (
                "Multiple CSV paths found for the same "
                f"scenario/location/user/esp/trial: {scenario_key}/{location_key}/"
                f"{user_key}/{esp_key}/{trial_key}. Existing: {trial_path.name}, "
                f"New: {file.name}"
            )
            warnings.warn(warning_msg, stacklevel=2)
            continue

        files[scenario_key][location_key][user_key][esp_key][trial_key] = file

    return files
