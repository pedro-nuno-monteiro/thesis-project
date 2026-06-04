import re
import warnings
from pathlib import Path

FileMap = dict[str, dict[str, dict[str, dict[str, dict[str, Path]]]]]
USER_NAMES = {
    "01": "Pedro Monteiro",
    "02": "Guilherme Cabaço",
    "03": "Henrique",
}

# scenario_id - location_x-location_y - user_id - esp_id - trial - timestamp.csv
# 1_A-1_00_10_01_15-05_14-59-23.csv
PATTERN = re.compile(
    r"^(?P<scenario>\d+)_(?P<location>(?:[A-G]-(?:[1-9]|1[0-4])|Z-0))_"
    r"(?P<user>\d+)_(?P<esp>\d+)_(?P<trial>\d+)_"
    r"(?P<timestamp>\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$",
    re.IGNORECASE,
)


# This function scans the specified directory for CSV files that match a specific naming pattern.
# It collects unique values for scenarios, locations, users, esps,
# and trials found in the file names and returns them as sorted lists.
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


# This function scans the specified directory for CSV files that match a specific naming pattern.
# It organizes the found files into a nested dictionary structure based on their metadata
# (scenario, location, user, esp, trial).
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


# This function extracts the locations associated with each user from the provided file map.
def get_locations_by_user(files: FileMap) -> dict[str, set[str]]:
    locations_by_user: dict[str, set[str]] = {}

    for locations_map in files.values():
        for location_key, users_map in locations_map.items():
            location = location_key.removeprefix("location_")

            for user_key in users_map:
                user_id = user_key.removeprefix("user_")
                locations_by_user.setdefault(user_id, set()).add(location)

    return locations_by_user


# This function prints a table of user locations based on the provided file map.
# It organizes the data by user and location, showing which locations each user has files for.
def print_user_location_tables(
    files: FileMap,
    row_letters: str = "ABCDEF",
    max_column: int = 14,
) -> None:
    locations_by_user = get_locations_by_user(files)

    columns = list(range(1, max_column + 1))
    header = "   " + " ".join(f"{column:>2}" for column in columns)
    user_ids = sorted(set(USER_NAMES) | set(locations_by_user))

    if not user_ids:
        print("No user locations found.")
        return

    for user_id in user_ids:
        user_name = USER_NAMES.get(user_id, "Unknown")
        user_locations = locations_by_user.get(user_id, set())

        print(f"\nUser {user_id} - {user_name}")
        print(header)

        for row_letter in row_letters:
            cells = [
                " X" if f"{row_letter}-{column}" in user_locations else " ."
                for column in columns
            ]
            print(f"{row_letter} |" + " ".join(cells))

        grid_locations = {
            f"{row_letter}-{column}"
            for row_letter in row_letters
            for column in columns
        }
        extra_locations = sorted(user_locations - grid_locations)

        if extra_locations:
            print(f"Other locations: {', '.join(extra_locations)}")

        if not user_locations:
            print("No files found for this user.")
