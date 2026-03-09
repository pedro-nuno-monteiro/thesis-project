import re
from pathlib import Path

FileMap = dict[str, dict[str, dict[str, dict[str, Path]]]]

# scenario_id - user_id - activity_id - esp_id - trial - date - time.csv
# example: 21_00_00_01_01_03_2026-03-05_15-30.csv
PATTERN = re.compile(
    r"^(?P<scenario>\d+)_(?P<user>\d+)_(?P<activity>\d+)_(?P<esp>\d+)_(?P<trial>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2})\.csv$",
)


def sort_meta_info(path: str) -> tuple[list[int], list[str], list[int], list[int], list[int]]:
    base: Path = Path(path)
    scenarios_id = set()
    users_id = set()
    activities_id = set()
    esps_id = set()
    trial_id = set()

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            print("skipped on no match, ", file.name)
            continue

        scenario = match.group("scenario")
        user = match.group("user")
        activity = match.group("activity")
        esp = match.group("esp")
        trial = match.group("trial")

        scenarios_id.add(scenario)
        users_id.add(user)
        activities_id.add(activity)
        esps_id.add(esp)
        trial_id.add(trial)

    return (
        sorted(scenarios_id),
        sorted(users_id),
        sorted(activities_id),
        sorted(esps_id),
        sorted(trial_id),
    )


def get_csv_files_generalistic(path: str) -> FileMap:
    files: FileMap = {}
    base: Path = Path(path)

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            continue

        scenario = match.group("scenario")
        user = match.group("user")
        activity = match.group("activity")
        esp = match.group("esp")
        trial = match.group("trial")

        scenario_key = f"scenario_{scenario}"
        user_key = f"user_{user}"
        activity_key = f"activity_{activity}"
        esp_key = f"esp_{esp}"
        trial_key = f"trial_{trial}"

        files.setdefault(scenario_key, {})
        files[scenario_key].setdefault(user_key, {})
        files[scenario_key][user_key].setdefault(activity_key, {})
        files[scenario_key][user_key][activity_key].setdefault(esp_key, {})
        files[scenario_key][user_key][activity_key][esp_key].setdefault(trial_key, [])

        # append file (support multiple timestamps)
        files[scenario_key][user_key][activity_key][esp_key][trial_key].append(file)

    return files
