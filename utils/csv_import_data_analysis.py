import re
from pathlib import Path

# user_id-activity_id-place_id-esp_id-timestamp.csv
# scenario_id - user_id - activity_id - esp_id - timestamp.csv
PATTERN = re.compile(
    r"^(?P<scenario>\d+)_(?P<user>\d+)_(?P<activity>\d+)_(?P<esp>\d+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$",
)


def sort_meta_info(path: str) -> tuple[list[int], list[str], list[int], list[int]]:
    base: Path = Path(path)
    scenarios_id = set()
    users_id = set()
    activities_id = set()
    esps_id = set()

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            print("skipped on no match, ", file.name)
            continue

        scenario = match.group("scenario")
        user = match.group("user")
        activity = match.group("activity")
        esp = match.group("esp")

        scenarios_id.add(scenario)
        users_id.add(user)
        activities_id.add(activity)
        esps_id.add(esp)

    return (
        sorted(users_id),
        sorted(activities_id),
        sorted(scenarios_id),
        sorted(esps_id),
    )


FileMap = dict[str, dict[str, dict[str, dict[str, Path]]]]


def get_csv_files_generalistic(path: str):
    files: FileMap = {}
    base: Path = Path(path)

    # legenda

    # USER ID
    # # 00 - ninguém
    # # 01 - Pepas

    # ACTIVITY
    # # 00 - empty room
    # # 01 - walking

    for file in base.glob("*.csv"):
        match = PATTERN.match(file.name)
        if not match:
            continue

        scenario = match.group("scenario")
        user = match.group("user")
        activity = match.group("activity")
        esp = match.group("esp")

        scenario_key = f"scenario_{scenario}"
        user_key = f"user_{user}"
        activity_key = f"activity_{activity}"
        esp_key = f"esp_{esp}"

        files.setdefault(scenario_key, {})
        files[scenario_key].setdefault(user_key, {})
        files[scenario_key][user_key].setdefault(activity_key, {})
        files[scenario_key][user_key][activity_key].setdefault(esp_key, [])

        # append file (support multiple timestamps)
        files[scenario_key][user_key][activity_key][esp_key].append(file)

    return files
