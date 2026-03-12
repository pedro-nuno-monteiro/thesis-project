from __future__ import annotations

import os
import re
import socket
import sys
import time
from pathlib import Path

UDP_IP = "0.0.0.0"
UDP_PORT = 5001
RECV_BUFFER_SIZE = 4096
SOCKET_TIMEOUT_SECONDS = 1.0
DEFAULT_SAVE_DIRECTORY = Path.cwd() / "csi_frames"

ESP_MAC_MAP = {
    "90:38:0C:EA:D3:78": "01",
    "90:38:0C:EA:D4:CC": "02",
    "C4:DE:E2:C0:98:E8": "03",
    "90:38:0C:EA:D5:04": "04",
    "D0:CF:13:ED:B7:D8": "05",
    "D0:CF:13:ED:9A:2C": "06",
    "D0:CF:13:ED:9A:8C": "07",
}

VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9.:-]+$")


def prompt_text(prompt: str) -> str:
    return input(prompt).strip()


def prompt_identifier(prompt: str) -> str:
    value = prompt_text(prompt)
    if not value:
        raise ValueError("This field cannot be empty.")
    if not VALID_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Only letters, numbers, '.', ':', and '-' are allowed in identifiers.",
        )
    return value


def prompt_duration_minutes() -> float:
    raw_value = prompt_text("* * Enter duration in minutes (0 for unlimited): ")
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        print("Invalid duration. Falling back to unlimited collection.")
        return 0.0


def resolve_save_directory() -> Path:
    configured_path = os.environ.get("CSI_SAVE_DIR")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_SAVE_DIRECTORY


def get_next_trial_number(
    directory: Path,
    scenario_id: str,
    user_id: str,
    activity_id: str,
    esp_id: str,
) -> int:
    pattern = re.compile(
        rf"^{re.escape(scenario_id)}_{re.escape(user_id)}_{re.escape(activity_id)}_"
        rf"{re.escape(esp_id)}_(\d+)_.*\.csv$",
    )

    last_trial = 0
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if match:
            last_trial = max(last_trial, int(match.group(1)))

    return last_trial + 1


def create_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.settimeout(SOCKET_TIMEOUT_SECONDS)
    sock.bind((UDP_IP, UDP_PORT))
    return sock


def prompt_session_metadata() -> tuple[str, str, str, float]:
    print("\n* User ID")
    print("* 00 - Nobody")
    print("* 01 - Pedro")
    user_id = prompt_identifier("* * Enter user ID: ")

    print("\n* Activity ID")
    print("* 00 - Empty Room")
    print("* 01 - Walking")
    activity_id = prompt_identifier("* * Enter activity: ")

    print("\n* Scenario ID")
    print(" * X.Y.Z.H.N")
    print(" * X - ESP/router layout (scenario 1 vs scenario 2) | 1/2")
    print(" * Y - frequency band (2.4 GHz vs 5 GHz) | 1/2")
    print(" * Z - ESP positioning (1 - layed on a table, 2 - standing, 3 - mix) | 1/2/3")
    print(" * H - ESP height (0 - floor, 1 - 1 m, 2 - 2m, 3 - random) | 0/1/2/3")
    print(" * N - ESPs used | 1/2/3/4/5")
    scenario_id = prompt_identifier("* * Enter scenario: ")

    print("\n* Collection Duration")
    duration_minutes = prompt_duration_minutes()

    return user_id, activity_id, scenario_id, duration_minutes


def parse_packet(data: bytes) -> tuple[str, str] | None:
    try:
        decoded_data = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None

    if not decoded_data or "," not in decoded_data:
        return None

    esp_mac, csi_data = decoded_data.split(",", 1)
    esp_mac = esp_mac.strip().upper()
    csi_data = csi_data.strip()

    if not esp_mac or not csi_data:
        return None

    return esp_mac, csi_data


def main() -> int:
    try:
        user_id, activity_id, scenario_id, duration_minutes = prompt_session_metadata()
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 1

    save_directory = resolve_save_directory()
    save_directory.mkdir(parents=True, exist_ok=True)

    try:
        sock = create_socket()
    except OSError as exc:
        print(f"Failed to start UDP server on {UDP_IP}:{UDP_PORT}: {exc}")
        return 1

    packet_count_by_esp: dict[str, int] = {}
    trial_by_esp: dict[str, int] = {}
    file_path_by_esp: dict[str, Path] = {}

    session_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.monotonic()

    print(f"\nUDP server listening on {UDP_IP}:{UDP_PORT}")
    print(f"Saving CSI frames to: {save_directory}")
    if duration_minutes > 0:
        print(f"Collection will run for {duration_minutes} minute(s)")
    else:
        print("Collection will run indefinitely (press Ctrl+C to stop)")
    input("Press ENTER to run")
    print("----------------------------------------------")

    try:
        while True:
            if duration_minutes > 0:
                elapsed_minutes = (time.monotonic() - start_time) / 60
                if elapsed_minutes >= duration_minutes:
                    print("\n----------------------------------------------")
                    print(
                        f"Duration of {duration_minutes} minute(s) reached. Stopping collection.",
                    )
                    print("----------------------------------------------")
                    break

            try:
                data, addr = sock.recvfrom(RECV_BUFFER_SIZE)
            except socket.timeout:
                continue

            parsed_packet = parse_packet(data)
            if parsed_packet is None:
                print(f"Invalid packet received from {addr}")
                print("----------------------------------------------")
                continue

            esp_mac, csi_data = parsed_packet
            esp_id = ESP_MAC_MAP.get(esp_mac, esp_mac.replace(":", ""))

            packet_count_by_esp[esp_id] = packet_count_by_esp.get(esp_id, 0) + 1
            print(f"Data received from: {esp_mac}")
            print(
                f"ESP {esp_id} - Packets received: {packet_count_by_esp[esp_id]}",
            )

            if esp_id not in trial_by_esp:
                trial_by_esp[esp_id] = get_next_trial_number(
                    save_directory,
                    scenario_id,
                    user_id,
                    activity_id,
                    esp_id,
                )
                file_path_by_esp[esp_id] = save_directory / (
                    f"{scenario_id}_{user_id}_{activity_id}_{esp_id}_"
                    f"{trial_by_esp[esp_id]:02d}_{session_timestamp}.csv"
                )
                print(
                    f"ESP {esp_id} - Using trial {trial_by_esp[esp_id]:02d} "
                    f"for {scenario_id}/{user_id}/{activity_id}/{esp_id}",
                )

            output_file = file_path_by_esp[esp_id]
            try:
                with output_file.open("a", encoding="utf-8", newline="") as file:
                    file.write(f"{csi_data}\n")
            except OSError as exc:
                print(f"Failed to write packet for ESP {esp_id}: {exc}")
                print("----------------------------------------------")
                continue

            print(f"CSI saved: ESP {esp_id} -> {output_file.name}")
            print("----------------------------------------------")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping collection.")
    finally:
        sock.close()

    print("UDP server stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
