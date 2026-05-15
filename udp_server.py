from __future__ import annotations

import os
import re
import socket
import sys
import time
from pathlib import Path

UDP_IP = "0.0.0.0"
DEFAULT_UDP_PORT = 5001
RECV_BUFFER_SIZE = 4096
SOCKET_TIMEOUT_SECONDS = 1.0
DEFAULT_SAVE_DIRECTORY = Path.cwd() / "csi_frames"

ESP_MAC_MAP = {
    "90:38:0C:EA:D3:78": "01",
    "90:38:0C:EA:D4:CC": "02",
    "C4:DE:E2:C0:98:E8": "03",
    "90:38:0C:EA:D5:04": "04",
    "C0:49:EF:44:D0:94": "05",
    "08:B6:1F:EE:4B:E0": "06",
    "C4:DE:E2:C0:AC:70": "07",
    "90:38:0C:ED:6C:98": "08",
    "90:38:0C:ED:6B:10": "09",
    "08:B6:1F:EF:81:A0": "10",

    "D0:CF:13:ED:B7:D8": "11",
    "D0:CF:13:ED:9A:2C": "12",
    "D0:CF:13:ED:9A:8C": "13",
    "D0:CF:13:ED:C4:AC": "14",
    "D0:CF:13:ED:9A:4C": "15",
    "D0:CF:13:ED:F9:20": "16",
    "D0:CF:13:ED:F9:A8": "17",
    "D0:CF:13:ED:FA:D8": "18",
    "D0:CF:13:ED:F8:F0": "19",
    "D0:CF:13:ED:81:68": "20",
}

VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9.:-]+$")
LOCATION_PATTERN = re.compile(r"^[A-G]-(?:[1-9]|1[0-4])$")


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


def prompt_user_id() -> str:
    value = prompt_identifier("* * Enter user ID: ")
    if not value.isdigit():
        raise ValueError("User ID must contain only digits.")
    return value.zfill(2)


def prompt_location() -> str:
    value = prompt_text("* * Enter location (e.g. A-4, D-2, F-10): ")
    if not value:
        raise ValueError("Location cannot be empty.")

    normalized_value = value.replace(" ", "").upper()
    if not LOCATION_PATTERN.fullmatch(normalized_value):
        raise ValueError(
            "Location must use a letter from A to G and a number from 1 to 14, e.g. A-4.",
        )

    return normalized_value


def prompt_run_duration_seconds() -> float:
    default_duration_seconds = 30.0
    raw_value = prompt_text(
        "* * Run for 30 seconds? Press ENTER for yes, or type anything else to set a custom duration: ",
    )

    if not raw_value:
        return default_duration_seconds

    raw_duration = prompt_text("* * Enter duration in seconds (0 for unlimited): ")
    try:
        return max(float(raw_duration), 0.0)
    except ValueError:
        print("Invalid duration. Falling back to unlimited collection.")
        return 0.0


def resolve_save_directory() -> Path:
    configured_path = os.environ.get("CSI_SAVE_DIR")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_SAVE_DIRECTORY


def resolve_udp_port() -> int:
    configured_port = os.environ.get("CSI_UDP_PORT")
    if not configured_port:
        return DEFAULT_UDP_PORT

    try:
        udp_port = int(configured_port)
    except ValueError as exc:
        raise ValueError("CSI_UDP_PORT must be an integer.") from exc

    if not 1 <= udp_port <= 65535:
        raise ValueError("CSI_UDP_PORT must be between 1 and 65535.")

    return udp_port


def get_next_trial_number(
    directory: Path,
    scenario_id: str,
    location: str,
    user_id: str,
    esp_id: str,
) -> int:
    pattern = re.compile(
        rf"^{re.escape(scenario_id)}_{re.escape(location)}_{re.escape(user_id)}_"
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


def create_socket(udp_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.settimeout(SOCKET_TIMEOUT_SECONDS)
    sock.bind((UDP_IP, udp_port))
    return sock


def prompt_session_metadata() -> tuple[str, str, str, float]:
    print("\n* Scenario ID")
    print("* 1, 2, 3, etc.")
    scenario_id = prompt_identifier("* * Enter scenario: ")

    print("\n* Location")
    print(" * Location label in x-y format (e.g., A-4, D-2, F-10)")
    location = prompt_location()

    print("\n* User ID")
    print("* 00 - Nobody")
    print("* 01 - Pedro")
    user_id = prompt_user_id()

    print("\n* Collection Duration")
    duration_seconds = prompt_run_duration_seconds()

    return user_id, scenario_id, location, duration_seconds


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


def get_esp_id(esp_mac: str) -> str:
    esp_id = ESP_MAC_MAP.get(esp_mac)

    if esp_id is None:
        esp_id = esp_mac.replace(":", "")
        print(f"Warning: unknown ESP MAC {esp_mac}. Using fallback ID {esp_id}.")

    return esp_id


def main() -> int:
    try:
        user_id, scenario_id, location, duration_seconds = prompt_session_metadata()
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 1

    try:
        udp_port = resolve_udp_port()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1

    save_directory = resolve_save_directory()
    save_directory.mkdir(parents=True, exist_ok=True)

    try:
        sock = create_socket(udp_port)
    except OSError as exc:
        print(
            f"Failed to start UDP server on {UDP_IP}:{udp_port}: {exc}. "
            "Another process is already using that port. Stop it or set CSI_UDP_PORT to a free UDP port.",
        )
        return 1

    packet_count_by_esp: dict[str, int] = {}
    trial_by_esp: dict[str, int] = {}
    file_path_by_esp: dict[str, Path] = {}

    session_timestamp = time.strftime("%d-%m_%H-%M-%S")

    print(f"\nUDP server listening on {UDP_IP}:{udp_port}")
    print(f"Saving CSI frames to: {save_directory}")

    if duration_seconds > 0:
        print(f"Collection will run for {duration_seconds} second(s)")
    else:
        print("Collection will run indefinitely (press Ctrl+C to stop)")

    input("Press ENTER to run")
    start_time = time.monotonic()

    print("----------------------------------------------")

    try:
        while True:
            if duration_seconds > 0:
                elapsed_seconds = time.monotonic() - start_time
                if elapsed_seconds >= duration_seconds:
                    print("\n----------------------------------------------")
                    print(
                        f"Duration of {duration_seconds} second(s) reached. Stopping collection.",
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
            esp_id = get_esp_id(esp_mac)

            packet_count_by_esp[esp_id] = packet_count_by_esp.get(esp_id, 0) + 1
            print(f"Data received from: {esp_mac}")
            print(f"ESP {esp_id} - Packets received: {packet_count_by_esp[esp_id]}")

            if esp_id not in trial_by_esp:
                trial_by_esp[esp_id] = get_next_trial_number(
                    save_directory,
                    scenario_id,
                    location,
                    user_id,
                    esp_id,
                )
                file_path_by_esp[esp_id] = save_directory / (
                    f"{scenario_id}_{location}_{user_id}_{esp_id}_"
                    f"{trial_by_esp[esp_id]:02d}_{session_timestamp}.csv"
                )

                print(
                    f"ESP {esp_id} - Using trial {trial_by_esp[esp_id]:02d} "
                    f"for {scenario_id}/{location}/{user_id}/{esp_id}",
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
