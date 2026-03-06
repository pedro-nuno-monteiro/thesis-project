import os
import socket
import time

# Perguntar ID do usuário e local
print("\n* User ID")
print("* 00 - Ninguém")
print("* 01 - Pedro")
user_id = input("* * Enter user ID: ").strip()

print("\n* Activity ID")
print("* 00 - Empty Room")
print("* 01 - Walking")
activity = input("* * Enter activity: ").strip()

# print("\n* Place ID")
# print("* 01 - Gab. Pedro (GP)")
# print("* 02 - Gab. Rafa  (GR)")
# print("* 03 - Gab. Lab.  (GL)")
# print("* 04 - GP + GL")
# print("* 05 - Lab. ")
# place = input("* * Enter location: ").strip()

print("\n* Scenario ID")
print("* 11 - Cen. 1, 2.4 GHz")
print("* 12 - Cen. 1, 5.0 GHz")
print("* 21 - Cen. 2, 2.4 GHz")
print("* 22 - Cen. 2, 5.0 GHz")
scenario = input("* * Enter scenario: ").strip()

print("\n* Collection Duration")
duration_minutes = input("* * Enter duration in minutes (0 for unlimited): ").strip()
try:
    duration_minutes = float(duration_minutes)
    duration_minutes = max(duration_minutes, 0)
except ValueError:
    print("Invalid duration. Setting to unlimited.")
    duration_minutes = 0

# Mapeamento de MACs
esp_mac_map = {
    "90:38:0C:EA:D3:78": "01",
    "90:38:0C:EA:D4:CC": "02",
    "C4:DE:E2:C0:98:E8": "03",
    "90:38:0C:EA:D5:04": "04",
    "D0:CF:13:ED:B7:D8": "05",
    "D0:CF:13:ED:9A:2C": "06",
    "D0:CF:13:ED:9A:8C": "07",
}

# Packet count tracking for each ESP
esp_packet_count = {}

# CSV HEADER (if needed)
# CSV_HEADER = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n"

# Configuração do servidor UDP
UDP_IP = "0.0.0.0"
UDP_PORT = 5001
save_directory = "/home/isac/Desktop/csi_frames"
os.makedirs(save_directory, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
sock.bind((UDP_IP, UDP_PORT))

print(f"\nUDP server on {UDP_IP}:{UDP_PORT}")
if duration_minutes > 0:
    print(f"Collection will run for {duration_minutes} minute(s)")
else:
    print("Collection will run indefinitely (press Ctrl+C to stop)")
input("Press ENTER to Run")
print("----------------------------------------------")

start_time = time.time()
timestamp = time.strftime("%Y-%m-%d_%H-%M")

while True:
    # Check if duration limit has been reached
    if duration_minutes > 0:
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes >= duration_minutes:
            print("\n----------------------------------------------")
            print(f"Duration of {duration_minutes} minute(s) reached. Stopping collection.")
            print("----------------------------------------------")
            break

    data, addr = sock.recvfrom(4096)

    try:
        decoded_data = data.decode("utf-8").strip()
        esp_mac, csi_data = decoded_data.split(",", 1)
    except ValueError:
        print(f"Dados inválidos recebidos de {addr}: {decoded_data}")
        continue

    esp_mac = esp_mac.upper()
    print("Data received from:", esp_mac)

    esp_id = esp_mac_map.get(esp_mac, esp_mac.replace(":", ""))

    # Increment packet count for this ESP
    if esp_id not in esp_packet_count:
        esp_packet_count[esp_id] = 0
    esp_packet_count[esp_id] += 1
    print(f"ESP {esp_id} - Packets received: {esp_packet_count[esp_id]}")

    filename = os.path.join(
        save_directory,
        f"{scenario}_{user_id}_{activity}_{esp_id}_{timestamp}.csv"
    )

    with open(filename, "a") as file:
        file.write(f"{csi_data}\n")

    print(f"CSI saved: ESP {esp_id} -> {os.path.basename(filename)}")
    print("----------------------------------------------")
