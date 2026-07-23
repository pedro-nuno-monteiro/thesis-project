"""Shared paths and experiment constants used by both notebooks."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = "/disco1500gb/Pedro_Data"
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "results"

ESP_IDS_BY_BAND = {
    "2.4 GHz": (*range(1, 6), *range(7, 11)),
    "5 GHz": tuple(range(11, 21)),
    "Fusion": (*range(1, 6), *range(7, 21)),
}
BANDS_TO_RUN = tuple(ESP_IDS_BY_BAND)
ANCHOR_GROUPS = {
    band: [f"esp_{esp_id:02d}" for esp_id in ESP_IDS_BY_BAND[band]]
    for band in ("2.4 GHz", "5 GHz")
}
ROOM_ANCHOR_PAIRS: dict[str, list[tuple[str, str]]] = {
    "Room 1": [
        ("esp_07", "esp_17"),
        ("esp_08", "esp_18"),
        ("esp_09", "esp_19"),
        ("esp_10", "esp_20"),
    ],
    "Room 2": [
        ("esp_01", "esp_11"),
        ("esp_02", "esp_12"),
        ("esp_03", "esp_13"),
    ],
    "Room 3": [
        ("esp_04", "esp_14"),
        ("esp_05", "esp_15"),
    ],
}
ROOM_CHANNEL_COUNTS = {
    room: len(anchor_pairs)
    for room, anchor_pairs in ROOM_ANCHOR_PAIRS.items()
}

EXPECTED_SUBCARRIERS = {"2.4 GHz": 50, "5 GHz": 56}
EXPECTED_ANCHORS = {
    band: len(anchor_group)
    for band, anchor_group in ANCHOR_GROUPS.items()
}

EMPTY_ROOM_LOCATION = "Z-0"
ROOM_1_COLUMNS = range(1, 10)
ROOM_2_A_COLUMNS = {13, 14}
ROOM_2_BC_COLUMNS = range(10, 15)
ROOM_3_EF_COLUMNS = range(10, 14)

DEFAULT_WINDOW_SIZE = 60
DEFAULT_OVERLAP_SIZE = 0

# only used in paper notebook, but kept here for consistency with the paper notebook
WINDOW_CONFIGS = [
    ("win30-step30", 30, 30),
    ("win60-step30", 60, 30),
    ("win60-step60", 60, 60),
    ("win120-step60", 120, 60),
    ("win120-step120", 120, 120),
]

# trials to use in training set
TRIALS_FOR_TRAINING_PROTOCOLS = ("01",)
SEEDS = (42, 43, 44)

# CNN model parameters
ARCHITECTURE = "band_branch"  # "band_branch" | "room_stacked"
CUDA_BATCH_SIZE = 256
CUDA_NUM_WORKERS = 8
CUDA_PIN_MEMORY = True
CUDA_PERSISTENT_WORKERS = True
CUDA_PREFETCH_FACTOR = 4

# CPU model parameters
CPU_BATCH_SIZE = 64
CPU_NUM_WORKERS = 0
CPU_PIN_MEMORY = False
CPU_PERSISTENT_WORKERS = False

# Classical ML grid-search execution and diagnostic settings
GRID_SEARCH_N_JOBS = 4
GRID_SEARCH_HEARTBEAT_SECONDS = 60.0

MAX_EPOCH_SECONDS = 3600.0

PLOT_DPI = 200
PLOT_FORMAT = "png"

# Capacity is intentionally frozen after the ablation study.
CNN_CONV1_FILTERS = 32
CNN_CONV2_FILTERS = 64
CNN_LATENT_DIM = 128
CNN_HEAD_HIDDEN = 256
CNN_DROPOUT = 0.3

# Default CNN parameters used in DL notebook
DEFAULT_CNN_PARAMS = {
    "model_label": "CNN",
    "conv1_filters": CNN_CONV1_FILTERS,
    "conv2_filters": CNN_CONV2_FILTERS,
    "latent_dim": CNN_LATENT_DIM,
    "head_hidden": CNN_HEAD_HIDDEN,
    "dropout": CNN_DROPOUT,
    "cuda_batch_size": CUDA_BATCH_SIZE,
    "cuda_num_workers": CUDA_NUM_WORKERS,
    "cuda_pin_memory": CUDA_PIN_MEMORY,
    "cuda_persistent_workers": CUDA_PERSISTENT_WORKERS,
    "cuda_prefetch_factor": CUDA_PREFETCH_FACTOR,
    "cpu_batch_size": CPU_BATCH_SIZE,
    "cpu_num_workers": CPU_NUM_WORKERS,
    "cpu_pin_memory": CPU_PIN_MEMORY,
    "cpu_persistent_workers": CPU_PERSISTENT_WORKERS,
    "max_epoch_seconds": MAX_EPOCH_SECONDS,
}
