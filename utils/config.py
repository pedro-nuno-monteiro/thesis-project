"""Project-wide experiment defaults.

Keep operational settings here so notebook and command-line entry points use the
same protocol, data-loader, plotting, and model-capacity configuration.
"""

from __future__ import annotations

TRIALS_FOR_TRAINING_PROTOCOLS = ("01",)
SEEDS = (42, 43, 44)

CUDA_BATCH_SIZE = 256
CUDA_NUM_WORKERS = 8
CUDA_PIN_MEMORY = True
CUDA_PERSISTENT_WORKERS = True
CUDA_PREFETCH_FACTOR = 4

CPU_BATCH_SIZE = 64
CPU_NUM_WORKERS = 0
CPU_PIN_MEMORY = False
CPU_PERSISTENT_WORKERS = False

MAX_EPOCH_SECONDS = 3600.0

PLOT_DPI = 200
PLOT_FORMAT = "png"

# Capacity is intentionally frozen after the ablation study.
CNN_CONV1_FILTERS = 32
CNN_CONV2_FILTERS = 64
CNN_LATENT_DIM = 128
CNN_HEAD_HIDDEN = 256
CNN_DROPOUT = 0.3

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
