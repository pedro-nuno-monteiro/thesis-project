"""PyTorch model definitions for the deep-learning experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from utils.config import (
    CNN_CONV1_FILTERS,
    CNN_CONV2_FILTERS,
    CNN_DROPOUT,
    CNN_HEAD_HIDDEN,
    CNN_LATENT_DIM,
)


class BandEncoder(nn.Module):
    """Encode one CSI band into a fixed-width latent representation."""

    def __init__(
        self,
        n_anchors: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        params = params or {}
        conv1_filters = int(params.get("conv1_filters", CNN_CONV1_FILTERS))
        conv2_filters = int(params.get("conv2_filters", CNN_CONV2_FILTERS))
        latent_dim = int(params.get("latent_dim", CNN_LATENT_DIM))
        self.features = nn.Sequential(
            nn.Conv2d(n_anchors, conv1_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv2_filters, latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


def _safe_branch_name(name: str) -> str:
    return "b_" + name.replace(".", "_").replace(" ", "_").replace("-", "_")


class DualBandCNN(nn.Module):
    """Run one CNN branch per band and fuse their latent features."""

    def __init__(
        self,
        branch_channels: dict[str, int],
        n_classes: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        params = params or {}
        latent_dim = int(params.get("latent_dim", CNN_LATENT_DIM))
        head_hidden = int(params.get("head_hidden", CNN_HEAD_HIDDEN))
        dropout = float(params.get("dropout", CNN_DROPOUT))
        self._order = list(branch_channels)
        self.branches = nn.ModuleDict(
            {
                _safe_branch_name(band): BandEncoder(n_anchors, params)
                for band, n_anchors in branch_channels.items()
            }
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * len(branch_channels), head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        latents = [
            self.branches[_safe_branch_name(band)](inputs[band]) for band in self._order
        ]
        return self.head(torch.cat(latents, dim=1))
