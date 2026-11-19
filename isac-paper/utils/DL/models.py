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
    ROOM_ANCHOR_PAIRS,
)


class BandEncoder(nn.Module):
    """Encode one CSI band into a fixed-width latent representation."""

    def __init__(
        self,
        n_anchors: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Build the convolutional encoder for a specified number of anchors."""
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
        """Encode a batch of band-specific CSI windows."""
        return self.features(inputs)


class RoomEncoder(BandEncoder):
    """Encode one room's anchor-pair channels with the unchanged band encoder."""


def _safe_branch_name(name: str) -> str:
    """Convert a display band name to a valid ``ModuleDict`` key."""
    return "b_" + name.replace(".", "_").replace(" ", "_").replace("-", "_")


class DualBandCNN(nn.Module):
    """Run one CNN branch per band and fuse their latent features."""

    def __init__(
        self,
        branch_channels: dict[str, int],
        n_classes: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Build one encoder per band and the shared classification head."""
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
        """Encode each band, concatenate its latent vector, and predict logits."""
        latents = [
            self.branches[_safe_branch_name(band)](inputs[band]) for band in self._order
        ]
        return self.head(torch.cat(latents, dim=1))


class RoomStackedCNN(nn.Module):
    """Encode fixed room branches and fuse their latent representations."""

    def __init__(
        self,
        branch_channels: dict[str, int],
        n_classes: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Build one unchanged encoder per configured room and a wider fusion head."""
        super().__init__()
        params = params or {}
        expected_rooms = tuple(ROOM_ANCHOR_PAIRS)
        observed_rooms = set(branch_channels)
        if observed_rooms != set(expected_rooms):
            missing = sorted(set(expected_rooms) - observed_rooms)
            unexpected = sorted(observed_rooms - set(expected_rooms))
            raise ValueError(
                "RoomStackedCNN branch_channels must match ROOM_ANCHOR_PAIRS; "
                f"missing={missing}, unexpected={unexpected}."
            )

        latent_dim = int(params.get("latent_dim", CNN_LATENT_DIM))
        head_hidden = int(params.get("head_hidden", CNN_HEAD_HIDDEN))
        dropout = float(params.get("dropout", CNN_DROPOUT))
        # The configuration order is authoritative even if the runtime arrays
        # were supplied through a mapping constructed in a different order.
        self._order = list(expected_rooms)
        self.branches = nn.ModuleDict(
            {
                _safe_branch_name(room): RoomEncoder(branch_channels[room], params)
                for room in expected_rooms
            }
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * len(expected_rooms), head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate room latents in configured room order and predict logits."""
        latents = [
            self.branches[_safe_branch_name(room)](inputs[room])
            for room in self._order
        ]
        return self.head(torch.cat(latents, dim=1))
