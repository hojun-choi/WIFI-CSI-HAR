"""CNN baseline definitions for project-specific experiments.

Rubric: model selection must be explainable in the report. CNN is kept as a
baseline for local time-subcarrier pattern extraction.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    """Small LeNet-style CNN for UT-HAR."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        # Rubric: CNN is the local time-subcarrier pattern baseline.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)
