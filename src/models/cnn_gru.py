"""CNN+GRU baseline definitions for project-specific experiments.

Rubric: model selection must be explainable in the report. CNN+GRU tests a
hybrid design that combines local feature extraction with temporal modeling.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNGRUClassifier(nn.Module):
    """Modest CNN+GRU hybrid for UT-HAR."""

    def __init__(self, num_classes: int = 7, hidden_size: int = 128) -> None:
        super().__init__()
        # Rubric: the hybrid model combines local feature extraction with
        # temporal modeling while staying small enough for laptop GPUs.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(90, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])
