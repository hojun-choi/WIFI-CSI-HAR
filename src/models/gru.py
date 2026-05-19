"""GRU baseline definitions for project-specific experiments.

Rubric: model selection must be explainable in the report. GRU is the
lightweight sequence baseline for the first dry-run and later comparisons.
"""

from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    """UT-HAR GRU classifier for CSI sequences with shape (batch, 250, 90)."""

    def __init__(
        self,
        input_size: int = 90,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        # Rubric: GRU is a lightweight sequence baseline for CSI time-series
        # and is a safe first model before broader architecture comparison.
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        logits = self.classifier(hidden[-1])
        return logits
