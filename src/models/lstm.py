"""LSTM baseline definitions for project-specific experiments.

Rubric: model selection must be explainable in the report. LSTM is included as
the representative recurrent baseline with stronger memory capacity than GRU.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """UT-HAR LSTM classifier for CSI sequences with shape (batch, 250, 90)."""

    def __init__(
        self,
        input_size: int = 90,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        # Rubric: LSTM is a representative recurrent sequence baseline.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])
