"""Shared UT-HAR benchmark model registry and constructors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from original_baseline.UT_HAR_model import (
    UT_HAR_BiLSTM,
    UT_HAR_CNN_GRU,
    UT_HAR_GRU,
    UT_HAR_LSTM,
    UT_HAR_LeNet,
    UT_HAR_MLP,
    UT_HAR_RNN,
    UT_HAR_ResNet18,
    UT_HAR_ResNet50,
    UT_HAR_ResNet101,
    UT_HAR_ViT,
)


@dataclass(frozen=True)
class BenchmarkModelSpec:
    canonical_name: str
    original_model_name: str
    epoch_policy: int
    input_mode: str
    exists_in_original_baseline: bool = True


class ChannelDimensionAdapter(nn.Module):
    """Inject a channel dimension for original 2D benchmark models."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.module(x)


ORIGINAL_UT_HAR_MODEL_SPECS = {
    "MLP": BenchmarkModelSpec(
        canonical_name="MLP",
        original_model_name="MLP",
        epoch_policy=200,
        input_mode="flat_or_sequence",
    ),
    "LeNet": BenchmarkModelSpec(
        canonical_name="LeNet",
        original_model_name="LeNet",
        epoch_policy=200,
        input_mode="channel_first_2d",
    ),
    "ResNet18": BenchmarkModelSpec(
        canonical_name="ResNet18",
        original_model_name="ResNet18",
        epoch_policy=200,
        input_mode="channel_first_2d",
    ),
    "ResNet50": BenchmarkModelSpec(
        canonical_name="ResNet50",
        original_model_name="ResNet50",
        epoch_policy=200,
        input_mode="channel_first_2d",
    ),
    "ResNet101": BenchmarkModelSpec(
        canonical_name="ResNet101",
        original_model_name="ResNet101",
        epoch_policy=200,
        input_mode="channel_first_2d",
    ),
    "RNN": BenchmarkModelSpec(
        canonical_name="RNN",
        original_model_name="RNN",
        epoch_policy=3000,
        input_mode="sequence",
    ),
    "GRU": BenchmarkModelSpec(
        canonical_name="GRU",
        original_model_name="GRU",
        epoch_policy=200,
        input_mode="sequence",
    ),
    "LSTM": BenchmarkModelSpec(
        canonical_name="LSTM",
        original_model_name="LSTM",
        epoch_policy=200,
        input_mode="sequence",
    ),
    "BiLSTM": BenchmarkModelSpec(
        canonical_name="BiLSTM",
        original_model_name="BiLSTM",
        epoch_policy=200,
        input_mode="sequence",
    ),
    "CNN+GRU": BenchmarkModelSpec(
        canonical_name="CNN+GRU",
        original_model_name="CNN+GRU",
        epoch_policy=200,
        input_mode="sequence",
    ),
    "ViT": BenchmarkModelSpec(
        canonical_name="ViT",
        original_model_name="ViT",
        epoch_policy=200,
        input_mode="channel_first_2d",
    ),
}

_NORMALIZED_NAME_TO_CANONICAL = {
    "mlp": "MLP",
    "lenet": "LeNet",
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn+gru": "CNN+GRU",
    "cnn_gru": "CNN+GRU",
    "cnn-gru": "CNN+GRU",
    "vit": "ViT",
}


def normalize_benchmark_model_name(model_name: str) -> str:
    normalized_key = model_name.strip().replace(" ", "").lower()
    if normalized_key not in _NORMALIZED_NAME_TO_CANONICAL:
        raise ValueError(f"Unsupported benchmark model name: {model_name}")
    return _NORMALIZED_NAME_TO_CANONICAL[normalized_key]


def list_original_uthar_model_names() -> list[str]:
    return list(ORIGINAL_UT_HAR_MODEL_SPECS.keys())


def get_original_uthar_model_spec(model_name: str) -> BenchmarkModelSpec:
    canonical_name = normalize_benchmark_model_name(model_name)
    return ORIGINAL_UT_HAR_MODEL_SPECS[canonical_name]


def build_benchmark_model(model_name: str) -> nn.Module:
    canonical_name = normalize_benchmark_model_name(model_name)
    if canonical_name == "MLP":
        return UT_HAR_MLP()
    if canonical_name == "LeNet":
        return ChannelDimensionAdapter(UT_HAR_LeNet())
    if canonical_name == "ResNet18":
        return ChannelDimensionAdapter(UT_HAR_ResNet18())
    if canonical_name == "ResNet50":
        return ChannelDimensionAdapter(UT_HAR_ResNet50())
    if canonical_name == "ResNet101":
        return ChannelDimensionAdapter(UT_HAR_ResNet101())
    if canonical_name == "RNN":
        return UT_HAR_RNN()
    if canonical_name == "GRU":
        return UT_HAR_GRU()
    if canonical_name == "LSTM":
        return UT_HAR_LSTM()
    if canonical_name == "BiLSTM":
        return UT_HAR_BiLSTM()
    if canonical_name == "CNN+GRU":
        return UT_HAR_CNN_GRU()
    if canonical_name == "ViT":
        return ChannelDimensionAdapter(UT_HAR_ViT())
    raise ValueError(f"Unsupported benchmark model name: {model_name}")

