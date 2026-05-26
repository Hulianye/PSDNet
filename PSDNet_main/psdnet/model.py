from __future__ import annotations

import torch
from torch import nn


class PSDNet(nn.Module):
    def __init__(self, n_bands: int, hidden_dim: int = 64, dropout: float = 0.1, nonnegative_output: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bands),
        )
        self.output_activation = nn.Softplus() if nonnegative_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_time_points = x.shape
        x = x.reshape(batch_size * num_channels, 1, num_time_points)
        x = self.encoder(x).squeeze(-1)
        x = x.reshape(batch_size, num_channels, -1).mean(dim=1)
        return self.output_activation(self.head(x))


class PSDNetV2(nn.Module):
    def __init__(self, n_bands: int, hidden_dim: int = 64, dropout: float = 0.1, num_segments: int = 8, nonnegative_output: bool = False):
        super().__init__()
        branch_dim = max(hidden_dim // 2, 16)
        self.num_segments = num_segments
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Conv1d(16, branch_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(num_segments),
                )
                for kernel_size in (15, 31, 63)
            ]
        )
        merged_dim = branch_dim * len(self.branches) * num_segments
        self.channel_projector = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.ReLU(),
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bands),
        )
        self.output_activation = nn.Softplus() if nonnegative_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_time_points = x.shape
        x = x.reshape(batch_size * num_channels, 1, num_time_points)
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = x.flatten(start_dim=1)
        x = self.channel_projector(x)
        x = x.reshape(batch_size, num_channels, -1)
        weights = torch.softmax(self.channel_attention(x), dim=1)
        x = (x * weights).sum(dim=1)
        return self.output_activation(self.head(x))


def create_model(model_type: str, **kwargs) -> nn.Module:
    if model_type == "psdnet":
        return PSDNet(
            n_bands=kwargs["n_bands"],
            hidden_dim=kwargs.get("hidden_dim", 64),
            dropout=kwargs.get("dropout", 0.1),
            nonnegative_output=kwargs.get("nonnegative_output", False),
        )
    if model_type == "psdnet_v2":
        return PSDNetV2(
            n_bands=kwargs["n_bands"],
            hidden_dim=kwargs.get("hidden_dim", 64),
            dropout=kwargs.get("dropout", 0.1),
            num_segments=kwargs.get("num_segments", 8),
            nonnegative_output=kwargs.get("nonnegative_output", False),
        )
    raise ValueError(f"Unknown model_type: {model_type}")
