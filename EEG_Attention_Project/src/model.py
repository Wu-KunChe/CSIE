"""The learned CNN temporal feature extractor AND classifier live together here."""
from __future__ import annotations
import torch
from torch import nn


class CNN1D(nn.Module):
    """Compact temporal CNN. Input (batch, channels, 500); output raw LOGITS.

    The first convolution learns spatial/channel combinations + local temporal
    patterns; later layers learn more abstract temporal features.
    Do NOT add Sigmoid here when using BCEWithLogitsLoss.
    """
    def __init__(self, n_channels: int, filters=(32,64), kernels=(51,15), dropout=0.30):
        super().__init__()
        if n_channels < 1 or len(filters) != 2 or len(kernels) != 2:
            raise ValueError('Unexpected model dimensions')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, filters[0], kernel_size=kernels[0], padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters[1], 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(f'Expected (batch,channels,time), got {tuple(X.shape)}')
        return self.classifier(self.features(X)).squeeze(-1)


def build_model(model_config: dict, n_channels: int, seed: int | None = None) -> CNN1D:
    if seed is not None:
        torch.manual_seed(int(seed))  # seed BEFORE creating the conv layer weights
    if model_config.get('name', '').lower() != 'cnn1d':
        raise ValueError('Only cnn1d is implemented in the baseline')
    return CNN1D(n_channels=n_channels,
                 filters=tuple(model_config.get('conv_filters',(32,64))),
                 kernels=tuple(model_config.get('kernels',(51,15))),
                 dropout=float(model_config.get('dropout',.3)))
