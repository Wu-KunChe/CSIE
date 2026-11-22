"""Compact EEGNet-8,2-style binary EEG model.

Input: tensor [batch, EEG_channels, time_samples] (e.g. [B, 2, 500]).
Output: *one logit* per epoch (BCEWithLogitsLoss, sigmoid only in evaluation).

Architecture follows Lawhern et al. 2018 EEGNet principle:
 temporal conv -> channel-spanning depthwise spatial conv ->
 depthwise+pointwise temporal separable conv -> flatten+linear.
This is a self-contained PyTorch implementation, not a pretrained model.
"""
from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, *, F1: int = 8,
                 D: int = 2, F2: int = 16, temporal_kernel: int = 64,
                 separable_kernel: int = 16, pool1: int = 4,
                 pool2: int = 8, dropout: float = 0.3,
                 batch_norm_momentum: float = 0.01,
                 spatial_max_norm: float = 1.0,
                 classifier_max_norm: float = 0.25):
        super().__init__()
        if n_channels < 1 or n_times < 32:
            raise ValueError('Invalid EEG channel count or epoch length')
        if min(F1, D, F2, temporal_kernel, separable_kernel, pool1, pool2) <= 0:
            raise ValueError('Network filters, kernels and pools must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.spatial_max_norm = float(spatial_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=batch_norm_momentum),
        )
        self.spatial = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                                 groups=F1, bias=False)
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(F1 * D, momentum=batch_norm_momentum),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2), groups=F1 * D, bias=False)
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F2, momentum=batch_norm_momentum),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        # Derive input size without assuming a specific EEG sampling rate/window.
        with torch.no_grad():
            old = self.training
            self.eval()
            shape = self._features(torch.zeros(1, n_channels, n_times)).shape
            self.train(old)
        flattened = int(shape[1] * shape[2] * shape[3])
        if flattened < 1:
            raise ValueError('Epoch too short for selected pooling configuration')
        self.classifier = nn.Linear(flattened, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,1,C,T]
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.block1(x)
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        return self.block2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_channels or x.shape[2] != self.n_times:
            raise ValueError(f'Expected EEG [batch,{self.n_channels},{self.n_times}], got {tuple(x.shape)}')
        features = self._features(x)
        return self.classifier(features.flatten(start_dim=1)).squeeze(1)

    @torch.no_grad()
    def constrain_weights(self) -> None:
        """EEGNet-like max-norm on spatial filters and classification weights."""
        if self.spatial_max_norm > 0:
            w = self.spatial.weight
            norms = torch.linalg.vector_norm(w.reshape(w.shape[0], -1), dim=1,
                                              keepdim=True).clamp_min(1e-12)
            factor = (self.spatial_max_norm / norms).clamp(max=1.0)
            w.mul_(factor.reshape(-1, 1, 1, 1))
        if self.classifier_max_norm > 0:
            w = self.classifier.weight
            norms = torch.linalg.vector_norm(w, dim=1, keepdim=True).clamp_min(1e-12)
            w.mul_((self.classifier_max_norm / norms).clamp(max=1.0))


def build_eegnet(config: dict, n_channels: int, n_times: int) -> EEGNet:
    options = {k: v for k, v in config.items() if k != 'variant'}
    return EEGNet(n_channels, n_times, **options)
