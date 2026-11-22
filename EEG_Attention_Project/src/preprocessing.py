"""Fold-wise numerical scaling only. B0 bandpass and resampling already done.

No E3 Wavelet, ICA or denoiser implementations are included at this stage.
"""
from __future__ import annotations
import numpy as np


class ChannelwiseStandardizer:
    """Fit one mean/std per channel, pooling TRAIN epochs and time samples only."""
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = float(epsilon)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> 'ChannelwiseStandardizer':
        if X_train.ndim != 3 or not len(X_train):
            raise ValueError('Need nonempty TRAIN (N,C,T) EEG')
        mean = X_train.astype(np.float64).mean(axis=(0,2), keepdims=True)
        std = X_train.astype(np.float64).std(axis=(0,2), keepdims=True)
        self.mean_ = mean.astype(np.float32)
        self.std_ = np.maximum(std, self.epsilon).astype(np.float32)
        return self

    def fit_from_batches(self, train_batches) -> 'ChannelwiseStandardizer':
        """Memory-saving streaming fit; pass TRAIN data batches ONLY."""
        total = None
        total_sq = None
        count = 0
        for X in train_batches:
            X = np.asarray(X)
            if X.ndim != 3 or not X.size:
                raise ValueError('Each training batch must be (N,C,T)')
            x64 = X.astype(np.float64, copy=False)
            s = x64.sum(axis=(0,2), keepdims=True)
            sq = np.square(x64).sum(axis=(0,2), keepdims=True)
            if total is None:
                total, total_sq = s, sq
            else:
                if total.shape != s.shape:
                    raise ValueError('Number of EEG channels changed across batches')
                total += s
                total_sq += sq
            count += X.shape[0]*X.shape[2]
        if count == 0:
            raise ValueError('No training samples provided')
        mean = total / count
        variance = np.maximum(total_sq/count - np.square(mean), 0.0)
        self.mean_ = mean.astype(np.float32)
        self.std_ = np.maximum(np.sqrt(variance), self.epsilon).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('Must fit on TRAIN subjects before transform')
        if X.ndim != 3 or X.shape[1] != self.mean_.shape[1]:
            raise ValueError('EEG shape / channel mismatch')
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def to_metadata(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError('Scaler not fitted')
        return {'mean_per_channel': self.mean_.reshape(-1).tolist(),
                'std_per_channel': self.std_.reshape(-1).tolist()}
