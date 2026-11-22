"""Frequency-band extraction for existing 4-second B0 EEG epochs.

The B0 epochs are already causally 1-30Hz bandpassed at 250 Hz; these
filters further restrict spectral content while preserving (N,C,1000).
Reflection padding reduces (not eliminates) epoch-edge transients.
This zero-phase offline diagnostic does not model streaming latency.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def make_sos(low_hz: float, high_hz: float, fs_hz: float = 250., order: int = 4):
    if not (0 < low_hz < high_hz < fs_hz / 2):
        raise ValueError(f"Invalid bandpass {low_hz}..{high_hz} at {fs_hz} Hz")
    if order < 1:
        raise ValueError("Butterworth order must be positive")
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_hz, output="sos")


def filter_epochs(X: np.ndarray, sos: np.ndarray, pad_samples: int = 250) -> np.ndarray:
    """Filter mini-batch [N,C,T] independently, without changing sample count.

    Reflection uses data ONLY inside each 4-second window; no test/train leakage.
    No normalisation is performed here: each fold separately fits TRAIN-only Z-score.
    """
    if X.ndim != 3 or X.shape[-1] < 20:
        raise ValueError(f"Expected EEG [N,C,T] with T>=20, got {X.shape}")
    if pad_samples <= 0 or pad_samples >= X.shape[-1]:
        raise ValueError("pad_samples must be between 1 and T-1")
    if not np.isfinite(X).all():
        raise ValueError("EEG contains NaN/Inf")
    # Float64 only for one mini-batch to preserve numerical stability.
    extended = np.pad(np.asarray(X, dtype=np.float64),
                      ((0, 0), (0, 0), (pad_samples, pad_samples)), mode="reflect")
    filtered = sosfiltfilt(sos, extended, axis=-1, padtype=None)
    result = filtered[..., pad_samples:-pad_samples].astype(np.float32)
    if result.shape != X.shape or not np.isfinite(result).all():
        raise RuntimeError("Filtered EEG has wrong shape or nonfinite values")
    return np.ascontiguousarray(result)
