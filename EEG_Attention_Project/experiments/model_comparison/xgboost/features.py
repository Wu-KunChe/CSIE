"""Fixed per-epoch EEG features, computed WITHOUT examining class labels.

Input: B0-filtered EEG of shape (epochs, EEG_channels, 500), in microvolts.
Welch frequency resolution is 1 Hz for fs=250 Hz, nperseg=250. Short
2-second windows do not support robust very-low-frequency band estimates;
this preset intentionally uses theta, alpha and beta rather than delta.

No feature learning across subjects is performed. This transformation may
be applied before subject splitting without fitting a scaler on held-out data.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

FEATURE_NAMES = (
    'log10_theta_power', 'log10_alpha_power', 'log10_beta_power',
    'relative_theta_power', 'relative_alpha_power', 'relative_beta_power',
    'log10_theta_alpha_ratio', 'log10_theta_beta_ratio',
    'log10_alpha_beta_ratio', 'log10_total_1to30_power',
    'log10_rms_uV', 'log10_variance_uV2',
    'hjorth_mobility', 'hjorth_complexity',
)


def compute_features(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Return float32 array (N, C, 14); no labels or normalization fitted."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3 or X.shape[0] < 1:
        raise ValueError('Expected EEG array [N, channels, time]')
    samples = int(cfg['expected_samples'])
    fs = float(cfg['sampling_rate_hz'])
    nperseg = int(cfg['welch_nperseg'])
    noverlap = int(cfg['welch_noverlap'])
    eps = float(cfg.get('epsilon', 1e-12))
    if X.shape[-1] != samples or not np.isfinite(X).all():
        raise ValueError(f'Unexpected sample length or non-finite EEG: {X.shape}')
    if not (0 < noverlap < nperseg <= samples) or fs <= 0 or eps <= 0:
        raise ValueError('Invalid Welch/feature settings')
    f, pxx = welch(X, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   axis=-1, detrend='constant', scaling='density')
    # Welch PSD units are uV^2/Hz; multiplying by df integrates band power.
    df = float(f[1] - f[0])

    def power(name: str, include_end: bool = False) -> np.ndarray:
        low, high = map(float, cfg[name])
        mask = ((f >= low) & (f <= high if include_end else f < high))
        if not mask.any():
            raise ValueError(f'No Welch bins for {name}: {low}-{high} Hz')
        return (pxx[..., mask].sum(axis=-1) * df).clip(min=eps)

    theta = power('theta_hz')
    alpha = power('alpha_hz')
    beta = power('beta_hz', include_end=True)
    total = power('total_hz', include_end=True)
    # One epoch/channel only; these are non-learned deterministic statistics.
    variance = np.mean((X - X.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    rms = np.sqrt(np.mean(X ** 2, axis=-1))
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var_d1 = np.var(d1, axis=-1)
    var_d2 = np.var(d2, axis=-1)
    mobility = np.sqrt(var_d1 / np.maximum(variance, eps))
    mobility_d1 = np.sqrt(var_d2 / np.maximum(var_d1, eps))
    complexity = mobility_d1 / np.maximum(mobility, eps)
    features = [
        np.log10(theta), np.log10(alpha), np.log10(beta),
        theta / total, alpha / total, beta / total,
        np.log10(theta / alpha), np.log10(theta / beta),
        np.log10(alpha / beta), np.log10(total),
        np.log10(np.maximum(rms, np.sqrt(eps))),
        np.log10(np.maximum(variance, eps)),
        mobility, complexity,
    ]
    result = np.stack(features, axis=-1).astype(np.float32)
    if result.shape != (X.shape[0], X.shape[1], len(FEATURE_NAMES)):
        raise AssertionError('Unexpected spectral feature output shape')
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite features; inspect extreme amplitude epochs')
    return result


def flatten_channel_features(array: np.ndarray, indices: list[int],
                             channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Convert N x C x F into N x (selected_C*F) and names for XGBoost."""
    picked = np.ascontiguousarray(array[:, indices, :])
    cols = [f'{channel_names[k]}__{feature}'
            for k in indices for feature in FEATURE_NAMES]
    return picked.reshape(len(picked), -1), cols
