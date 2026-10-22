"""Temporal smoothing (spec sheet section 5.3).

Runtime MUST use a CAUSAL filter so the live display and detector never depend
on future frames. We use an exponential moving average -- the same idiom the
foundational DroneTestv2.3.py loop uses on throttle/pitch/yaw.

Centered Savitzky-Golay and zero-phase filtfilt are provided for OFFLINE
analysis/figures ONLY and must not be used as live filters (they look ahead).
"""

from __future__ import annotations

import numpy as np


class CausalEMA:
    """Causal exponential moving average for the live signal.

    Invalid frames (d_norm is None) are passed through as None and do NOT
    advance the filter state, so a gap cannot smear smoothed values across it.
    """

    def __init__(self, alpha: float = 0.4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._y: float | None = None

    def reset(self):
        self._y = None

    def update(self, x: float | None) -> float | None:
        if x is None:
            return None
        if self._y is None:
            self._y = x
        else:
            self._y = self._y * (1 - self.alpha) + x * self.alpha
        return self._y


# --- Offline-only filters (figures / analysis, never the live path) ---

def savgol_offline(signal, window: int = 9, poly: int = 2):
    """Centered Savitzky-Golay smoothing. OFFLINE ONLY (non-causal)."""
    from scipy.signal import savgol_filter

    sig = np.asarray(signal, dtype=float)
    if len(sig) < window:
        return sig.copy()
    if window % 2 == 0:
        window += 1
    return savgol_filter(sig, window, poly)


def filtfilt_offline(signal, cutoff_hz: float, fs: float, order: int = 2):
    """Zero-phase Butterworth low-pass. OFFLINE ONLY (non-causal)."""
    from scipy.signal import butter, filtfilt

    sig = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    pad = 3 * max(len(a), len(b))
    if len(sig) <= pad:
        return sig.copy()
    return filtfilt(b, a, sig)
