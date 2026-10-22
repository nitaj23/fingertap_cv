"""Session metrics (spec sheet section 5.5).

  - Count: accepted complete cycles
  - Frequency: accepted taps / VALID assessment duration (gap time removed)
  - ITI: time between accepted close events
  - Timing variability: CV_ITI = std(ITI) / mean(ITI)
  - Amplitude: local open distance - close distance per cycle
  - Performance change: predefined early-vs-late differences

Never use 1 - CV (can be negative). A bounded display score uses 1/(1+CV_ITI),
labelled as derived.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import Config
from .detector import TapEvent


@dataclass
class SessionMetrics:
    tap_count: int
    frequency_hz: float
    valid_duration_s: float
    iti_mean_s: float
    iti_std_s: float
    cv_iti: float
    regularity_score: float      # 1/(1+CV_ITI), derived display score
    amplitude_mean: float
    amplitude_median: float
    early_frequency_hz: float
    late_frequency_hz: float
    freq_change_hz: float        # late - early
    early_amplitude_median: float
    late_amplitude_median: float
    amplitude_change_pct: float  # (late - early) / early * 100
    uncertain_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(
    events: list[TapEvent],
    valid_duration_s: float,
    config: Config | None = None,
) -> SessionMetrics:
    """Compute all session metrics from confident (non-uncertain) tap events."""
    cfg = config or Config()
    confident = [e for e in events if not e.uncertain]
    uncertain_count = sum(1 for e in events if e.uncertain)

    n = len(confident)
    close_times = np.array([e.close_t for e in confident], dtype=float)
    amps = np.array([e.amplitude for e in confident], dtype=float)

    frequency = _safe_div(n, valid_duration_s)

    if n >= 2:
        itis = np.diff(close_times)
        iti_mean = float(np.mean(itis))
        iti_std = float(np.std(itis))
        cv_iti = _safe_div(iti_std, iti_mean)
    else:
        iti_mean = iti_std = cv_iti = 0.0

    regularity = _safe_div(1.0, 1.0 + cv_iti) if n >= 2 else 0.0

    amp_mean = float(np.mean(amps)) if n else 0.0
    amp_median = float(np.median(amps)) if n else 0.0

    # Early vs late windows (spec section 5.5): first/last fraction of taps.
    early_freq = late_freq = 0.0
    early_amp_med = late_amp_med = 0.0
    amp_change_pct = 0.0
    if n >= 2:
        k = max(1, int(round(n * cfg.early_late_frac)))
        early = confident[:k]
        late = confident[-k:]
        early_freq = _window_frequency(early)
        late_freq = _window_frequency(late)
        early_amp_med = float(np.median([e.amplitude for e in early]))
        late_amp_med = float(np.median([e.amplitude for e in late]))
        amp_change_pct = _safe_div(late_amp_med - early_amp_med, early_amp_med) * 100.0

    return SessionMetrics(
        tap_count=n,
        frequency_hz=frequency,
        valid_duration_s=valid_duration_s,
        iti_mean_s=iti_mean,
        iti_std_s=iti_std,
        cv_iti=cv_iti,
        regularity_score=regularity,
        amplitude_mean=amp_mean,
        amplitude_median=amp_median,
        early_frequency_hz=early_freq,
        late_frequency_hz=late_freq,
        freq_change_hz=late_freq - early_freq,
        early_amplitude_median=early_amp_med,
        late_amplitude_median=late_amp_med,
        amplitude_change_pct=amp_change_pct,
        uncertain_count=uncertain_count,
    )


def _window_frequency(window: list[TapEvent]) -> float:
    """Tap rate within a window: its (n-1) intervals over their total span."""
    if len(window) < 2:
        return 0.0
    span = window[-1].close_t - window[0].close_t
    return _safe_div(len(window) - 1, span)
