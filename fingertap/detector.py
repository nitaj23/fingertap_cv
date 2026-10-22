"""Hysteresis tap-detection state machine (spec sheet section 5.4).

A single threshold on d_norm chatters near the threshold and over-counts. A
two-state hysteresis detector with two thresholds avoids this:

  - OPEN -> CLOSED when d_norm falls below T_low
  - while CLOSED, record the candidate close at the local minimum
  - CLOSED -> OPEN when d_norm rises above T_high (T_high > T_low)
  - count a tap ONLY after the full OPEN -> CLOSED -> OPEN cycle completes
  - enforce minimum closed time and minimum cycle duration (refractory)

Detection SUSPENDS during invalid gaps; a cycle spanning a gap is marked
uncertain and excluded from the confident count (safety requirement:
predicted/interpolated samples must never independently create a tap).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class TapEvent:
    """One detected tap (completed OPEN->CLOSED->OPEN cycle)."""

    close_t: float          # timestamp of the contact (local minimum)
    close_norm: float       # d_norm at contact
    open_norm: float        # local-open level preceding the contact
    amplitude: float        # open_norm - close_norm
    cycle_start_t: float    # when it entered CLOSED
    cycle_end_t: float      # when it returned to OPEN
    uncertain: bool         # True if the cycle spanned an invalid gap


class TapDetector:
    """Feed validated samples in time order; collect TapEvents."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"

    def __init__(self, t_low: float, t_high: float, config: Config | None = None):
        if not t_high > t_low:
            raise ValueError("T_high must be greater than T_low")
        self.cfg = config or Config()
        self.t_low = t_low
        self.t_high = t_high
        self.reset()

    def reset(self):
        self.state = self.OPEN
        self.events: list[TapEvent] = []
        # local-open level tracked while OPEN (the peak before a contact)
        self._open_level: float | None = None
        # contact tracking while CLOSED
        self._min_norm: float | None = None
        self._min_t: float | None = None
        self._enter_closed_t: float | None = None
        self._closed_spanned_gap = False
        self._last_tap_end_t: float | None = None

    def update(self, t: float, d_norm: float | None, valid: bool):
        """Process one sample. Returns a TapEvent if a tap completed, else None."""
        if not valid or d_norm is None:
            # Suspend: if we were mid-cycle, mark it uncertain. We do NOT change
            # state here so the detector resumes cleanly when valid data returns.
            if self.state == self.CLOSED:
                self._closed_spanned_gap = True
            else:
                # An invalid gap during OPEN invalidates the current open level
                # too, so a cycle straddling it is flagged.
                self._open_level = None
            return None

        if self.state == self.OPEN:
            # Track the local-open level (highest d_norm seen while open).
            if self._open_level is None or d_norm > self._open_level:
                self._open_level = d_norm
            if d_norm < self.t_low:
                self.state = self.CLOSED
                self._enter_closed_t = t
                self._min_norm = d_norm
                self._min_t = t
                self._closed_spanned_gap = (self._open_level is None)
            return None

        # state == CLOSED (_min_norm / _min_t / _enter_closed_t set on entry)
        if d_norm < self._min_norm:
            self._min_norm = d_norm
            self._min_t = t
        if d_norm > self.t_high:
            return self._complete_cycle(t)
        return None

    def _complete_cycle(self, t: float):
        cfg = self.cfg
        cycle_start = self._enter_closed_t

        # Refractory / debounce: reject cycles shorter than the minimum
        # physically plausible tap (contact + return).
        reject = (t - cycle_start) < max(cfg.min_cycle_s, cfg.min_closed_s)

        # Return to OPEN regardless of acceptance.
        self.state = self.OPEN
        open_level = self._open_level
        close_norm = self._min_norm
        close_t = self._min_t
        spanned_gap = self._closed_spanned_gap or (open_level is None)

        # Prepare for next cycle's open level: start fresh from this frame.
        self._open_level = None
        self._min_norm = None
        self._min_t = None
        self._enter_closed_t = None
        self._closed_spanned_gap = False

        if reject:
            return None

        self._last_tap_end_t = t
        amp = (open_level - close_norm) if open_level is not None else 0.0
        ev = TapEvent(
            close_t=close_t,
            close_norm=close_norm,
            open_norm=open_level if open_level is not None else close_norm,
            amplitude=amp,
            cycle_start_t=cycle_start,
            cycle_end_t=t,
            uncertain=spanned_gap,
        )
        self.events.append(ev)
        return ev

    @property
    def confident_events(self) -> list[TapEvent]:
        return [e for e in self.events if not e.uncertain]


def derive_thresholds(d_norm_values, config: Config):
    """Compute (T_low, T_high) from robust percentiles of observed d_norm.

    Falls back to the fixed config absolutes when there is too little data
    (spec section 5.4: adapt to the participant rather than fixed absolutes).
    """
    import numpy as np

    vals = np.asarray([v for v in d_norm_values if v is not None], dtype=float)
    if vals.size < 10:
        return config.t_low, config.t_high
    t_low = float(np.percentile(vals, config.t_low_pct))
    t_high = float(np.percentile(vals, config.t_high_pct))
    if not t_high > t_low:  # degenerate signal; back off to defaults
        return config.t_low, config.t_high
    return t_low, t_high
