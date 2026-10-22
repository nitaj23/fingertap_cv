"""Distance and normalization stage (spec sheet section 5.2).

The tap signal is the thumb-tip <-> index-tip separation, normalized by a hand
reference length that does NOT change during a tap so the signal is (to first
order) invariant to how far the hand is from the camera.

    d_raw(t)  = || p4 - p8 ||
    d_ref(t)  = || p0 - p9 ||          (default reference)
    d_norm(t) = d_raw(t) / d_ref(t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Config


def _lm_xy(lmList, idx):
    """Return (x, y) for landmark index idx.

    findPosition returns all 21 landmarks in order, so lmList[idx] is that
    landmark directly (IndexError if the list is short, caught by the caller).
    """
    row = lmList[idx]
    return float(row[1]), float(row[2])


def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


@dataclass
class Sample:
    """One processed frame of the tap signal."""

    t: float                 # perf_counter timestamp (seconds)
    d_raw: float | None      # raw pixel distance, None if invalid
    d_ref: float | None      # smoothed reference length, None if invalid
    d_norm: float | None     # normalized distance, None if invalid
    valid: bool              # usable this frame?
    reason: str = ""         # why invalid, for auditing


class SignalProcessor:
    """Turns per-frame landmark lists into validated normalized samples.

    Keeps a small amount of state (previous d_ref and d_norm) so it can flag
    implausible jumps and lightly smooth the reference length.
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self._ref_ema: float | None = None
        self._prev_norm: float | None = None

    def reset(self):
        self._ref_ema = None
        self._prev_norm = None

    def process(self, t: float, lmList, valid: bool) -> Sample:
        """Process one frame. `valid` is MediaPipe hand-presence from the detector."""
        if not valid or not lmList:
            # Flagged gap: detector will suspend counting here (FR-4).
            return Sample(t, None, None, None, False, "no_hand")

        cfg = self.cfg
        try:
            thumb = _lm_xy(lmList, 4)
            index = _lm_xy(lmList, 8)
            ref_a = _lm_xy(lmList, cfg.ref_a)
            ref_b = _lm_xy(lmList, cfg.ref_b)
        except IndexError:
            return Sample(t, None, None, None, False, "missing_landmark")

        d_raw = euclidean(thumb, index)
        d_ref_now = euclidean(ref_a, ref_b)

        # Reference must be large enough to divide by and stable frame-to-frame.
        if d_ref_now < cfg.ref_min_px:
            return Sample(t, d_raw, None, None, False, "ref_too_small")

        if self._ref_ema is not None:
            jump = abs(d_ref_now - self._ref_ema) / self._ref_ema
            if jump > cfg.ref_max_jump_frac:
                return Sample(t, d_raw, self._ref_ema, None, False, "ref_jump")
            self._ref_ema = (
                self._ref_ema * (1 - cfg.ref_ema_alpha) + d_ref_now * cfg.ref_ema_alpha
            )
        else:
            self._ref_ema = d_ref_now

        d_norm = d_raw / self._ref_ema

        # Reject an impossible instantaneous jump in the normalized signal.
        if self._prev_norm is not None and abs(d_norm - self._prev_norm) > cfg.dnorm_max_jump:
            return Sample(t, d_raw, self._ref_ema, None, False, "norm_jump")

        self._prev_norm = d_norm
        return Sample(t, d_raw, self._ref_ema, d_norm, True, "")
