"""Recording-quality diagnostics (spec sheet section 8.3).

Retain FPS, valid-landmark percentage, rolling d_ref stability, gap count, and
longest gap. Assign reliable / borderline / discard using documented thresholds
(tuned on development recordings, not final test recordings).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import Config
from .signal import Sample


@dataclass
class QualityReport:
    fps: float
    valid_fraction: float
    gap_count: int
    longest_gap_s: float
    ref_cv: float            # coefficient of variation of d_ref over valid frames
    rating: str              # reliable | borderline | discard
    n_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def assess(samples: list[Sample], config: Config | None = None) -> QualityReport:
    cfg = config or Config()
    n = len(samples)
    if n == 0:
        return QualityReport(0.0, 0.0, 0, 0.0, 0.0, "discard", 0)

    valids = [s for s in samples if s.valid]
    valid_frac = len(valids) / n

    duration = samples[-1].t - samples[0].t
    fps = (n - 1) / duration if duration > 0 else 0.0

    # Gaps: maximal runs of invalid samples.
    gap_count, longest_gap = _gap_stats(samples)

    refs = np.array([s.d_ref for s in valids if s.d_ref is not None], dtype=float)
    if refs.size >= 2 and float(np.mean(refs)) > 0:
        ref_cv = float(np.std(refs) / np.mean(refs))
    else:
        ref_cv = 0.0

    rating = _rate(fps, valid_frac, longest_gap, ref_cv, cfg)
    return QualityReport(fps, valid_frac, gap_count, longest_gap, ref_cv, rating, n)


def _gap_stats(samples: list[Sample]):
    gap_count = 0
    longest = 0.0
    in_gap = False
    gap_start_t = None
    prev_t = samples[0].t
    for s in samples:
        if not s.valid:
            if not in_gap:
                in_gap = True
                gap_count += 1
                gap_start_t = prev_t
        else:
            if in_gap:
                longest = max(longest, s.t - gap_start_t)
                in_gap = False
        prev_t = s.t
    if in_gap:
        longest = max(longest, samples[-1].t - gap_start_t)
    return gap_count, longest


def _rate(fps, valid_frac, longest_gap, ref_cv, cfg: Config) -> str:
    if (
        valid_frac < cfg.min_valid_frac
        or fps < cfg.min_fps
        or longest_gap > cfg.max_longest_gap_s
    ):
        return "discard"
    if (
        valid_frac >= cfg.borderline_valid_frac
        and ref_cv <= cfg.ref_stability_cv_max
    ):
        return "reliable"
    return "borderline"
