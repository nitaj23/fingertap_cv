"""Central configuration for the finger-tapping pipeline.

Every parameter that affects a measurement lives here so a session can
serialize exactly what produced its numbers (spec sheet: reproducibility /
auditability non-functional requirements). No magic numbers elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field


# MediaPipe landmark indices (spec sheet section 3: required landmarks).
THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
PINKY_MCP = 17


@dataclass
class Config:
    """All tunable parameters for one capture/analysis session."""

    # --- Capture (spec section 5.1) ---
    cam_index: int = 0
    width: int = 640
    height: int = 480

    # --- Landmark reference length (spec section 5.2) ---
    # Default reference is wrist(0) -> middle MCP(9); alternate is index MCP(5)
    # -> pinky MCP(17), selectable for ablation without touching code.
    ref_a: int = WRIST
    ref_b: int = MIDDLE_MCP

    # d_ref sanity bounds (pixels) and max fractional jump between frames.
    ref_min_px: float = 20.0
    ref_max_jump_frac: float = 0.5
    # Light causal smoothing applied to d_ref before it normalizes d_raw.
    ref_ema_alpha: float = 0.3

    # Reject an implausible frame-to-frame jump in d_norm as a bad landmark.
    dnorm_max_jump: float = 0.9

    # --- Runtime causal smoothing of d_norm (spec section 5.3) ---
    smooth_alpha: float = 0.4  # exponential moving average factor

    # --- Hysteresis tap detector (spec section 5.4) ---
    # Defaults used when auto-calibration from percentiles is disabled.
    t_low: float = 0.35
    t_high: float = 0.60
    min_closed_s: float = 0.060   # minimum time spent below T_low
    min_cycle_s: float = 0.150    # refractory: minimum OPEN->CLOSED->OPEN duration

    # Derive thresholds from robust percentiles of observed d_norm instead of
    # the fixed absolutes above. Percentiles chosen inside the open/close band.
    auto_threshold: bool = True
    t_low_pct: float = 25.0
    t_high_pct: float = 55.0

    # --- Metrics (spec section 5.5) ---
    # Early/late windows as a fraction of accepted taps each.
    early_late_frac: float = 0.33

    # --- Quality thresholds (spec section 8.3) ---
    min_fps: float = 15.0
    min_valid_frac: float = 0.90        # >= -> not discard
    borderline_valid_frac: float = 0.97  # >= -> reliable (if other checks pass)
    max_longest_gap_s: float = 0.30
    ref_stability_cv_max: float = 0.08   # rolling d_ref coefficient of variation

    # --- MediaPipe detector ---
    detection_con: float = 0.7
    tracking_con: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
