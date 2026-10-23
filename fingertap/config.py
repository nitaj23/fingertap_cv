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
    # EMA is a low-pass filter: a small alpha heavily attenuates FAST taps
    # (their short swings get damped below the hysteresis band and are missed),
    # while slow exaggerated taps survive. 0.65 keeps enough high-frequency
    # amplitude to detect fast, consistent tapping; the hysteresis band still
    # absorbs the residual chatter.
    smooth_alpha: float = 0.65  # exponential moving average factor

    # --- Hysteresis tap detector (spec section 5.4) ---
    # Defaults used when auto-calibration from percentiles is disabled.
    t_low: float = 0.35
    t_high: float = 0.60
    min_closed_s: float = 0.060   # minimum time spent below T_low
    # Refractory: minimum OPEN->CLOSED->OPEN duration. This is a hard ceiling on
    # tap rate (1 / min_cycle_s). 0.150 capped detection at ~6.7 taps/s and
    # clipped genuinely fast tapping; 0.100 raises the ceiling to ~10 taps/s
    # while still rejecting sub-100 ms chatter.
    min_cycle_s: float = 0.100    # refractory: minimum OPEN->CLOSED->OPEN duration

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

    # --- Optional preprocessing experiments (spec section 9.1) ---
    # Config-gated course ablations, applied between resize and MediaPipe; all
    # default to no-ops so the runtime pipeline stays the minimal five-stage
    # flow. See fingertap/preprocess.py and experiments/ablation.py.
    enhance: str = "none"           # none | gamma | clahe (Lab 2 enhancement)
    gamma: float = 1.5              # >1 brightens shadows (power-law transform)
    clahe_clip: float = 2.0         # CLAHE contrast-clip limit
    clahe_grid: int = 8             # CLAHE tile grid (grid x grid)
    use_hsv_roi: bool = False       # Lab 4 HSV skin segmentation -> ROI crop
    hsv_lower: tuple = (0, 30, 60)  # skin HSV lower bound (H 0-179)
    hsv_upper: tuple = (25, 160, 255)
    roi_morph_kernel: int = 5       # open/close kernel size for mask cleanup
    roi_pad_frac: float = 0.20      # bbox padding as a fraction of its size
    roi_min_area_frac: float = 0.02  # reject blobs smaller than this of frame

    def to_dict(self) -> dict:
        return asdict(self)
