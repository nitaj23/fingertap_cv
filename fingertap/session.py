"""Session orchestration shared by the live UI and the offline runner.

This is the single source of event logic (spec section 10.2: live and offline
modes must share the same event logic). It consumes frames as
(timestamp, lmList, valid) and:

  - builds normalized samples (signal.SignalProcessor)
  - applies the causal EMA (smoothing.CausalEMA) for the live signal
  - runs a LIVE detector with provisional thresholds for the on-screen count
  - on finalize(), re-derives thresholds from the whole session and re-runs the
    detector to produce the authoritative events, metrics, and quality report
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2 as cv

import hand_track_module as htmod

from .config import Config
from .signal import SignalProcessor, Sample
from .smoothing import CausalEMA
from .preprocess import preprocess
from .detector import TapDetector, TapEvent, derive_thresholds
from .metrics import compute_metrics, SessionMetrics
from .quality import assess, QualityReport


@dataclass
class FrameResult:
    """What the live UI needs to render one frame."""

    t: float
    d_norm: float | None
    d_norm_smooth: float | None
    valid: bool
    state: str
    tap_count: int           # provisional live count (confident only)
    new_event: TapEvent | None


@dataclass
class SessionResult:
    """Authoritative end-of-session output."""

    samples: list[Sample]
    smoothed: list[float | None]
    events: list[TapEvent]
    metrics: SessionMetrics
    quality: QualityReport
    t_low: float
    t_high: float
    config: Config
    meta: dict = field(default_factory=dict)


class SessionProcessor:
    """Stateful per-frame processor plus a finalize() for authoritative results."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.signal = SignalProcessor(self.cfg)
        self.ema = CausalEMA(self.cfg.smooth_alpha)
        # Live detector uses provisional (config default) thresholds.
        self.live_detector = TapDetector(self.cfg.t_low, self.cfg.t_high, self.cfg)
        self.samples: list[Sample] = []
        self.smoothed: list[float | None] = []

    def process_frame(self, t: float, lmList, valid: bool) -> FrameResult:
        sample = self.signal.process(t, lmList, valid)
        self.samples.append(sample)
        sm = self.ema.update(sample.d_norm)
        self.smoothed.append(sm)
        ev = self.live_detector.update(t, sm, sample.valid)
        return FrameResult(
            t=t,
            d_norm=sample.d_norm,
            d_norm_smooth=sm,
            valid=sample.valid,
            state=self.live_detector.state,
            tap_count=len(self.live_detector.confident_events),
            new_event=ev,
        )

    def valid_duration(self) -> float:
        """Total session time minus time spent in invalid gaps."""
        if len(self.samples) < 2:
            return 0.0
        total = self.samples[-1].t - self.samples[0].t
        gap = 0.0
        prev = self.samples[0]
        for s in self.samples[1:]:
            if not s.valid or not prev.valid:
                gap += s.t - prev.t
            prev = s
        return max(0.0, total - gap)

    def finalize(self, meta: dict | None = None) -> SessionResult:
        """Re-derive thresholds from the whole session and recompute everything."""
        cfg = self.cfg
        if cfg.auto_threshold:
            t_low, t_high = derive_thresholds(self.smoothed, cfg)
        else:
            t_low, t_high = cfg.t_low, cfg.t_high

        det = TapDetector(t_low, t_high, cfg)
        for s, sm in zip(self.samples, self.smoothed):
            det.update(s.t, sm, s.valid)

        metrics = compute_metrics(det.events, self.valid_duration(), cfg)
        quality = assess(self.samples, cfg)
        return SessionResult(
            samples=self.samples,
            smoothed=self.smoothed,
            events=det.events,
            metrics=metrics,
            quality=quality,
            t_low=t_low,
            t_high=t_high,
            config=cfg,
            meta=meta or {},
        )


def run_offline(video_path: str, config: Config | None = None,
                detector_con: float | None = None) -> SessionResult:
    """Run the full pipeline on a recorded file. Uses the file's own frame times
    derived from CAP_PROP_POS_MSEC when available, else a nominal FPS."""
    cfg = config or Config()
    detector = htmod.handDetector(
        detectionCon=detector_con or cfg.detection_con,
        trackCon=cfg.tracking_con,
    )
    proc = SessionProcessor(cfg)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    nominal_fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (cfg.width, cfg.height))
            frame = preprocess(frame, cfg)  # optional enhance / HSV ROI (spec 9.1)
            pos_ms = cap.get(cv.CAP_PROP_POS_MSEC)
            t = (pos_ms / 1000.0) if pos_ms and pos_ms > 0 else frame_idx / nominal_fps
            detector.findHands(frame, draw=False)
            lmList, valid = detector.findPositionValid(frame, draw=False)
            proc.process_frame(t, lmList, valid)
            frame_idx += 1
    finally:
        cap.release()

    return proc.finalize(meta={"source": video_path, "frames": frame_idx})
