"""Optional frame preprocessing experiments (spec sheet section 9.1 ablations).

Three course techniques are wired in as *optional*, config-gated stages that run
between frame resize and MediaPipe. They default OFF so the runtime pipeline
stays the minimal five-stage flow (spec section 1.2); enable them to run the
section 9.1 ablations and measure whether they actually help.

  - Low-light enhancement (Lab 2): gamma (power-law) correction or CLAHE
    (adaptive histogram equalization), to recover landmarks on dim/side-lit
    frames.
  - HSV skin ROI (Lab 4): segment the hand by colour, clean the mask with
    morphology, crop to the largest blob, and run MediaPipe on that ROI.
  - Canny edges (Lab 4): a demonstration edge map whose dual-threshold
    hysteresis mirrors the tap detector's OPEN/CLOSED hysteresis band.

Both runtime stages preserve the 640x480 frame size, and the tap signal is a
RATIO (d_norm = d_raw / d_ref), so cropping/rescaling the ROI leaves it
unchanged. Prefer the simpler (disabled) path unless an ablation shows a
meaningful gain in valid-frame coverage or event F1.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .config import Config


def apply_gamma(frame, gamma: float):
    """Power-law (gamma) intensity transform via a lookup table (Lab 2).

    gamma > 1 brightens shadows; gamma < 1 darkens. Implemented as a 256-entry
    LUT so it runs per-frame at negligible cost.
    """
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                   dtype=np.uint8)
    return cv.LUT(frame, lut)


def apply_clahe(frame, clip: float, grid: int):
    """Contrast-limited adaptive histogram equalization on luminance (Lab 2).

    Equalizes only the L channel in LAB space so colour is preserved -- this is
    the histogram-equalization family applied where it helps detection without
    shifting hue (which would disturb the HSV ROI stage).
    """
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)


def enhance_frame(frame, cfg: Config):
    """Dispatch the configured enhancement (none | gamma | clahe)."""
    if cfg.enhance == "gamma":
        return apply_gamma(frame, cfg.gamma)
    if cfg.enhance == "clahe":
        return apply_clahe(frame, cfg.clahe_clip, cfg.clahe_grid)
    return frame


def hand_roi_bbox(frame, cfg: Config):
    """HSV skin segmentation -> largest blob bbox (x, y, w, h), or None (Lab 4).

    Thresholds the hue/sat/val skin band, cleans the mask with a morphological
    open-then-close, and takes the largest contour. Returns None when nothing
    plausible is found so the caller falls back to the full frame (an ROI stage
    must never starve a working full-frame detection).
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lo = np.array(cfg.hsv_lower, dtype=np.uint8)
    hi = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv.inRange(hsv, lo, hi)

    k = np.ones((cfg.roi_morph_kernel, cfg.roi_morph_kernel), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv.contourArea)
    h, w = frame.shape[:2]
    if cv.contourArea(c) < cfg.roi_min_area_frac * w * h:
        return None
    return cv.boundingRect(c)


def apply_roi(frame, cfg: Config):
    """Crop to the padded HSV hand ROI and rescale back to WxH.

    Rescaling preserves frame size so downstream geometry and the UI aspect
    ratio are unchanged; the ratio-based d_norm is invariant to the rescale.
    Falls back to the untouched frame when no hand blob is found.
    """
    bbox = hand_roi_bbox(frame, cfg)
    if bbox is None:
        return frame
    x, y, bw, bh = bbox
    h, w = frame.shape[:2]
    pad_x = int(bw * cfg.roi_pad_frac)
    pad_y = int(bh * cfg.roi_pad_frac)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    return cv.resize(crop, (cfg.width, cfg.height))


def preprocess(frame, cfg: Config):
    """Apply the enabled experimental stages in order: enhance, then HSV ROI.

    With the default Config both are no-ops, so this is a transparent pass
    through on the production path.
    """
    frame = enhance_frame(frame, cfg)
    if cfg.use_hsv_roi:
        frame = apply_roi(frame, cfg)
    return frame


def canny_edges(frame, low: int = 50, high: int = 150):
    """Canny hysteresis edge map (Lab 4).

    Canny keeps strong edges (> high), drops weak ones (< low), and keeps
    in-between edges only if connected to strong ones -- the same dual-threshold
    hysteresis the tap detector applies to the 1-D d_norm signal. Returns a
    single-channel edge image.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.Canny(gray, low, high)
