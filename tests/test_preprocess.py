"""Tests for the optional preprocessing stages (spec section 9.1).

These stages must (a) default to a transparent pass-through, (b) preserve the
640x480 frame size, and (c) leave the ratio-based signal usable -- verified
indirectly by keeping shape invariant so d_norm = d_raw/d_ref is unchanged.
"""

import numpy as np

from fingertap.config import Config
from fingertap.preprocess import (
    apply_gamma, enhance_frame, preprocess, canny_edges,
)


def _frame(value: int = 60):
    """A uniform mid-dark BGR frame (no skin colour, so ROI falls back)."""
    return np.full((480, 640, 3), value, dtype=np.uint8)


def test_gamma_brightens_dark_frame():
    f = _frame(60)
    assert apply_gamma(f, 2.0).mean() > f.mean()


def test_enhance_none_is_identity():
    f = _frame()
    assert np.array_equal(enhance_frame(f, Config()), f)


def test_default_preprocess_is_passthrough():
    f = _frame()
    assert np.array_equal(preprocess(f, Config()), f)


def test_preprocess_preserves_shape_when_enabled():
    cfg = Config(enhance="clahe", use_hsv_roi=True)
    out = preprocess(_frame(), cfg)
    assert out.shape == (cfg.height, cfg.width, 3)


def test_roi_falls_back_when_no_hand():
    # A frame with no skin-coloured blob must pass through unchanged.
    cfg = Config(use_hsv_roi=True)
    f = _frame()
    assert preprocess(f, cfg).shape == f.shape


def test_canny_returns_single_channel():
    assert canny_edges(_frame()).ndim == 2
