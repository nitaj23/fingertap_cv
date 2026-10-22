"""Unit tests for distance and normalization (spec section 10.1)."""

import math

from fingertap.config import Config
from fingertap.signal import SignalProcessor, euclidean


def _hand(thumb, index, wrist, mmcp):
    """Build a 21-landmark list placing the four landmarks we use."""
    lm = [[i, 0, 0] for i in range(21)]
    lm[4] = [4, thumb[0], thumb[1]]
    lm[8] = [8, index[0], index[1]]
    lm[0] = [0, wrist[0], wrist[1]]
    lm[9] = [9, mmcp[0], mmcp[1]]
    return lm


def test_known_coordinates_distance():
    assert euclidean((0, 0), (3, 4)) == 5.0


def test_uniform_scaling_leaves_dnorm_unchanged():
    cfg = Config(ref_ema_alpha=1.0)  # ema follows input exactly
    sp = SignalProcessor(cfg)
    lm1 = _hand((100, 100), (160, 180), (100, 100), (100, 190))  # ref=90
    s1 = sp.process(0.0, lm1, True)

    sp2 = SignalProcessor(cfg)
    scale = 2.0
    lm2 = _hand((200, 200), (320, 360), (200, 200), (200, 380))  # same shape x2
    s2 = sp2.process(0.0, lm2, True)

    assert s1.valid and s2.valid
    assert math.isclose(s1.d_norm, s2.d_norm, rel_tol=1e-9)


def test_invalid_when_no_hand():
    sp = SignalProcessor()
    s = sp.process(0.0, [], False)
    assert not s.valid and s.d_norm is None and s.reason == "no_hand"


def test_near_zero_ref_rejected():
    cfg = Config(ref_min_px=20.0)
    sp = SignalProcessor(cfg)
    lm = _hand((100, 100), (160, 180), (100, 100), (100, 105))  # ref=5 px
    s = sp.process(0.0, lm, True)
    assert not s.valid and s.reason == "ref_too_small"


def test_ref_jump_rejected():
    cfg = Config(ref_max_jump_frac=0.3, ref_ema_alpha=0.5)
    sp = SignalProcessor(cfg)
    sp.process(0.0, _hand((100, 100), (160, 180), (100, 100), (100, 190)), True)  # ref 90
    s = sp.process(0.1, _hand((100, 100), (160, 180), (100, 100), (100, 400)), True)  # ref 300
    assert not s.valid and s.reason == "ref_jump"
