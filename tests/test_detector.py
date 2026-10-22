"""Unit tests for the hysteresis tap detector (spec section 10.1)."""

from fingertap.config import Config
from fingertap.detector import TapDetector


def make_cycles(n, open_val=1.0, close_val=0.1, period_s=0.5, dt=0.02, phase_open=0.5):
    """Generate a triangular open<->close signal of n full cycles.

    Each cycle: open_val -> close_val -> open_val over period_s.
    Returns list of (t, d_norm).
    """
    samples = []
    t = 0.0
    steps = int(round(period_s / dt))
    half = steps // 2
    for _ in range(n):
        for k in range(steps):
            if k < half:
                frac = k / half
                v = open_val + (close_val - open_val) * frac
            else:
                frac = (k - half) / (steps - half)
                v = close_val + (open_val - close_val) * frac
            samples.append((t, v))
            t += dt
    # trailing open samples so the last cycle closes back to OPEN
    for _ in range(half):
        samples.append((t, open_val))
        t += dt
    return samples


def run(samples, cfg=None, t_low=0.35, t_high=0.60):
    cfg = cfg or Config(min_closed_s=0.02, min_cycle_s=0.1)
    det = TapDetector(t_low, t_high, cfg)
    for t, v in samples:
        det.update(t, v, True)
    return det


def test_complete_cycles_counted_once():
    det = run(make_cycles(5))
    assert len(det.confident_events) == 5


def test_shallow_crossing_rejected():
    # Signal dips to 0.5 only -- never below T_low=0.35, so no taps.
    det = run(make_cycles(4, close_val=0.5))
    assert len(det.confident_events) == 0


def test_incomplete_cycle_not_counted():
    # Close but never rise back above T_high -> no completed cycle.
    cfg = Config(min_closed_s=0.02, min_cycle_s=0.1)
    det = TapDetector(0.35, 0.60, cfg)
    for t, v in [(0.0, 1.0), (0.02, 0.5), (0.04, 0.2), (0.06, 0.1), (0.08, 0.15)]:
        det.update(t, v, True)
    assert len(det.confident_events) == 0


def test_no_tap_across_invalid_gap():
    cfg = Config(min_closed_s=0.02, min_cycle_s=0.1)
    det = TapDetector(0.35, 0.60, cfg)
    # open -> close, then gap, then open: the cycle spans the gap.
    seq = [(0.00, 1.0, True), (0.02, 0.2, True), (0.04, 0.1, True),
           (0.06, None, False), (0.08, None, False),
           (0.10, 0.5, True), (0.12, 1.0, True)]
    for t, v, ok in seq:
        det.update(t, v, ok)
    # Any produced event must be flagged uncertain -> zero confident taps.
    assert len(det.confident_events) == 0


def test_refractory_rejects_too_brief_contact():
    cfg = Config(min_closed_s=0.10, min_cycle_s=0.30)
    # Very fast dips (period 0.1s) fall under the min cycle duration.
    det = run(make_cycles(6, period_s=0.1, dt=0.01), cfg=cfg)
    assert len(det.confident_events) == 0


def test_amplitude_is_open_minus_close():
    det = run(make_cycles(3, open_val=1.0, close_val=0.1))
    amps = [e.amplitude for e in det.confident_events]
    assert all(a > 0.7 for a in amps)  # ~0.9 expected, allow discretization
