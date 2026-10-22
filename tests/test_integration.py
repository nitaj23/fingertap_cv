"""Integration tests: full SessionProcessor pipeline (spec section 10.2)."""

from fingertap.config import Config
from fingertap.session import SessionProcessor


def _frame(d_raw_px):
    """Build a 21-landmark hand with a fixed reference (wrist->mmcp = 100 px)
    and thumb/index separated by d_raw_px along x."""
    lm = [[i, 0, 0] for i in range(21)]
    lm[0] = [0, 50, 300]       # wrist
    lm[9] = [9, 50, 200]       # middle MCP -> ref length 100 px
    lm[8] = [8, 50, 100]       # index tip (fixed)
    lm[4] = [4, 50 + d_raw_px, 100]  # thumb tip, moves to change separation
    return lm


def _tap_stream(n_cycles, dt=0.02, period=0.5, open_px=90, close_px=8):
    """Yield (t, lmList, valid) frames with triangular open/close motion."""
    t = 0.0
    steps = int(period / dt)
    half = steps // 2
    for _ in range(n_cycles):
        for k in range(steps):
            frac = k / half if k < half else (steps - k) / half
            frac = max(0.0, min(1.0, frac))
            px = open_px - (open_px - close_px) * (1 - frac)  # open->close->open
            yield (t, _frame(px), True)
            t += dt
    for _ in range(half):
        yield (t, _frame(open_px), True)
        t += dt


def test_full_pipeline_counts_taps():
    cfg = Config(auto_threshold=True, min_closed_s=0.02, min_cycle_s=0.1,
                 smooth_alpha=0.6)
    proc = SessionProcessor(cfg)
    for t, lm, valid in _tap_stream(8):
        proc.process_frame(t, lm, valid)
    result = proc.finalize()
    # Allow +/-1 from discretization/smoothing edge effects.
    assert abs(result.metrics.tap_count - 8) <= 1
    assert result.metrics.frequency_hz > 0
    assert result.quality.valid_fraction == 1.0


def test_hand_loss_produces_no_phantom_taps():
    cfg = Config(auto_threshold=False, t_low=0.35, t_high=0.60,
                 min_closed_s=0.02, min_cycle_s=0.1)
    proc = SessionProcessor(cfg)
    t = 0.0
    # Open hand, then a long invalid gap, then open again -- no real taps.
    for _ in range(20):
        proc.process_frame(t, _frame(90), True)
        t += 0.02
    for _ in range(10):
        proc.process_frame(t, [], False)
        t += 0.02
    for _ in range(20):
        proc.process_frame(t, _frame(90), True)
        t += 0.02
    result = proc.finalize()
    assert result.metrics.tap_count == 0
    assert result.quality.gap_count >= 1
    assert result.quality.longest_gap_s > 0


def test_valid_duration_removes_gap_time():
    cfg = Config()
    proc = SessionProcessor(cfg)
    t = 0.0
    for _ in range(10):
        proc.process_frame(t, _frame(90), True)
        t += 0.1          # 0.0 .. 0.9
    for _ in range(5):
        proc.process_frame(t, [], False)
        t += 0.1          # gap 0.9 .. 1.4
    for _ in range(10):
        proc.process_frame(t, _frame(90), True)
        t += 0.1
    vd = proc.valid_duration()
    total = proc.samples[-1].t - proc.samples[0].t
    assert vd < total      # gap time removed
