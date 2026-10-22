"""Unit tests for metrics and evaluation matching (spec section 10.1)."""

import math

from fingertap.config import Config
from fingertap.detector import TapEvent
from fingertap.metrics import compute_metrics
from eval.evaluate import match_events


def _event(close_t, amp=1.0, uncertain=False):
    return TapEvent(close_t=close_t, close_norm=0.1, open_norm=0.1 + amp,
                    amplitude=amp, cycle_start_t=close_t - 0.1,
                    cycle_end_t=close_t + 0.1, uncertain=uncertain)


def test_known_itis_give_correct_mean_and_cv():
    # closes at 0,0.5,1.0,1.5,2.0 -> ITI constant 0.5 -> CV 0.
    events = [_event(t) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    m = compute_metrics(events, valid_duration_s=2.0)
    assert m.tap_count == 5
    assert math.isclose(m.iti_mean_s, 0.5, rel_tol=1e-9)
    assert math.isclose(m.cv_iti, 0.0, abs_tol=1e-9)
    assert math.isclose(m.regularity_score, 1.0, rel_tol=1e-9)


def test_frequency_uses_valid_duration():
    events = [_event(t) for t in (0.0, 1.0, 2.0, 3.0)]
    m = compute_metrics(events, valid_duration_s=4.0)
    assert math.isclose(m.frequency_hz, 1.0, rel_tol=1e-9)


def test_uncertain_events_excluded():
    events = [_event(0.0), _event(0.5, uncertain=True), _event(1.0)]
    m = compute_metrics(events, valid_duration_s=1.0)
    assert m.tap_count == 2 and m.uncertain_count == 1


def test_early_late_amplitude_change():
    # amplitudes decreasing: early high, late low -> negative change.
    amps = [1.0, 1.0, 0.95, 0.9, 0.6, 0.5]
    events = [_event(i * 0.5, amp=a) for i, a in enumerate(amps)]
    m = compute_metrics(events, valid_duration_s=3.0, config=Config(early_late_frac=0.33))
    assert m.amplitude_change_pct < 0


def test_event_matching_precision_recall():
    detected = [0.10, 0.60, 1.10]
    truth = [0.12, 0.58, 1.50]  # first two match within 100ms, third is off
    r = match_events(detected, truth, tolerance_s=0.10)
    assert r.tp == 2 and r.fp == 1 and r.fn == 1
    assert math.isclose(r.precision, 2 / 3, rel_tol=1e-9)
    assert math.isclose(r.recall, 2 / 3, rel_tol=1e-9)


def test_event_matching_one_to_one():
    # Two detections near one truth: only one may match.
    detected = [0.10, 0.11]
    truth = [0.10]
    r = match_events(detected, truth, tolerance_s=0.10)
    assert r.tp == 1 and r.fp == 1 and r.fn == 0
