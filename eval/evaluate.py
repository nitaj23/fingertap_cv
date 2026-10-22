"""Event-matching evaluation (spec sheet section 8.1).

Matches detected close events to manually annotated contact events with a
greedy one-to-one nearest-time assignment inside a tolerance window, then
reports precision, recall, F1, count error, and close-time MAE. Sensitivity
can be reported at several tolerances (+/-50, +/-100, +/-150 ms).

Ground-truth CSV schema (spec 7.3):
    event_id, frame_idx, timestamp_sec, annotator_id
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class MatchResult:
    tolerance_s: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    count_error: int           # detected - annotated
    count_error_pct: float
    timing_mae_s: float        # mean |detected - matched gt| over TPs
    n_detected: int
    n_truth: int

    def to_dict(self) -> dict:
        return asdict(self)


def match_events(detected_times, truth_times, tolerance_s: float) -> MatchResult:
    """Greedy one-to-one matching by ascending pair distance."""
    det = sorted(float(t) for t in detected_times)
    gt = sorted(float(t) for t in truth_times)

    # Build candidate pairs within tolerance, sort by distance, assign greedily.
    pairs = []
    for i, d in enumerate(det):
        for j, g in enumerate(gt):
            dt = abs(d - g)
            if dt <= tolerance_s:
                pairs.append((dt, i, j))
    pairs.sort()

    used_det, used_gt = set(), set()
    errors = []
    for dt, i, j in pairs:
        if i in used_det or j in used_gt:
            continue
        used_det.add(i)
        used_gt.add(j)
        errors.append(dt)

    tp = len(errors)
    fp = len(det) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    count_error = len(det) - len(gt)
    count_error_pct = (count_error / len(gt) * 100.0) if gt else 0.0
    timing_mae = float(np.mean(errors)) if errors else 0.0

    return MatchResult(
        tolerance_s=tolerance_s,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        count_error=count_error, count_error_pct=count_error_pct,
        timing_mae_s=timing_mae,
        n_detected=len(det), n_truth=len(gt),
    )


def load_truth_csv(path: str):
    times = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            times.append(float(row["timestamp_sec"]))
    return times


def load_detected_csv(path: str, confident_only: bool = True):
    """Load close times from an exported events CSV (export.export_events_csv)."""
    times = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if confident_only and int(row.get("uncertain", 0)):
                continue
            times.append(float(row["close_time_s"]))
    return times


def evaluate_file(video_path: str, truth_path: str, tolerances=(0.05, 0.10, 0.15)):
    """Run the offline pipeline on a video and score it against annotations."""
    from fingertap.session import run_offline

    result = run_offline(video_path)
    detected = [e.close_t for e in result.events if not e.uncertain]
    truth = load_truth_csv(truth_path)
    return result, [match_events(detected, truth, tol) for tol in tolerances]


def _print_report(matches):
    print(f"{'tol(ms)':>8} {'P':>6} {'R':>6} {'F1':>6} "
          f"{'cnt_err':>8} {'MAE(ms)':>8} {'TP/FP/FN'}")
    for m in matches:
        print(f"{m.tolerance_s*1000:8.0f} {m.precision:6.3f} {m.recall:6.3f} "
              f"{m.f1:6.3f} {m.count_error:8d} {m.timing_mae_s*1000:8.1f} "
              f"{m.tp}/{m.fp}/{m.fn}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate detected taps vs annotations.")
    ap.add_argument("--video", help="recorded video to run the pipeline on")
    ap.add_argument("--events", help="exported events CSV (skip running the pipeline)")
    ap.add_argument("--truth", required=True, help="ground-truth annotation CSV")
    ap.add_argument("--tolerances", default="50,100,150",
                    help="comma-separated ms tolerances")
    args = ap.parse_args()

    tols = tuple(float(x) / 1000.0 for x in args.tolerances.split(","))
    truth = load_truth_csv(args.truth)

    if args.events:
        detected = load_detected_csv(args.events)
        matches = [match_events(detected, truth, t) for t in tols]
    elif args.video:
        _, matches = evaluate_file(args.video, args.truth, tols)
    else:
        ap.error("provide either --video or --events")

    _print_report(matches)


if __name__ == "__main__":
    main()
