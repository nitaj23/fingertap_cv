"""Section 9.1 ablation harness: does the optional preprocessing help?

Runs the offline pipeline on one clip under several Config variants and prints
valid-frame coverage, mean FPS, tap count, and -- when a ground-truth CSV is
given -- event precision/recall/F1 at a matching tolerance. The spec rule is to
prefer the simpler variant unless a harder one wins clearly.

    python -m experiments.ablation clip.mp4
    python -m experiments.ablation clip.mp4 --truth truth.csv --tol 0.10
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from fingertap.config import Config
from fingertap.session import run_offline
from eval.evaluate import match_events, load_truth_csv


def variants() -> dict[str, Config]:
    """The ablation grid: baseline vs. each enabled course technique."""
    base = Config()
    return {
        "baseline": base,
        "gamma": replace(base, enhance="gamma"),
        "clahe": replace(base, enhance="clahe"),
        "hsv_roi": replace(base, use_hsv_roi=True),
        "clahe+roi": replace(base, enhance="clahe", use_hsv_roi=True),
    }


def main():
    ap = argparse.ArgumentParser(description="Preprocessing ablation (spec 9.1).")
    ap.add_argument("video", help="recorded clip to analyse")
    ap.add_argument("--truth", help="ground-truth annotation CSV (optional)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="event match tolerance in seconds (default 0.10)")
    args = ap.parse_args()

    truth = load_truth_csv(args.truth) if args.truth else None

    print(f"{'variant':<12} {'valid%':>7} {'fps':>6} {'taps':>5} "
          f"{'F1':>6} {'P':>6} {'R':>6}")
    for name, cfg in variants().items():
        res = run_offline(args.video, config=cfg)
        q, m = res.quality, res.metrics
        if truth is not None:
            detected = [e.close_t for e in res.events if not e.uncertain]
            mt = match_events(detected, truth, args.tol)
            f1, p, r = mt.f1, mt.precision, mt.recall
        else:
            f1 = p = r = float("nan")
        print(f"{name:<12} {q.valid_fraction * 100:7.1f} {q.fps:6.1f} "
              f"{m.tap_count:5d} {f1:6.3f} {p:6.3f} {r:6.3f}")


if __name__ == "__main__":
    main()
