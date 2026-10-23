"""Lab 4 demonstration: Otsu thresholding + Canny hysteresis on a hand frame.

Shows the single-threshold (Otsu binary) vs. dual-threshold (Canny hysteresis)
distinction, which is the same idea the tap detector uses on the 1-D d_norm
signal: a single threshold chatters, two thresholds with a band do not.

    python -m experiments.edges_demo clip.mp4 --out output
    python -m experiments.edges_demo frame.png --low 40 --high 120
"""

from __future__ import annotations

import argparse
import os

import cv2 as cv

from fingertap.config import Config
from fingertap.preprocess import canny_edges


def _read_first_frame(source: str):
    cap = cv.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if ok:
        return frame
    return cv.imread(source)  # fall back to a still image


def main():
    ap = argparse.ArgumentParser(description="Canny / threshold demo (Lab 4).")
    ap.add_argument("source", help="image or video file")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--low", type=int, default=50, help="Canny low threshold")
    ap.add_argument("--high", type=int, default=150, help="Canny high threshold")
    args = ap.parse_args()
    cfg = Config()

    frame = _read_first_frame(args.source)
    if frame is None:
        raise SystemExit(f"cannot read {args.source}")
    frame = cv.resize(frame, (cfg.width, cfg.height))

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    edges = canny_edges(frame, args.low, args.high)

    os.makedirs(args.out, exist_ok=True)
    cv.imwrite(os.path.join(args.out, "demo_gray.png"), gray)
    cv.imwrite(os.path.join(args.out, "demo_otsu.png"), otsu)
    cv.imwrite(os.path.join(args.out, "demo_canny.png"), edges)
    print(f"wrote demo_gray.png, demo_otsu.png, demo_canny.png to {args.out}")
    print(f"Canny {args.low}->{args.high}: single-threshold (Otsu) vs. "
          f"dual-threshold hysteresis -- the same principle as the tap "
          f"detector's T_low/T_high band.")


if __name__ == "__main__":
    main()
