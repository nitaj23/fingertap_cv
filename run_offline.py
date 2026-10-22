"""Run the finger-tapping pipeline on a recorded video and export artifacts.

Live and offline modes share the same event logic (spec section 10.2).

    python run_offline.py path/to/clip.mp4 --subject S001 --out output
"""

from __future__ import annotations

import argparse

from fingertap.session import run_offline
from fingertap.export import export_all


def main():
    ap = argparse.ArgumentParser(description="Offline finger-tapping analysis.")
    ap.add_argument("video", help="recorded video file")
    ap.add_argument("--subject", default="subject", help="subject code for filenames")
    ap.add_argument("--out", default="output", help="output directory")
    args = ap.parse_args()

    result = run_offline(args.video)
    m, q = result.metrics, result.quality
    print(f"taps={m.tap_count}  freq={m.frequency_hz:.3f} Hz  "
          f"ITI_mean={m.iti_mean_s:.3f}s  CV_ITI={m.cv_iti:.3f}")
    print(f"amplitude median={m.amplitude_median:.3f}  "
          f"early/late amp change={m.amplitude_change_pct:+.1f}%")
    print(f"quality={q.rating}  valid={q.valid_fraction*100:.1f}%  "
          f"fps={q.fps:.1f}  longest_gap={q.longest_gap_s:.3f}s")
    print(f"thresholds: T_low={result.t_low:.3f}  T_high={result.t_high:.3f}")

    paths = export_all(result, out_dir=args.out, subject_code=args.subject)
    print("\nExported:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
