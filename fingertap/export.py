"""Session output artifacts (spec sheet section 7.4).

  - Per-frame log  (CSV)  : frame, time, distances, validity, reason
  - Detected events (CSV) : time, cycle amplitude, uncertainty flag
  - Session summary (JSON): metrics, quality, condition, configuration
  - Figures         (PNG) : full signal/events, ITI, amplitude series
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from .session import SessionResult


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_frames_csv(result: SessionResult, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "d_raw", "d_ref", "d_norm",
                    "d_norm_smooth", "valid", "reason"])
        for i, (s, sm) in enumerate(zip(result.samples, result.smoothed)):
            w.writerow([
                i,
                f"{s.t:.6f}",
                "" if s.d_raw is None else f"{s.d_raw:.4f}",
                "" if s.d_ref is None else f"{s.d_ref:.4f}",
                "" if s.d_norm is None else f"{s.d_norm:.6f}",
                "" if sm is None else f"{sm:.6f}",
                int(s.valid),
                s.reason,
            ])


def export_events_csv(result: SessionResult, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "close_time_s", "close_norm", "open_norm",
                    "amplitude", "cycle_start_s", "cycle_end_s", "uncertain"])
        for i, e in enumerate(result.events):
            w.writerow([
                i,
                f"{e.close_t:.6f}",
                f"{e.close_norm:.6f}",
                f"{e.open_norm:.6f}",
                f"{e.amplitude:.6f}",
                f"{e.cycle_start_t:.6f}",
                f"{e.cycle_end_t:.6f}",
                int(e.uncertain),
            ])


def export_summary_json(result: SessionResult, path: str):
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": result.meta,
        "thresholds": {"t_low": result.t_low, "t_high": result.t_high},
        "mediapipe_api": "solutions.hands",
        "metrics": result.metrics.to_dict(),
        "quality": result.quality.to_dict(),
        "config": result.config.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def export_figures(result: SessionResult, path_prefix: str):
    """Write signal+events, ITI, and amplitude PNGs. Returns list of paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # 1) Full signal with detected taps + thresholds.
    ts = [s.t for s in result.samples]
    norm = [s.d_norm for s in result.samples]
    sm = result.smoothed
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, norm, color="0.7", lw=0.8, label="d_norm")
    ax.plot(ts, sm, color="C0", lw=1.2, label="d_norm (causal EMA)")
    ax.axhline(result.t_low, color="C3", ls="--", lw=0.8, label="T_low")
    ax.axhline(result.t_high, color="C2", ls="--", lw=0.8, label="T_high")
    for e in result.events:
        ax.axvline(e.close_t, color="C1" if not e.uncertain else "0.5",
                   ls=":" if e.uncertain else "-", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized distance")
    ax.set_title("Finger-tapping signal and detected taps")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = f"{path_prefix}_signal.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    confident = [e for e in result.events if not e.uncertain]

    # 2) Inter-tap interval series.
    if len(confident) >= 2:
        closes = [e.close_t for e in confident]
        itis = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(1, len(itis) + 1), itis, "o-", color="C4")
        ax.set_xlabel("tap interval #")
        ax.set_ylabel("ITI (s)")
        ax.set_title(f"Inter-tap intervals (CV = {result.metrics.cv_iti:.3f})")
        fig.tight_layout()
        p = f"{path_prefix}_iti.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    # 3) Amplitude series.
    if confident:
        amps = [e.amplitude for e in confident]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(1, len(amps) + 1), amps, "s-", color="C5")
        ax.set_xlabel("tap #")
        ax.set_ylabel("amplitude (open - close)")
        ax.set_title("Per-tap amplitude")
        fig.tight_layout()
        p = f"{path_prefix}_amplitude.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    return paths


def export_all(result: SessionResult, out_dir: str, subject_code: str = "subject") -> dict:
    """Write all artifacts into out_dir; returns a map of artifact -> path."""
    _ensure_dir(out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{subject_code}_{stamp}")
    frames = f"{base}_frames.csv"
    events = f"{base}_events.csv"
    summary = f"{base}_summary.json"
    export_frames_csv(result, frames)
    export_events_csv(result, events)
    export_summary_json(result, summary)
    figs = export_figures(result, base)
    return {"frames": frames, "events": events, "summary": summary, "figures": figs}
