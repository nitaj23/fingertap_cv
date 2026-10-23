"""Minimal single-window Tkinter app (spec sheet section 6).

Capture/processing runs on a worker thread and feeds a queue; the Tk main
thread polls via after() to render the frame (PIL ImageTk) and update the
scrolling signal plot -- this keeps the UI responsive and respects Tk's
single-thread rule.

Design goal: a clean, modern, compact interface -- a soft neutral palette,
white cards, one accent colour, a scrollable resizable window, and a two-column
layout (video beside metric tiles) so nothing is cut off on small screens.

The live view shows the spec section 6 minimum (video, signal WITH detected
taps, count, FPS, quality). The full spec section 5.5 metric set -- frequency,
ITI, ITI CV, amplitude, and early-vs-late change -- plus the section 8.3 quality
diagnostics are laid out in a results panel that appears when a session ends.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox

import cv2 as cv
from PIL import Image, ImageTk

import hand_track_module as htmod

from .config import Config, THUMB_TIP, INDEX_TIP
from .session import SessionProcessor
from .export import export_all


# --- Modern design tokens --------------------------------------------------
BG = "#eef1f6"          # soft window background
CARD = "#ffffff"        # cards / tiles
CARD_ALT = "#f7f8fb"    # subtle inset / secondary surface
BORDER = "#dfe3ea"      # hairline borders
INK = "#141922"         # primary text
MUTED = "#79828f"       # secondary text / captions
FAINT = "#aab2bf"       # tertiary / placeholder
ACCENT = "#4f46e5"      # single accent colour (indigo)
ACCENT_HOVER = "#4338ca"
ACCENT_INK = "#ffffff"
SOFT = "#eef0fb"        # tint for secondary buttons
SOFT_HOVER = "#e3e5fb"
TRACK = "#4f46e5"       # signal line
TRACK_FILL = "#ecebfb"  # soft fill under the signal
GRID = "#eaedf3"        # gaps / faint lines
GUIDE = "#cfd4de"       # threshold guides
TAPDOT = "#ec4899"      # detected-tap marker (pink, distinct from track)
GOOD, WARN, BAD = "#15a34a", "#d97706", "#dc2626"

FONT = "Segoe UI"
F_TITLE = (FONT, 18, "bold")
F_SUB = (FONT, 9)
F_LABEL = (FONT, 8, "bold")
F_INPUT = (FONT, 10)
F_BTN = (FONT, 10, "bold")
F_TILE_NUM = (FONT, 27, "bold")
F_TILE_CAP = (FONT, 8, "bold")
F_SECTION = (FONT, 8, "bold")
F_STATUS = (FONT, 9)
F_GROUP = (FONT, 10, "bold")
F_METRIC_CAP = (FONT, 9)
F_METRIC_VAL = (FONT, 10, "bold")

QUALITY_COLOR = {"reliable": GOOD, "borderline": WARN, "discard": BAD}


class CaptureWorker(threading.Thread):
    """Grabs frames, runs detection + processing, pushes results to a queue."""

    def __init__(self, cfg: Config, out_q: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.out_q = out_q
        # NB: not self._stop -- that name shadows threading.Thread._stop()
        # and breaks is_alive().
        self._stop_event = threading.Event()
        self.proc = SessionProcessor(cfg)
        self.error: str | None = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        detector = htmod.handDetector(
            detectionCon=self.cfg.detection_con, trackCon=self.cfg.tracking_con
        )
        cap = cv.VideoCapture(self.cfg.cam_index)
        cap.set(3, self.cfg.width)
        cap.set(4, self.cfg.height)
        if not cap.isOpened():
            self.error = "Could not open camera."
            self.out_q.put(("error", self.error))
            return

        t0 = time.perf_counter()
        prev = t0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv.resize(frame, (self.cfg.width, self.cfg.height))
                t = time.perf_counter() - t0

                detector.findHands(frame, draw=True)
                lmList, valid = detector.findPositionValid(frame, draw=False)

                if valid:
                    p4 = (lmList[THUMB_TIP][1], lmList[THUMB_TIP][2])
                    p8 = (lmList[INDEX_TIP][1], lmList[INDEX_TIP][2])
                    cv.line(frame, p4, p8, (0, 255, 255), 2)
                    cv.circle(frame, p4, 8, (255, 0, 255), cv.FILLED)
                    cv.circle(frame, p8, 8, (255, 0, 255), cv.FILLED)

                res = self.proc.process_frame(t, lmList, valid)

                now = time.perf_counter()
                inst_fps = 1.0 / (now - prev) if now > prev else 0.0
                prev = now

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                self.out_q.put(("frame", rgb, res, inst_fps))
        finally:
            cap.release()
            self.out_q.put(("done", None))


class App:
    DISP_W, DISP_H = 480, 360   # on-screen video size (down-scaled from capture)
    PLOT_W, PLOT_H = 760, 170
    HISTORY = 300               # samples shown in the scrolling plot

    def __init__(self, root: tk.Tk, cfg: Config | None = None):
        self.root = root
        self.cfg = cfg or Config()
        self.root.title("Finger-Tapping Measurement")
        self.root.configure(bg=BG)
        self.root.geometry("900x820")
        self.root.minsize(560, 480)
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.worker: CaptureWorker | None = None
        self.last_result = None
        # (t, d_norm_smooth, valid) for the scrolling plot.
        self.hist = deque(maxlen=self.HISTORY)
        # close timestamps of detected taps, for plot markers.
        self.tap_times = deque(maxlen=self.HISTORY)
        self.tiles: dict[str, tk.Label] = {}
        self.result_tiles: dict[str, tk.Label] = {}
        self.vars: dict[str, tk.StringVar] = {}

        body = self._build_scroll_area()
        self._build_widgets(body)

    # --- scrollable container ---
    def _build_scroll_area(self) -> tk.Frame:
        """A canvas + vertical scrollbar holding one inner frame."""
        self._canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(self.root, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(self._canvas, bg=BG)
        self._win = self._canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        # Keep the inner frame as wide as the canvas (centred content handles the rest).
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfigure(self._win, width=e.width))
        # Mouse-wheel scrolling (Windows/Mac delta convention).
        self.root.bind_all("<MouseWheel>",
                           lambda e: self._canvas.yview_scroll(int(-e.delta / 120), "units"))
        return inner

    # --- small layout helpers ---
    def _card(self, parent):
        return tk.Frame(parent, bg=CARD, highlightthickness=1,
                        highlightbackground=BORDER, bd=0)

    def _field(self, parent, label, key, default, col):
        wrap = tk.Frame(parent, bg=BG)
        wrap.grid(row=0, column=col, sticky="w", padx=(0, 12))
        tk.Label(wrap, text=label.upper(), font=F_LABEL, fg=MUTED, bg=BG).pack(
            anchor="w", pady=(0, 2))
        var = tk.StringVar(value=default)
        tk.Entry(wrap, textvariable=var, width=11, font=F_INPUT, fg=INK,
                 relief="flat", bg=CARD, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT).pack(
            anchor="w", ipady=4)
        self.vars[key] = var

    def _button(self, parent, text, cmd, primary=False):
        fg = ACCENT_INK if primary else ACCENT
        bg = ACCENT if primary else SOFT
        hover = ACCENT_HOVER if primary else SOFT_HOVER
        b = tk.Button(parent, text=text, command=cmd, font=F_BTN, fg=fg, bg=bg,
                      activeforeground=fg, activebackground=hover, relief="flat",
                      bd=0, padx=20, pady=9, cursor="hand2",
                      disabledforeground=FAINT)
        b._base_bg = bg  # remember for hover restore
        b.bind("<Enter>", lambda e: b.configure(bg=hover) if str(b["state"]) != "disabled" else None)
        b.bind("<Leave>", lambda e: b.configure(bg=b._base_bg))
        return b

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=F_SECTION, fg=MUTED, bg=BG).pack(
            anchor="w", pady=(0, 6))

    # --- layout ---
    def _build_widgets(self, root):
        pad = tk.Frame(root, bg=BG, padx=24, pady=20)
        pad.pack(fill="both", expand=True)
        pad.columnconfigure(0, weight=1)

        # Header
        head = tk.Frame(pad, bg=BG)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        bar = tk.Frame(head, bg=ACCENT, width=4, height=38)
        bar.pack(side="left", padx=(0, 12))
        bar.pack_propagate(False)
        htext = tk.Frame(head, bg=BG)
        htext.pack(side="left")
        tk.Label(htext, text="Finger-Tapping Measurement", font=F_TITLE,
                 fg=INK, bg=BG).pack(anchor="w")
        tk.Label(htext, text="Webcam hand-motor performance · research prototype",
                 font=F_SUB, fg=MUTED, bg=BG).pack(anchor="w", pady=(1, 0))

        # Inputs + controls bar
        controls = tk.Frame(pad, bg=BG)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        fields = tk.Frame(controls, bg=BG)
        fields.pack(side="left")
        for col, (lbl, key, dflt) in enumerate([
            ("Subject", "subject", "S001"), ("Hand", "hand", "right"),
            ("Duration s", "duration", "10"), ("Condition", "condition", "baseline"),
        ]):
            self._field(fields, lbl, key, dflt, col)

        ctrls = tk.Frame(controls, bg=BG)
        ctrls.pack(side="right", anchor="s")
        self.start_btn = self._button(ctrls, "Start", self.start, primary=True)
        self.stop_btn = self._button(ctrls, "Stop", self.stop)
        self.export_btn = self._button(ctrls, "Export", self.export)
        for b in (self.start_btn, self.stop_btn, self.export_btn):
            b.pack(side="left", padx=(0, 8))
        self.stop_btn["state"] = "disabled"
        self.export_btn["state"] = "disabled"

        # Content: two columns -- video (left) + metric tiles stacked (right)
        content = tk.Frame(pad, bg=BG)
        content.grid(row=2, column=0, sticky="ew")
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)

        # Video gets at least half the width; its cell and the metrics cell
        # share the row equally (uniform), and the frame scales with the window.
        content.columnconfigure(0, weight=1, uniform="main")
        content.columnconfigure(1, weight=1, uniform="main")

        vcard = self._card(content)
        vcard.grid(row=0, column=0, sticky="nsew")
        self.vhold = tk.Frame(vcard, bg="#0d0f14", height=self.DISP_H)
        self.vhold.pack(fill="both", expand=True, padx=1, pady=1)
        self.vhold.pack_propagate(False)
        self.video_label = tk.Label(self.vhold, bg="#0d0f14", text="Press Start",
                                    fg=FAINT, font=(FONT, 11))
        self.video_label.pack(expand=True, fill="both")
        # Current on-screen video size; recomputed as the holder resizes so the
        # frame fills the widget at the capture's 4:3 aspect ratio.
        self._video_size = (self.DISP_W, self.DISP_H)
        self.vhold.bind("<Configure>", self._on_video_resize)

        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
        for i in range(4):
            right.rowconfigure(i, weight=1)
        right.columnconfigure(0, weight=1)
        for i, (key, cap) in enumerate([
            ("count", "TAPS"), ("freq", "FREQ (Hz)"),
            ("fps", "FPS"), ("quality", "QUALITY"),
        ]):
            self._stacked_tile(right, key, cap, i)

        # Signal plot card (full width)
        plot_wrap = tk.Frame(pad, bg=BG)
        plot_wrap.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        self._section(plot_wrap, "NORMALIZED THUMB–INDEX DISTANCE  ·  ● = DETECTED TAP")
        pcard = self._card(plot_wrap)
        pcard.pack(fill="x")
        self.plot_canvas = tk.Canvas(pcard, height=self.PLOT_H, bg=CARD,
                                     highlightthickness=0)
        self.plot_canvas.pack(fill="x", padx=1, pady=1)
        self.plot_canvas.bind("<Configure>", lambda e: self._render_plot())

        # Results panel (full spec 5.5 metrics + 8.3 quality) -- hidden until a
        # session finishes.
        self.results_wrap = tk.Frame(pad, bg=BG)
        self._build_results(self.results_wrap)

        # Status line
        self.status = tk.StringVar(value="Idle · enter details and press Start.")
        self.status_lbl = tk.Label(pad, textvariable=self.status, font=F_STATUS,
                                    fg=MUTED, bg=BG, anchor="w")
        self.status_lbl.grid(row=5, column=0, sticky="ew", pady=(16, 0))

    def _stacked_tile(self, parent, key, caption, row):
        cell = self._card(parent)
        cell.grid(row=row, column=0, sticky="nsew", pady=(0 if row == 0 else 10, 0))
        val = tk.Label(cell, text="—", font=F_TILE_NUM, fg=INK, bg=CARD)
        val.pack(pady=(10, 0))
        tk.Label(cell, text=caption, font=F_TILE_CAP, fg=MUTED, bg=CARD).pack(
            pady=(2, 10))
        self.tiles[key] = val

    # --- results panel ---
    def _build_results(self, parent):
        self._section(parent, "SESSION RESULTS")
        grid = tk.Frame(parent, bg=BG)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1, uniform="r")
        grid.columnconfigure(1, weight=1, uniform="r")

        self._group(grid, "Rhythm & rate", [
            ("count", "Taps"),
            ("freq", "Frequency"),
            ("iti_mean", "ITI mean"),
            ("cv_iti", "ITI CV"),
            ("regularity", "Regularity 1/(1+CV)"),
        ], 0, 0)
        self._group(grid, "Amplitude", [
            ("amp_mean", "Mean"),
            ("amp_median", "Median"),
            ("early_amp", "Early median"),
            ("late_amp", "Late median"),
            ("amp_change", "Early→late change"),
        ], 0, 1)
        self._group(grid, "Within-session change", [
            ("early_freq", "Early frequency"),
            ("late_freq", "Late frequency"),
            ("freq_change", "Frequency change"),
            ("uncertain", "Uncertain taps (gaps)"),
        ], 1, 0)
        self._group(grid, "Recording quality", [
            ("rating", "Rating"),
            ("valid", "Valid frames"),
            ("qfps", "Mean FPS"),
            ("gaps", "Gap count"),
            ("longest_gap", "Longest gap"),
            ("ref_cv", "Reference CV"),
            ("thresholds", "Thresholds Tlo/Thi"),
        ], 1, 1)

    def _group(self, parent, title, items, row, col):
        card = self._card(parent)
        card.grid(row=row, column=col, sticky="nsew",
                  padx=(0 if col == 0 else 12, 0),
                  pady=(0 if row == 0 else 12, 0))
        tk.Label(card, text=title, font=F_GROUP, fg=INK, bg=CARD).pack(
            anchor="w", padx=14, pady=(12, 4))
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(0, 4))
        for key, cap in items:
            self._metric_row(card, key, cap)
        tk.Frame(card, bg=CARD, height=8).pack()

    def _metric_row(self, parent, key, caption):
        r = tk.Frame(parent, bg=CARD)
        r.pack(fill="x", padx=14, pady=3)
        tk.Label(r, text=caption, font=F_METRIC_CAP, fg=MUTED, bg=CARD).pack(
            side="left")
        v = tk.Label(r, text="—", font=F_METRIC_VAL, fg=INK, bg=CARD)
        v.pack(side="right")
        self.result_tiles[key] = v

    def _show_results(self, show: bool):
        if show:
            self.results_wrap.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        else:
            self.results_wrap.grid_forget()

    # --- tile helpers ---
    def _set_tile(self, key, text, color=INK):
        self.tiles[key].configure(text=text, fg=color)

    def _reset_tiles(self):
        for key in ("count", "freq", "fps", "quality"):
            self._set_tile(key, "—")
        for lbl in self.result_tiles.values():
            lbl.configure(text="—", fg=INK)

    # --- control ---
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.hist.clear()
        self.tap_times.clear()
        self.last_result = None
        self._reset_tiles()
        self._show_results(False)
        self.worker = CaptureWorker(self.cfg, self.q)
        self.worker.start()
        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.export_btn["state"] = "disabled"
        self.status.set("Recording…")
        self._auto_stop_after()
        self.root.after(15, self._poll)

    def _auto_stop_after(self):
        try:
            dur = float(self.vars["duration"].get())
        except ValueError:
            dur = 10.0
        if dur > 0:
            self.root.after(int(dur * 1000), self.stop)

    def stop(self):
        if not (self.worker and self.worker.is_alive()):
            return
        self.worker.stop()

    def export(self):
        if not self.last_result:
            messagebox.showinfo("Export", "No session to export yet.")
            return
        self.last_result.meta.update({
            "subject": self.vars["subject"].get(),
            "hand": self.vars["hand"].get(),
            "condition": self.vars["condition"].get(),
        })
        paths = export_all(self.last_result, out_dir="output",
                           subject_code=self.vars["subject"].get())
        self.status.set(f"Exported → {paths['summary']}")
        messagebox.showinfo("Export", f"Saved session artifacts:\n{paths['summary']}")

    # --- polling / rendering ---
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    _, rgb, res, fps = msg
                    self._render_frame(rgb)
                    self.hist.append((res.t, res.d_norm_smooth, res.valid))
                    if res.new_event is not None:
                        self.tap_times.append(res.new_event.close_t)
                    self._render_plot()
                    self._render_live(res, fps)
                elif kind == "error":
                    messagebox.showerror("Camera", msg[1])
                    self._finish()
                    return
                elif kind == "done":
                    self._finish()
                    return
        except queue.Empty:
            pass
        if self.worker and self.worker.is_alive():
            self.root.after(15, self._poll)

    def _finish(self):
        self.start_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        if not self.worker:
            return
        self.last_result = self.worker.proc.finalize(meta={
            "subject": self.vars["subject"].get(),
            "hand": self.vars["hand"].get(),
            "condition": self.vars["condition"].get(),
        })
        self.export_btn["state"] = "normal"
        self._render_results(self.last_result)

    def _on_video_resize(self, e):
        """Keep the video area at the capture's 4:3 aspect as the window grows."""
        w = max(240, e.width)
        h = int(round(w * 3 / 4))
        if (w, h) == self._video_size:
            return
        self._video_size = (w, h)
        self.vhold.configure(height=h)

    def _render_frame(self, rgb):
        w, h = self._video_size
        img = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(img)
        self.video_label.configure(image=tkimg, text="")
        self.video_label.image = tkimg  # keep a reference

    def _render_live(self, res, fps):
        self._set_tile("count", str(res.tap_count))
        self._set_tile("fps", f"{fps:.0f}",
                       INK if fps >= self.cfg.min_fps else WARN)
        self._set_tile("quality", "live" if res.valid else "gap",
                       MUTED if res.valid else WARN)

    def _render_results(self, result):
        """Populate the end-of-session panel with the full metric set."""
        m, qr = result.metrics, result.quality

        # Live KPI tiles -> authoritative final values.
        self._set_tile("count", str(m.tap_count))
        self._set_tile("freq", f"{m.frequency_hz:.2f}")
        self._set_tile("quality", qr.rating, QUALITY_COLOR.get(qr.rating, INK))

        def put(key, text, color=INK):
            if key in self.result_tiles:
                self.result_tiles[key].configure(text=text, fg=color)

        # Rhythm & rate
        put("count", str(m.tap_count))
        put("freq", f"{m.frequency_hz:.2f} Hz")
        put("iti_mean", f"{m.iti_mean_s * 1000:.0f} ms" if m.iti_mean_s else "—")
        put("cv_iti", f"{m.cv_iti:.3f}" if m.tap_count >= 2 else "—")
        put("regularity", f"{m.regularity_score:.3f}" if m.tap_count >= 2 else "—")
        # Amplitude
        put("amp_mean", f"{m.amplitude_mean:.3f}")
        put("amp_median", f"{m.amplitude_median:.3f}")
        put("early_amp", f"{m.early_amplitude_median:.3f}")
        put("late_amp", f"{m.late_amplitude_median:.3f}")
        put("amp_change", f"{m.amplitude_change_pct:+.1f}%" if m.tap_count >= 2 else "—")
        # Within-session change
        put("early_freq", f"{m.early_frequency_hz:.2f} Hz" if m.tap_count >= 2 else "—")
        put("late_freq", f"{m.late_frequency_hz:.2f} Hz" if m.tap_count >= 2 else "—")
        put("freq_change", f"{m.freq_change_hz:+.2f} Hz" if m.tap_count >= 2 else "—")
        put("uncertain", str(m.uncertain_count),
            WARN if m.uncertain_count else INK)
        # Recording quality
        put("rating", qr.rating, QUALITY_COLOR.get(qr.rating, INK))
        put("valid", f"{qr.valid_fraction * 100:.0f}%",
            INK if qr.valid_fraction >= self.cfg.min_valid_frac else WARN)
        put("qfps", f"{qr.fps:.1f}",
            INK if qr.fps >= self.cfg.min_fps else WARN)
        put("gaps", str(qr.gap_count))
        put("longest_gap", f"{qr.longest_gap_s * 1000:.0f} ms",
            INK if qr.longest_gap_s <= self.cfg.max_longest_gap_s else WARN)
        put("ref_cv", f"{qr.ref_cv:.3f}",
            INK if qr.ref_cv <= self.cfg.ref_stability_cv_max else WARN)
        put("thresholds", f"{result.t_low:.2f} / {result.t_high:.2f}")

        self._show_results(True)
        self.status.set(
            f"Done · {m.tap_count} taps · {m.frequency_hz:.2f} Hz · "
            f"quality {qr.rating} · press Export to save artifacts."
        )

    def _render_plot(self):
        c = self.plot_canvas
        c.delete("all")
        w = c.winfo_width() or self.PLOT_W
        h = self.PLOT_H
        entries = list(self.hist)
        real = [v for _t, v, _ok in entries if v is not None]
        if not real:
            return
        lo, hi = min(real), max(real)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        dx = w / max(1, self.HISTORY - 1)
        pad_top, pad_bot = 12, 10

        def y(v):
            return h - pad_bot - (v - lo) / (hi - lo) * (h - pad_top - pad_bot)

        # Threshold guide lines (the fixed live hysteresis band).
        for thr in (self.cfg.t_low, self.cfg.t_high):
            if lo <= thr <= hi:
                yy = y(thr)
                c.create_line(0, yy, w, yy, fill=GUIDE, dash=(2, 4))

        # Soft fill + signal line, broken across invalid gaps.
        times = [t for t, _v, _ok in entries]
        prev = None
        for i, (_t, v, _ok) in enumerate(entries):
            x = i * dx
            if v is None:
                c.create_line(x, 0, x, h, fill=GRID)
                prev = None
                continue
            yy = y(v)
            if prev is not None:
                c.create_polygon(prev[0], prev[1], x, yy, x, h, prev[0], h,
                                 fill=TRACK_FILL, outline="")
                c.create_line(prev[0], prev[1], x, yy, fill=TRACK, width=2)
            prev = (x, yy)

        # Detected-tap markers: place each tap's close time at its nearest
        # sample in the visible window.
        if times:
            for ct in self.tap_times:
                if ct < times[0] or ct > times[-1]:
                    continue
                j = min(range(len(times)), key=lambda k: abs(times[k] - ct))
                v = entries[j][1]
                if v is None:
                    continue
                x, yy = j * dx, y(entries[j][1])
                c.create_oval(x - 4, yy - 4, x + 4, yy + 4,
                              fill=TAPDOT, outline=CARD, width=1)


def _enable_hidpi():
    """Make the process DPI-aware on Windows so Tk renders crisp text.

    Without this, Windows bitmap-stretches the whole window on displays scaled
    above 100% (125/150/175%), which blurs every glyph. Declaring awareness lets
    us render at native pixel density instead.
    """
    if sys.platform != "win32":
        return
    import ctypes

    for call in (
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),  # per-monitor v2
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),  # system DPI
        lambda: ctypes.windll.user32.SetProcessDPIAware(),       # legacy fallback
    ):
        try:
            call()
            return
        except Exception:
            continue


def main():
    _enable_hidpi()
    root = tk.Tk()
    # Scale Tk's point-sized fonts to the monitor's real DPI (72 pt == 1 inch).
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
