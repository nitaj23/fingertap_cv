"""Minimal single-window Tkinter app (spec sheet section 6).

Capture/processing runs on a worker thread and feeds a queue; the Tk main
thread polls via after() to render the frame (PIL ImageTk) and update the
scrolling signal plot -- this keeps the UI responsive and respects Tk's
single-thread rule.

Design goal: a clean, modern, compact interface -- a soft neutral palette,
white cards, one accent colour, a scrollable resizable window, and a two-column
layout (video beside metric tiles) so nothing is cut off on small screens.
"""

from __future__ import annotations

import queue
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
BG = "#eceef3"          # soft window background
CARD = "#ffffff"        # cards / tiles
BORDER = "#e1e4ea"      # hairline borders
INK = "#141922"         # primary text
MUTED = "#7b8494"       # secondary text / captions
ACCENT = "#4f46e5"      # single accent colour (indigo)
ACCENT_HOVER = "#4338ca"
ACCENT_INK = "#ffffff"
SOFT = "#eef0fb"        # tint for secondary buttons
TRACK = "#4f46e5"       # signal line
GRID = "#e9ecf2"        # gaps / faint lines
GUIDE = "#d6dae2"       # threshold guides
GOOD, WARN, BAD = "#15a34a", "#d97706", "#dc2626"

FONT = "Segoe UI"
F_TITLE = (FONT, 17, "bold")
F_SUB = (FONT, 9)
F_LABEL = (FONT, 8, "bold")
F_INPUT = (FONT, 10)
F_BTN = (FONT, 10, "bold")
F_TILE_NUM = (FONT, 26, "bold")
F_TILE_CAP = (FONT, 8, "bold")
F_SECTION = (FONT, 8, "bold")
F_STATUS = (FONT, 9)

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
    PLOT_W, PLOT_H = 760, 150
    HISTORY = 300               # samples shown in the scrolling plot

    def __init__(self, root: tk.Tk, cfg: Config | None = None):
        self.root = root
        self.cfg = cfg or Config()
        self.root.title("Finger-Tapping Measurement")
        self.root.configure(bg=BG)
        self.root.geometry("840x720")
        self.root.minsize(560, 480)
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.worker: CaptureWorker | None = None
        self.last_result = None
        self.hist = deque(maxlen=self.HISTORY)  # (d_norm_smooth, valid)
        self.tiles: dict[str, tk.Label] = {}
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
        hover = ACCENT_HOVER if primary else "#e3e5fb"
        b = tk.Button(parent, text=text, command=cmd, font=F_BTN, fg=fg, bg=bg,
                      activeforeground=fg, activebackground=hover, relief="flat",
                      bd=0, padx=20, pady=9, cursor="hand2",
                      disabledforeground=MUTED)
        b._base_bg = bg  # remember for hover restore
        b.bind("<Enter>", lambda e: b.configure(bg=hover) if str(b["state"]) != "disabled" else None)
        b.bind("<Leave>", lambda e: b.configure(bg=b._base_bg))
        return b

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=F_SECTION, fg=MUTED, bg=BG).pack(
            anchor="w", pady=(0, 6))

    # --- layout ---
    def _build_widgets(self, root):
        pad = tk.Frame(root, bg=BG, padx=22, pady=18)
        pad.pack(fill="both", expand=True)
        pad.columnconfigure(0, weight=1)

        # Header
        head = tk.Frame(pad, bg=BG)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        tk.Label(head, text="Finger-Tapping Measurement", font=F_TITLE,
                 fg=INK, bg=BG).pack(anchor="w")
        tk.Label(head, text="Webcam hand-motor performance · research prototype",
                 font=F_SUB, fg=MUTED, bg=BG).pack(anchor="w", pady=(1, 0))

        # Inputs + controls bar
        bar = tk.Frame(pad, bg=BG)
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        fields = tk.Frame(bar, bg=BG)
        fields.pack(side="left")
        for col, (lbl, key, dflt) in enumerate([
            ("Subject", "subject", "S001"), ("Hand", "hand", "right"),
            ("Duration s", "duration", "10"), ("Condition", "condition", "baseline"),
        ]):
            self._field(fields, lbl, key, dflt, col)

        ctrls = tk.Frame(bar, bg=BG)
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

        vcard = self._card(content)
        vcard.grid(row=0, column=0, sticky="n")
        vhold = tk.Frame(vcard, bg="#0d0f14", width=self.DISP_W, height=self.DISP_H)
        vhold.pack(padx=1, pady=1)
        vhold.pack_propagate(False)
        self.video_label = tk.Label(vhold, bg="#0d0f14", text="Press Start",
                                    fg=MUTED, font=(FONT, 11))
        self.video_label.pack(expand=True, fill="both")

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
        plot_wrap.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        self._section(plot_wrap, "NORMALIZED THUMB–INDEX DISTANCE")
        pcard = self._card(plot_wrap)
        pcard.pack(fill="x")
        self.plot_canvas = tk.Canvas(pcard, height=self.PLOT_H, bg=CARD,
                                     highlightthickness=0)
        self.plot_canvas.pack(fill="x", padx=1, pady=1)
        self.plot_canvas.bind("<Configure>", lambda e: self._render_plot())

        # Status line
        self.status = tk.StringVar(value="Idle · enter details and press Start.")
        tk.Label(pad, textvariable=self.status, font=F_STATUS, fg=MUTED, bg=BG,
                 anchor="w").grid(row=4, column=0, sticky="ew", pady=(14, 0))

    def _stacked_tile(self, parent, key, caption, row):
        cell = self._card(parent)
        cell.grid(row=row, column=0, sticky="nsew", pady=(0 if row == 0 else 10, 0))
        val = tk.Label(cell, text="—", font=F_TILE_NUM, fg=INK, bg=CARD)
        val.pack(pady=(10, 0))
        tk.Label(cell, text=caption, font=F_TILE_CAP, fg=MUTED, bg=CARD).pack(
            pady=(2, 10))
        self.tiles[key] = val

    # --- tile helpers ---
    def _set_tile(self, key, text, color=INK):
        self.tiles[key].configure(text=text, fg=color)

    def _reset_tiles(self):
        for key in ("count", "freq", "fps", "quality"):
            self._set_tile(key, "—")

    # --- control ---
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.hist.clear()
        self.last_result = None
        self._reset_tiles()
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
                    self.hist.append((res.d_norm_smooth, res.valid))
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
        m, qr = self.last_result.metrics, self.last_result.quality
        self._set_tile("count", str(m.tap_count))
        self._set_tile("freq", f"{m.frequency_hz:.2f}")
        self._set_tile("quality", qr.rating, QUALITY_COLOR.get(qr.rating, INK))
        self.status.set(
            f"Done · CV_ITI {m.cv_iti:.3f} · "
            f"amplitude Δ {m.amplitude_change_pct:+.1f}% · "
            f"valid {qr.valid_fraction * 100:.0f}% · "
            f"thresholds {self.last_result.t_low:.2f}/{self.last_result.t_high:.2f}"
        )

    def _render_frame(self, rgb):
        img = Image.fromarray(rgb).resize((self.DISP_W, self.DISP_H))
        tkimg = ImageTk.PhotoImage(img)
        self.video_label.configure(image=tkimg, text="")
        self.video_label.image = tkimg  # keep a reference

    def _render_live(self, res, fps):
        self._set_tile("count", str(res.tap_count))
        self._set_tile("fps", f"{fps:.0f}",
                       INK if fps >= self.cfg.min_fps else WARN)
        self._set_tile("quality", "live" if res.valid else "gap",
                       MUTED if res.valid else WARN)

    def _render_plot(self):
        c = self.plot_canvas
        c.delete("all")
        w = c.winfo_width() or self.PLOT_W
        h = self.PLOT_H
        vals = [v for v, _ in self.hist]
        real = [v for v in vals if v is not None]
        if not real:
            return
        lo, hi = min(real), max(real)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        dx = w / max(1, self.HISTORY - 1)

        def y(v):
            return h - (v - lo) / (hi - lo) * (h - 12) - 6

        for thr in (self.cfg.t_low, self.cfg.t_high):
            if lo <= thr <= hi:
                yy = y(thr)
                c.create_line(0, yy, w, yy, fill=GUIDE, dash=(2, 4))

        prev = None
        for i, (v, _valid) in enumerate(self.hist):
            x = i * dx
            if v is None:
                c.create_line(x, 0, x, h, fill=GRID)
                prev = None
                continue
            if prev is not None:
                c.create_line(prev[0], prev[1], x, y(v), fill=TRACK, width=2)
            prev = (x, y(v))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
