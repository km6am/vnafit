"""The calibration window: connect a standard, press a button, repeat.

Separate from gui.py because it is a self-contained job with its own state --
which standards have been captured, on what span -- and folding that into a
window that is already tracking a fit, a target and a schematic makes both
harder to follow.

The order is deliberate.  LOAD first, because it is the standard most often
forgotten and the one whose absence is least visible afterwards: without it
there is no directivity term and a return loss is decorative.  THRU last,
because it is the only one that needs both cables joined and is the one people
skip when they only want S11.

Every capture is taken with the instrument's OWN correction off.  Calibrating
through an existing calibration is circular, and the result looks fine.
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .cal import CalError, Calibration
from .touchstone import Touchstone

# (key, button label, what to connect, required?)
STEPS = [
    ("load",      "Load",      "a 50 Ω termination on PORT 1", True),
    ("open",      "Open",      "nothing on PORT 1, or an open standard", True),
    ("short",     "Short",     "a short on PORT 1", True),
    ("thru",      "Thru",      "the two cables joined to each other", False),
    ("isolation", "Isolation", "both cables terminated, not joined", False),
]


class CalWindow(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("vnafit - calibration")
        self.captured = {}
        self.grid_f = None
        self._build()
        self._refresh()

    # ------------------------------------------------------------------ ui
    def _build(self):
        pad = dict(padx=8, pady=4)
        head = ttk.Frame(self, padding=8)
        head.pack(fill="x")
        ttk.Label(head, text="Sweep the standards, then save.", 
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(head, foreground="#5f6873", wraplength=560, justify="left",
                  text="Each capture is taken with the instrument's own "
                       "correction switched OFF and restored afterwards -- "
                       "calibrating through an existing calibration is "
                       "circular, and the result looks perfectly fine."
                  ).pack(anchor="w", pady=(2, 0))

        span = ttk.Frame(self, padding=(8, 2))
        span.pack(fill="x")
        # Default to the span the main window is already sweeping: a cal that
        # does not cover the measurement is the commonest way to get one wrong.
        self.v_start = tk.StringVar(value=getattr(app, "v_start", None)
                                    and app.v_start.get() or "130")
        self.v_stop = tk.StringVar(value=getattr(app, "v_stop", None)
                                   and app.v_stop.get() or "165")
        self.v_points = tk.StringVar(value="401")
        for lab, var, w in (("start MHz", self.v_start, 9),
                            ("stop", self.v_stop, 9), ("points", self.v_points, 6)):
            ttk.Label(span, text=lab).pack(side="left", padx=(0, 3))
            ttk.Entry(span, textvariable=var, width=w).pack(side="left", padx=(0, 10))
        ttk.Label(span, foreground="#5f6873",
                  text="every standard must use the same span").pack(side="left")

        body = ttk.Frame(self, padding=8)
        body.pack(fill="both", expand=True)
        self.rows = {}
        for key, label, what, need in STEPS:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=2)
            b = ttk.Button(row, text=label, width=10,
                           command=lambda k=key: self.capture(k))
            b.pack(side="left")
            state = tk.Label(row, text="", width=3, font=("TkDefaultFont", 13))
            state.pack(side="left")
            ttk.Label(row, text=("connect " + what) +
                      ("" if need else "   (optional)")).pack(side="left")
            self.rows[key] = (b, state)

        self.status = ttk.Label(self, text="", padding=(8, 2),
                                foreground="#5f6873", wraplength=580,
                                justify="left")
        self.status.pack(fill="x")

        foot = ttk.Frame(self, padding=8)
        foot.pack(fill="x")
        ttk.Label(foot, text="notes").pack(side="left")
        self.v_notes = tk.StringVar()
        ttk.Entry(foot, textvariable=self.v_notes, width=44).pack(
            side="left", padx=6)
        self.savebtn = ttk.Button(foot, text="Save calibration...",
                                  command=self.save, state="disabled")
        self.savebtn.pack(side="right")
        ttk.Button(foot, text="Close", command=self.destroy).pack(
            side="right", padx=6)

    # ------------------------------------------------------------- capture
    def _cfg(self):
        try:
            return (float(self.v_start.get()) * 1e6, float(self.v_stop.get()) * 1e6,
                    int(self.v_points.get()))
        except ValueError:
            raise CalError("start, stop and points must be numbers")

    def capture(self, key):
        dev = getattr(self.app, "dev", None)
        if dev is None:
            messagebox.showinfo("calibration",
                                "Connect a VNA first -- a calibration has to be "
                                "measured, and a replay file cannot be one.")
            return
        try:
            start, stop, points = self._cfg()
            with dev.uncorrected():
                f, s11, s21 = dev.scan(start, stop, points)
        except Exception as e:                              # noqa: BLE001
            messagebox.showerror("calibration", f"{type(e).__name__}: {e}")
            return
        if self.grid_f is not None and (len(f) != len(self.grid_f)
                                        or not np.allclose(f, self.grid_f)):
            messagebox.showerror(
                "calibration",
                "That sweep came back on a different frequency grid from the "
                "standards already captured.  Change the span and start again, "
                "or re-capture the earlier ones.")
            return
        self.grid_f = f
        S = np.zeros((len(f), 2, 2), complex)
        S[:, 0, 0], S[:, 1, 0] = s11, s21
        self.captured[key] = Touchstone(f, S)
        self._refresh()

    def _refresh(self):
        for key, _l, _w, need in STEPS:
            b, state = self.rows[key]
            got = key in self.captured
            state.configure(text="✓" if got else ("○" if need else "·"),
                            fg="#1a7f37" if got else "#8a8a8a")
        have = set(self.captured)
        ready = {"open", "short", "load"} <= have
        self.savebtn.configure(state="normal" if ready else "disabled")
        if not ready:
            missing = [k for k in ("load", "open", "short") if k not in have]
            self.status.configure(
                text="still needed: " + ", ".join(missing))
        elif "thru" not in have:
            self.status.configure(
                text="Ready to save. Without a THRU this corrects S11 only; "
                     "S21 will be left exactly as measured.")
        elif "isolation" not in have:
            self.status.configure(
                text="Ready to save. Without an ISOLATION sweep the crosstalk "
                     "term is assumed zero, which is right until you are "
                     "looking near the instrument's leakage floor.")
        else:
            self.status.configure(text="All five captured.")

    # ---------------------------------------------------------------- save
    def save(self):
        try:
            cal = Calibration.from_standards(
                self.captured["open"], self.captured["short"],
                self.captured["load"], self.captured.get("thru"),
                self.captured.get("isolation"),
                notes=self.v_notes.get(),
                instrument=getattr(getattr(self.app, "dev", None), "_info", None))
        except CalError as e:
            messagebox.showerror("calibration", str(e))
            return
        path = filedialog.asksaveasfilename(
            title="save calibration", defaultextension=".calz",
            filetypes=[("vnafit calibration", "*.calz"), ("all", "*")])
        if not path:
            return
        cal.save(path)
        self.app.set_calibration(cal, path)
        messagebox.showinfo("calibration",
                            f"{os.path.basename(path)}\n\n{cal.describe()}")
        self.destroy()
