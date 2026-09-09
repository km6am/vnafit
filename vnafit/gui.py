"""Live tuning: a measured sweep with the netlist model drawn over it.

The workflow this exists for: turn a screw, watch the measured trace move toward
the model, stop when they sit on top of each other.  So the model has to redraw
faster than you can turn a screw, and the component values have to be editable
without leaving the window.  Both are met -- a 401-point netlist evaluation is
well under a millisecond.

Layout: the traces above the schematic, then one editable row per netlist
parameter with a slider, then a readout.

The schematic is not decoration.  It is where the fit set lives: every
parameter free to be fitted outlines the elements it drives, in its own colour,
and clicking an outline takes it out of the set or puts it back.  Without it the
parameter column is a list of names with no connection to the circuit, which is
the state this window was in for a while and it was hard to use.

Which parameters CAN be fitted is computed from the netlist by `structure`, in a
worker thread because it takes a second or two.  It is a property of the
circuit, so it is worth waiting for and it never has to be redone while tuning.

Two things on screen are deliberate and worth knowing about:

  * the residual is split into PASSBAND and SKIRTS.  Every model agrees in the
    passband; the skirts are where the information about topology lives, and a
    single rms number hides exactly the part you are trying to see.

  * the coupling-matrix model can be overlaid alongside the netlist.  Having
    both on one screen is how you catch either of them lying.

Design conventions follow the older tool so the two look like siblings:
measured S21 in blue, measured S11 in orange, model in dashed black, the 2 m
band shaded, and the same hysteretic quantised autoranging (expand at once,
shrink reluctantly -- an axis that rescales on every sweep is unreadable while
you are turning something).
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import mna, schematic as SCH, structure as STRUCT, touchstone
from .schematic import eng as _eng
from .acq import Acq, FileSource
from .card import annotate, param_colors
from .coupled import db
from .netlist import NetlistError, load as load_netlist

C_MEAS21, C_MEAS11 = "#1f6fd0", "#e06010"
C_MODEL, C_ALT = "#000000", "#8a2be2"
C_TARGET = "#2f7d32"
OK, WARN, BAD, DIM = "#1a7f37", "#a5730b", "#c62828", "#8a8a8a"
HAM_2M = (144e6, 148e6)


def short_device(info, limit=26):
    """A device banner cut down to something a toolbar can hold.

    The H4 answers `info` with four lines of firmware banner -- board, years,
    copyright, licence, project URL -- and putting that verbatim in a label
    stretched the toolbar past the width of the window, so connecting a VNA
    resized the whole thing.  Keep the board name and drop the rest.
    """
    txt = " ".join(str(info or "").split())
    for part in txt.split("|"):
        part = part.strip()
        if part.lower().startswith("board:"):
            txt = part.split(":", 1)[1].strip()
            break
    else:
        txt = txt.split("|")[0].strip() or "VNA"
    return txt if len(txt) <= limit else txt[:limit - 1].rstrip() + "\u2026"


def _hex(rgba):
    """A matplotlib colour as Tk wants it."""
    r, g, b = (int(round(255 * c)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    def __init__(self, root, netlist_path=None, replay=None):
        self.root = root
        root.title("vnafit - live tuning")
        self.dev = self.acq = None
        self.nl = self.built = None
        self.netlist_path = netlist_path
        self.meas = None                 # (f, s11, s21)
        self.vars = {}                   # param -> tk.StringVar
        self.sliders = {}
        self.base = {}                   # param -> nominal, for slider centring
        self.rows = []                   # param names that have an editable row
        self.fit_on = {}                 # param -> tk.BooleanVar, in the fit set
        self.st = None                   # structure.Structure, once analysed
        self.colors = {}
        self.drawn = []                  # [(param, [elements], [rects], None)]
        self._prec = {}                  # param -> Cramer-Rao fractional bound
        self.import_report = None        # what a SPICE import had to infer
        self.target = {}                 # the values to tune TOWARD, frozen
        self._zoom = None                # (lo, hi, kind) for the inset
        self._gauge_note = ""
        self.cal = None                  # a Calibration, applied to every sweep
        self.cal_path = None
        self._cal_warned = False
        self._inset = None
        self._pending = None             # analysis running in a worker
        self._fitting = None             # a fit running in a worker
        self.fitres = None
        self._srcname = "no source"
        self._last_count = -1
        self._ylim21 = (-70, 3)
        self._ylim11 = (-28, 1)

        self._build()
        # Closing the window must hand the trace back.  Without this the
        # instrument is left paused and looking hung, which is the same failure
        # a SIGTERM used to cause -- see vna.release_all.
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if netlist_path:
            self.load_netlist(netlist_path)
        if replay:
            self.load_replay(replay)

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = ttk.Frame(self.root, padding=6)
        outer.pack(fill="both", expand=True)

        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        ttk.Button(bar, text="Netlist...", command=self.pick_netlist).pack(side="left")
        ttk.Button(bar, text="Replay .s2p...", command=self.pick_replay).pack(side="left", padx=4)
        ttk.Button(bar, text="Connect VNA", command=self.connect).pack(side="left")
        self.runbtn = ttk.Button(bar, text="Start", command=self.toggle, state="disabled")
        self.runbtn.pack(side="left", padx=4)
        self.fitbtn = ttk.Button(bar, text="Fit", command=self.do_fit)
        self.fitbtn.pack(side="left", padx=(0, 4))
        ttk.Button(bar, text="Save .s2p...", command=self.save).pack(side="left")

        ttk.Label(bar, text="start MHz").pack(side="left", padx=(12, 2))
        self.v_start = tk.StringVar(value="135.4")
        ttk.Entry(bar, textvariable=self.v_start, width=8).pack(side="left")
        ttk.Label(bar, text="stop").pack(side="left", padx=(6, 2))
        self.v_stop = tk.StringVar(value="155.4")
        ttk.Entry(bar, textvariable=self.v_stop, width=8).pack(side="left")
        ttk.Label(bar, text="points").pack(side="left", padx=(6, 2))
        self.v_points = tk.StringVar(value="401")
        ttk.Entry(bar, textvariable=self.v_points, width=6).pack(side="left")
        ttk.Label(bar, text="avg").pack(side="left", padx=(6, 2))
        self.v_avg = tk.StringVar(value="1")
        ttk.Entry(bar, textvariable=self.v_avg, width=4).pack(side="left")

        self.v_alt = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="coupling-matrix overlay", variable=self.v_alt,
                        command=self.redraw).pack(side="left", padx=12)
        # Off by default, and that is a judgement rather than caution: while you
        # are turning a screw the model should stay put so you can see how far
        # off you are.  A model that chases the trace shows a good fit at every
        # position of the screw and tells you nothing about where to stop.
        # Tick it for extraction -- watching the parameters settle -- not for
        # tuning.
        self.v_autofit = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="refit each sweep", variable=self.v_autofit
                        ).pack(side="left")
        self.v_target = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="target", variable=self.v_target,
                        command=self.redraw).pack(side="left", padx=(10, 0))
        ttk.Button(bar, text="set target", command=self.set_target
                   ).pack(side="left", padx=(2, 0))
        ttk.Button(bar, text="Cal...", command=self.open_cal
                   ).pack(side="left", padx=(8, 0))
        self.v_cal = tk.BooleanVar(value=True)
        self.calbtn = ttk.Checkbutton(bar, text="apply cal", variable=self.v_cal,
                                      command=self.redraw, state="disabled")
        self.calbtn.pack(side="left")
        self.v_zoom = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="zoom", variable=self.v_zoom,
                        command=self.redraw).pack(side="left", padx=(6, 0))

        # WHAT is on the other end, kept apart from the status line at the
        # bottom.  That line carries transient messages -- a sweep counter, a
        # model error, a refused parameter -- and each one used to wipe out the
        # only place the window said whether anything was connected at all.
        self.link = tk.Label(bar, text="\u25cf  no source", fg=DIM, width=40,
                             anchor="e", font=("TkDefaultFont", 11))
        self.link.pack(side="right", padx=(10, 2))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=(6, 0))

        # Traces over schematic, in a draggable split: the schematic is worth a
        # lot of room while you decide what to fit and almost none while you are
        # turning a screw, and only you know which of those you are doing.
        split = ttk.Panedwindow(body, orient="vertical")
        split.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(9.5, 4.6), dpi=100)
        self.ax21 = self.fig.add_subplot(211)
        self.ax11 = self.fig.add_subplot(212, sharex=self.ax21)
        self._setup_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=split)
        split.add(self.canvas.get_tk_widget(), weight=3)

        self.sfig = Figure(figsize=(9.5, 2.4), dpi=100)
        self.scanvas = FigureCanvasTkAgg(self.sfig, master=split)
        split.add(self.scanvas.get_tk_widget(), weight=2)
        self.scanvas.mpl_connect("button_press_event", self._schematic_click)
        self._draw_schematic()

        side = ttk.Frame(body, width=330)
        side.pack(side="left", fill="y", padx=(8, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="MODEL PARAMETERS   (tick = free to fit)",
                  foreground=DIM).pack(anchor="w")
        self.params_frame = ttk.Frame(side)
        self.params_frame.pack(fill="x", pady=(4, 8))
        ttk.Separator(side, orient="horizontal").pack(fill="x")
        self.readout = tk.Text(side, width=36, height=18, borderwidth=0,
                               font=("TkFixedFont", 10), wrap="none")
        self.readout.pack(fill="both", expand=True, pady=(6, 0))
        self.readout.configure(state="disabled")

        self.status = ttk.Label(outer, text="load a netlist to begin", foreground=DIM)
        self.status.pack(fill="x", pady=(4, 0))

    def _setup_axes(self):
        for ax, lab, ylim in ((self.ax21, "|S21| dB", self._ylim21),
                              (self.ax11, "|S11| dB", self._ylim11)):
            ax.set_ylabel(lab)
            ax.set_ylim(*ylim)
            ax.grid(alpha=.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axvspan(HAM_2M[0] / 1e6, HAM_2M[1] / 1e6, color="#2e8b57", alpha=.09)
        self.ax11.set_xlabel("MHz")
        # The target goes on FIRST and stays behind: it is the thing you are
        # aiming at, not the thing you are reading, and drawn on top it hides
        # the two curves whose difference you are watching.
        self.l_t21, = self.ax21.plot([], [], color=C_TARGET, lw=2.6, alpha=.45,
                                     label="target (as loaded)", zorder=1)
        self.l_t11, = self.ax11.plot([], [], color=C_TARGET, lw=2.6, alpha=.45,
                                     zorder=1)
        self.l_m21, = self.ax21.plot([], [], color=C_MEAS21, lw=2.1, label="measured")
        self.l_f21, = self.ax21.plot([], [], color=C_MODEL, lw=1.0, ls="--", label="netlist")
        self.l_a21, = self.ax21.plot([], [], color=C_ALT, lw=1.0, ls=":", label="coupling matrix")
        self.l_m11, = self.ax11.plot([], [], color=C_MEAS11, lw=1.8)
        self.l_f11, = self.ax11.plot([], [], color=C_MODEL, lw=1.0, ls="--")
        self.ax21.legend(loc="upper right", fontsize=8, framealpha=.9)
        self.fig.tight_layout()

    # ----------------------------------------------------------------- loading
    def pick_netlist(self):
        p = filedialog.askopenfilename(
            title="netlist or SPICE netlist",
            filetypes=[("netlist", "*.net *.cir *.sp *.spi *.txt"), ("all", "*")])
        if p:
            self.load_netlist(p)

    def load_netlist(self, path):
        try:
            from . import spice as SP
            self.nl, report = SP.load_any(path)
        except (NetlistError, OSError, ValueError) as e:
            messagebox.showerror("netlist", str(e))
            return
        # An LTspice file arrives with no ports and no parameters; both are
        # inferred, and a silent guess about which node is a port models a
        # different network, so what was inferred is shown.  NOT in a
        # messagebox: that is modal, so it blocks load_netlist before the
        # identifiability worker is even started -- and it hangs outright in any
        # context without someone to click it.
        self.import_report = report.text() if report is not None else None
        self.netlist_path = path
        self.base = self.nl.resolve_params()
        # Frozen HERE, before anything can move it.  The file's own values are
        # the design, and nothing -- a fit, a slider, the role grouping -- ever
        # writes to this dict again.
        self.set_target(quiet=True)
        self.st = None
        self._pending = None
        self._rebuild_rows(STRUCT.independent(self.nl))
        self.status.configure(text=f"{os.path.basename(path)}: "
                                   f"{len(self.nl.elements)} elements, "
                                   f"{len(self.rows)} parameters "
                                   f"- working out which can be fitted...")
        if self.nl.ac:
            self.v_start.set(f"{self.nl.ac[2]/1e6:.4g}")
            self.v_stop.set(f"{self.nl.ac[3]/1e6:.4g}")
        self._draw_schematic()
        self._analyse()
        self.redraw()

    # ------------------------------------------------- what can be fitted
    def _analyse(self):
        """Run the structural analysis off the event loop.

        It takes a second or two -- long enough that doing it inline freezes the
        window on every netlist load.  It depends on the circuit alone, so it
        runs once and its answer stands for the whole tuning session.
        """
        nl, box = self.nl, {}
        lo = float(self.v_start.get() or 100) * 1e6
        hi = float(self.v_stop.get() or 200) * 1e6

        def work():
            try:
                box["r"] = STRUCT.analyse_reduced(nl, np.linspace(lo, hi, 400))
            except Exception as e:                          # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        self._pending = (t, box, nl)
        self.root.after(150, self._analysis_poll)

    def _analysis_poll(self):
        if not self._pending:
            return
        t, box, nl = self._pending
        if t.is_alive():
            self.root.after(150, self._analysis_poll)
            return
        self._pending = None
        if nl is not self.nl:
            return                       # a different netlist was loaded meanwhile
        if "e" in box:
            self.status.configure(text=f"identifiability: {box['e']}", foreground=WARN)
            return
        self.st, notes = box["r"]
        self.colors = param_colors(self.st.names)
        # The coordinates the analysis chose, plus any literal that is neither
        # one of them nor feeds one.  A parameter that FEEDS a coordinate must
        # not get a row: overriding both `al` and `lsh = al*turns^2` pins the
        # very dependency being edited, and turning `al` then does nothing at
        # all -- measured, 0.0 MHz of movement where freeing `lsh` gives 17.
        feeds = set()
        for n in self.st.names:
            feeds |= self._ancestors(n)
        keep = list(self.st.names) + [n for n in STRUCT.independent(self.nl)
                                      if n not in self.st.names and n not in feeds]
        # Ticked by default = determined AND measurable.  Ticking everything
        # merely determined turned a 12-parameter LTspice import loose with
        # eight parameters whose best case was several hundred percent.
        good = set(self.st.measurable())
        # A soft direction has to be GAUGE-FIXED, not fitted.  Leaving the
        # impedance scale free let a fit park at 3.6 nH with 300 pF capacitors
        # and a convincing residual, because the data cannot tell that apart
        # from 30 nH and 32 pF.  One member of each soft direction is pinned,
        # preferring an inductance -- the conventional gauge and usually the
        # value you trust most, since it is printed on the part.
        for v in self.st.soft_directions():
            members = [(abs(v[i]), n) for i, n in enumerate(self.st.names)
                       if abs(v[i]) > 0.15 and n in good]
            if not members:
                continue
            kinds = {e.name: e.kind for e in self.nl.elements}
            ind = [n for _c, n in members
                   if {kinds.get(e) for e in self.st.drives(n)} == {"L"}]
            pick = ind[0] if ind else max(members)[1]
            good.discard(pick)
            self._gauge_note = (
                f"{pick} pinned: it is in a direction the data can barely see "
                f"({', '.join(n for _c, n in members)} move together), so "
                f"fitting it walks instead of measuring")
        self._rebuild_rows(keep, free=good)
        weak = len(self.st.names) - len(good)
        msg = f"{len(good)} of {len(self.rows)} parameters ticked"
        if weak:
            msg += (f"; {weak} determined but not measurable at 0.05 dB noise "
                    f"-- tick one to try anyway")
        if getattr(self, "_gauge_note", ""):
            msg += "; " + self._gauge_note
        elif notes:
            msg += f"; {notes[0]}"
        self.status.configure(text=msg, foreground=DIM)
        self._draw_schematic()

    def _ancestors(self, name, seen=None):
        """Every .param `name` is computed from, transitively."""
        seen = seen or set()
        raw = self.nl.params.get(name)
        out = set()
        for x in getattr(raw, "names", ()):
            if x not in seen:
                out |= {x} | self._ancestors(x, seen | {x})
        return out

    def _why_held(self, name):
        """Why a parameter is not in the fit set -- in the column, next to it.

        Three different reasons, and they are not interchangeable: a gauge can
        never be fitted, an unmeasurable one could be but should not be at this
        noise, and one you pinned yourself is your decision.  A single "held"
        label for all three is what makes people widen bounds and refit instead
        of thinking about which case they are in.
        """
        if self.st is None:
            return ""
        if name not in self.st.names:
            return "not separable - set it"
        prec = self._prec.get(name)
        if prec is not None and prec >= self.st.MEASURABLE:
            return (f"best case {100*prec:.0f}%" if prec < 10 else "best case >10x")
        return "you pinned it"

    def _rebuild_rows(self, names, free=()):
        # Carry the values across.  Rebuilding reads them from self.base, so a
        # toggle -- a click on the schematic, a tick in the column -- threw away
        # everything tuned since the netlist was loaded.  Committing the boxes
        # into base first also keeps the sliders honest: they multiply base, so
        # they must be re-centred on what is actually on screen.
        for n, var in self.vars.items():
            try:
                self.base[n] = float(var.get())
            except (ValueError, TypeError):
                pass
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.vars.clear()
        self.sliders.clear()
        self.fit_on.clear()
        self.rows = [n for n in names if n in self.base]
        if not self.rows:
            ttk.Label(self.params_frame, foreground=DIM, wraplength=310,
                      text="This netlist has no .param lines, so there is nothing "
                           "to tune here.  Move the values you want to turn into "
                           ".param and refer to them from the elements.").pack(anchor="w")
        self._prec = (self.st.precision() if self.st is not None else {})
        free = set(free)
        fit = [n for n in self.rows if n in free]
        user = [n for n in self.rows if n not in free]
        for head, group, hint in (
                ("FIT - the fitter moves these", fit, None),
                ("YOURS - the fit will not touch these", user,
                 "set them by hand; the reason each one is here is beside it")):
            if not group:
                continue
            ttk.Label(self.params_frame, text=head, foreground=DIM
                      ).pack(anchor="w", pady=(6, 0))
            if hint:
                ttk.Label(self.params_frame, text=hint, foreground=DIM,
                          wraplength=310, font=("TkDefaultFont", 9)
                          ).pack(anchor="w")
            for name in group:
                self._param_row(name, self.base[name], name in free)

    def _param_row(self, name, val, free=False):
        row = ttk.Frame(self.params_frame)
        row.pack(fill="x", pady=1)
        on = tk.BooleanVar(value=bool(free))
        self.fit_on[name] = on
        ttk.Checkbutton(row, variable=on,
                        command=lambda n=name: self._toggle_fit(n, from_box=True)
                        ).pack(side="left")
        col = self.colors.get(name)
        lab = tk.Label(row, text=name, width=7, anchor="w",
                       fg=(_hex(col) if (free and col is not None) else DIM))
        lab.pack(side="left")
        if not free:
            tk.Label(row, text=self._why_held(name), fg=DIM, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="right")
        var = tk.StringVar(value=f"{val:.6g}")
        self.vars[name] = var
        ent = ttk.Entry(row, textvariable=var, width=12)
        ent.pack(side="left")
        ent.bind("<Return>", lambda _e, n=name: self._entry_changed(n))
        ent.bind("<FocusOut>", lambda _e, n=name: self._entry_changed(n))
        # The slider is in LOG space and spans a decade either side, because the
        # parameters here run from 1e-13 to 1e9 and a linear slider is useless
        # across that.  Its position is relative to the value in the box.
        s = tk.Scale(self.params_frame, from_=-1.0, to=1.0, resolution=0.002,
                     orient="horizontal", showvalue=False, length=310,
                     command=lambda v, n=name: self._slider_moved(n, float(v)))
        s.set(0.0)
        s.pack(fill="x")
        self.sliders[name] = s

    def _toggle_fit(self, name, from_box=False):
        """Put a parameter into the free set, or take it out.

        Taking one out is always allowed and is often the right move -- on a
        real TinyFilter sweep, freeing `l` alongside the capacitors ran the
        condition number to 5.6e3 and halved the inductance; pinning it at its
        printed value gave the same fit at condition 54.

        Putting one in is refused when the analysis says the data cannot
        separate it, with the reason.  Pinning such a parameter at a physical
        value is legitimate; fitting it is not, and a regulariser that hides
        the difference returns the prior's confidence interval as if it were
        the data's.
        """
        on = self.fit_on.get(name)
        if on is None:
            return
        if on.get() and (self.st is None or name not in self.st.names):
            on.set(False)
            why = "the analysis has not finished yet" if self.st is None else \
                  ("this parameter is not separable from the others -- pin it at "
                   "a value you trust instead")
            self.status.configure(text=f"{name}: {why}", foreground=WARN)
            return
        if not from_box:
            on.set(not on.get())
        self._rebuild_rows(self.rows, free={n for n, v in self.fit_on.items()
                                            if v.get()})
        self._draw_schematic()

    # ----------------------------------------------------------- calibration
    def open_cal(self):
        from .calwin import CalWindow
        CalWindow(self.root, self)

    def set_calibration(self, cal, path=None):
        """Adopt a calibration for everything measured from now on."""
        self.cal, self.cal_path = cal, path
        self.calbtn.configure(state="normal")
        self.status.configure(
            text=f"calibration: {os.path.basename(path) if path else 'in memory'}"
                 f" -- {cal.start/1e6:.3f}-{cal.stop/1e6:.3f} MHz, "
                 f"{cal.points} points, "
                 f"{'S11 and S21' if cal.has_thru else 'S11 only'}",
            foreground=OK)
        self.redraw()

    def load_cal(self, path):
        from .cal import Calibration
        self.set_calibration(Calibration.load(path), path)

    def corrected(self, data):
        """Apply the calibration to a raw sweep, or hand it back untouched.

        A cal that does not cover the sweep is reported ONCE and then ignored,
        rather than raising on every sweep of a live loop -- but it is never
        applied outside its range, because that is the failure it exists to
        prevent.
        """
        if data is None or self.cal is None or not self.v_cal.get():
            return data
        f, s11, s21 = data
        S = np.zeros((len(f), 2, 2), complex)
        S[:, 0, 0], S[:, 1, 0] = s11, s21
        try:
            out = self.cal.apply(touchstone.Touchstone(np.asarray(f, float), S))
        except Exception as e:                              # noqa: BLE001
            if not self._cal_warned:
                self._cal_warned = True
                self.status.configure(text=f"calibration not applied: {e}",
                                      foreground=WARN)
            return data
        self._cal_warned = False
        return f, out.s11, out.s21

    # ---------------------------------------------------------------- target
    def set_target(self, quiet=False):
        """Freeze the values to tune toward."""
        self.target = dict(self.base if quiet else self.nl.resolve_params(
            self.current()))
        self._zoom = None
        if not quiet:
            self.status.configure(text="target set to the values on screen",
                                  foreground=DIM)
            self.redraw()

    def target_model(self, f):
        if not (self.nl and self.target):
            return None
        try:
            b = mna.build(self.nl, self.target,
                          w_ref=2 * np.pi * float(np.sqrt(f[0] * f[-1])))
            return b.solve(f)
        except (NetlistError, ValueError, np.linalg.LinAlgError):
            return None

    # Tight on the insertion.  At 3 widths either side the passband was a small
    # bump in the middle of a mostly-empty box, and the thing you are actually
    # reading -- insertion loss and ripple, a dB or two of it -- was a few
    # pixels tall.  1.25 widths and 8 dB puts the top of the passband across
    # the whole inset.
    ZOOM_SPAN = 1.25         # feature widths either side of it
    ZOOM_DEPTH = 8.0         # dB below the peak
    ZOOM_NOTCH_DEPTH = 20.0  # dB above the null

    def feature(self, f, d):
        """(lo, hi, kind) around the thing being tuned.

        A notch is an INTERIOR minimum with the response at both ends more than
        20 dB above it; anything else is read as a peak, which covers bandpass,
        lowpass and highpass alike.  Taken from the TARGET rather than the
        measurement so the window holds still while you tune -- one that
        re-centres every sweep is unreadable exactly when you are turning
        something.
        """
        n = len(f)
        if n < 8:
            return None
        lo_i, hi_i = int(0.05 * n), int(0.95 * n)
        j = int(np.argmin(d))
        notch = (lo_i < j < hi_i and min(d[0], d[-1]) > d[j] + 20.0)
        i = j if notch else int(np.argmax(d))
        level = d[i] + 20.0 if notch else d[i] - 3.0
        inside = (np.flatnonzero(d <= level) if notch
                  else np.flatnonzero(d >= level))
        inside = inside[(inside >= lo_i - 1) & (inside <= hi_i + 1)] \
            if len(inside) else inside
        w = (f[inside[-1]] - f[inside[0]]) if len(inside) > 1 else 0.0
        w = max(w, 0.01 * f[i])
        return (f[i] - self.ZOOM_SPAN * w, f[i] + self.ZOOM_SPAN * w,
                "notch" if notch else "peak")

    def _zoom_axes(self, f, St, S):
        """The inset: the feature, close up, with all three traces in it."""
        if not self.v_zoom.get() or self._zoom is None:
            if self._inset is not None:
                self._inset.remove()
                self._inset = None
            return
        lo, hi, kind = self._zoom
        if self._inset is None:
            # Lower left: stopband on a bandpass, flat baseline on a notch --
            # empty in both.
            self._inset = self.ax21.inset_axes([0.05, 0.08, 0.36, 0.46])
            self._inset.tick_params(labelsize=6.5, length=2, pad=1)
            self._inset.grid(alpha=.25)
        ax = self._inset
        for ln in list(ax.lines):
            ln.remove()
        if St is not None:
            ax.plot(f / 1e6, db(St[:, 1, 0]), color=C_TARGET, lw=2.4, alpha=.45)
        if S is not None:
            ax.plot(f / 1e6, db(S[:, 1, 0]), color=C_MODEL, lw=1.0, ls="--")
        if self.meas is not None:
            fm, _m11, m21 = self.meas
            ax.plot(np.asarray(fm, float) / 1e6, db(m21), color=C_MEAS21, lw=1.6)
        ax.set_xlim(lo / 1e6, hi / 1e6)
        # Around the FEATURE, not around everything in the window.  Fitting the
        # whole window's range put 75 dB on a 3 cm axis and flattened the very
        # shape being tuned.  A peak is worth 20 dB below its top; a notch is
        # worth 25 dB above its bottom.
        ref = db(St[:, 1, 0]) if St is not None else None
        if ref is not None:
            k = (f >= lo) & (f <= hi)
            if k.any():
                if kind == "notch":
                    v = float(ref[k].min())
                    ax.set_ylim(v - 2.0, v + self.ZOOM_NOTCH_DEPTH)
                else:
                    v = float(ref[k].max())
                    ax.set_ylim(v - self.ZOOM_DEPTH, v + 1.0)
        ax.set_title(f"{kind}, {'null' if kind == 'notch' else 'insertion'} "
                     f"detail", fontsize=6.5, pad=1.5, color=DIM)

    def tuning_lines(self):
        """What to turn, in numbers the impedance scale cannot move.

        This used to compare fitted picofarads against target picofarads, which
        is worthless the moment a fit walks along the impedance scale: it asked
        for a 32 pF trimmer to be changed to 301 pF, a pure scale artefact.
        `tune.advice` works in resonator frequencies and capacitance RATIOS,
        which are invariant along that direction.

        Taken from the model now on screen rather than from the last fit, so it
        follows a slider as well as a fit.
        """
        if not (self.nl and self.target):
            return []
        try:
            from . import tune
            return tune.advice(self.nl, self.nl.resolve_params(self.current()),
                               self.target)
        except Exception:                                   # noqa: BLE001
            return []

    # ------------------------------------------------------------------- fit
    def _pinned_netlist(self):
        """The netlist with every box value baked in.

        `fit.Problem` seeds from, and holds fixed, whatever the NETLIST says --
        so without this a fit would ignore the values on screen and hold the
        pinned parameters at their file values, which is not what the window is
        showing anyone.
        """
        import copy
        nl = copy.copy(self.nl)
        nl.params = dict(self.nl.params)
        for name in self.rows:
            try:
                nl.params[name] = float(self.vars[name].get())
            except (ValueError, KeyError, TypeError):
                pass
        return nl

    def do_fit(self, quiet=False):
        """Fit the free parameters to the measured trace.

        The window overlays a model and lets you turn it by hand; until this
        existed nothing ever asked the model to move by itself, so a filter
        could sit connected with the two traces apart and nothing would happen.
        """
        if self._fitting:
            return
        if self.nl is None or self.meas is None:
            self.status.configure(text="fit needs a netlist and a sweep",
                                  foreground=WARN)
            return
        free = [n for n in self.rows
                if self.fit_on.get(n) and self.fit_on[n].get()]
        if not free:
            self.status.configure(text="fit: nothing is ticked", foreground=WARN)
            return
        got = self.passband_pts()
        if got and got[0] < self.PASSBAND_PTS:
            n, bw, (lo, hi) = got
            self.status.configure(
                text=f"fitting on only {n} points inside the {bw/1e6:.2f} MHz "
                     f"passband - narrow the sweep to {lo/1e6:.1f}-{hi/1e6:.1f} "
                     f"MHz and refit", foreground=WARN)

        from . import fit as F
        fm, m11, m21 = self.meas
        nl, box = self._pinned_netlist(), {}

        def work():
            try:
                prob = F.Problem(nl, np.asarray(fm, float), s11=m11, s21=m21,
                                 free=free, band="auto")
                box["r"] = prob.run()
                box["p"] = prob
            except Exception as e:                          # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=work, daemon=True)
        t.start()
        self._fitting = (t, box)
        if not quiet:                    # no flicker when refitting every sweep
            self.fitbtn.configure(text="fitting...", state="disabled")
        self.root.after(120, self._fit_poll)

    def _fit_poll(self):
        if not self._fitting:
            return
        t, box = self._fitting
        if t.is_alive():
            self.root.after(120, self._fit_poll)
            return
        self._fitting = None
        self.fitbtn.configure(text="Fit", state="normal")
        if "e" in box:
            self.status.configure(text=f"fit: {box['e']}", foreground=BAD)
            return
        res, prob = box["r"], box["p"]
        self.fitres = res
        for name, val in res.values.items():
            if name in self.vars:
                self.vars[name].set(f"{val:.6g}")
                self.base[name] = val
                if name in self.sliders:
                    self.sliders[name].set(0.0)
        rail = [n for n, bad in zip(prob.names, res.at_bound) if bad]
        lo, hi = prob.band_used or (prob.f[0], prob.f[-1])
        msg = (f"fit {res.rms:.3f} dB rms over {int(prob.keep.sum())} points, "
               f"{lo/1e6:.1f}-{hi/1e6:.1f} MHz")
        b = prob.band
        if b is not None and (b[0] < prob.f[0] or b[1] > prob.f[-1]):
            msg += "  (whole sweep - the auto window is wider than it)"
        if rail:
            msg += (f"  --  {', '.join(rail)} AT A BOUND: that is the limit, not "
                    f"a measurement.  Pin it rather than widening the bound.")
        self.status.configure(text=msg, foreground=(BAD if rail else OK))
        self.redraw()
        self._draw_schematic()

    # ------------------------------------------------------------- schematic
    def _draw_schematic(self):
        self.sfig.clear()
        self.drawn = []
        ax = self.sfig.add_subplot(111)
        if not self.nl:
            ax.axis("off")
            ax.text(0.5, 0.5, "no netlist loaded", ha="center", va="center",
                    color=DIM, transform=ax.transAxes)
            self.scanvas.draw_idle()
            return
        try:
            _f, _a, lay = SCH.draw(self.nl, ax=ax, title=None,
                                   params=self.nl.resolve_params(self.current()))
        except Exception as e:                              # noqa: BLE001
            ax.axis("off")
            ax.text(0.5, 0.5, f"schematic: {e}", ha="center", va="center",
                    color=BAD, transform=ax.transAxes, fontsize=8)
            self.scanvas.draw_idle()
            return
        if self.st is not None:
            fit = [n for n in self.st.names
                   if n in self.fit_on and self.fit_on[n].get()]
            # EVERY coordinate is drawn, pinned ones dashed and faint.  Drawing
            # only the free ones meant pinning a parameter erased its outline,
            # which read as a click destroying something and left nothing to
            # click to get it back.
            shown = list(self.st.names)
            held = {e for e in (lay.boxes or {})
                    if not any(p in self.st.epar.get(e, ()) for p in shown)}
            self.drawn = annotate(ax, self.st, lay, self.colors, only=shown,
                                  pinned=held,
                                  dim=[n for n in shown if n not in fit])
            ax.text(0.0, 1.0, "click an outline to pin or free that parameter; "
                              "dashed = pinned",
                    transform=ax.transAxes, fontsize=7.5, color=DIM,
                    ha="left", va="top")
        lo, hi = lay.ylim or ax.get_ylim()
        ax.set_ylim(lo, max(hi, 0.95))
        self.sfig.tight_layout(pad=0.2)
        self.scanvas.draw_idle()

    def _schematic_click(self, ev):
        """Toggle whatever outline was clicked.

        Smallest hit wins.  Outlines nest -- `l` and `qu` both wrap L1 -- and
        taking the first match in drawing order toggled whichever happened to
        be listed first, so clicking the inner box changed the outer
        parameter.  Area is the only thing that distinguishes them.
        """
        if ev.inaxes is None or self.st is None:
            return
        hits = []
        for pn, els, rects, _t in self.drawn:
            for r in rects:
                if r.contains_point((ev.x, ev.y)):
                    w, h = r.get_width(), r.get_height()
                    hits.append((w * h, pn, els))
        if not hits:
            return
        _area, pn, els = min(hits, key=lambda t: t[0])
        if pn is None:                   # an element no coordinate touches
            who = sorted(self.st.epar.get(els[0], ())) or ["a literal value"]
            self.status.configure(
                text=f"{els[0]} is held by {', '.join(who)}", foreground=DIM)
            return
        self._toggle_fit(pn)
        on = self.fit_on[pn].get()
        if on or pn in self.st.names:
            self.status.configure(
                text=(f"{pn} free to fit" if on else
                      f"{pn} pinned at {self.vars[pn].get()} - click it again "
                      f"to free it"), foreground=(OK if on else DIM))

    def _entry_changed(self, name):
        try:
            self.base[name] = float(self.vars[name].get())
        except ValueError:
            return
        self.sliders[name].set(0.0)
        self.redraw()
        self._draw_schematic()

    def _slider_moved(self, name, pos):
        val = self.base[name] * (10.0 ** pos)
        self.vars[name].set(f"{val:.6g}")
        self.redraw()

    def current(self):
        """Parameter overrides from the boxes, as the model sees them.

        Only the rows -- which are built to be free of dependencies on each
        other.  Sending every resolved parameter, derived ones included, pins
        whatever a row feeds and silently disables that row.
        """
        out = {}
        for name in self.rows:
            var = self.vars.get(name)
            if var is None:
                continue
            try:
                out[name] = float(var.get())
            except ValueError:
                out[name] = self.base[name]
        return out

    def pick_replay(self):
        p = filedialog.askopenfilename(title="Touchstone",
                                       filetypes=[("Touchstone", "*.s2p *.s1p"),
                                                  ("all", "*")])
        if p:
            self.load_replay(p)

    NAME_CHARS = 26

    def _link(self, name, state="", color=DIM):
        """The one place that says what we are talking to.

        The NAME is clamped and the STATE is appended afterwards, never the
        other way round: truncating the whole string would drop the sweep
        counter off the end, which is the half that changes.  The label has a
        fixed width so no state change shifts the rest of the toolbar.
        """
        name = " ".join(str(name).split())
        if len(name) > self.NAME_CHARS:
            name = name[:self.NAME_CHARS - 1].rstrip() + "\u2026"
        txt = f"{name} - {state}" if state else name
        self.link.configure(text=f"\u25cf  {txt}", fg=color)

    def _link_idle(self):
        """Restore the indicator to whatever the source is, not running."""
        if self.dev is None:
            self._link("no source", color=DIM)
        elif isinstance(self.dev, FileSource):
            # No state word for a file: `self.acq` outlives a stop(), so keying
            # off it labelled a stopped replay "replaying".
            self._link(self._srcname, "", C_MEAS21)
        else:
            self._link(self._srcname, "idle", OK)

    def _set_device(self, dev):
        """Adopt a source, handing back whatever we were driving before.

        Switching from the instrument to a replay file used to just overwrite
        self.dev, which left the H4 paused with nothing holding a reference to
        it -- an instrument frozen by a window the user had moved on from.
        """
        self.stop()
        old = self.dev
        self.dev = dev
        if old is not None and old is not dev:
            try:
                old.close()
            except Exception:                               # noqa: BLE001
                pass
        self._link_idle()

    def on_close(self):
        self.stop()
        self._link("closing", "handing the trace back", DIM)
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:                               # noqa: BLE001
                pass
            self.dev = None
        self.root.destroy()

    def load_replay(self, path):
        try:
            ts = touchstone.load(path)
        except (ValueError, OSError) as e:
            messagebox.showerror("Touchstone", str(e))
            return
        if ts.nports < 2:
            messagebox.showerror("Touchstone", f"{path} is 1-port; tuning needs S21")
            return
        self._srcname = f"replay {os.path.basename(path)}"
        self._set_device(FileSource(ts))
        self.v_start.set(f"{ts.f[0]/1e6:.4g}")
        self.v_stop.set(f"{ts.f[-1]/1e6:.4g}")
        self.runbtn.configure(state="normal")
        self.status.configure(text=f"replay {os.path.basename(path)} - "
                                   f"{len(ts)} points, no hardware needed")

    def connect(self):
        try:
            from .vna import NanoVNA
            dev = NanoVNA()
        except Exception as e:                          # noqa: BLE001
            self._link("no VNA found", color=BAD)
            messagebox.showerror("VNA", f"{e}")
            return
        self._srcname = short_device(getattr(dev, "_info", "VNA"))
        self._set_device(dev)
        self.runbtn.configure(state="normal")
        self.status.configure(text=f"connected: {self.dev._info}")

    # -------------------------------------------------------------- acquisition
    def toggle(self):
        if self.acq and self.acq.is_alive():
            self.stop()
        else:
            self.start()

    def _cfg(self):
        num = lambda v, d: (float(v.get()) if v.get().strip() else d)
        return dict(start=num(self.v_start, 135.4) * 1e6,
                    stop=num(self.v_stop, 155.4) * 1e6,
                    points=int(num(self.v_points, 401)),
                    segments=1, average=int(num(self.v_avg, 1)))

    def start(self):
        if not self.dev:
            messagebox.showinfo("vnafit", "Connect a VNA or load a .s2p to replay first")
            return
        self.acq = Acq(self.dev, self._cfg())
        self.acq.start()
        self.runbtn.configure(text="Stop")
        self._link(self._srcname, "sweeping", OK)
        self.root.after(120, self.tick)

    def stop(self):
        if self.acq:
            self.acq.stop()
        self.runbtn.configure(text="Start")
        self._link_idle()

    def tick(self):
        if not self.acq or not self.acq.is_alive():
            self.runbtn.configure(text="Start")
            # The thread can die on its own -- a pulled USB cable ends it -- and
            # the window used to sit there looking connected.
            self._link(self._srcname, "stopped", WARN)
            return
        self.acq.cfg.update(self._cfg())
        data, err, count = self.acq.latest()
        if err:
            self.status.configure(text=f"sweep: {err}", foreground=BAD)
            self._link(self._srcname, str(err), BAD)
        elif data is not None and count != self._last_count:
            self._link(self._srcname, f"sweep {count}", OK)
            self._last_count = count
            self.meas = self.corrected(data)
            self.redraw()
            self.status.configure(text=f"sweep {count}", foreground=DIM)
            # A fit is 0.05-0.15 s on 1226 points with six free parameters,
            # against a second or more per sweep, so this keeps up with room to
            # spare.  If one is still running the sweep is simply skipped.
            if self.v_autofit.get() and not self._fitting:
                self.do_fit(quiet=True)
        self.root.after(120, self.tick)

    # ------------------------------------------------------------------ drawing
    def model(self, f):
        if not self.nl:
            return None
        try:
            b = mna.build(self.nl, self.current(),
                          w_ref=2 * np.pi * float(np.sqrt(f[0] * f[-1])))
            return b.solve(f)
        except (NetlistError, ValueError, np.linalg.LinAlgError) as e:
            self.status.configure(text=f"model: {e}", foreground=BAD)
            return None

    def redraw(self):
        f = None
        if self.meas is not None:
            f, m11, m21 = self.meas
            f = np.asarray(f, float)
            self.l_m21.set_data(f / 1e6, db(m21))
            self.l_m11.set_data(f / 1e6, db(m11))
        elif self.nl and self.nl.ac:
            f = np.linspace(self.nl.ac[2], self.nl.ac[3], self.nl.ac[1])
        if f is None or len(f) < 2:
            self.canvas.draw_idle()
            return

        S = self.model(f)
        if S is not None:
            self.l_f21.set_data(f / 1e6, db(S[:, 1, 0]))
            self.l_f11.set_data(f / 1e6, db(S[:, 0, 0]))

        St = self.target_model(f) if self.v_target.get() else None
        if St is not None:
            self.l_t21.set_data(f / 1e6, db(St[:, 1, 0]))
            self.l_t11.set_data(f / 1e6, db(St[:, 0, 0]))
            if self._zoom is None:
                self._zoom = self.feature(f, db(St[:, 1, 0]))
        else:
            self.l_t21.set_data([], [])
            self.l_t11.set_data([], [])
        self._zoom_axes(f, St, S)
        self._St = St
        if self.v_alt.get() and S is not None:
            self._draw_alt(f, S)
        else:
            self.l_a21.set_data([], [])

        self.ax21.set_xlim(f[0] / 1e6, f[-1] / 1e6)
        self._autorange(f, S)
        self._readout(f, S)
        self.canvas.draw_idle()

    def _draw_alt(self, f, S):
        """Fit the coupling-matrix model to the NETLIST and overlay it.

        Not a second fit to the data -- a translation of the model now on
        screen into the (f0, k, Qe) language the bench procedure is written in.
        Where the dotted line peels away from the dashed one is exactly where
        the narrowband model stops being able to describe this circuit.
        """
        from . import coupled as M
        try:
            d = db(S[:, 1, 0])
            f0 = f[int(np.argmax(d))]
            r = M.fit(f, S[:, 0, 0], S[:, 1, 0], 2, 1000.0,
                      seed=dict(f0s=[f0, f0], ks=[0.01], Qe1=60, Qen=60))
            _, b = M.response(f, r["f0s"], r["ks"], r["Qe1"], r["Qen"], 1000.0)
            self.l_a21.set_data(f / 1e6, db(b))
            self._alt = r
        except Exception:                                # noqa: BLE001
            self.l_a21.set_data([], [])
            self._alt = None

    def _autorange(self, f, S):
        """Quantised and hysteretic: expand at once, shrink reluctantly.

        An axis that rescales on every sweep is unreadable while you are turning
        something, which is precisely when you are looking at it.
        """
        vals = []
        if self.meas is not None:
            vals.append(db(self.meas[2]))
        if S is not None:
            vals.append(db(S[:, 1, 0]))
        if vals:
            lo = max(-130.0, min(v.min() for v in vals) - 4)
            lo = float(np.floor(lo / 10.0) * 10.0)
            cur = self._ylim21[0]
            if lo < cur or lo > cur + 20:
                self._ylim21 = (lo, 3)
                self.ax21.set_ylim(*self._ylim21)
        vals11 = []
        if self.meas is not None:
            vals11.append(db(self.meas[1]))
        if S is not None:
            vals11.append(db(S[:, 0, 0]))
        if vals11:
            lo = max(-60.0, min(v.min() for v in vals11) - 3)
            lo = float(np.floor(lo / 5.0) * 5.0)
            cur = self._ylim11[0]
            if lo < cur or lo > cur + 10:
                self._ylim11 = (lo, 1)
                self.ax11.set_ylim(*self._ylim11)

    def _readout(self, f, S):
        lines = []
        if S is not None:
            d = db(S[:, 1, 0])
            i = int(np.argmax(d))
            lines += [f"model   peak {d[i]:+7.2f} dB", f"        at   {f[i]/1e6:9.4f} MHz"]
            bw = self._bw(f, d)
            if bw:
                lines.append(f"        3 dB {bw/1e6:9.4f} MHz")
        if self.meas is not None:
            fm, m11, m21 = self.meas
            dm = db(m21)
            j = int(np.argmax(dm))
            lines += ["", f"meas    peak {dm[j]:+7.2f} dB", f"        at   {fm[j]/1e6:9.4f} MHz"]
            bw = self._bw(np.asarray(fm, float), dm)
            if bw:
                lines.append(f"        3 dB {bw/1e6:9.4f} MHz")
            lines.append(f"        RL   {-20*np.log10(max(np.abs(m11).min(),1e-9)):9.2f} dB")
            got = self.passband_pts()
            if got:
                n, _bw, (lo, hi) = got
                lines.append(f"        pts in 3 dB {n:4d}")
                if n < self.PASSBAND_PTS:
                    lines += [f"        THIN - a fit will be",
                              f"        driven by the stopband.",
                              f"        try {lo/1e6:.1f}-{hi/1e6:.1f} MHz"]

        if S is not None and self.meas is not None and len(f) == len(self.meas[0]):
            dd = db(S[:, 1, 0]) - db(self.meas[2])
            pk = f[int(np.argmax(db(self.meas[2])))]
            near = np.abs(f - pk) < 0.02 * pk
            far = ~near
            # Split deliberately: every model agrees in the passband, so one
            # number hides the part that carries the information.
            lines += ["", "residual (model - meas)"]
            if near.any():
                lines.append(f"  passband {np.sqrt(np.mean(dd[near]**2)):7.3f} dB")
            if far.any():
                lines.append(f"  skirts   {np.sqrt(np.mean(dd[far]**2)):7.3f} dB")
        St = getattr(self, "_St", None)
        if St is not None and self.meas is not None and len(f) == len(self.meas[0]):
            dt = db(St[:, 1, 0]) - db(self.meas[2])
            pk = f[int(np.argmax(db(St[:, 1, 0])))]
            near = np.abs(f - pk) < 0.02 * pk
            lines += ["", "tune to target"]
            if near.any():
                lines.append(f"  trace passband {np.sqrt(np.mean(dt[near]**2)):7.3f} dB")
            if (~near).any():
                lines.append(f"        skirts   {np.sqrt(np.mean(dt[~near]**2)):7.3f} dB")
        tl = self.tuning_lines()
        if tl:
            lines += [""] + tl
        if self.fitres is not None:
            r = self.fitres
            lines += ["", f"last fit {r.rms:7.3f} dB rms",
                      f"  chi2/dof {r.chi2_red:8.1f}"]
            if r.chi2_red > 4.0:
                # Not a scolding: above ~4 the leftover is model error, and
                # every error bar computed from it is too small.
                lines.append(f"  residual is {np.sqrt(r.chi2_red):.0f}x the")
                lines.append(f"  assumed noise -- model")
                lines.append(f"  error, not measurement")
        if self.import_report:
            lines += ["", "imported from SPICE"] + [
                "  " + ln.strip() for ln in self.import_report.splitlines()]
        alt = getattr(self, "_alt", None)
        if self.v_alt.get() and alt:
            lines += ["", "as a coupling matrix",
                      f"  k    {alt['ks'][0]:9.5f}",
                      f"  Qe   {alt['Qe1']:6.1f} /{alt['Qen']:6.1f}",
                      f"  fit  {alt['rms']:7.3f} dB"]
        self.readout.configure(state="normal")
        self.readout.delete("1.0", "end")
        self.readout.insert("1.0", "\n".join(lines))
        self.readout.configure(state="disabled")

    PASSBAND_PTS = 15            # inside the 3 dB width, below which say so

    def passband_pts(self):
        """How many measured points land inside the measured 3 dB width.

        The number that decides whether a fit means anything, and the one this
        project has been caught by: a 401-point sweep looks generous until it is
        spread over 220 MHz and the 10 MHz passband gets 18 of them.  The peak,
        the width and every parameter that acts through them are then being
        fitted to a handful of samples, and the fit will still report a
        confident rms because the 380 stopband points agree beautifully.

        Returns (points, bandwidth, suggested_span) or None.
        """
        if self.meas is None:
            return None
        fm, _m11, m21 = self.meas
        fm, dm = np.asarray(fm, float), db(m21)
        bw = self._bw(fm, dm)
        if not bw:
            return None
        pk = fm[int(np.argmax(dm))]
        n = int(np.sum(np.abs(fm - pk) <= bw / 2))
        return n, bw, (pk - 3 * bw, pk + 3 * bw)

    @staticmethod
    def _bw(f, d, level=3.0):
        inside = np.flatnonzero(d >= d.max() - level)
        if len(inside) < 2 or inside[0] == 0 or inside[-1] == len(f) - 1:
            return None
        return float(f[inside[-1]] - f[inside[0]])

    # -------------------------------------------------------------------- files
    def save(self):
        if self.meas is None:
            messagebox.showinfo("vnafit", "nothing measured yet")
            return
        p = filedialog.asksaveasfilename(defaultextension=".s2p",
                                         filetypes=[("Touchstone", "*.s2p")])
        if not p:
            return
        f, s11, s21 = self.meas
        S = np.zeros((len(f), 2, 2), complex)
        S[:, 0, 0], S[:, 1, 0] = s11, s21
        touchstone.save(p, f, S, comments=[
            "vnafit live capture",
            f"device {getattr(self.dev, 'portname', '?')}",
            f"netlist {os.path.basename(self.netlist_path or '-')}",
            "S12 and S22 are NOT measured by this instrument; written as zero"])
        self.status.configure(text=f"wrote {os.path.basename(p)}", foreground=OK)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    netlist = replay = None
    for a in argv:
        if a.endswith((".s2p", ".s1p")):
            replay = a
        elif not a.startswith("-"):
            netlist = a
    root = tk.Tk()
    app = App(root, netlist, replay)
    try:
        root.mainloop()
    finally:
        # Belt and braces: an exception out of the event loop must not leave the
        # instrument frozen either.
        if getattr(app, "dev", None) is not None:
            try:
                app.dev.close()
            except Exception:                               # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
