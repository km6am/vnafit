"""Design a helical filter and load it, without leaving the tuning window.

Nothing here computes anything.  `helical.design` builds the netlist and the
main window already draws it, cards it, fits it, zooms the passband and applies
a calibration -- this is the four numbers that choose which filter, and a live
readout of what those numbers imply on the bench.

The design is regenerated on every keystroke because that is the question being
asked: what would three sections look like instead of two?  Making that a button
press turns a comparison into a chore.
"""
import tkinter as tk
from tkinter import ttk

from . import helical


class HelicalWindow(tk.Toplevel):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.title("vnafit - helical filter")
        self._build()
        self.refresh()

    def _build(self):
        head = ttk.Frame(self, padding=8)
        head.pack(fill="x")
        ttk.Label(head, text="Helical filter", font=("TkDefaultFont", 12, "bold")
                  ).pack(anchor="w")
        ttk.Label(head, foreground="#5f6873", wraplength=520, justify="left",
                  text="Parallel L||C resonators magnetically coupled through "
                       "the aperture, tapped at the ends.  Choosing an order "
                       "rebuilds the model; everything else in the window then "
                       "works on it unchanged."
                  ).pack(anchor="w", pady=(2, 0))

        form = ttk.Frame(self, padding=8)
        form.pack(fill="x")
        self.v = {}
        rows = (("order", "sections", "2"), ("f0", "centre MHz", "146.0"),
                ("bw", "bandwidth MHz", "2.0"),
                ("ripple", "ripple dB (0 = Butterworth)", "0.1"),
                ("qu", "unloaded Q", "670"))
        for i, (key, label, default) in enumerate(rows):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=default)
            self.v[key] = var
            if key == "order":
                w = ttk.Spinbox(form, from_=1, to=helical.MAX_SECTIONS, width=6,
                                textvariable=var, command=self.refresh)
            else:
                w = ttk.Entry(form, textvariable=var, width=12)
            w.grid(row=i, column=1, sticky="w", padx=8)
            var.trace_add("write", lambda *_a: self.refresh())

        # The bench sequence.  Each is the same filter with everything but the
        # thing being set stripped out, so it is a smaller netlist and not a
        # different mode.
        bench = ttk.LabelFrame(self, text=" Bench step ", padding=8)
        bench.pack(fill="x", padx=8, pady=(2, 0))
        self.v["stage"] = tk.StringVar(value="measure")
        self.v["section"] = tk.StringVar(value="1")
        row = ttk.Frame(bench); row.pack(fill="x")
        for st in helical.STAGES:
            ttk.Radiobutton(row, text=st, value=st, variable=self.v["stage"],
                            command=self.refresh).pack(side="left", padx=(0, 10))
        ttk.Label(row, text="section").pack(side="left", padx=(10, 3))
        ttk.Spinbox(row, from_=1, to=helical.MAX_SECTIONS, width=4,
                    textvariable=self.v["section"], command=self.refresh
                    ).pack(side="left")
        # Both variables get a trace, not just a `command` on the widget: a
        # radiobutton's command fires on a click and not when the variable is
        # set any other way, so the panel and the model can drift apart.
        for key in ("stage", "section"):
            self.v[key].trace_add("write", lambda *_a: self.refresh())
        self.hint = ttk.Label(bench, foreground="#1a7f37", wraplength=520,
                              justify="left", text="")
        self.hint.pack(anchor="w", pady=(6, 0))

        self.out = tk.Text(self, width=62, height=11, borderwidth=0, wrap="none",
                           font=("TkFixedFont", 11))
        self.out.pack(fill="both", expand=True, padx=8)
        self.out.configure(state="disabled")

        foot = ttk.Frame(self, padding=8)
        foot.pack(fill="x")
        ttk.Button(foot, text="Close", command=self.destroy).pack(side="right")
        self.loadbtn = ttk.Button(foot, text="Use this model",
                                  command=self.load)
        self.loadbtn.pack(side="right", padx=6)

    # ------------------------------------------------------------------ work
    def _design(self):
        g = lambda k: float(self.v[k].get())
        return helical.design(int(g("order")), g("f0") * 1e6, g("bw") * 1e6,
                              g("ripple"), qu=g("qu"),
                              stage=self.v["stage"].get(),
                              section=int(g("section")))

    def refresh(self):
        try:
            nl = self._design()
            text = (helical.summary(nl.targets) + "\n\n"
                    + f"{len(nl.elements)} elements, "
                    + f"{len(nl.fits)} parameters that may be fitted:\n  "
                    + ", ".join(f.param for f in nl.fits)
                    + "\n\nlref is not among them.  With tapped ports every\n"
                      "impedance scales together, so L is an exact gauge and a\n"
                      "fit that frees it slides until it hits a bound.")
            self.hint.configure(text=helical.stage_hint(nl))
            self.loadbtn.configure(state="normal")
        except Exception as e:                              # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            self.hint.configure(text="")
            self.loadbtn.configure(state="disabled")
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", text)
        self.out.configure(state="disabled")

    def load(self):
        try:
            nl = self._design()
        except Exception:                                   # noqa: BLE001
            return
        stage = self.v["stage"].get()
        self.app.adopt_netlist(
            nl, f"helical {self.v['order'].get()}-section"
                + ("" if stage == "measure"
                   else f", {stage} {self.v['section'].get()}"))
        self.destroy()
