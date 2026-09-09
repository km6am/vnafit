"""Calibrations you own: derive them, keep them, apply them, stitch them.

The instrument's own calibration lives in five slots with no record of the span
it was taken over, no notes, and no way to get it back out except as numbers.
That is enough to lose an afternoon to -- this project did, on a filter whose
"flat 2.4 dB loss" was a cal being interpolated a long way outside where it was
taken.  A calibration you cannot audit is a number you have to trust.

So a `Calibration` here carries its error terms AND the standards they came
from, the span, the point count, the instrument's identity, the date and your
notes, in one file.  It can always be re-derived, which means it can be checked.

**The algebra is the firmware's, not an invention.**  Read from `apply_error_term`
and the `eterm_calc_*` functions of NanoVNA-D so that a sweep corrected here and
the same sweep corrected on the instrument agree:

    S11c = (S11m - ED) / (ER + ES*(S11m - ED))
    S21c = (S21m - EX) / ET                       ET = S21_thru - EX
           * (1 - ES*S11c)                        enhanced response

and from the standards, with ED = S11_load and O, S the open and short
measurements less ED,

    ES = (O + S) / (O - S)        ER = O * (1 - ES)

which is the ordinary three-term solve, and reduces to what the firmware does
when the standards are ideal.  `Touchstone.normalize(thru)` turns out to be
exactly the non-enhanced half of the S21 correction.

**What this cannot do**, because the hardware cannot: a full 12-term two-port
calibration -- there is no S22 or S12 to solve the reverse terms from -- and
uploading a calibration back into the instrument, for which the firmware
exposes no command at all.
"""
import json
import os
import time

import numpy as np

from .touchstone import Touchstone

TERMS = ("ED", "ES", "ER", "ET", "EX")

# Ideal standards.  Real ones have a delay and a fringing capacitance, and a
# model for them belongs here later -- but assuming ideal and SAYING SO beats
# quietly applying somebody's coefficients for a kit you may not own.
IDEAL = {"open": 1.0 + 0j, "short": -1.0 + 0j, "load": 0.0 + 0j}


# Optional notes about the setup.  All of them are useful and none is required;
# whatever is filled in is kept and shown, and whatever is not is not mentioned.
FIXTURE_FIELDS = (
    ("reference_plane", "reference plane", "cable ends, instrument, fixture"),
    ("cables", "cables", "type and length"),
    ("kit", "standards kit", ""),
    ("thru", "thru", "a barrel has length; a cable has more"),
    ("temperature", "temperature", ""),
    ("notes", "notes", ""),
)
FIXTURE_KEYS = tuple(k for k, _l, _h in FIXTURE_FIELDS)


def _were_raw(standards):
    """True / False / None: were the standards swept with correction OFF?

    They have to be.  Calibrating from sweeps the instrument already corrected
    is circular, and it does not fail loudly -- it gives error terms near unity
    that look like an unusually good fixture.  Read from the comments this
    project writes; a file from elsewhere says nothing, and None is reported
    rather than guessed.
    """
    seen = set()
    for ts in standards.values():
        for c in getattr(ts, "comments", ()):
            if "instrument correction" in c.lower():
                seen.add("off" if "off" in c.lower() else "on")
    return None if not seen else (seen == {"off"})


class CalError(ValueError):
    """A calibration that cannot be built, or cannot be applied where asked."""


def _same_grid(a, b, what):
    if len(a) != len(b) or not np.allclose(a, b):
        raise CalError(
            f"{what} is on a different frequency grid ({len(b)} points, "
            f"{b[0]/1e6:.4f}-{b[-1]/1e6:.4f} MHz) from the first standard "
            f"({len(a)} points, {a[0]/1e6:.4f}-{a[-1]/1e6:.4f} MHz).  Every "
            f"standard has to be swept with identical settings.")


class Calibration:
    """Error terms on a frequency grid, with everything needed to check them."""

    def __init__(self, f, terms, meta=None, standards=None):
        self.f = np.asarray(f, float)
        self.terms = {k: np.asarray(v, complex) for k, v in terms.items()}
        for k in ("ED", "ES", "ER"):
            if k not in self.terms:
                raise CalError(f"a calibration needs at least ED, ES and ER; "
                               f"{k} is missing")
        for k, v in self.terms.items():
            if len(v) != len(self.f):
                raise CalError(f"{k} has {len(v)} points, the grid has "
                               f"{len(self.f)}")
        self.meta = dict(meta or {})
        self.standards = dict(standards or {})
        self.meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.meta.setdefault("notes", "")

    # ------------------------------------------------------------- properties
    start = property(lambda self: float(self.f[0]))
    stop = property(lambda self: float(self.f[-1]))
    points = property(lambda self: len(self.f))
    has_thru = property(lambda self: "ET" in self.terms)
    spacing = property(lambda self: (self.stop - self.start) / max(self.points - 1, 1))

    def __repr__(self):
        return (f"<Calibration {self.start/1e6:.4f}-{self.stop/1e6:.4f} MHz, "
                f"{self.points} pts, {'S11+S21' if self.has_thru else 'S11 only'}"
                f"{', ' + self.meta['notes'] if self.meta.get('notes') else ''}>")

    def describe(self):
        m = self.meta
        out = [f"{self.start/1e6:.4f}-{self.stop/1e6:.4f} MHz, {self.points} "
               f"points, {self.spacing/1e3:.3f} kHz spacing",
               "corrects:        " + ("S11 and S21" if self.has_thru
                                      else "S11 only (no thru)"),
               f"made:            {m.get('source', 'unknown')} on {m.get('created')}"]
        if m.get("instrument"):
            out.append(f"device:          {m['instrument']}")
        for key, label, _hint in FIXTURE_FIELDS:
            if m.get(key):
                out.append(f"{(label + ':'):<17}{m[key]}")
        if m.get("standards_raw") is False:
            out.append("WARNING: a standard was swept with the instrument's own "
                       "correction on, which makes this calibration circular")
        if self.standards:
            out.append("standards:       " + ", ".join(sorted(self.standards)))
        else:
            out.append("standards:       not embedded, so this cal cannot be re-derived")
        return "\n".join(out)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_standards(cls, open_, short_, load, thru=None, isoln=None,
                       notes="", instrument=None, ideal=None, fixture=None):
        """Solve the three-term model, and the thru terms if given."""
        ideal = {**IDEAL, **(ideal or {})}
        f = np.asarray(open_.f, float)
        for name, ts in (("short", short_), ("load", load),
                         ("thru", thru), ("isolation", isoln)):
            if ts is not None:
                _same_grid(f, ts.f, f"the {name} standard")

        ED = np.asarray(load.s11, complex)
        O = np.asarray(open_.s11, complex) - ED
        S = np.asarray(short_.s11, complex) - ED
        den = O - S
        if np.any(np.abs(den) < 1e-12):
            raise CalError(
                "the open and short measured the same thing at some "
                "frequencies, so the calibration cannot be solved.  Usually "
                "that means one standard was not actually connected.")
        ES = (O + S) / den
        ER = O * (1.0 - ES)

        terms = {"ED": ED, "ES": ES, "ER": ER}
        std = {"open": open_, "short": short_, "load": load}
        if isoln is not None:
            terms["EX"] = np.asarray(isoln.s21, complex)
            std["isolation"] = isoln
        if thru is not None:
            EX = terms.get("EX", np.zeros_like(ED))
            ET = np.asarray(thru.s21, complex) - EX
            if np.any(np.abs(ET) < 1e-12):
                raise CalError("the thru measured zero transmission at some "
                               "frequencies -- was it actually connected?")
            terms["ET"] = ET
            std["thru"] = thru
        meta = {"source": "standards", "notes": notes,
                "instrument": instrument,
                "standards_raw": _were_raw(std),
                "standards_ideal": {k: [v.real, v.imag] for k, v in ideal.items()}}
        meta.update({k: v for k, v in (fixture or {}).items()
                     if k in FIXTURE_KEYS and v})
        return cls(f, terms, standards=std, meta=meta)

    @classmethod
    def from_instrument(cls, dev, slot=None, notes=""):
        """Read the terms the instrument itself is holding.

        Downloading works (`data 2..6`); there is no way to put one back, so
        this is an archive and a starting point, not a round trip.
        """
        if slot is not None:
            dev.recall_cal(int(slot))
        st = dev.cal_status()
        got = dev.cal_terms()
        n = max(len(v) for v in got.values())
        if not n:
            raise CalError("the instrument returned no calibration data; "
                           "`cal` says: " + " ".join(st["raw"]) or "(nothing)")
        # The instrument does not report the grid its cal was taken on.  That
        # is the whole reason this module exists; record the honest answer.
        f = np.arange(n, dtype=float)
        terms = {k: v for k, v in got.items() if len(v) == n}
        return cls(f, terms, meta={
            "source": f"instrument slot {slot}" if slot is not None
                      else "instrument, current",
            "notes": notes, "instrument": getattr(dev, "_info", None),
            "grid_unknown": True,
            "cal_status": " ".join(st["raw"]).strip()})

    # ------------------------------------------------------------- applying
    def interpolated(self, f):
        """The terms on another grid, refusing to go outside our own."""
        f = np.asarray(f, float)
        if f[0] < self.start - 1e-6 or f[-1] > self.stop + 1e-6:
            raise CalError(
                f"this calibration covers {self.start/1e6:.4f}-"
                f"{self.stop/1e6:.4f} MHz and the sweep asks for "
                f"{f[0]/1e6:.4f}-{f[-1]/1e6:.4f} MHz.  Extrapolating error "
                f"terms is how a cal quietly stops meaning anything; take one "
                f"over the span you intend to use, or stitch.")
        out = {}
        for k, v in self.terms.items():
            out[k] = (np.interp(f, self.f, v.real)
                      + 1j * np.interp(f, self.f, v.imag))
        return out

    def apply(self, ts, enhanced=True):
        """Correct a raw sweep.  Returns a new Touchstone."""
        t = self.interpolated(ts.f)
        s = np.array(ts.s, complex)
        d = np.asarray(ts.s11, complex) - t["ED"]
        s11c = d / (t["ER"] + t["ES"] * d)
        s[:, 0, 0] = s11c
        note = [f"corrected by vnafit using {self.meta.get('source', 'a cal')}"
                f" ({self.start/1e6:.4f}-{self.stop/1e6:.4f} MHz, "
                f"{self.points} cal points)"]
        if self.has_thru and ts.nports > 1:
            s21c = (np.asarray(ts.s21, complex) - t.get(
                "EX", np.zeros(len(ts.f), complex))) / t["ET"]
            if enhanced:
                s21c = s21c * (1.0 - t["ES"] * s11c)
            s[:, 1, 0] = s21c
            note.append("S21 corrected"
                        + (" (enhanced response)" if enhanced else ""))
        else:
            note.append("S11 only -- this calibration has no thru, so S21 is "
                        "left as measured")
        ratio = len(ts.f) / self.points
        if ratio > 1.05:
            # The criticism this project levelled at scan_hires applies to us.
            note.append(f"NOTE: the sweep has {ratio:.1f}x more points than the "
                        f"calibration, so the terms are interpolated between "
                        f"cal points -- the resolution is the cal's, not the "
                        f"sweep's")
        if self.meta.get("notes"):
            note.append("cal notes: " + self.meta["notes"])
        return Touchstone(ts.f, s, ts.z0, list(ts.comments) + note, ts.path)

    # ------------------------------------------------------------------- io
    def save(self, path):
        arrays = {f"term_{k}": v for k, v in self.terms.items()}
        arrays["f"] = self.f
        for name, ts in self.standards.items():
            arrays[f"std_{name}_f"] = ts.f
            arrays[f"std_{name}_s"] = ts.s
        meta = dict(self.meta)
        meta["terms"] = sorted(self.terms)
        meta["standards"] = sorted(self.standards)
        # Through a FILE OBJECT, not a path: np.savez_compressed appends ".npz"
        # to any name that does not already end in it, so saving "2m.calz"
        # silently produced "2m.calz.npz" and load() then could not find it.
        with open(path, "wb") as fh:
            np.savez_compressed(fh, meta=json.dumps(meta), **arrays)
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        terms = {k: z[f"term_{k}"] for k in meta.get("terms", TERMS)
                 if f"term_{k}" in z}
        std = {}
        for name in meta.get("standards", []):
            if f"std_{name}_s" in z:
                std[name] = Touchstone(z[f"std_{name}_f"], z[f"std_{name}_s"])
        return cls(z["f"], terms, meta=meta, standards=std)

    def rederive(self):
        """Rebuild from the embedded standards.  The audit this file exists for."""
        if not {"open", "short", "load"} <= set(self.standards):
            raise CalError("this calibration has no embedded standards, so it "
                           "cannot be re-derived -- only trusted")
        return Calibration.from_standards(
            self.standards["open"], self.standards["short"],
            self.standards["load"], self.standards.get("thru"),
            self.standards.get("isolation"),
            notes=self.meta.get("notes", ""),
            instrument=self.meta.get("instrument"))


class CalBank:
    """Several calibrations covering different spans, used as one.

    Where two overlap the FINER one wins outright.  They are never blended:
    terms taken an hour and a few degrees apart do not average into something
    meaningful, they average into a discontinuity you cannot see afterwards.
    """

    def __init__(self, cals):
        self.cals = sorted(cals, key=lambda c: c.start)
        if not self.cals:
            raise CalError("a bank needs at least one calibration")

    def __repr__(self):
        return f"<CalBank {len(self.cals)} cals, {self.coverage()}>"

    def coverage(self):
        return ", ".join(f"{c.start/1e6:.3f}-{c.stop/1e6:.3f}" for c in self.cals) + " MHz"

    def pick(self, f):
        """Which calibration covers each frequency.  None where none does."""
        f = np.asarray(f, float)
        out = np.full(len(f), -1, int)
        for i, c in enumerate(self.cals):
            inside = (f >= c.start - 1e-6) & (f <= c.stop + 1e-6)
            better = inside & ((out < 0) | np.array(
                [c.spacing < self.cals[j].spacing if j >= 0 else True
                 for j in out]))
            out[better] = i
        return out

    def gaps(self, f):
        who = self.pick(f)
        f = np.asarray(f, float)
        miss = f[who < 0]
        if not len(miss):
            return []
        breaks = np.flatnonzero(np.diff(miss) > 1.5 * np.median(np.diff(f)))
        edges = np.split(miss, breaks + 1)
        return [(float(e[0]), float(e[-1])) for e in edges if len(e)]

    def apply(self, ts, enhanced=True):
        who = self.pick(ts.f)
        gaps = self.gaps(ts.f)
        if gaps:
            raise CalError(
                "no calibration covers " + ", ".join(
                    f"{a/1e6:.4f}-{b/1e6:.4f} MHz" for a, b in gaps) +
                f".  This bank covers {self.coverage()}.")
        s = np.array(ts.s, complex)
        used, edges = [], []
        for i in sorted(set(who)):
            m = who == i
            part = Touchstone(ts.f[m], ts.s[m], ts.z0)
            got = self.cals[i].apply(part, enhanced=enhanced)
            s[m] = got.s
            used.append(self.cals[i])
            edges.append(f"{ts.f[m][0]/1e6:.4f}-{ts.f[m][-1]/1e6:.4f} MHz from "
                         f"{self.cals[i].meta.get('source', 'a cal')}"
                         f" ({self.cals[i].spacing/1e3:.3f} kHz)")
        return Touchstone(ts.f, s, ts.z0,
                          list(ts.comments) + ["stitched calibration:"] +
                          ["  " + e for e in edges], ts.path)
