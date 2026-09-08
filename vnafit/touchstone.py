"""Touchstone (.s1p / .s2p) reading and writing.

There were three parsers across the older trees and only one of them honoured
the option line.  The other two assumed `Hz S RI R 50` unconditionally, which is
right for what this bench happens to save and silently wrong for a file from
anywhere else -- a NanoVNA-Saver export in MA or DB format reads as garbage with
no error at all.  This is the one parser.

Notes on the format that matter here:

  * The option line is `# <freq unit> <parameter> <format> R <impedance>`, any
    field may be omitted, and the defaults are `GHz S MA R 50`.  Defaulting to
    Hz -- as the old parsers effectively did -- is a 1e9 error on a file that
    omits the unit.
  * A 2-port data line is  f  S11 S21 S12 S22  (note S21 BEFORE S12; that
    column order is unique to 2-ports and catches people out).
  * Comment lines start with `!`.  Instrument metadata lives there, so the
    comments are kept rather than discarded.
"""
import os
import re

import numpy as np

def _db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-15)) if x is not None else None


UNIT = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
FORMATS = ("ri", "ma", "db")


class Touchstone:
    """Frequencies in Hz plus an (nfreq, nports, nports) S-matrix."""

    def __init__(self, f, s, z0=50.0, comments=None, path=None):
        self.f = np.asarray(f, float)
        self.s = np.asarray(s, complex)
        if self.s.ndim != 3 or self.s.shape[1] != self.s.shape[2]:
            raise ValueError(f"S must be (nfreq, n, n), got {self.s.shape}")
        if len(self.f) != len(self.s):
            raise ValueError("frequency and S length disagree")
        self.z0 = float(z0)
        self.comments = list(comments or [])
        self.path = path

    nports = property(lambda self: self.s.shape[1])
    s11 = property(lambda self: self.s[:, 0, 0])
    s21 = property(lambda self: self.s[:, 1, 0] if self.nports > 1 else None)
    s12 = property(lambda self: self.s[:, 0, 1] if self.nports > 1 else None)
    s22 = property(lambda self: self.s[:, 1, 1] if self.nports > 1 else None)

    # ------------------------------------------------- for a notebook session
    # Plain numpy arrays, named the way you would say them out loud, so a custom
    # plot is one line and needs no helper of your own.  Everything here is a
    # view or a fresh array -- nothing mutates the object.
    f_mhz = property(lambda self: self.f / 1e6)
    f_ghz = property(lambda self: self.f / 1e9)
    s11_db = property(lambda self: _db(self.s11))
    s21_db = property(lambda self: _db(self.s21))
    s12_db = property(lambda self: _db(self.s12))
    s22_db = property(lambda self: _db(self.s22))
    s11_deg = property(lambda self: np.angle(self.s11, deg=True))
    s21_deg = property(lambda self: np.angle(self.s21, deg=True))
    return_loss = property(lambda self: -_db(self.s11))
    insertion_loss = property(lambda self: -_db(self.s21))

    @property
    def vswr(self):
        m = np.clip(np.abs(self.s11), 0, 0.999999)
        return (1 + m) / (1 - m)

    @property
    def z_in(self):
        """Input impedance looking into port 1, in ohms."""
        return self.z0 * (1 + self.s11) / (1 - self.s11)

    @property
    def group_delay(self):
        """S21 group delay in seconds, -d(phase)/d(omega).

        Differentiated on the UNWRAPPED phase; without unwrapping every 2-pi
        crossing becomes a spike that looks like a resonance.
        """
        ph = np.unwrap(np.angle(self.s21))
        return -np.gradient(ph, 2 * np.pi * self.f)

    def at(self, freq):
        """Everything at the point nearest `freq`, as a dict.  `d.at(146e6)`."""
        i = int(np.argmin(np.abs(self.f - freq)))
        out = dict(i=i, f=float(self.f[i]), f_mhz=float(self.f[i] / 1e6))
        for name in ("s11", "s21", "s12", "s22"):
            v = getattr(self, name)
            if v is not None:
                out[name] = complex(v[i])
                out[name + "_db"] = float(_db(v)[i])
        out["vswr"] = float(self.vswr[i])
        return out

    def band(self, lo=None, hi=None):
        """A copy restricted to a frequency range.  `d.band(140e6, 150e6)`."""
        m = np.ones(len(self.f), bool)
        if lo is not None:
            m &= self.f >= lo
        if hi is not None:
            m &= self.f <= hi
        if not m.any():
            raise ValueError(f"no points between {lo} and {hi}; the file covers "
                             f"{self.f[0]:.0f}-{self.f[-1]:.0f} Hz")
        return Touchstone(self.f[m], self.s[m], self.z0, self.comments, self.path)

    def normalize(self, thru):
        """Divide S21 by a THRU sweep, removing the fixture from transmission.

        Response normalisation, not a calibration: it takes out the cables' loss
        and delay but does nothing for mismatch, and it cannot fix S11 at all.
        Still, it is one minute at the bench and it settles the commonest
        question there is -- "is that 2 dB the filter or my setup?".

        `thru` is another Touchstone taken with the DUT replaced by the two
        cables joined together, on the same frequency grid.
        """
        if len(thru) != len(self) or not np.allclose(thru.f, self.f):
            raise ValueError(
                f"the thru sweep is on a different grid ({len(thru)} points, "
                f"{thru.f[0]/1e6:.4f}-{thru.f[-1]/1e6:.4f} MHz) from this one "
                f"({len(self)} points, {self.f[0]/1e6:.4f}-{self.f[-1]/1e6:.4f} "
                f"MHz).  Sweep both with identical settings.")
        s = self.s.copy()
        ref = np.where(np.abs(thru.s21) > 1e-12, thru.s21, 1e-12)
        s[:, 1, 0] = self.s21 / ref
        if self.nports > 1:
            s[:, 0, 1] = self.s12 / ref
        return Touchstone(self.f, s, self.z0,
                          self.comments + [f"normalised by thru "
                                           f"{os.path.basename(thru.path or '')}"],
                          self.path)

    def peak(self, what="s21"):
        """Frequency and level of the strongest point of `what`."""
        v = getattr(self, what)
        i = int(np.argmax(np.abs(v)))
        return float(self.f[i]), float(_db(v)[i])

    def bandwidth(self, level_db=3.0, what="s21"):
        """Width at `level_db` below the peak, or None if the sweep is too
        narrow to contain both crossings -- None rather than a number computed
        off the edge of the data."""
        v = _db(getattr(self, what))
        inside = np.flatnonzero(v >= v.max() - level_db)
        if len(inside) < 2 or inside[0] == 0 or inside[-1] == len(self.f) - 1:
            return None
        return float(self.f[inside[-1]] - self.f[inside[0]])

    def columns(self):
        """A plain dict of arrays -- hand straight to pandas or to a plot."""
        out = {"f": self.f, "f_mhz": self.f_mhz}
        for name in ("s11", "s21", "s12", "s22"):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
                out[name + "_db"] = _db(v)
        out["vswr"] = self.vswr
        return out

    def plot(self, ax=None, what=("s21", "s11"), **kw):
        """A quick look.  Returns the axis, so you can keep drawing on it."""
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=kw.pop("figsize", (8, 4.2)))
        for name in ([what] if isinstance(what, str) else what):
            v = getattr(self, name)
            if v is None:
                continue
            ax.plot(self.f_mhz, _db(v),
                    label=f"{name} {os.path.basename(self.path or '')}".strip(),
                    **kw)
        ax.set_xlabel("MHz")
        ax.set_ylabel("dB")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        return ax

    def __len__(self):
        return len(self.f)

    def __repr__(self):
        name = os.path.basename(self.path) if self.path else "in memory"
        extra = ""
        if self.nports > 1 and len(self.f):
            fp, pk = self.peak("s21")
            extra = f", peak S21 {pk:+.2f} dB at {fp/1e6:.4f} MHz"
        return (f"<{name}: {self.nports}-port, {len(self.f)} pts, "
                f"{self.f[0]/1e6:.4f}-{self.f[-1]/1e6:.4f} MHz{extra}>")

    def meta(self):
        """Instrument metadata this bench writes into the comment block."""
        out = {}
        for c in self.comments:
            m = re.search(r"device\s+(\S+)\s*(?:\(([^)]*)\))?", c)
            if m:
                out["port"], out["device"] = m.group(1), (m.group(2) or "").strip()
            m = re.search(r"points\s+(\d+)", c)
            if m:
                out["points"] = int(m.group(1))
            m = re.search(r"span\s+([\d.]+)-([\d.]+)\s*MHz", c)
            if m:
                out["span_mhz"] = (float(m.group(1)), float(m.group(2)))
            m = re.match(r"\s*saved\s+(.*)", c)
            if m:
                out["saved"] = m.group(1).strip()
        return out

    def s21_unphysical(self, tol=1.0001):
        """True where |S21| exceeds unity -- a passive DUT cannot, so this is a
        calibration or normalisation artefact, not data."""
        return np.abs(self.s21) > tol if self.nports > 1 else np.zeros(0, bool)


def parse(text, path=None):
    unit, fmt, z0 = 1e9, "ma", 50.0          # Touchstone defaults: GHz S MA R 50
    comments, rows = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!"):
            comments.append(line[1:].rstrip())
            continue
        line = line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            toks = line[1:].split()
            i = 0
            while i < len(toks):
                t = toks[i].lower()
                if t in UNIT:
                    unit = UNIT[t]
                elif t in FORMATS:
                    fmt = t
                elif t == "r" and i + 1 < len(toks):
                    z0 = float(toks[i + 1])
                    i += 1
                i += 1
            continue
        rows.append([float(x) for x in line.replace(",", " ").split()])

    if not rows:
        raise ValueError(f"no data rows in {path or 'touchstone text'}")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("ragged data rows -- multi-line 3+ port files are not "
                         "supported; use a 1- or 2-port file")
    npairs = (width - 1) // 2
    if npairs == 1:
        n = 1
    elif npairs == 4:
        n = 2
    else:
        raise ValueError(f"{npairs} S-parameter pairs per row: only 1-port and "
                         f"2-port Touchstone files are supported")

    a = np.asarray(rows, float)
    f = a[:, 0] * unit
    pairs = [_to_complex(a[:, 1 + 2 * i], a[:, 2 + 2 * i], fmt) for i in range(npairs)]

    s = np.zeros((len(f), n, n), complex)
    if n == 1:
        s[:, 0, 0] = pairs[0]
    else:
        # The 2-port column order is S11 S21 S12 S22 -- S21 comes SECOND.
        s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1] = pairs
    return Touchstone(f, s, z0, comments, path)


def _to_complex(x, y, fmt):
    if fmt == "ri":
        return x + 1j * y
    if fmt == "ma":
        return x * np.exp(1j * np.deg2rad(y))
    if fmt == "db":
        return 10 ** (x / 20.0) * np.exp(1j * np.deg2rad(y))
    raise ValueError(f"unknown Touchstone format {fmt!r}")


def load(path):
    with open(path, "r", errors="replace") as fh:
        return parse(fh.read(), path=os.path.abspath(path))


def save(path, f, s, z0=50.0, comments=()):
    """Write a .s1p or .s2p in `Hz S RI R <z0>`.

    Real S12/S22 are written when they are given.  The older writer padded both
    with zeros, which round-trips through a reader that ignores them and lies to
    one that does not.
    """
    f = np.asarray(f, float)
    s = np.asarray(s, complex)
    if s.ndim == 1:
        s = s.reshape(-1, 1, 1)
    n = s.shape[1]
    if n not in (1, 2):
        raise ValueError("only 1- and 2-port files can be written")
    with open(path, "w") as fh:
        for c in comments:
            fh.write(f"! {c}\n")
        fh.write(f"# Hz S RI R {z0:g}\n")
        for i, fr in enumerate(f):
            if n == 1:
                cols = [s[i, 0, 0]]
            else:
                cols = [s[i, 0, 0], s[i, 1, 0], s[i, 0, 1], s[i, 1, 1]]
            fh.write(f" {fr:.0f} " +
                     " ".join(f"{c.real:.9g} {c.imag:.9g}" for c in cols) + "\n")
    return path


def from_built(built, f, comments=()):
    """A model's own response as a Touchstone object, for symmetric handling of
    measured and modelled traces downstream."""
    return Touchstone(f, built.solve(f), built.ports[0].z0, list(comments))


def load_many(pattern):
    """Every file matching a glob, sorted.  `load_many("sweeps/*.s2p")`."""
    import glob as _glob
    return [load(p) for p in sorted(_glob.glob(pattern))]
