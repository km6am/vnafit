"""Names, kept ON THIS COMPUTER, for calibrations that live ON THE DEVICE.

That asymmetry is the whole design.  The calibration itself never leaves the
instrument -- there is no firmware command to write one back, so it cannot --
and the label never goes onto the instrument, because there is nowhere to put
it.  What ties the two together is a hash of the slot's contents.

The H4 stores its calibrations and remembers nothing about them -- not what
they were for, not the span, not when.  Slot 2 is "slot 2".  So the label lives
here instead.

A label stored somewhere else is a promise about something you cannot see, and
the way it goes wrong is obvious: you recalibrate slot 2 on the bench, come back
a month later, and the software cheerfully tells you it is the 2 m cal it no
longer is.  So a label is stored WITH A HASH OF THE SLOT'S CONTENTS, and is only
ever shown when the contents still hash the same.  When they do not, the label
is not shown, not guessed at, and not kept -- it is deleted, and you are told
the slot changed.

That also solves an identity problem the instrument creates and cannot fix: its
`info` banner is the same on every H4 running the same firmware, so two units
cannot be told apart by it.  The hash does not care.  A different instrument's
slot 2 holds different error terms, so it will not match, and the label simply
will not appear.
"""
import hashlib
import json
import os

import numpy as np

TERM_ORDER = ("ED", "ES", "ER", "ET", "EX")


def home():
    """Where per-user state lives.  Never inside the installed package."""
    return os.environ.get("VNAFIT_HOME") or os.path.join(
        os.path.expanduser("~"), ".vnafit")


def slot_hash(terms):
    """A stable fingerprint of a slot's calibration data.

    Over the terms in a fixed order, as canonical little-endian complex128, so
    the same slot hashes the same on any machine.

    Feed it `NanoVNA.cal_fingerprint()`, not `cal_terms()`.  `data 2..6` returns
    the calibration INTERPOLATED onto the current sweep, so terms read at 101
    points and at 401 hash differently -- measured on an H4, and a
    normalised-position digest across those two differed by a median of 45%.
    `cal_fingerprint` reads at a fixed sweep and puts the instrument's back.
    """
    h = hashlib.sha256()
    for name in TERM_ORDER:
        v = terms.get(name)
        if v is None:
            h.update(b"-")
            continue
        a = np.ascontiguousarray(np.asarray(v, "<c16"))
        h.update(name.encode())
        h.update(str(a.size).encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()[:32]


class SlotLabels:
    """Labels keyed by (instrument, slot), validated by content hash."""

    def __init__(self, path=None):
        self.path = path or os.path.join(home(), "slots.json")
        try:
            with open(self.path) as fh:
                self.data = json.load(fh)
        except (OSError, ValueError):
            self.data = {}

    # ------------------------------------------------------------------ io
    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.path)          # never a half-written store

    @staticmethod
    def _key(instrument, slot):
        return f"{instrument or 'unknown'}|{int(slot)}"

    # --------------------------------------------------------------- lookup
    def get(self, instrument, slot, terms):
        """(label, state) for a slot whose contents are `terms`.

        On "labelled" the whole record comes back -- the label and whatever
        fixture detail was stored with it -- not just the name.

        state is "labelled", "changed" (the label was dropped just now) or
        "unlabelled".  A changed slot never returns its old label, not even
        alongside a warning: a label shown next to a caveat is still a label,
        and it is the wrong one.
        """
        key = self._key(instrument, slot)
        rec = self.data.get(key)
        if not rec:
            return None, "unlabelled"
        got = slot_hash(terms)
        if rec.get("hash") != got:
            del self.data[key]
            self._save()
            return None, "changed"
        return rec, "labelled"

    def set(self, instrument, slot, terms, label, fixture=None):
        from .cal import FIXTURE_KEYS
        rec = {k: v for k, v in (fixture or {}).items() if k in FIXTURE_KEYS and v}
        rec.update({
            "label": label,
            "hash": slot_hash(terms),
            "points": int(max((np.size(v) for v in terms.values()), default=0)),
            "noted": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self.data[self._key(instrument, slot)] = rec
        self._save()
        return label

    def forget(self, instrument, slot):
        self.data.pop(self._key(instrument, slot), None)
        self._save()

    def known(self, instrument=None):
        """Everything remembered, optionally for one instrument."""
        out = []
        for key, rec in sorted(self.data.items()):
            inst, _, slot = key.rpartition("|")
            if instrument is not None and inst != (instrument or "unknown"):
                continue
            out.append({"instrument": inst, "slot": int(slot), **rec})
        return out
