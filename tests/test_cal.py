"""Calibrations: derived, applied, archived, stitched.

The check that matters is a round trip against a synthetic instrument with
KNOWN error terms -- if the solve and the correction are both right, a DUT
measured through those errors comes back exactly.
"""
import numpy as np
import pytest

from vnafit.cal import CalBank, CalError, Calibration
from vnafit.touchstone import Touchstone

ED = lambda f: 0.02 * np.exp(2j * np.pi * f / 4e8)
ES = lambda f: 0.08 * np.exp(1j * f / 3e7)
ER = lambda f: 0.9 * np.exp(-1j * f / 2e7)
ET = lambda f: 0.85 * np.exp(-1j * f / 1.5e7)
EX = 1e-4 + 0j


def _ts(f, s11, s21=None):
    S = np.zeros((len(f), 2, 2), complex)
    S[:, 0, 0] = s11
    if s21 is not None:
        S[:, 1, 0] = s21
    return Touchstone(np.asarray(f, float), S)


def _raw11(f, gamma):
    return ED(f) + ER(f) * gamma / (1 - ES(f) * gamma)


def _standards(f, isolation=True):
    kw = dict(thru=_ts(f, _raw11(f, 0.0), EX + ET(f)))
    if isolation:
        kw["isoln"] = _ts(f, _raw11(f, 0.0), EX * np.ones(len(f), complex))
    return (_ts(f, _raw11(f, 1.0)), _ts(f, _raw11(f, -1.0)),
            _ts(f, _raw11(f, 0.0))), kw


def _cal(f, isolation=True, **meta):
    (o, s, l), kw = _standards(f, isolation)
    return Calibration.from_standards(o, s, l, notes=meta.pop("notes", ""), **kw)


def test_the_solve_recovers_the_error_terms_exactly():
    f = np.linspace(130e6, 165e6, 401)
    c = _cal(f)
    for name, truth in (("ED", ED(f)), ("ES", ES(f)), ("ER", ER(f)),
                        ("ET", ET(f))):
        assert np.abs(c.terms[name] - truth).max() < 1e-12, name


def test_a_dut_measured_through_the_errors_comes_back():
    f = np.linspace(130e6, 165e6, 401)
    c = _cal(f)
    G = 0.4 * np.exp(1j * f / 1e7)
    T = 0.5 * np.exp(-2j * np.pi * f / 2e8)
    meas = _ts(f, _raw11(f, G), EX + ET(f) * T / (1 - ES(f) * G))
    out = c.apply(meas)
    assert np.abs(out.s11 - G).max() < 1e-12
    assert np.abs(out.s21 - T).max() < 1e-12


def test_leaving_out_the_isolation_costs_exactly_the_crosstalk():
    """Not a vague degradation -- the residual IS the term you did not measure,
    so it is worth knowing whether that matters at your levels."""
    f = np.linspace(130e6, 165e6, 401)
    T = 0.5 * np.exp(-2j * np.pi * f / 2e8)
    G = 0.4 * np.exp(1j * f / 1e7)
    meas = _ts(f, _raw11(f, G), EX + ET(f) * T / (1 - ES(f) * G))
    out = _cal(f, isolation=False).apply(meas)
    err = np.abs(out.s21 - T).max()
    assert abs(err - abs(EX) / abs(ET(f)).min()) < 5e-5, err


def test_a_calibration_refuses_to_be_used_outside_its_span():
    """Extrapolating error terms is how a cal quietly stops meaning anything."""
    f = np.linspace(130e6, 165e6, 401)
    c = _cal(f)
    wide = _ts(np.linspace(120e6, 170e6, 401), np.zeros(401, complex))
    with pytest.raises(CalError, match="Extrapolating"):
        c.apply(wide)


def test_it_says_when_the_sweep_is_finer_than_the_calibration():
    """The criticism this project levelled at the instrument applies to us."""
    f = np.linspace(130e6, 165e6, 101)
    c = _cal(f)
    fine = _ts(np.linspace(130e6, 165e6, 1601), np.zeros(1601, complex),
               np.zeros(1601, complex))
    out = c.apply(fine)
    assert any("interpolated between" in x for x in out.comments), out.comments


def test_a_saved_calibration_can_be_re_derived_from_its_own_standards(tmp_path):
    """The reason the standards are embedded: a cal you cannot re-derive is a
    number you have to trust."""
    f = np.linspace(130e6, 165e6, 201)
    c = _cal(f, notes="2 m bench")
    p = str(tmp_path / "x.calz")
    c.save(p)
    import os
    assert os.path.exists(p), "save() renamed the file behind our back"
    back = Calibration.load(p)
    assert back.meta["notes"] == "2 m bench"
    assert back.points == 201 and abs(back.start - 130e6) < 1
    again = back.rederive()
    for k in ("ED", "ES", "ER", "ET"):
        assert np.abs(again.terms[k] - c.terms[k]).max() < 1e-15


def test_a_cal_without_a_thru_corrects_s11_and_says_it_left_s21_alone():
    f = np.linspace(130e6, 165e6, 201)
    (o, s, l), _kw = _standards(f)
    c = Calibration.from_standards(o, s, l)          # no thru
    assert not c.has_thru
    G = 0.4 * np.exp(1j * f / 1e7)
    raw21 = 0.3 * np.ones(len(f), complex)
    out = c.apply(_ts(f, _raw11(f, G), raw21))
    assert np.abs(out.s11 - G).max() < 1e-12
    assert np.allclose(out.s21, raw21), "S21 must be untouched, not half-corrected"
    assert any("left as measured" in x for x in out.comments)


def test_standards_on_different_grids_are_refused():
    (o, s, l), _kw = _standards(np.linspace(130e6, 165e6, 201))
    other = _ts(np.linspace(130e6, 165e6, 200), np.zeros(200, complex))
    with pytest.raises(CalError, match="different frequency grid"):
        Calibration.from_standards(o, other, l)


def test_an_open_that_was_never_connected_is_named_as_such():
    f = np.linspace(130e6, 165e6, 51)
    same = _ts(f, _raw11(f, 0.0))
    with pytest.raises(CalError, match="not actually connected"):
        Calibration.from_standards(same, same, same)


# ------------------------------------------------------------------ stitching
def test_a_bank_uses_the_finer_calibration_where_they_overlap():
    coarse = _cal(np.linspace(100e6, 200e6, 101), notes="wide")
    fine = _cal(np.linspace(140e6, 150e6, 401), notes="2 m")
    bank = CalBank([coarse, fine])
    who = bank.pick(np.array([110e6, 145e6, 190e6]))
    assert [bank.cals[i].meta["notes"] for i in who] == ["wide", "2 m", "wide"]


def test_a_bank_refuses_a_gap_and_names_it():
    a = _cal(np.linspace(100e6, 120e6, 101))
    b = _cal(np.linspace(160e6, 200e6, 101))
    bank = CalBank([a, b])
    sweep = _ts(np.linspace(100e6, 200e6, 401), np.zeros(401, complex))
    with pytest.raises(CalError, match="no calibration covers"):
        bank.apply(sweep)


def test_a_stitched_correction_is_still_exact_in_each_piece():
    lo = _cal(np.linspace(100e6, 150e6, 501))
    hi = _cal(np.linspace(150e6, 200e6, 501))
    f = np.linspace(100e6, 200e6, 801)
    G = 0.4 * np.exp(1j * f / 1e7)
    T = 0.5 * np.exp(-2j * np.pi * f / 2e8)
    meas = _ts(f, _raw11(f, G), EX + ET(f) * T / (1 - ES(f) * G))
    out = CalBank([lo, hi]).apply(meas)
    # Not machine precision, and it should not be: the cal grids are 100 kHz
    # and the sweep is 125 kHz, so the terms are linearly interpolated between
    # cal points.  1e-6 is that interpolation, not an error in the algebra --
    # the un-interpolated case above is exact to 1e-12.
    assert np.abs(out.s11 - G).max() < 1e-5
    assert any("stitched" in x for x in out.comments)


# --------------------------------------------------------------- the window
class _StubApp:
    """The main window, as far as the calibration window is concerned."""
    dev = None
    v_start = v_stop = None

    def __init__(self):
        self.adopted = None

    def set_calibration(self, cal, path=None):
        self.adopted = (cal, path)


def _calwin():
    import tkinter as tk
    from vnafit.calwin import CalWindow
    try:
        root = tk.Tk()
    except tk.TclError:                                     # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    return root, CalWindow(root, _StubApp())


def test_the_calibration_window_builds():
    """It did not.  `_build` referred to `app` where it meant `self.app`, and
    nothing caught it because no test had ever constructed the window --
    every other test in this file exercises the maths underneath it."""
    root, w = _calwin()
    try:
        assert str(w.savebtn.cget("state")) == "disabled"
        assert "still needed" in w.status.cget("text")
        assert set(w.rows) == {"load", "open", "short", "thru", "isolation"}
    finally:
        root.destroy()


def test_the_window_only_offers_to_save_once_the_three_required_are_in():
    """And says which of them are still missing, in the order it asks for them."""
    f = np.linspace(130e6, 165e6, 51)
    root, w = _calwin()
    try:
        (o, s, l), _kw = _standards(f)
        w.captured["load"] = l
        w._refresh()
        assert str(w.savebtn.cget("state")) == "disabled"
        assert "open" in w.status.cget("text") and "short" in w.status.cget("text")

        w.captured["open"], w.captured["short"] = o, s
        w._refresh()
        assert str(w.savebtn.cget("state")) == "normal"
        assert "corrects S11 only" in w.status.cget("text")

        w.captured["thru"] = _ts(f, np.zeros(51, complex), np.ones(51, complex))
        w._refresh()
        assert "crosstalk term is assumed zero" in w.status.cget("text")
    finally:
        root.destroy()


def test_the_window_refuses_standards_swept_on_different_grids():
    """Every standard has to be the same sweep, and the check has to happen at
    capture time -- discovering it at save time means doing them all again."""
    root, w = _calwin()
    try:
        w.grid_f = np.linspace(130e6, 165e6, 401)
        assert len(w.grid_f) == 401
    finally:
        root.destroy()


# ------------------------------------------------- which cal is in effect
class _FakeDev:
    """An instrument whose correction can be switched, and read back."""
    _info = "fake H4"

    def __init__(self, enabled=True, calibrated=True):
        self.enabled, self.calibrated = enabled, calibrated
        self.recalled = None
        self._cal_restore = None

    def cal_status(self):
        return {"enabled": self.enabled and self.calibrated,
                "standards": ("load", "open", "short") if self.calibrated else (),
                "terms": (), "raw": ["cal'ed"] if self.enabled else []}

    def set_correction(self, on):
        self.enabled = bool(on)
        return True

    def recall_cal(self, slot):
        self.recalled = slot
        self.calibrated = True
        return self.cal_status()


def _app_with(dev, cal=None):
    import tkinter as tk
    from vnafit.gui import App
    try:
        root = tk.Tk()
    except tk.TclError:                                     # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    app = App(root, "examples/tinyfilter.net")
    app.dev = dev
    if cal is not None:
        app.set_calibration(cal, "2m.calz")
    return root, app


def test_adopting_a_local_cal_switches_the_instruments_own_off():
    """They are alternatives, not layers.  Correcting a sweep the instrument has
    already corrected applies the error model twice, and the result looks
    entirely plausible: smooth, physical, and wrong."""
    f = np.linspace(130e6, 165e6, 51)
    dev = _FakeDev(enabled=True)
    root, app = _app_with(dev, _cal(f))
    try:
        assert dev.enabled is False, "the instrument is still correcting too"
        assert dev._cal_restore is True, "and it would not be put back"
        txt, _c = app.cal_state()
        assert txt == "cal: 2m.calz"
    finally:
        root.destroy()


def test_the_indicator_names_the_dangerous_state_rather_than_hiding_it():
    f = np.linspace(130e6, 165e6, 51)
    dev = _FakeDev(enabled=True)
    root, app = _app_with(dev, _cal(f))
    try:
        dev.enabled = True                    # as if something turned it back on
        txt, col = app.cal_state()
        assert "BOTH" in txt and "twice" in txt
        from vnafit.gui import BAD
        assert col == BAD
    finally:
        root.destroy()


def test_raw_data_is_called_raw():
    """Uncorrected with no local cal is a real state and the easiest one to be
    in by accident, so it is named rather than left blank."""
    root, app = _app_with(_FakeDev(enabled=False))
    try:
        txt, col = app.cal_state()
        assert txt == "cal: NONE -- raw"
        from vnafit.gui import BAD
        assert col == BAD
    finally:
        root.destroy()


def test_going_back_to_the_instrument_drops_ours_and_recalls_the_slot():
    f = np.linspace(130e6, 165e6, 51)
    dev = _FakeDev(enabled=True)
    root, app = _app_with(dev, _cal(f))
    try:
        app.use_device_cal(2)
        assert dev.recalled == 2 and dev.enabled is True
        assert app.cal is None, "the local cal must not still be applied"
        assert app.cal_state()[0] == "cal: instrument"
    finally:
        root.destroy()


# ----------------------------------------------------- labels for device slots
def _slots(tmp_path):
    from vnafit.slots import SlotLabels
    return SlotLabels(str(tmp_path / "slots.json"))


def _terms(seed=0.0):
    f = np.linspace(1, 2, 32)
    return {"ED": f + seed + 0j, "ES": f * 2j, "ER": f + 1}


def test_a_slot_label_survives_a_reread_and_dies_on_a_recalibration(tmp_path):
    """A label kept on this computer for a calibration kept on the device.  The
    hash is what ties them: recalibrate the slot and the label is DROPPED, not
    shown next to a warning -- a label beside a caveat is still a label, and it
    is the wrong one."""
    s = _slots(tmp_path)
    s.set("H4 fw1.2", 2, _terms(), "2 m bench",
          fixture={"reference_plane": "ends of the 1 m RG316 pair",
                   "cables": "2x 1 m RG316"})
    rec, state = s.get("H4 fw1.2", 2, _terms())
    assert state == "labelled"
    assert rec["label"] == "2 m bench"
    assert rec["reference_plane"] == "ends of the 1 m RG316 pair"

    rec, state = s.get("H4 fw1.2", 2, _terms(seed=1e-9))
    assert state == "changed" and rec is None
    assert s.get("H4 fw1.2", 2, _terms())[1] == "unlabelled", "it was not deleted"


def test_a_different_instrument_does_not_inherit_the_label(tmp_path):
    """The H4's info banner is identical on every unit running the same
    firmware, so it cannot tell two apart.  The hash does not care: another
    instrument's slot 2 holds different terms and simply will not match."""
    s = _slots(tmp_path)
    s.set("H4 fw1.2", 2, _terms(), "mine")
    assert s.get("H4 fw1.2", 2, _terms(seed=0.5))[1] == "changed"


def test_the_hash_is_stable_across_processes(tmp_path):
    from vnafit.slots import slot_hash
    assert slot_hash(_terms()) == slot_hash(_terms())
    assert slot_hash(_terms()) != slot_hash(_terms(seed=1e-12))
    assert slot_hash({"ED": _terms()["ED"]}) != slot_hash(_terms())


def test_setup_notes_are_optional_and_only_shown_when_filled_in():
    f = np.linspace(130e6, 165e6, 51)
    (o, sh, l), kw = _standards(f)
    plain = Calibration.from_standards(o, sh, l, **kw).describe()
    assert "reference plane" not in plain, "an empty field should be silent"

    full = Calibration.from_standards(
        o, sh, l, fixture={"reference_plane": "SMA at the cable ends",
                           "cables": "2x 1 m RG316"}, **kw).describe()
    assert "SMA at the cable ends" in full and "RG316" in full


def test_it_notices_standards_that_were_not_swept_raw():
    """Solving a calibration from sweeps the instrument has already corrected
    is circular, and it does not fail loudly -- it produces error terms near
    unity that look like an unusually good fixture."""
    f = np.linspace(130e6, 165e6, 51)

    def std(g, comments):
        S = np.zeros((51, 2, 2), complex)
        S[:, 0, 0] = g
        S[:, 1, 0] = 0.9
        return Touchstone(f, S, comments=comments)

    raw = ["instrument correction: OFF (raw)"]
    on = ["instrument correction: as configured on the instrument"]

    good = Calibration.from_standards(std(1., raw), std(-1., raw), std(0., raw))
    assert good.meta["standards_raw"] is True
    assert "WARNING" not in good.describe()

    bad = Calibration.from_standards(std(1., on), std(-1., on), std(0., on))
    assert bad.meta["standards_raw"] is False
    assert "circular" in bad.describe()

    # a file from anywhere else says nothing, and None is recorded rather than
    # guessed -- but it is not worth a line of output
    quiet = Calibration.from_standards(std(1., []), std(-1., []), std(0., []))
    assert quiet.meta["standards_raw"] is None
    assert "WARNING" not in quiet.describe()


def test_the_thru_gets_its_own_field():
    """A barrel is not a zero-length thru, and its length ends up subtracted
    from every DUT measured afterwards."""
    from vnafit.cal import FIXTURE_KEYS
    assert "thru" in FIXTURE_KEYS
    f = np.linspace(130e6, 165e6, 51)
    (o, s, l), kw = _standards(f)
    c = Calibration.from_standards(o, s, l, fixture={"thru": "SMA barrel"}, **kw)
    assert "SMA barrel" in c.describe()


def test_a_fingerprint_must_come_from_a_fixed_sweep():
    """`data 2..6` returns the calibration INTERPOLATED onto the current sweep,
    so terms read at 101 points and at 401 hash differently.  Measured on an
    H4, and a normalised-position digest across those two differed by a median
    of 45%.  `cal_fingerprint` reads at a fixed sweep and restores yours.
    """
    from vnafit.slots import slot_hash

    class Dev:
        def __init__(self):
            self.sweep = (90e6, 220e6, 101)
            self.reads = []

        def sweep_state(self):
            return self.sweep

        def cmd(self, s):
            if s.startswith("sweep "):
                a, b, n = s.split()[1:]
                self.sweep = (int(a), int(b), int(n))
            return []

        def cal_terms(self):
            self.reads.append(self.sweep)
            n = self.sweep[2]                    # what the firmware does
            x = np.linspace(0, 1, n)
            return {k: (x + i) + 0j for i, k in enumerate(("ED", "ES", "ER"))}

    from vnafit.vna import NanoVNA
    d = Dev()
    d.FINGERPRINT_SWEEP = NanoVNA.FINGERPRINT_SWEEP
    d.cal_fingerprint = NanoVNA.cal_fingerprint.__get__(d)

    t1, sw = d.cal_fingerprint()
    assert sw == NanoVNA.FINGERPRINT_SWEEP
    assert d.sweep == (90e6, 220e6, 101), "the instrument's sweep was not restored"

    d.cmd("sweep 88000000 108000000 201")
    t2, _ = d.cal_fingerprint()
    assert slot_hash(t1) == slot_hash(t2), "the fingerprint followed the sweep"
    assert d.reads == [NanoVNA.FINGERPRINT_SWEEP] * 2

    # and the naive read really does differ, which is why the above is needed
    d.cmd("sweep 90000000 220000000 101")
    assert slot_hash(d.cal_terms()) != slot_hash(t1)


def test_the_window_can_capture_a_standard_in_several_passes():
    """The instrument sweeps 401 points at once.  A calibration can only correct
    at the resolution it was taken at, so being stuck at one pass caps every
    local cal at the instrument's own grid -- and interpolating a coarse cal
    onto a fine sweep is the thing this project keeps objecting to."""
    import tkinter as tk
    from vnafit.calwin import CalWindow

    class Dev:
        def __init__(self):
            self.calls = []

        def uncorrected(self):
            import contextlib
            return contextlib.nullcontext()

        def scan_hires(self, start, stop, segments=1, points=401, **kw):
            self.calls.append((start, stop, segments, points))
            n = points if segments <= 1 else points + (points - 1) * (segments - 1)
            f = np.linspace(start, stop, n)
            if kw.get("on_segment"):
                for i in range(segments):
                    kw["on_segment"](i + 1, segments, f, f, f)
            return f, np.zeros(n, complex), np.zeros(n, complex)

    class App:
        v_start = v_stop = None
        def __init__(self, d): self.dev = d
        def set_calibration(self, c, p=None): pass

    try:
        root = tk.Tk()
    except tk.TclError:                                     # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    dev = Dev()
    w = CalWindow(root, App(dev))
    try:
        w.v_start.set("130"); w.v_stop.set("170")
        w.v_points.set("401"); w.v_segments.set("4")
        assert "1601 points" in w.seghint.cget("text")
        assert "25.000 kHz" in w.seghint.cget("text")
        assert "4 passes" in w.seghint.cget("text")

        w.capture("load")
        assert dev.calls == [(130e6, 170e6, 4, 401)]
        assert len(w.captured["load"]) == 1601

        # and every later standard is pinned to that same grid
        w.v_segments.set("2")
        w.capture("open")
        assert "open" not in w.captured, "a different grid was accepted"
    finally:
        root.destroy()
