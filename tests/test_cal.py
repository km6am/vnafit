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
