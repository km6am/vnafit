"""The live tuning readout.

Driven from a Touchstone file through FileSource, so the whole loop is testable
with no instrument.  What matters here is that it keeps producing something
useful when the model is wrong or the fit fails -- which is the normal case with
a hand-entered model, and the moment you least want a blank screen.
"""
import numpy as np
import pytest

from vnafit import mna, touchstone, track
from vnafit.acq import FileSource
from vnafit.netlist import parse

NET = """.param f0=146Meg L=330n qu=800 k=0.015 qe=100
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={sqrt(50/(2*pi*f0*L*qe))}
L1 n1 0 {L} Q={qu}
C1 n1 0 {1/(4*pi*pi*f0*f0*L)}
L2 n2 0 {L} Q={qu}
C2 n2 0 {1/(4*pi*pi*f0*f0*L)}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={sqrt(50/(2*pi*f0*L*qe))}"""


@pytest.fixture
def src(tmp_path):
    nl = parse(NET)
    f = np.linspace(130e6, 165e6, 401)
    S = mna.build(nl, w_ref=2 * np.pi * 146e6).solve(f)
    p = tmp_path / "dut.s2p"
    touchstone.save(str(p), f, S)
    return FileSource(touchstone.load(str(p)))


def test_model_free_numbers_are_produced(src):
    f, s11, s21 = src.scan_hires(130e6, 165e6, 1, 401)
    row = track.measure_row(f, s11, s21, 2, 800.0)
    assert 140e6 < row["f_peak"] < 152e6
    assert -3 < row["peak_db"] < 0
    assert 1e6 < row["bw3"] < 8e6
    assert row["rl_db"] > 5


def test_a_failed_fit_does_not_stop_the_readout(src):
    """With both hands on a trimmer, a blank screen is the worst outcome.

    The model-free half of the row must survive a fit that cannot converge, and
    the formatter must render the missing fields rather than raising.
    """
    f, s11, s21 = src.scan_hires(130e6, 165e6, 1, 401)
    row = track.measure_row(f, s11, s21, 9, 800.0)   # 9 poles on a 2-pole trace
    assert row["f_peak"] > 0 and np.isfinite(row["peak_db"])
    row.pop("k", None)
    row.pop("qe1", None)
    row.pop("fit_rms", None)
    line = track.format_row(1, row)
    assert "-" in line and str(round(row["f_peak"] / 1e6, 4))[:7] in line


def test_a_deliberately_wrong_model_still_scores(src):
    """The normal case: the netlist is a guess.  It must be scored, not refused,
    and passband and skirts reported separately -- one number hides the half
    that is telling you something."""
    f, s11, s21 = src.scan_hires(130e6, 165e6, 1, 401)
    wrong = mna.build(parse(NET.replace("k=0.015", "k=0.045")))
    sc = track.netlist_score(wrong, f, s21)
    assert sc["passband"] > 1.0, "a 3x k error should show in the passband"
    assert np.isfinite(sc["skirts"])
    right = mna.build(parse(NET))
    assert track.netlist_score(right, f, s21)["passband"] < 0.01


def test_the_loop_logs_every_sweep(tmp_path, src):
    out = tmp_path / "log"
    rows = track.run(src, dict(start=130e6, stop=165e6, points=401, segments=1),
                     seconds=1.2, poles=2, qu=800.0, out_dir=str(out),
                     every=0.05, verbose=False)
    assert len(rows) >= 3
    lines = (out / "track.csv").read_text().strip().splitlines()
    assert len(lines) == len(rows) + 1
    assert (out / "latest.s2p").exists()


def test_the_loop_survives_a_dead_instrument():
    class Flaky:
        n = 0
        def scan_hires(self, *a, **kw):
            Flaky.n += 1
            raise IOError("cable fell out")
        def close(self):
            pass
    rows = track.run(Flaky(), dict(start=130e6, stop=165e6, points=401),
                     seconds=0.6, poles=2, qu=800.0, every=0.05, verbose=False)
    assert rows == []
    assert Flaky.n >= 1
