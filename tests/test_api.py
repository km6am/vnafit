"""The notebook-facing API.

`from vnafit import load` should give you arrays you can plot without writing a
helper of your own, and the object should say something useful when you just
type its name.
"""
import numpy as np
import pytest

import vnafit
from vnafit import load, netlist, sweep


@pytest.fixture
def d(tmp_path):
    nl = vnafit.parse_netlist("""two pole
.port 1 p1 0
.port 2 p2 0
Cin p1 n1 1.5p
L1 n1 0 330n Q=600
C1 n1 0 1.9p
Cm n1 n2 0.2p
L2 n2 0 330n Q=600
C2 n2 0 1.9p
Cout p2 n2 1.5p""")
    # C1/C2 are 1.9p, not 3.6p: the coupling and port capacitors land on the
    # same node, so the TOTAL node capacitance is what sets f0.  Ignoring that
    # put this fixture's passband off the bottom of its own sweep.
    f = np.linspace(120e6, 175e6, 401)
    _f, S = sweep(nl, f)
    p = tmp_path / "d.s2p"
    vnafit.save(str(p), f, S, comments=["synthetic"])
    return load(str(p))


def test_the_repr_says_something_useful(d):
    r = repr(d)
    assert "d.s2p" in r and "2-port" in r and "401 pts" in r and "peak S21" in r


def test_the_arrays_are_there_and_are_plain_numpy(d):
    for name in ("f", "f_mhz", "s21_db", "s11_db", "vswr", "group_delay",
                 "return_loss", "insertion_loss", "z_in"):
        v = getattr(d, name)
        assert isinstance(v, np.ndarray) and len(v) == len(d)
    assert np.allclose(d.f_mhz, d.f / 1e6)
    assert np.allclose(d.return_loss, -d.s11_db)


def test_at_and_band_and_peak(d):
    fp, pk = d.peak()
    row = d.at(fp)
    assert abs(row["s21_db"] - pk) < 1e-9
    assert set(("f", "f_mhz", "s11", "s21", "vswr")) <= set(row)
    b = d.band(140e6, 150e6)
    assert b.f[0] >= 140e6 and b.f[-1] <= 150e6 and len(b) < len(d)
    assert len(d) == 401, "band() must not mutate the original"


def test_bandwidth_returns_none_rather_than_a_number_off_the_edge(d):
    assert d.bandwidth(3.0) is not None
    narrow = d.band(146e6, 147e6)      # entirely inside the passband
    assert narrow.bandwidth(3.0) is None


def test_group_delay_is_unwrapped(d):
    """Without unwrapping, every 2-pi crossing becomes a spike that reads as a
    resonance.  A raw-angle derivative would be orders of magnitude larger."""
    gd = d.group_delay
    assert np.all(np.abs(gd) < 1e-5)
    raw = -np.gradient(np.angle(d.s21), 2 * np.pi * d.f)
    assert np.abs(raw).max() > 20 * np.abs(gd).max()


def test_band_refuses_an_empty_range(d):
    with pytest.raises(ValueError, match="no points between"):
        d.band(1e9, 2e9)


def test_sweep_accepts_the_measurement_itself(d, tmp_path):
    nl = vnafit.parse_netlist(".port 1 p1 0\n.port 2 p2 0\nR1 p1 p2 50\n"
                              "R2 p2 0 1e9")
    f, S = sweep(nl, d)              # not d.f -- the object
    assert len(f) == len(d) and S.shape == (len(d), 2, 2)


def test_importing_vnafit_does_not_pull_in_matplotlib():
    """A notebook that only wants numbers should not pay for a GUI toolkit."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, vnafit; "
         "print(','.join(m for m in ('matplotlib','tkinter') if m in sys.modules))"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_load_many(tmp_path):
    from vnafit import load_many
    nl = vnafit.parse_netlist(".port 1 p1 0\n.port 2 p2 0\nR1 p1 p2 50\n"
                              "R2 p2 0 1e9")
    f, S = sweep(nl, np.linspace(1e6, 2e6, 11))
    for i in range(3):
        vnafit.save(str(tmp_path / f"s{i}.s2p"), f, S)
    got = load_many(str(tmp_path / "*.s2p"))
    assert len(got) == 3 and all(len(g) == 11 for g in got)
