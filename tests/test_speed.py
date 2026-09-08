"""Speed is a design constraint here, so it gets a test rather than a comment.

The architecture was planned around an estimate of ~10-15 ms per sweep
evaluation, which would have forced netlist fitting out of the interactive loop
and left the coupling-matrix model as the only live fitter.  The measurement
refuted that estimate by about 6x:

    nfreq  dim   total    assemble   solve
      401    3   0.09 ms   0.01 ms   0.07 ms
      401   10   0.43 ms   0.07 ms   0.36 ms
      401   20   1.71 ms   0.54 ms   1.17 ms
      401   30   4.11 ms   1.06 ms   3.06 ms
      401   60  16.70 ms   3.07 ms  13.63 ms
     1001   20   4.46 ms   1.21 ms   3.25 ms

A 3-pole filter netlist is dim ~10-14, so a forward evaluation is well under a
millisecond and a finite-difference fit at p=10 is ~0.2 s.  Netlist fitting is
therefore interactive after all, and the planned split was dropped.

These bounds are loose on purpose -- they are here to catch an architectural
regression (an accidental per-frequency Python loop, a lost affine assembly),
not to police a few percent on a busy machine.
"""
import time

import numpy as np
import pytest

from vnafit import mna
from vnafit.netlist import parse

THREE_POLE = """.param L=330n C=3.6p
.port 1 p 0
.port 2 q 0
Cs1 p a 1.5p
L1 a 0 {L} Q=200
C1 a 0 {C}
Cm1 a b 0.2p
L2 b 0 {L} Q=200
C2 b 0 {C}
Cm2 b c 0.2p
L3 c 0 {L} Q=200
C3 c 0 {C}
Cs2 q c 1.5p"""


def _time(fn, rep=20):
    fn()                                    # warm up
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep


def test_forward_evaluation_is_interactive():
    b = mna.build(parse(THREE_POLE), w_ref=2 * np.pi * 145e6)
    f = np.linspace(120e6, 170e6, 401)
    dt = _time(lambda: b.solve(f))
    assert b.dim <= 20, f"3-pole netlist unexpectedly large: dim={b.dim}"
    assert dt < 0.010, f"{dt*1e3:.2f} ms per 401-point sweep -- too slow to tune against"


def test_rebuild_is_cheap_enough_to_fit():
    """A fit rebuilds per parameter vector, so build+solve is the real unit."""
    nl = parse(THREE_POLE)
    f = np.linspace(120e6, 170e6, 401)
    dt = _time(lambda: mna.build(nl, {"l": 331e-9}, w_ref=2 * np.pi * 145e6).solve(f))
    assert dt < 0.030, f"{dt*1e3:.2f} ms per build+solve -- a fit would crawl"


def test_assembly_does_not_scale_with_frequency_count():
    """Guards the affine form: A = G + jw*W means the element loop runs once
    per build, not once per frequency.  If someone reintroduces a per-frequency
    stamp loop, cost per point stops being flat and this catches it."""
    b = mna.build(parse(THREE_POLE), w_ref=2 * np.pi * 145e6)
    small = _time(lambda: b.solve(np.linspace(120e6, 170e6, 101)))
    big = _time(lambda: b.solve(np.linspace(120e6, 170e6, 1616)))
    per_point_small = small / 101
    per_point_big = big / 1616
    assert per_point_big < 2.0 * per_point_small, (
        f"cost per point rose {per_point_big/per_point_small:.1f}x from 101 to 1616 "
        "points -- assembly is probably no longer affine")
