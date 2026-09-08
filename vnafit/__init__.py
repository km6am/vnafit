"""vnafit -- netlist-model filter measurement, tuning and fitting.

For a notebook or an IPython session the useful entry points are here:

    from vnafit import load, load_many, netlist, sweep

    d = load("capture.s2p")
    d                       # -> <capture.s2p: 2-port, 401 pts, 130-165 MHz, ...>
    plt.plot(d.f_mhz, d.s21_db)
    d.at(146e6)             # everything at one frequency
    d.band(140e6, 150e6)    # a slice, as a new object
    d.bandwidth(3.0)        # None if the sweep does not contain both crossings

    m = netlist("model.net")
    f, S = sweep(m, d.f)    # the model on the measurement's own grid
    plt.plot(d.f_mhz, 20*np.log10(abs(S[:, 1, 0])))

Importing this module pulls in numpy and scipy but NOT matplotlib or tkinter --
those load only when you actually plot or open the GUI.  `vnafit.vna` is
standalone and imports neither.
"""
from .touchstone import Touchstone, load, load_many, save          # noqa: F401
from .netlist import NetlistError, parse as parse_netlist          # noqa: F401
from .netlist import load as netlist                               # noqa: F401


def sweep(nl, f, overrides=None, **kw):
    """Evaluate a netlist on a frequency grid.  Returns (f, S).

    `f` may be an array, or a Touchstone object -- passing the measurement
    directly is the common case and saves you writing `d.f` every time.
    """
    import numpy as _np
    from . import mna
    if hasattr(f, "f"):
        f = f.f
    f = _np.atleast_1d(_np.asarray(f, float))
    return f, mna.build(nl, overrides, **kw).solve(f)


__all__ = ["Touchstone", "load", "load_many", "save", "netlist",
           "parse_netlist", "NetlistError", "sweep"]
