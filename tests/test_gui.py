"""GUI logic, exercised without a display where possible.

A Tk window cannot be asserted about very deeply, but the parts that actually
carry risk are not Tk: the replay source that stands in for the instrument, the
acquisition thread, and the parameter plumbing between the entry boxes and the
model.  Those are testable, and they are what breaks.
"""
import numpy as np
import pytest

from vnafit import mna, touchstone
from vnafit.gui import OK
from vnafit.acq import Acq, FileSource
from vnafit.netlist import parse

NET = """.param f0=145.4Meg L=330n qe=46 k=0.0121 qu=670
.port 1 p1 0
.port 2 p2 0
X1 p1 0 n1 0 n={sqrt(50/(2*pi*f0*L*qe))}
L1 n1 0 {L} Q={qu}
C1 n1 0 {1/(4*pi*pi*f0*f0*L)}
L2 n2 0 {L} Q={qu}
C2 n2 0 {1/(4*pi*pi*f0*f0*L)}
K1 L1 L2 {k}
X2 p2 0 n2 0 n={sqrt(50/(2*pi*f0*L*qe))}
.ac lin 401 135.4Meg 155.4Meg
"""


@pytest.fixture
def ts(tmp_path):
    nl = parse(NET)
    f = np.linspace(135.4e6, 155.4e6, 401)
    S = mna.build(nl, w_ref=2 * np.pi * 145.4e6).solve(f)
    p = tmp_path / "synthetic.s2p"
    touchstone.save(str(p), f, S, comments=["synthetic"])
    return touchstone.load(str(p))


def test_file_source_duck_types_the_instrument(ts):
    src = FileSource(ts)
    f, s11, s21 = src.scan_hires(140e6, 150e6, 1, 401)
    assert len(f) == len(s11) == len(s21) > 10
    assert f[0] >= 140e6 and f[-1] <= 150e6
    src.close()


def test_file_source_says_so_when_the_span_is_wrong(ts):
    with pytest.raises(IOError, match="the file covers"):
        FileSource(ts).scan_hires(300e6, 400e6, 1, 401)


def test_acquisition_thread_delivers_sweeps(ts):
    acq = Acq(FileSource(ts), dict(start=140e6, stop=150e6, points=401,
                                   segments=1, average=2))
    acq.start()
    try:
        for _ in range(100):
            data, err, count = acq.latest()
            if count:
                break
            import time
            time.sleep(0.05)
        assert err is None, err
        assert count >= 1
        assert data is not None and len(data[0]) > 10
    finally:
        acq.stop()
        acq.join(timeout=2)


def test_acquisition_holds_errors_rather_than_dying():
    """A knocked cable must pause the display, not kill the thread."""
    class Broken:
        portname = "<broken>"
        def scan_hires(self, *a, **kw):
            raise IOError("cable fell out")
    acq = Acq(Broken(), dict(start=140e6, stop=150e6, points=401,
                             segments=1, average=1))
    acq.start()
    try:
        import time
        for _ in range(60):
            _, err, _ = acq.latest()
            if err:
                break
            time.sleep(0.05)
        assert err == "cable fell out"
        assert acq.is_alive(), "the thread died instead of reporting"
    finally:
        acq.stop()
        acq.join(timeout=3)


def test_touchstone_round_trip_through_the_replay_path(ts):
    """What the GUI saves must read back as what it drew."""
    src = FileSource(ts)
    f, s11, s21 = src.scan_hires(ts.f[0], ts.f[-1], 1, len(ts))
    assert np.allclose(np.abs(s21), np.abs(ts.s21))
    assert np.allclose(np.abs(s11), np.abs(ts.s11))


# --------------------------------------------------- widgets, if a display exists
tk = pytest.importorskip("tkinter")


def _shut(root, app=None):
    """Tear a window down in an order Tk survives.

    Creating and destroying a Tk root per test, each with two matplotlib
    canvases attached, aborted the whole run inside a later garbage collection
    about one run in three -- a canvas being finalised after the interpreter
    that owned its widgets had gone.  Dropping the canvases first, then the
    root, then collecting while everything is still valid, makes it reliable.
    """
    import gc
    for attr in ("canvas", "scanvas"):
        c = getattr(app, attr, None)
        if c is not None:
            try:
                c.get_tk_widget().destroy()
            except Exception:                               # noqa: BLE001
                pass
    if app is not None:
        app.canvas = app.scanvas = None
        app._inset = None
    try:
        root.destroy()
    except Exception:                                       # noqa: BLE001
        pass
    gc.collect()


def _root():
    try:
        r = tk.Tk()
        r.withdraw()
        return r
    except tk.TclError:
        pytest.skip("no display")


def test_parameter_edits_reach_the_model(tmp_path, ts):
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert set(app.vars) == {"f0", "l", "qe", "k", "qu"}

        f = np.linspace(140e6, 150e6, 201)
        before = app.model(f)
        app.vars["k"].set("0.03")
        after = app.model(f)
        assert before is not None and after is not None
        assert not np.allclose(np.abs(before[:, 1, 0]), np.abs(after[:, 1, 0])), \
            "editing k did not change the model"

        # the log slider multiplies the value in the box
        app.base["k"] = 0.01
        app._slider_moved("k", 1.0)
        assert abs(float(app.vars["k"].get()) - 0.1) < 1e-9
        app._slider_moved("k", -1.0)
        assert abs(float(app.vars["k"].get()) - 0.001) < 1e-9
    finally:
        _shut(root, locals().get('app'))


def test_a_bad_value_does_not_crash_the_window(tmp_path):
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        app.vars["l"].set("not a number")
        assert app.current()["l"] == app.base["l"], "should fall back, not raise"
        # A negative inductance parses as a number but is physically nonsense,
        # and here it reaches a sqrt() inside the tap expression.  The message
        # has to name the expression and the parameters -- "math domain error"
        # tells whoever is dragging a slider precisely nothing.
        app.vars["l"].set("-5")
        assert app.model(np.linspace(140e6, 150e6, 51)) is None
        msg = app.status.cget("text")
        assert "sqrt" in msg and "is undefined" in msg, msg
        assert "l=-5" in msg, f"the offending parameter is not named: {msg}"
    finally:
        _shut(root, locals().get('app'))


def test_closing_the_window_hands_the_instrument_back(tmp_path, ts):
    """A frozen trace behind a closed window is the same failure as a SIGTERM."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    closed = []

    class Dev(FileSource):
        def close(self):
            closed.append(True)

    try:
        app = App(root, str(netpath))
        app.dev = Dev(ts)
        app.on_close()
        assert closed == [True], "the device was not closed"
        assert app.dev is None
    finally:
        try:
            _shut(root, locals().get('app'))
        except Exception:
            pass


def test_switching_source_releases_the_previous_one(tmp_path, ts):
    """Loading a replay file while the H4 is connected must not orphan it."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    closed = []

    class Dev(FileSource):
        def close(self):
            closed.append(True)

    try:
        app = App(root, str(netpath))
        app.dev = Dev(ts)
        app._set_device(FileSource(ts))
        assert closed == [True], "the previous source was left open"
        assert isinstance(app.dev, FileSource)
    finally:
        _shut(root, locals().get('app'))


DERIVED = """a derived inductance, and a gauge
.param L=68n qu=150 cm=1.2p cin=5.0p c1=20p c2=22p c3=20p
.param al=3.5n turns=4 lsh={al*turns*turns}
.port 1 p1 0 Z0=50
.port 2 p2 0 Z0=50
Cin p1 n1 {cin}
L1  n1 0 {L} Q={qu}
C1  n1 0 {c1}
Cm1 n1 n2 {cm}
L2  n2 0 {lsh} Q={qu}
C2  n2 0 {c2}
Cm2 n2 n3 {cm}
L3  n3 0 {L} Q={qu}
C3  n3 0 {c3}
Cout p2 n3 {cin}
.ac lin 401 60Meg 300Meg
"""


def _settled(app, root=None):
    """Wait for the identifiability worker, without pumping the Tk event loop.

    Spinning on `root.update()` deadlocked here -- reliably, but only when the
    earlier tests in this file had already created and destroyed Tk roots, so a
    single update() never returned.  Joining the worker and calling the poll
    directly tests the same logic and depends on nothing about Tk's timers.
    """
    if app._pending is None:
        return app.st is not None
    thread, _box, _nl = app._pending
    thread.join(60)
    app._analysis_poll()
    return app.st is not None


def test_a_row_is_never_created_for_something_another_row_feeds(tmp_path):
    """Overriding `al` AND `lsh = al*turns^2` pins the dependency being edited.

    The window used to show a row for every resolved parameter and send them
    all as overrides, so turning `al` moved the response by exactly nothing --
    measured: the peak stayed at 137.400 MHz, where freeing `lsh` moves it to
    120.3.  Rows are now the coordinates the structural analysis chose, plus
    literals that feed none of them.
    """
    from vnafit.gui import App
    netpath = tmp_path / "d.net"
    netpath.write_text(DERIVED)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert _settled(app), "the structural analysis never finished"
        assert "lsh" in app.rows
        assert "al" not in app.rows and "turns" not in app.rows
        assert "al" not in app.current() and "lsh" in app.current()

        f = np.linspace(120e6, 180e6, 201)
        peak = lambda: f[np.argmax(np.abs(app.model(f)[:, 1, 0]))] / 1e6
        before = peak()
        app.vars["lsh"].set(f"{float(app.vars['lsh'].get()) * 1.5:.6g}")
        assert abs(peak() - before) > 5.0, "editing lsh did not move the model"
    finally:
        _shut(root, locals().get('app'))


def test_a_parameter_the_data_cannot_separate_is_refused_from_the_fit_set(tmp_path):
    """Taking a parameter OUT is always allowed; putting one in is not.

    `c2` here is the member pinned to break an internal-node impedance gauge.
    Fitting it is not a measurement, and a regulariser that hid the difference
    would return the prior's interval as if it were the data's.
    """
    from vnafit.gui import App
    netpath = tmp_path / "d.net"
    netpath.write_text(DERIVED)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert _settled(app)
        free = {n for n, v in app.fit_on.items() if v.get()}
        assert free == set(app.st.names) and "c2" not in free

        app.fit_on["c2"].set(True)
        app._toggle_fit("c2", from_box=True)
        assert not app.fit_on["c2"].get(), "a gauge was allowed into the fit set"
        assert "not separable" in app.status.cget("text")

        # and out again, which is the move that matters on real data.  Its
        # outline stays -- dashed -- so it can be clicked back.
        app._toggle_fit("l")
        assert not app.fit_on["l"].get()
        style = next(r[2][0].get_linestyle() for r in app.drawn if r[0] == "l")
        assert style != "-"
    finally:
        _shut(root, locals().get('app'))


def test_the_link_indicator_survives_a_status_message(tmp_path, ts):
    """Connection state and transient messages need separate places.

    The window had one status line, so a sweep counter, a model error or a
    refused parameter each wiped out the only place that said whether anything
    was connected at all.
    """
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    p = tmp_path / "r.s2p"
    touchstone.save(str(p), ts.f, ts.s, comments=["replay"])
    root = _root()
    try:
        app = App(root, str(netpath))
        assert "no source" in app.link.cget("text")

        app.load_replay(str(p))
        assert "r.s2p" in app.link.cget("text")

        app.status.configure(text="c2: not separable")
        assert "r.s2p" in app.link.cget("text"), \
            "a status message wiped out the connection indicator"

        app.start()
        assert "sweeping" in app.link.cget("text")
        app.stop()
        assert "sweeping" not in app.link.cget("text")

        # a firmware banner must not stretch the toolbar: the H4 answers `info`
        # with four lines of copyright and a project URL, and putting that in
        # the label resized the window when a VNA was connected
        from vnafit.gui import short_device
        banner = ("Board: NanoVNA-H 4 | 2019-2023 Copyright @DiSlord (based on "
                  "@edy555 source) | Licensed under GPL. | "
                  "https://github.com/DiSlord/NanoVNA-D")
        assert short_device(banner) == "NanoVNA-H 4"
        app._link(short_device(banner), "sweep 1234", OK)
        assert len(app.link.cget("text")) <= app.link.cget("width")
        # and the state stays visible when the NAME is the long part
        app._link("replay " + "x" * 80, "sweep 7", OK)
        assert app.link.cget("text").endswith("sweep 7")
        assert len(app.link.cget("text")) <= app.link.cget("width")

        # a source that dies on its own -- a pulled USB cable -- must show
        app.acq = None
        app.tick()
        assert "stopped" in app.link.cget("text")
        assert app.link.cget("fg") != OK
    finally:
        _shut(root, locals().get('app'))


def _click(app, param):
    """Synthesise a click at the centre of `param`'s first outline."""
    import types
    rect = next(rects[0] for pn, _e, rects, _t in app.drawn if pn == param)
    ax = rect.axes
    cx = rect.get_x() + rect.get_width() / 2
    cy = rect.get_y() + rect.get_height() / 2
    x, y = ax.transData.transform((cx, cy))
    app._schematic_click(types.SimpleNamespace(inaxes=ax, x=x, y=y))


def test_a_pinned_parameter_keeps_its_outline_and_can_be_clicked_back(tmp_path):
    """Pinning used to erase the outline, which read as a click deleting
    something and left nothing on the drawing to click to undo it."""
    from vnafit.gui import App
    netpath = tmp_path / "d.net"
    netpath.write_text(DERIVED)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert _settled(app)
        style = lambda p: next((r[2][0].get_linestyle()
                                for r in app.drawn if r[0] == p), None)
        assert style("l") == "-"

        _click(app, "l")
        assert not app.fit_on["l"].get(), "the click did not pin l"
        assert style("l") is not None, "pinning erased the outline"
        assert style("l") != "-", "a pinned outline must be distinguishable"
        assert "click it again" in app.status.cget("text")

        _click(app, "l")
        assert app.fit_on["l"].get(), "a pinned outline could not be clicked back"
        assert style("l") == "-"
    finally:
        _shut(root, locals().get('app'))


def test_the_smallest_outline_under_a_click_wins(tmp_path):
    """Outlines nest -- `l` and `qu` both wrap L1 -- and taking the first hit
    in drawing order toggled whichever was listed first, so clicking the inner
    box changed the outer parameter."""
    from vnafit.gui import App
    netpath = tmp_path / "d.net"
    netpath.write_text(DERIVED)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert _settled(app)
        area = lambda p: next(r[2][0].get_width() * r[2][0].get_height()
                              for r in app.drawn if r[0] == p)
        assert area("l") < area("qu"), "this test needs l nested inside qu"

        _click(app, "l")
        assert not app.fit_on["l"].get()
        assert app.fit_on["qu"].get(), "the click toggled the enclosing outline"
    finally:
        _shut(root, locals().get('app'))


def test_a_toggle_does_not_discard_tuned_values(tmp_path):
    """Toggling rebuilds the parameter column, which reads values from
    self.base -- so it threw away everything tuned since the netlist loaded."""
    from vnafit.gui import App
    netpath = tmp_path / "d.net"
    netpath.write_text(DERIVED)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert _settled(app)
        app.vars["c1"].set("4e-11")
        app._toggle_fit("l")
        assert app.vars["c1"].get() == "4e-11"
        assert abs(app.base["c1"] - 4e-11) < 1e-18, \
            "the slider would jump back to the old value"
    finally:
        _shut(root, locals().get('app'))


def _fit_now(app):
    """Run the Fit button to completion without the Tk event loop."""
    app.do_fit()
    assert app._fitting, app.status.cget("text")
    thread, _box = app._fitting
    thread.join(300)
    app._fit_poll()


def test_the_fit_button_moves_the_model_toward_the_data(tmp_path, ts):
    """Until this existed the window only ever OVERLAID a model: a filter could
    sit connected with the two traces far apart and nothing would happen."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        app.meas = (ts.f, ts.s[:, 0, 0], ts.s[:, 1, 0])
        for n in app.rows:                       # fit k alone, from 2x off
            app.fit_on[n].set(n == "k")
        app.vars["k"].set("0.024")
        app.base["k"] = 0.024

        _fit_now(app)
        assert app.fitres is not None, app.status.cget("text")
        assert abs(float(app.vars["k"].get()) - 0.0121) / 0.0121 < 0.02, \
            "the fit did not recover k"
        assert app.fitres.rms < 0.05
    finally:
        _shut(root, locals().get('app'))


def test_a_fit_holds_pinned_parameters_at_the_values_on_screen(tmp_path, ts):
    """fit.Problem seeds from, and holds fixed, whatever the NETLIST says -- so
    a fit would otherwise ignore the boxes and silently use the file's values
    for everything pinned."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        app.meas = (ts.f, ts.s[:, 0, 0], ts.s[:, 1, 0])
        app.vars["qu"].set("321")               # not the file's 670
        for n in app.rows:
            app.fit_on[n].set(n == "k")

        assert app._pinned_netlist().params["qu"] == 321.0
        _fit_now(app)
        assert app.vars["qu"].get() == "321", "a pinned parameter was moved"
    finally:
        _shut(root, locals().get('app'))


def test_a_thin_passband_is_reported(tmp_path):
    """A 401-point sweep looks generous until it is spread over 220 MHz and the
    passband gets 18 of them -- and the fit still reports a confident rms,
    because the stopband points agree beautifully."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    nl = parse(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        for span, expect_thin in ((20e6, False), (250e6, True)):
            f = np.linspace(145.4e6 - span / 2, 145.4e6 + span / 2, 401)
            S = mna.build(nl, w_ref=2 * np.pi * 145.4e6).solve(f)
            app.meas = (f, S[:, 0, 0], S[:, 1, 0])
            n, bw, (lo, hi) = app.passband_pts()
            assert (n < app.PASSBAND_PTS) is expect_thin, (span, n)
            app.redraw()
            assert ("THIN" in app.readout.get("1.0", "end")) is expect_thin
            if expect_thin:
                assert lo < 145.4e6 < hi and (hi - lo) < span
    finally:
        _shut(root, locals().get('app'))


def test_refit_each_sweep_is_off_unless_asked_and_then_keeps_up(tmp_path, ts):
    """`Fit` is one-shot; ticking `refit each sweep` makes the loop close.

    Off by default on purpose: while you are turning a screw the model should
    stay put, because one that chases the trace fits well at every position of
    the screw and so says nothing about where to stop.
    """
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    p = tmp_path / "r.s2p"
    touchstone.save(str(p), ts.f, ts.s, comments=["replay"])
    root = _root()
    try:
        app = App(root, str(netpath))
        app.load_replay(str(p))
        for n in app.rows:
            app.fit_on[n].set(n == "k")
        app.vars["k"].set("0.024")
        app.base["k"] = 0.024
        assert not app.v_autofit.get(), "auto-refit must be opt-in"

        # a sweep with it off must not fit
        app.start()
        for _ in range(200):
            app.tick()
            if app.meas is not None:
                break
        assert app.fitres is None, "a sweep refitted with the box unticked"

        app.v_autofit.set(True)
        app._last_count = -1
        for _ in range(200):
            app.tick()
            if app._fitting:
                break
        assert app._fitting, "ticking the box did not start a fit"
        thread, _box = app._fitting
        thread.join(300)
        app._fit_poll()
        assert app.fitres is not None
        assert abs(float(app.vars["k"].get()) - 0.0121) / 0.0121 < 0.02
        assert app.acq.is_alive(), "fitting stopped the acquisition"
    finally:
        app.stop()
        _shut(root, locals().get('app'))


SPICE_SYM = """* an LTspice export of a symmetric 3-pole
V1 vsrc 0 AC 1
Rsrc vsrc p1 50
Rload p2 0 50
Cin p1 n1 4.3431p
L1 n1 l1q 30n
RL1 l1q 0 0.139
C1 n1 0 34.546p
Cm1 n1 n2 0.8877p
L2 n2 l2q 30n
RL2 l2q 0 0.139
C2 n2 0 37.835p
Cm2 n2 n3 0.8877p
L3 n3 l3q 30n
RL3 l3q 0 0.139
C3 n3 0 34.546p
Cout p2 n3 4.3431p
.ac lin 401 125Meg 180Meg
.end
"""


def test_a_spice_file_loads_in_the_window_without_a_modal(tmp_path):
    """The import report used to go into a messagebox, which is MODAL: it
    blocked load_netlist before the identifiability worker was started, and in
    any context with nobody to click it, it never returned at all."""
    from vnafit.gui import App
    p = tmp_path / "lt.net"
    p.write_text(SPICE_SYM)
    root = _root()
    try:
        app = App(root, str(p))                  # must not block
        assert app.nl is not None
        assert app.import_report and "port 1" in app.import_report
        # grouped by the role heuristic: one `l`, one `cin`, one loss
        assert "l" in app.rows and "cin" in app.rows, app.rows
        assert _settled(app)
        app.redraw()
        assert "imported from SPICE" in app.readout.get("1.0", "end")
    finally:
        _shut(root, locals().get('app'))


def test_the_column_separates_what_can_be_fitted_from_what_you_set(tmp_path):
    """With only S11 and S21, the two ends of a SYMMETRIC filter are nearly
    indistinguishable, so per-element promotion leaves most parameters
    determined but unmeasurable.  Ticking them all by default turned thirteen
    parameters loose, eight of them with best cases in the hundreds of percent.
    """
    from vnafit.gui import App
    p = tmp_path / "lt.net"
    p.write_text(SPICE_SYM)
    root = _root()
    try:
        app = App(root, str(p))
        assert _settled(app)
        ticked = {n for n in app.rows if app.fit_on[n].get()}
        assert ticked, "nothing was fittable after grouping"
        # `l` is now pinned on purpose -- it carries the impedance scale, which
        # is soft here -- so the trimmers are what must be free.
        assert {"c1", "c2", "c3"} <= ticked and "l" not in ticked, ticked
        # after grouping this circuit has nothing left that is unmeasurable,
        # which is the point of the heuristic -- so the reasons are exercised
        # one at a time rather than expecting one netlist to show all three
        assert all(app._why_held(n) for n in app.rows if n not in ticked)

        app._toggle_fit("c2")
        assert app._why_held("c2") == "you pinned it"

        app._prec["c3"] = 4.83            # as an unmeasurable one would read
        app.fit_on["c3"].set(False)
        assert app._why_held("c3") == "best case 483%"

        app.st.names = [n for n in app.st.names if n != "c1"]
        app.fit_on["c1"].set(False)
        assert app._why_held("c1") == "not separable - set it"
    finally:
        _shut(root, locals().get('app'))


def test_the_target_is_the_files_own_values_and_a_fit_does_not_move_it(tmp_path, ts):
    """Priority 1 of this project is tuning to match a model, and until there
    was a target the model moved every time you fitted -- there was nothing
    fixed to tune toward."""
    from vnafit.gui import App
    netpath = tmp_path / "m.net"
    netpath.write_text(NET)
    root = _root()
    try:
        app = App(root, str(netpath))
        assert app.target["k"] == 0.0121 and app.target["qu"] == 670.0

        app.meas = (ts.f, ts.s[:, 0, 0], ts.s[:, 1, 0])
        for n in app.rows:
            app.fit_on[n].set(n == "k")
        app.vars["k"].set("0.024")
        app.base["k"] = 0.024
        _fit_now(app)
        assert app.fitres is not None
        assert app.target["k"] == 0.0121, "the fit moved the target"

        # and an edit moves the working model, not the target
        f = np.linspace(140e6, 150e6, 101)
        before = app.target_model(f)
        app.vars["qu"].set("120")
        assert not np.allclose(np.abs(app.model(f)[:, 1, 0]),
                               np.abs(before[:, 1, 0])), "the edit did nothing"
        assert np.allclose(np.abs(app.target_model(f)[:, 1, 0]),
                           np.abs(before[:, 1, 0])), "the edit moved the target"
    finally:
        _shut(root, locals().get('app'))


def test_the_tuning_table_says_which_way_to_turn(tmp_path):
    """It reads the model on screen, so it follows a slider as well as a fit --
    and it is in frequencies and ratios, not picofarads, so the impedance scale
    cannot turn it into nonsense."""
    from vnafit.gui import App
    root = _root()
    app = None
    try:
        app = App(root, "examples/tinyfilter_ltspice.net")
        assert _settled(app)
        assert not [l for l in app.tuning_lines() if "<- turn" in l], \
            "the model starts AT the target, so nothing should want turning"

        base = float(app.vars["c2"].get())
        for factor, sign in ((1.06, "-"), (0.94, "+")):
            app.vars["c2"].set(f"{base * factor:.6g}")
            line = [l for l in app.tuning_lines() if l.strip().startswith("C2")]
            assert line, app.tuning_lines()
            # more C -> lower f -> the trimmer must come back down, and vice versa
            assert f" {sign}" in line[0], (factor, line[0])
    finally:
        _shut(root, app)


def test_the_zoom_window_finds_the_feature_and_then_holds_still(tmp_path):
    """A window that re-centres on every sweep is unreadable exactly when you
    are turning something."""
    from vnafit.gui import App
    root = _root()
    try:
        app = App(root, "examples/tinyfilter_ltspice.net")
        app.redraw()
        assert app._zoom and app._zoom[2] == "peak"
        lo, hi = app._zoom[0], app._zoom[1]
        assert lo < 146e6 < hi

        app.vars["c1"].set("6e-11")          # detune hard
        app.redraw()
        assert (app._zoom[0], app._zoom[1]) == (lo, hi), "the window moved"
    finally:
        _shut(root, locals().get('app'))


def test_a_notch_is_zoomed_on_its_minimum_not_its_maximum(tmp_path):
    from vnafit.gui import App
    root = _root()
    try:
        app = App(root, "examples/topologies/notch_trap.net")
        app.redraw()
        assert app._zoom and app._zoom[2] == "notch", app._zoom
        lo, hi = app._zoom[0], app._zoom[1]
        assert lo < 146e6 < hi, (lo, hi)
    finally:
        _shut(root, locals().get('app'))


def test_a_soft_direction_is_pinned_rather_than_fitted(tmp_path):
    """The impedance scale on the TinyFilter sits at a relative singular value
    of 1.8e-4.  Left free, a fit parked at 3.6 nH with 300 pF capacitors and a
    convincing residual -- the data cannot tell that apart from 30 nH and 32 pF.

    The Cramer-Rao bound does not catch it and is not wrong to miss it: it says
    `l` is known to 2.6%, the right answer to "if the model is exact and the
    noise is 0.05 dB".  The model is wrong by 0.33 dB here.
    """
    from vnafit.gui import App
    root = _root()
    app = None
    try:
        app = App(root, "examples/tinyfilter_ltspice.net")
        assert _settled(app)
        assert app.st.soft_directions(), "no soft direction found to pin"
        ticked = {n for n in app.rows if app.fit_on[n].get()}
        assert "l" not in ticked, "the impedance scale was left free"
        assert {"c1", "c2", "c3"} <= ticked, "the trimmers must stay free"
        assert "pinned" in app.status.cget("text")

        # and it is a choice, not a prohibition: you can still tick it
        app.fit_on["l"].set(True)
        app._toggle_fit("l", from_box=True)
        assert app.fit_on["l"].get()
    finally:
        _shut(root, app)
