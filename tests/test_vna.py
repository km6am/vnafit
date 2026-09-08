"""The instrument driver's diagnostics.

The sweep path itself needs hardware, so what is tested here is the part that
was actually getting people lost: telling "wrong port" apart from "something
else already has this port".
"""
import os

import pytest

from vnafit.vna import port_holder


def test_port_holder_never_raises_and_always_explains():
    for path in ("/dev/definitely-not-a-port", "/nonexistent/zzz", __file__):
        msg = port_holder(path)
        assert isinstance(msg, str) and msg.startswith(" - ")


def test_unheld_port_gets_the_fallback_hint():
    msg = port_holder("/dev/definitely-not-a-port")
    assert "wrong port" in msg
    assert "already has it open" not in msg


def test_a_file_this_process_holds_is_not_blamed_on_someone_else(tmp_path):
    """The caller's own handle must not be reported as a conflict.

    Without the self-PID filter every failure would blame the program doing the
    complaining.
    """
    p = tmp_path / "held"
    p.write_text("x")
    with open(p, "r"):
        msg = port_holder(str(p))
    assert f"pid {os.getpid()}" not in msg


def test_the_message_names_a_real_holder_when_there_is_one(tmp_path):
    """The case that motivated this: a second program holding the port makes the
    probe fail in a way that looks exactly like a wrong port number.

    Observed live -- another program was still running and holding
    /dev/cu.usbmodem4001, and the driver reported "wrong port?" about an
    instrument that was plugged in and working perfectly.
    """
    import subprocess
    import sys
    import time
    p = tmp_path / "held2"
    p.write_text("x")
    child = subprocess.Popen(
        [sys.executable, "-c",
         f"f=open({str(p)!r}); import time; time.sleep(10)"])
    try:
        time.sleep(1.0)
        msg = port_holder(str(p))
        if "already has it open" in msg:          # lsof present and permitted
            assert f"pid {child.pid}" in msg
            assert "Close it and try again" in msg
    finally:
        child.terminate()
        child.wait(timeout=5)


# ------------------------------------------------ leaving the instrument alone
class FakeDev:
    """Stands in for an open instrument in the release registry."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self._closed = False

    def resume(self):
        self.calls.append("resume")
        return True

    def close(self):
        if self._closed:
            self.calls.append("close-again")
            return
        self._closed = True
        if self.fail:
            raise IOError("port already gone")
        self.calls.append("close")


def test_release_all_closes_every_open_instrument():
    """The safety net.

    `close()` in a finally block is not enough on its own: SIGTERM -- what a
    pkill, a kill, or a supervisor shutdown all send -- terminates CPython
    without running finally, and the instrument is left with its trace frozen,
    looking hung.  That happened during this project.
    """
    from vnafit import vna as V
    a, b = FakeDev(), FakeDev()
    V._OPEN.update({a, b})
    try:
        V.release_all()
        assert a.calls == ["close"] and b.calls == ["close"]
    finally:
        V._OPEN.discard(a)
        V._OPEN.discard(b)


def test_release_all_survives_a_device_that_cannot_be_closed():
    """One dead port must not stop the others being handed back."""
    from vnafit import vna as V
    bad, good = FakeDev(fail=True), FakeDev()
    V._OPEN.update({bad, good})
    try:
        V.release_all()
        assert good.calls == ["close"]
    finally:
        V._OPEN.discard(bad)
        V._OPEN.discard(good)


def test_hooks_are_installed_once_and_only_over_defaults():
    """A library has no business stamping on an application's signal handlers."""
    import signal
    from vnafit import vna as V

    mine = lambda *a: None
    prev = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, mine)
    was = V._HOOKS_INSTALLED
    V._HOOKS_INSTALLED = False
    try:
        V._install_hooks()
        assert signal.getsignal(signal.SIGTERM) is mine, "clobbered a live handler"
        assert V._HOOKS_INSTALLED
        V._install_hooks()          # second call must be a no-op
    finally:
        signal.signal(signal.SIGTERM, prev)
        V._HOOKS_INSTALLED = was


def test_close_is_idempotent_and_the_context_manager_uses_it():
    from vnafit.vna import NanoVNA
    assert hasattr(NanoVNA, "__enter__") and hasattr(NanoVNA, "__exit__")
    d = NanoVNA.__new__(NanoVNA)          # no hardware, no __init__
    d._closed = True                      # already closed
    d.close()                             # must not raise or touch a serial port


# ------------------------------------------------------------------- standalone
HEAVY = ("scipy", "matplotlib", "pandas", "tkinter", "sklearn")


def _vna_source():
    import vnafit.vna
    return vnafit.vna.__file__


def test_the_driver_imports_nothing_from_the_rest_of_vnafit():
    """It must stay copy-pasteable into someone else's project.

    Enforced by reading the AST rather than by convention, because the moment
    one `from . import touchstone` creeps in the file stops being a drop-in and
    nobody notices until they try.
    """
    import ast
    tree = ast.parse(open(_vna_source()).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import of {node.module!r}"
            root = (node.module or "").split(".")[0]
        elif isinstance(node, ast.Import):
            root = node.names[0].name.split(".")[0]
        else:
            continue
        assert root != "vnafit", f"imports vnafit.{node}"
        assert root not in HEAVY, f"pulls in {root}, which is not a driver dependency"


def test_importing_the_driver_does_not_drag_in_the_science_stack():
    """A subprocess, so the rest of the test session's imports cannot mask it."""
    import subprocess
    import sys
    code = ("import sys; import vnafit.vna; "
            "bad=[m for m in %r if m in sys.modules]; "
            "print(','.join(bad))" % (HEAVY,))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"driver pulled in {out.stdout.strip()}"


def test_the_driver_runs_as_a_file_on_its_own(tmp_path):
    """Copied out on its own it must still be a working command."""
    import shutil
    import subprocess
    import sys
    dst = tmp_path / "vna.py"
    shutil.copy(_vna_source(), dst)
    out = subprocess.run([sys.executable, str(dst), "--help"],
                         capture_output=True, text=True, timeout=60,
                         cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert "--sweep" in out.stdout and "standalone" in out.stdout.lower()


def test_find_refuses_a_port_that_is_obviously_not_a_nanovna(monkeypatch):
    """Falling back to 'whatever serial port exists' is worse than useless.

    With the instrument unplugged it used to pick the Bluetooth port, burn the
    probe timeout on it, and report that Bluetooth did not answer as a NanoVNA
    -- which sends you hunting for a driver problem instead of a USB cable.
    """
    from vnafit.vna import NanoVNA
    fake = [dict(device="/dev/cu.Bluetooth-Incoming-Port", desc="n/a",
                 likely=False, label="/dev/cu.Bluetooth-Incoming-Port")]
    monkeypatch.setattr(NanoVNA, "list_ports", staticmethod(lambda: fake))
    assert NanoVNA.find() is None
    assert NanoVNA.find(any_port=True) == "/dev/cu.Bluetooth-Incoming-Port"
    with pytest.raises(IOError, match="none of them identifies as a NanoVNA"):
        NanoVNA()
