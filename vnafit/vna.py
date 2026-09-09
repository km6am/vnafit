"""NanoVNA-H4 driver (DiSlord text shell over USB CDC).

STANDALONE BY DESIGN.  This file imports nothing from the rest of vnafit and
nothing heavier than numpy (pyserial is imported lazily, inside the methods that
need it), so it can be copied on its own into any project that wants to drive an
H4.  A test enforces that; if you add an import here, make it stdlib or numpy.

It is also runnable on its own:

    python vna.py                              list serial ports
    python vna.py --sweep out.s2p              sweep and write Touchstone
    python vna.py --sweep out.s2p --start 140e6 --stop 152e6 --points 401

Vendored verbatim from the author's earlier helical-filter tuning tool.
Deliberately not
rewritten: nearly every line that looks defensive here is a bug that actually
happened on the bench, and scikit-rf has no driver for this instrument at all
(its NanoVNAv2 class speaks the V2/LiteVNA *binary* protocol and will not talk
to an H4).  The `Replay` class duck-types the driver, so the whole tool runs
file-only with no hardware present.

The H4 enumerates as /dev/cu.usbmodem*.  Commands are newline-terminated and
the device echoes them back before the data, then prints the 'ch> ' prompt.
"""
import atexit, contextlib, os, signal, threading, time, numpy as np

PROMPT = b"ch> "
# USB CDC identities seen on NanoVNA-H / -H4 (STM32 virtual COM port)
NANOVNA_IDS = {(0x0483, 0x5740), (0x16c0, 0x0483), (0x0483, 0xdf11)}
NAME_HINTS = ("nanovna", "stm32", "virtual com", "chibios")

# Every live instrument, so that whatever happens to this process the trace is
# handed back.  `close()` alone is not enough: a SIGTERM -- which is what a
# `pkill`, a `kill`, or a supervisor shutting the job down all send -- terminates
# CPython without running `finally`, and the instrument is left frozen with its
# screen stopped, looking hung.  Observed exactly that way.
_OPEN = set()
_HOOKS_INSTALLED = False


def release_all(_signum=None, _frame=None):
    """Resume every open instrument.  Safe to call twice, never raises."""
    for dev in list(_OPEN):
        try:
            dev.close()
        except Exception:                                   # noqa: BLE001
            pass
    if _signum is not None:
        # Restore the default action and re-raise, so the exit status still
        # says "killed by signal" rather than quietly pretending otherwise.
        try:
            signal.signal(_signum, signal.SIG_DFL)
            os.kill(os.getpid(), _signum)
        except Exception:                                   # noqa: BLE001
            raise SystemExit(128 + int(_signum))


def _install_hooks():
    """atexit plus SIGTERM/SIGINT, installed once and only where it is polite.

    A library has no business stamping on an application's signal handlers, so
    this only replaces handlers that are still at their default, and only from
    the main thread where installing one is even legal.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True
    atexit.register(release_all)
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            if signal.getsignal(sig) in (signal.SIG_DFL, signal.default_int_handler):
                signal.signal(sig, release_all)
        except (ValueError, OSError):
            pass


def port_holder(port):
    """A clause naming whatever else has the port open, or a fallback hint.

    On macOS and Linux a serial port is not locked, so a second program can open
    it and simply read nothing back -- both clients then see garbage or silence.
    The failure looks identical to "wrong port", which sends you hunting for the
    wrong thing.  Observed live: another program was still running and
    holding /dev/cu.usbmodem4001, and this driver reported "wrong port?" about
    an instrument that was plugged in and working.
    """
    try:
        import subprocess
        out = subprocess.run(["lsof", "-t", "-w", port], capture_output=True,
                             text=True, timeout=4).stdout.split()
        pids = [int(x) for x in out if x.isdigit() and int(x) != os.getpid()]
        if pids:
            who = []
            for pid in pids[:3]:
                try:
                    cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                         capture_output=True, text=True,
                                         timeout=4).stdout.strip()
                except Exception:
                    cmd = ""
                who.append(f"pid {pid}" + (f" ({cmd.split()[-1]})" if cmd else ""))
            return (" - another program already has it open: "
                    + ", ".join(who) + ".  Close it and try again.")
    except Exception:
        pass
    return " - wrong port, or the instrument is asleep?"


class NanoVNA:
    def __init__(self, port=None, timeout=6.0, probe_timeout=1.5):
        import serial                       # pyserial
        self.link_err = None
        self.portname = port or self.find()
        if not self.portname:
            seen = self.list_ports()
            if seen:
                raise IOError(
                    "no NanoVNA found.  Serial ports on this machine are: "
                    + ", ".join(p['label'] for p in seen)
                    + " -- none of them identifies as a NanoVNA.  Check the USB "
                      "cable and that the instrument is powered on; pass an "
                      "explicit port to override.")
            raise IOError("no serial ports at all - is the H4 plugged in and awake?")
        self.ser = serial.Serial(self.portname, 115200, timeout=probe_timeout)
        self._lock = threading.RLock()      # one reader at a time; the GUI thread
                                            # and the acquisition thread share this
        time.sleep(0.12)
        # Drain whatever the instrument is still emitting BEFORE talking to it.
        # A sweep that was interrupted - the app killed, a cable knocked, a
        # second client stealing bytes - leaves the tail of its scan sitting in
        # the pipe, and the next command reads that as its own reply.  Observed
        # live: info() came back as "996240 0.963592576 ... | 139670000 ...",
        # which is S-parameter rows, and every later read stayed one response
        # behind.
        self._flush(2.0)
        if not self._probe():               # don't adopt a Bluetooth or debug port
            self.ser.close()
            raise IOError(f"{self.portname} did not answer as a NanoVNA "
                          f"(no 'ch>' prompt){port_holder(self.portname)}")
        self.ser.timeout = timeout
        # Cache the identity NOW.  Asking for it later, while the acquisition
        # thread is sweeping, is a concurrent read of the same port and pyserial
        # raises "device reports readiness to read but returned no data".
        try:
            self._info = (" | ".join(self.cmd("info")[:4])
                          or " ".join(self.cmd("version")) or self.portname)
        except Exception:
            self._info = self.portname
        self._closed = False
        _OPEN.add(self)
        _install_hooks()

    def _flush(self, seconds=2.0):
        """Read and discard until the port goes quiet, or `seconds` elapse."""
        t0 = time.time(); quiet = 0
        while time.time() - t0 < seconds:
            try:
                n = self.ser.in_waiting
                chunk = self.ser.read(n) if n else b""
            except Exception:
                break
            if chunk:
                quiet = 0
            else:
                quiet += 1
                if quiet >= 3:
                    break
                time.sleep(0.05)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def _probe(self, tries=2):
        """Send a bare CR and look for the shell prompt.  Fails in ~1.5 s per try
        rather than blocking on a long read timeout for a port that will never
        answer."""
        for _ in range(tries):
            self.ser.reset_input_buffer()
            self.ser.write(b"\r")
            if PROMPT in self._read_to_prompt():
                return True
            self._flush(0.5)                # desynced, not dead - try again clean
        return False

    @staticmethod
    def list_ports():
        """Every serial port, likely NanoVNAs first.  Works on macOS
        (/dev/cu.*), Linux (/dev/ttyACM*) and Windows (COMn)."""
        from serial.tools import list_ports
        out = []
        for p in list_ports.comports():
            desc = (p.description or "").strip()
            likely = ((p.vid, p.pid) in NANOVNA_IDS
                      or any(h in desc.lower() for h in NAME_HINTS))
            out.append(dict(device=p.device, desc=desc, likely=likely,
                            label=f"{p.device}" + (f"  ({desc})" if desc and desc != "n/a" else "")))
        return sorted(out, key=lambda d: (not d['likely'], d['device']))

    @staticmethod
    def find(any_port=False):
        """The most likely NanoVNA port, or None.

        Only returns a port whose USB identity or description actually looks
        like a NanoVNA.  Falling back to "whatever serial port exists" is worse
        than useless: with the instrument unplugged it picks
        /dev/cu.Bluetooth-Incoming-Port, spends the probe timeout on it, and
        then reports that the Bluetooth port did not answer as a NanoVNA --
        which sends you looking for a driver problem instead of a USB cable.
        Pass any_port=True to try everything anyway.
        """
        ports = NanoVNA.list_ports()
        likely = [p for p in ports if p['likely']]
        if likely:
            return likely[0]['device']
        if any_port and ports:
            return ports[0]['device']
        return None

    def _drain(self):
        self.ser.reset_input_buffer()
        self.ser.write(b"\r")
        self._read_to_prompt()

    def _read_to_prompt(self):
        """Sets self.link_err when the read itself failed, rather than losing it.

        Swallowing it silently is how a wedged USB CDC endpoint turns into
        "sweep returned 0 points": pyserial raises "device reports readiness to
        read but returned no data", the buffer comes back empty, and every
        layer above reports a data problem for what is actually a dead link.
        """
        import serial
        buf = b""
        self.link_err = None
        while True:
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except serial.SerialException as e:
                self.link_err = str(e)
                break                       # unplugged mid-read, or a lost race
            if not chunk:
                break
            buf += chunk
            if buf.endswith(PROMPT):
                break
        return buf

    def cmd(self, s):
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write((s + "\r").encode())
            raw = self._read_to_prompt().decode(errors="replace")
        lines = raw.replace("\r", "").split("\n")
        # drop the echoed command and the trailing prompt
        if lines and lines[0].strip().startswith(s.split()[0]):
            lines = lines[1:]
        # A partial read can leave the tail of the prompt on its own line ("h>",
        # ">"), which is not a data row and is not caught by startswith("ch>").
        def junk(l):
            t = l.strip()
            return (not t) or t.startswith("ch>") or t.rstrip(">").strip() in ("", "c", "h", "ch")
        return [l for l in lines if not junk(l)]

    def info(self):
        """Cached - never touches the port, so it is safe to call while the
        acquisition thread is sweeping."""
        return getattr(self, "_info", self.portname)

    def set_bandwidth(self, code):
        """IF bandwidth on DiSlord firmware: `bandwidth {0..7}`, lower index =
        wider = faster, higher index = narrower = lower noise floor.

        TESTED on an H4: the command is honoured (sweep time scales cleanly,
        0.19 s at code 0 to 0.54 s at code 7).  But it does NOT buy stopband
        depth on a filter measurement - 8x narrower IF plus 8x averaging moved
        a -79 dB floor by 3.1 dB, where noise-limited would give ~18.  That
        floor is port-to-port leakage, which no amount of filtering removes.
        Use it to quieten the trace, not to see deeper.  Failures are swallowed."""
        try:
            self.cmd(f"bandwidth {int(code)}")
            return True
        except Exception:
            return False

    # ----------------------------------------------------------- calibration
    # Read from the DiSlord NanoVNA-D source (cmd_cal in main.c), not from
    # memory.  A first version of this guessed that bare `cal` replies "on" or
    # "off"; it does not.  It prints the set bits of cal_status as words from
    #
    #     items[] = { "load","open","short","thru","isoln","Es","Er","Et","cal'ed" }
    #
    # so an uncalibrated instrument answers with an empty line, and a fully
    # calibrated one with something like "load open short thru isoln Es Er Et
    # cal'ed".  `cal on` sets CALSTAT_APPLY and `cal off` clears it, and that
    # bit is the one printed as `cal'ed` -- which is why APPLIED below is that
    # word and not any of the others: the rest mean a standard was COLLECTED,
    # which is a different thing from the correction being in use.
    CAL_STATUS, CAL_ON, CAL_OFF, CAL_RECALL = "cal", "cal on", "cal off", "recall %d"
    CAL_APPLIED = "cal'ed"
    CAL_COLLECT = ("open", "short", "thru")
    # `#define CALSTAT_ED CALSTAT_LOAD` and `#define CALSTAT_EX CALSTAT_ISOLN`
    # -- the same BITS, so once a calibration is finished the words "load" and
    # "isoln" mean the ED and EX terms exist, not that a standard is merely
    # sitting collected.  Reporting them as standards made a complete
    # calibration read as a half-finished one: a real H4 answers
    # "load isoln Es Er Et cal'ed", which is all five terms and applied.
    CAL_TERM_WORDS = {"load": "ED", "isoln": "EX",
                      "Es": "ES", "Er": "ER", "Et": "ET"}

    def cal_status(self):
        """What the instrument says about its own correction.

        Returns a dict with

            enabled     True/False/None -- correction is being APPLIED
            terms       which error terms the instrument holds
            complete    ED, ES and ER present, i.e. S11 can be corrected
            collected   raw standards still flagged as collected
            raw         the reply, verbatim

        `enabled` is None only when the instrument could not be asked.  An
        EMPTY reply is a real answer -- uncalibrated -- and is reported as
        False, not as unknown.
        """
        try:
            out = self.cmd(self.CAL_STATUS)
        except Exception as e:                              # noqa: BLE001
            return {"raw": [], "enabled": None, "terms": (), "complete": False,
                    "collected": (), "error": str(e)}
        words = " ".join(out).split()
        terms = tuple(v for k, v in self.CAL_TERM_WORDS.items() if k in words)
        return {"raw": out,
                "enabled": self.CAL_APPLIED in words,
                "terms": terms,
                "complete": {"ED", "ES", "ER"} <= set(terms),
                "collected": tuple(w for w in self.CAL_COLLECT if w in words)}

    # A fixed sweep used only for fingerprinting a slot.  `data 2..6` does NOT
    # return the stored calibration -- it returns that calibration INTERPOLATED
    # onto whatever span and point count the instrument is currently sweeping.
    # Measured on an H4: the same untouched slot hashed three different ways at
    # 101, 201 and 401 points, and a normalised-position digest across the two
    # extremes differed by a median of 45%.  So a fingerprint is only meaningful
    # if it is always taken at the same setting.
    FINGERPRINT_SWEEP = (1_000_000, 900_000_000, 401)

    def sweep_state(self):
        """(start, stop, points) the instrument is set to, or None."""
        try:
            p = " ".join(self.cmd("sweep")).split()
            return (int(p[0]), int(p[1]), int(p[2])) if len(p) >= 3 else None
        except Exception:                                   # noqa: BLE001
            return None

    def cal_fingerprint(self):
        """The slot's error terms read at a FIXED sweep, so they compare.

        Puts the instrument's sweep back afterwards.  Returns (terms, sweep).
        """
        before = self.sweep_state()
        a, b, n = self.FINGERPRINT_SWEEP
        try:
            self.cmd(f"sweep {a} {b} {n}")
            terms = self.cal_terms()
        finally:
            if before:
                self.cmd("sweep %d %d %d" % before)
        return terms, self.FINGERPRINT_SWEEP

    def cal_terms(self):
        """Download the instrument's calibration error terms.

        `data 2..6` returns cal_data[0..4] -- the five terms ED, ES, ER, ET, EX,
        INTERPOLATED ONTO THE CURRENT SWEEP.  Use `cal_fingerprint()` when the
        answer has to be comparable between calls.  There is no command to write
        them back:
        the firmware exposes no upload, so a calibration can be read off the
        instrument and archived, but only re-created by running the standards
        again or by `recall`ing a slot.
        """
        names = ("ED", "ES", "ER", "ET", "EX")
        out = {}
        for i, name in enumerate(names):
            rows = self.cmd(f"data {i + 2}")
            v = []
            for r in rows:
                p = r.split()
                if len(p) >= 2:
                    try:
                        v.append(float(p[0]) + 1j * float(p[1]))
                    except ValueError:
                        pass
            out[name] = np.array(v)
        return out

    def set_correction(self, on):
        """Turn the instrument's own error correction on or off.

        VERIFIED BY READING BACK.  If the firmware does not understand the
        command the state will not have changed, and this raises rather than
        returning a cheerful True -- measuring uncorrected when you think you
        are corrected, or the reverse, is worse than not being able to switch.
        """
        want = bool(on)
        self.cmd(self.CAL_ON if want else self.CAL_OFF)
        st = self.cal_status()
        got = st["enabled"]
        if got is None:
            raise IOError(
                f"sent {self.CAL_ON if want else self.CAL_OFF!r} but the "
                f"instrument could not be asked what state it is in: "
                f"{st.get('error', 'no reply')}")
        if got and not want:
            pass
        if got != want and want and not st["terms"]:
            raise IOError(
                "correction cannot be switched on: this instrument has no "
                "calibration collected (bare `cal` returns nothing).  There is "
                "no factory default to fall back on -- run the standards, or "
                "`recall` a slot that has one.")
        if got != want:
            raise IOError(f"asked for correction {'on' if want else 'off'}, "
                          f"instrument reports {'on' if got else 'off'}")
        self._cal_forced = want
        return True

    def recall_cal(self, slot):
        """Load one of the instrument's stored calibrations."""
        self.cmd(self.CAL_RECALL % int(slot))
        return self.cal_status()

    @contextlib.contextmanager
    def uncorrected(self):
        """Sweep with the instrument's correction off, then put it back.

        Restored through the same registry that hands the display back, so an
        abandoned process cannot leave the instrument silently uncalibrated --
        which is a worse thing to leave behind than a frozen screen, because it
        looks completely normal.
        """
        before = self.cal_status()["enabled"]
        self.set_correction(False)
        self._cal_restore = before
        try:
            yield self
        finally:
            self._restore_cal()

    def _restore_cal(self):
        """Put the correction back the way it was found.  Never raises."""
        want = getattr(self, "_cal_restore", None)
        if want is None:
            return
        self._cal_restore = None
        try:
            self.cmd(self.CAL_ON if want else self.CAL_OFF)
        except Exception:                                   # noqa: BLE001
            pass

    def resume(self):
        """Hand the display back to the instrument.

        `scan` sweeps without touching the screen, so the trace on the H4
        freezes for as long as we are driving it.  That is right while we are
        sweeping - it is what makes `scan` fast - but leaving it frozen when we
        stop makes the instrument look hung.  Send `resume` whenever we give up
        control: on stop, on close, and when switching away from the device.
        """
        try:
            self.cmd("resume"); return True
        except Exception:
            return False

    def pause(self):
        try:
            self.cmd("pause"); return True
        except Exception:
            return False

    # ---------------------------------------------------------------- sweeps
    def scan(self, start, stop, points=401):
        """One pass.  Tries the fast `scan` first, falls back to sweep+data.
        Held under the lock so the legacy sweep+data sequence stays atomic."""
        with self._lock:
            return self._scan(start, stop, points)

    def _scan(self, start, stop, points=401):
        out = self.cmd(f"scan {int(start)} {int(stop)} {int(points)} 7")
        f, s11, s21 = [], [], []
        for l in out:
            p = l.split()
            if len(p) < 5:
                continue
            try:                                  # parse all five BEFORE appending,
                v = [float(x) for x in p[:5]]     # or a bad line desyncs the lists
            except ValueError:
                continue
            f.append(v[0]); s11.append(v[1] + 1j*v[2]); s21.append(v[3] + 1j*v[4])
        if len(f) >= 3:
            n = min(len(f), len(s11), len(s21))
            return np.array(f[:n]), np.array(s11[:n]), np.array(s21[:n])
        if getattr(self, "link_err", None):
            raise IOError(
                f"the instrument stopped answering ({self.link_err}). "
                "Unplug the NanoVNA, plug it back in, then press `rescan`.")
        return self._scan_legacy(start, stop, points)

    def _scan_legacy(self, start, stop, points):
        self.cmd(f"sweep {int(start)} {int(stop)} {int(points)}")
        time.sleep(0.05)
        # Same defence as _scan: a usage line, a command echo or a prompt
        # fragment among the rows must be skipped, not crash the sweep.
        f = []
        for x in self.cmd("frequencies"):
            p = x.split()
            if not p:
                continue
            try:
                f.append(float(p[0]))
            except ValueError:
                continue
        f = np.array(f)
        def grab(ch):
            v = []
            for l in self.cmd(f"data {ch}"):
                p = l.split()
                if len(p) < 2:
                    continue
                try:
                    v.append(float(p[0]) + 1j*float(p[1]))
                except ValueError:
                    continue
            return np.array(v)
        s11, s21 = grab(0), grab(1)
        n = min(len(f), len(s11), len(s21))
        return f[:n], s11[:n], s21[:n]

    def scan_hires(self, start, stop, segments=1, points=401, settle=0.0, on_segment=None):
        """Stitch contiguous segments.  NOTE: the instrument interpolates its
        stored calibration across whatever span you ask for, so a cal taken
        over a wide span is being interpolated inside each narrow segment.
        Calibrate over the span you intend to use where accuracy matters."""
        if segments <= 1:
            r = self.scan(start, stop, points)
            if on_segment: on_segment(1, 1, *r)
            return r
        edges = np.linspace(start, stop, segments+1)
        F, A, B = [], [], []
        for i in range(segments):
            f, s11, s21 = self.scan(edges[i], edges[i+1], points)
            if i:                                    # drop duplicated edge point
                f, s11, s21 = f[1:], s11[1:], s21[1:]
            F.append(f); A.append(s11); B.append(s21)
            if on_segment:
                on_segment(i+1, segments, np.concatenate(F),
                           np.concatenate(A), np.concatenate(B))
            if settle: time.sleep(settle)
        return np.concatenate(F), np.concatenate(A), np.concatenate(B)

    def close(self):
        """Hand the trace back and let go of the port.  Idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        _OPEN.discard(self)
        self._restore_cal()           # never leave it silently uncalibrated
        self.resume()                 # never leave the screen frozen
        try: self.ser.close()
        except Exception: pass

    # `with NanoVNA() as dev:` is the form that cannot forget.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Replay:
    """Stand-in that serves a .s2p file, so the whole analysis path can be
    exercised with no hardware attached.  Parses the Touchstone option line
    rather than assuming Hz / real-imaginary."""
    UNIT = dict(HZ=1.0, KHZ=1e3, MHZ=1e6, GHZ=1e9)

    def __init__(self, path):
        unit, fmt = 1e0, "RI"
        f, s11, s21 = [], [], []
        for ln in open(path, errors="replace"):
            ln = ln.strip()
            if not ln or ln[0] == "!":
                continue
            if ln[0] == "#":
                tok = ln[1:].upper().split()
                for t in tok:
                    if t in self.UNIT: unit = self.UNIT[t]
                    elif t in ("RI", "MA", "DB"): fmt = t
                continue
            p = ln.split()
            if len(p) < 5:
                continue
            def cx(a, b):
                a, b = float(a), float(b)
                if fmt == "RI": return a + 1j*b
                if fmt == "MA": return a*np.exp(1j*np.radians(b))
                return 10**(a/20)*np.exp(1j*np.radians(b))      # DB
            f.append(float(p[0])*unit)
            s11.append(cx(p[1], p[2]))
            s21.append(cx(p[3], p[4]))
        self.f, self.s11, self.s21 = map(np.array, (f, s11, s21))
        self.format, self.portname = fmt, path

    def s11_is_physical(self, tol=1.05, frac=0.10):
        """A passive DUT cannot reflect more than it receives.  Transmission-only
        instruments (a tinySA with a tracking generator, for instance) still
        emit an S11 column, and it is meaningless."""
        a = np.abs(self.s11)
        return float(np.mean(a > tol)) <= frac
    def info(self):  return f"replay {self.portname} ({len(self.f)} pts)"
    def scan(self, start, stop, points=401):
        m = (self.f >= start) & (self.f <= stop)
        return self.f[m], self.s11[m], self.s21[m]
    def scan_hires(self, start, stop, segments=1, points=401, settle=0.0,
                   on_segment=None, seg_delay=0.0):
        """Honours `segments` so the progress display behaves as it will live.
        `seg_delay` fakes acquisition time so a replayed sweep can be watched."""
        if segments <= 1:
            r = self.scan(start, stop, points)
            if on_segment: on_segment(1, 1, *r)
            return r
        edges = np.linspace(start, stop, segments+1)
        F, A, B = [], [], []
        for i in range(segments):
            f, s11, s21 = self.scan(edges[i], edges[i+1], points)
            if i and len(f): f, s11, s21 = f[1:], s11[1:], s21[1:]
            F.append(f); A.append(s11); B.append(s21)
            if seg_delay: time.sleep(seg_delay)
            if on_segment:
                on_segment(i+1, segments, np.concatenate(F),
                           np.concatenate(A), np.concatenate(B))
        return np.concatenate(F), np.concatenate(A), np.concatenate(B)
    def close(self): pass
    def resume(self): return True     # a replay file has no screen
    def pause(self): return True

    # A file is whatever it was when it was captured.  Reporting `enabled=None`
    # rather than True or False is the honest answer: the correction state is
    # not knowable from a Touchstone, and pretending otherwise would let an
    # application believe it had switched something.
    def cal_status(self):
        return {"raw": ["replay: correction state is a property of the capture"],
                "enabled": None}

    def set_correction(self, on):
        raise IOError("a replay file cannot change its calibration; it was "
                      "captured however it was captured")

    def recall_cal(self, slot):
        return self.cal_status()

    @contextlib.contextmanager
    def uncorrected(self):
        yield self                    # nothing to switch, nothing to restore


# --------------------------------------------------------------------- as a tool
def _write_s2p(path, f, s11, s21, comments=()):
    """Minimal Touchstone writer, here so this file stays self-contained.

    S12 and S22 are written as zero because the H4 does not measure them -- it
    is a 2-port instrument with a 1-port bridge.  That is a real limitation of
    the hardware, not a shortcut, and it is stated in the file rather than left
    for the reader to discover.
    """
    with open(path, "w") as fh:
        for c in comments:
            fh.write(f"! {c}\n")
        fh.write("! S12 and S22 are NOT measured by a NanoVNA and are zero here\n")
        fh.write("# Hz S RI R 50\n")
        for fr, a, b in zip(f, s11, s21):
            fh.write(f" {fr:.0f} {a.real:.9g} {a.imag:.9g} "
                     f"{b.real:.9g} {b.imag:.9g} 0 0 0 0\n")
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NanoVNA-H4 driver, standalone")
    ap.add_argument("--port", help="serial port; autodetected if omitted")
    ap.add_argument("--sweep", metavar="OUT.s2p", help="sweep and write Touchstone")
    ap.add_argument("--start", type=float, default=140e6)
    ap.add_argument("--stop", type=float, default=152e6)
    ap.add_argument("--points", type=int, default=401)
    ap.add_argument("--segments", type=int, default=1,
                    help="stitch N segments for more points than the hardware "
                         "does in one pass")
    ap.add_argument("--average", type=int, default=1)
    ap.add_argument("--ifbw", type=int, help="DiSlord bandwidth code, 0 narrowest")
    ap.add_argument("--no-cal", action="store_true",
                    help="sweep with the instrument's own correction OFF, and "
                         "put it back afterwards")
    ap.add_argument("--recall", type=int, metavar="SLOT",
                    help="load a stored calibration slot before sweeping")
    ap.add_argument("--probe-cal", action="store_true",
                    help="report what this firmware exposes about calibration "
                         "and change nothing")
    args = ap.parse_args(argv)

    if args.probe_cal:
        # Read-only.  This exists because the calibration command spellings are
        # the one part of this driver not verified against hardware, and the
        # instrument is a better authority than anybody's memory.
        with NanoVNA(args.port) as dev:
            print(f"device {dev._info}\n")
            for c in ("help", "cal", "info"):
                try:
                    out = dev.cmd(c)
                except Exception as e:                      # noqa: BLE001
                    out = [f"<{type(e).__name__}: {e}>"]
                print(f"--- {c} ---")
                print("\n".join(out[:40]) or "  (no reply)")
                print()
            st = dev.cal_status()
            print(f"parsed correction state: {st['enabled']!r}"
                  f"{'  <- unreadable; set NanoVNA.CAL_* for this firmware'
                     if st['enabled'] is None else ''}")
        return 0

    if not args.sweep:
        ports = NanoVNA.list_ports()
        if not ports:
            print("no serial ports found")
            return 1
        for p in ports:
            print(("* " if p["likely"] else "  ") + p["label"])
        print("\n* = looks like a NanoVNA")
        return 0

    with NanoVNA(args.port) as dev:
        print(f"device {dev._info}")
        if args.recall is not None:
            print(f"recall {args.recall}: {dev.recall_cal(args.recall)['raw']}")
        if args.ifbw is not None:
            dev.set_bandwidth(args.ifbw)
        ctx = dev.uncorrected() if args.no_cal else contextlib.nullcontext()
        with ctx:
            if args.no_cal:
                print("correction OFF for this sweep; it is restored on exit")
            acc = None
            for _ in range(max(1, args.average)):
                f, a, b = dev.scan_hires(args.start, args.stop, args.segments,
                                         args.points)
                acc = (f, a, b) if acc is None else (f, acc[1] + a, acc[2] + b)
        n = max(1, args.average)
        f, s11, s21 = acc[0], acc[1] / n, acc[2] / n
        _write_s2p(args.sweep, f, s11, s21, comments=[
            f"NanoVNA sweep via {os.path.basename(__file__)}",
            f"device {dev.portname}",
            f"span {args.start/1e6:.4f}-{args.stop/1e6:.4f} MHz, "
            f"{len(f)} points, {args.segments} segment(s), {n} average(s)",
            f"instrument correction: {'OFF (raw)' if args.no_cal else 'as configured'}"])
        d = 20 * np.log10(np.maximum(np.abs(s21), 1e-12))
        print(f"wrote {args.sweep}: {len(f)} points, "
              f"{f[0]/1e6:.4f}-{f[-1]/1e6:.4f} MHz")
        print(f"  peak |S21| {d.max():+.2f} dB at {f[int(np.argmax(d))]/1e6:.4f} MHz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
