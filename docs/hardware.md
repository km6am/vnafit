# The instrument

Driving a NanoVNA-H4, and the rules this project keeps about doing so.

## The NanoVNA driver is standalone

`vnafit/vna.py` imports nothing from the rest of vnafit and nothing heavier than
numpy — pyserial is imported lazily, inside the methods that need it. Copy the
single file into any project that wants to drive an H4 and it works. Three tests
enforce that (an AST scan for stray imports, a subprocess check that importing it
does not drag in scipy/matplotlib/tkinter, and a run of the copied file on its
own), so it cannot quietly stop being true.

It is also a working command by itself:

```bash
python vna.py                                        # list ports, mark likely ones
python vna.py --sweep out.s2p --start 130e6 --stop 165e6 --points 401 --average 2
```

```
* /dev/cu.usbmodem4001  (NanoVNA-H4)
  /dev/cu.Bluetooth-Incoming-Port
  /dev/cu.debug-console
* = looks like a NanoVNA
```

Port autodetection only returns a port whose USB identity or description
actually looks like a NanoVNA. Falling back to "whatever serial port exists" is
worse than useless: with the instrument unplugged it picked
`/dev/cu.Bluetooth-Incoming-Port`, spent the probe timeout on it, and reported
that Bluetooth did not answer as a NanoVNA — which sends you hunting for a driver
problem instead of a USB cable. It now says:

```
no NanoVNA found.  Serial ports on this machine are: /dev/cu.Bluetooth-Incoming-Port,
/dev/cu.debug-console -- none of them identifies as a NanoVNA.  Check the USB cable
and that the instrument is powered on; pass an explicit port to override.
```

## The instrument is always handed back

Driving the H4 with `scan` freezes its display — that is what makes `scan` fast —
so anything that stops driving it must send `resume`, or the instrument sits
there looking hung.

A `close()` in a `finally` block is not enough, and this project found out the
hard way: **SIGTERM terminates CPython without running `finally`**, and a `pkill`,
a `kill`, or a supervisor shutting a job down all send exactly that. So:

- every open instrument is in a registry, and `atexit` plus SIGTERM/SIGINT/SIGHUP
  handlers resume them all on the way out (re-raising so the exit status still
  reads *killed by signal*, not a quiet success);
- the handlers are only installed over *default* handlers, and only from the main
  thread — a library has no business stamping on an application's signal handling;
- `close()` is idempotent, and `with NanoVNA() as dev:` is the form that cannot
  forget;
- closing the GUI window releases the instrument, and so does switching from the
  hardware to a replay file — which previously just overwrote the reference and
  orphaned a frozen instrument.

Verified against the hardware: a process that deliberately paused the H4 and was
then sent SIGTERM exited 143 and left the port free and the trace running.

## Environment

Self-contained, and it does not trust whatever `python` happens to be on PATH:

```bash
./bootstrap          # create the env (conda if present, else a venv)
./bootstrap --check  # report what is usable, create nothing
./run                # launch with an interpreter that actually works
```

`environment.yml` is the primary path because **tkinter is not pip-installable**
and conda ships it on all three platforms. `pyproject.toml` covers pip/venv, but
only on an interpreter that already has `_tkinter` (a python.org build, or
`brew install python-tk` — a bare Homebrew `python3` does not).

The launcher gates on **Python ≥ 3.10 as well as the imports**. That is not
belt-and-braces: a login shell that activates a conda environment leaves
`python` as whatever that env holds,
and Anaconda's Python 2.7 environment imports numpy, scipy, matplotlib, pyserial
*and* tkinter quite happily — so an imports-only check selects Python 2.7.
