# The NanoVNA driver, on its own

`vnafit/vna.py` is a single file that drives a NanoVNA-H4 over USB and depends on
nothing but the standard library, numpy, and pyserial — and pyserial only inside
the methods that need it. Copy the file into any project and it works; nothing
else in vnafit has to come with it.

Three tests hold that line: an AST scan for stray imports, a subprocess check
that importing it does not pull in scipy, matplotlib or tkinter, and a run of the
copied file on its own.

```python
from vna import NanoVNA

with NanoVNA() as dev:                       # autodetects the port
    f, s11, s21 = dev.scan(130e6, 165e6, 401)
```

`f` is Hz as `float64`; `s11` and `s21` are `complex128` linear reflection and
transmission. All three are the same length, which is **not necessarily the
`points` you asked for** — the instrument decides, and a malformed line is
dropped rather than desynchronising the arrays. Index by value, not by position.

## Reference

### `NanoVNA(port=None, timeout=6.0, probe_timeout=1.5)`

Opens the instrument. `port=None` autodetects. Raises `IOError` if the port does
not answer as a NanoVNA, naming what it found. Use it as a context manager, or
call `close()` yourself — see *Handing the instrument back* below, which is not
optional.

| | |
|---|---|
| `scan(start, stop, points=401)` | one pass. Returns `(f, s11, s21)`. Tries the firmware's fast `scan` and falls back to `sweep`+`data` on older firmware. Raises `IOError` if the link has died. |
| `scan_hires(start, stop, segments=1, points=401, settle=0.0, on_segment=None)` | the same, stitched from contiguous segments for more points than the instrument holds at once. `on_segment(i, n, f, s11, s21)` is called after each with everything so far, for a progress display. |
| `info()` | the cached identity string. Never touches the port, so it is safe to call while another thread is sweeping. |
| `set_bandwidth(code)` | DiSlord IF bandwidth, `0`–`7`. Lower is wider and faster; higher is narrower and quieter. |
| `resume()` / `pause()` | hand the display back / take it. `scan` pauses implicitly. |
| `close()` | resume and release the port. Idempotent. |
| `NanoVNA.find(any_port=False)` | the most likely port, or `None`. |
| `NanoVNA.list_ports()` | every serial port, likely NanoVNAs first. |
| `cmd(s)` | send a raw firmware command, return its reply lines. Escape hatch; everything above is built on it. |

### `Replay(path)`

Duck-types `NanoVNA` from a Touchstone file, so an application can be developed
and tested with no instrument attached. `scan` and `scan_hires` interpolate onto
the requested span; `resume`, `pause` and `close` do nothing. It also carries
`s11_is_physical()`, which flags a capture whose reflection exceeds unity — a
passive DUT cannot reflect more than it receives, so that means a bad
calibration, not a strange filter.

## Calibration

The driver can switch the instrument's own error correction, and put it back.

```python
with NanoVNA() as dev:
    with dev.uncorrected():                   # correction off
        f, s11, s21 = dev.scan(130e6, 165e6, 401)
    # correction is back on here, and on close, and on SIGTERM
```

| | |
|---|---|
| `cal_status()` | `{"raw": [...], "enabled": True/False/None}`. `None` means the reply could not be read — the honest answer, and deliberately not `False`. |
| `set_correction(on)` | switch it, **verified by reading the state back**. Raises if the firmware did not understand, rather than returning a cheerful `True`. |
| `recall_cal(slot)` | load one of the instrument's stored calibrations. |
| `uncorrected()` | context manager: correction off for the block, restored after. |

**The command spellings are the one part of this file not verified against
hardware.** They are constants — `CAL_STATUS`, `CAL_ON`, `CAL_OFF`,
`CAL_RECALL` — so a firmware that spells them differently needs one line
changed, and every method reads the state back so a wrong spelling surfaces as
an error instead of a silent no-op. Run `python vna.py --probe-cal` to see what
your firmware actually exposes; it changes nothing.

**Restored the same way the display is.** An abandoned process cannot leave the
instrument silently uncalibrated, which is a worse thing to walk away from than
a frozen screen because it looks completely normal.

## Two things that will bite you

**The H4 measures S11 and S21 only.** There is no S12 or S22: it is a
1.5-port instrument. Every writer in this project stores those as zero and says
so in the Touchstone comments. If you need the full 2×2, sweep the DUT from each
end and combine.

**`scan_hires` interpolates the calibration.** The instrument interpolates its
stored cal across whatever span you ask for, so a cal taken over a wide span is
being interpolated inside each narrow segment. Calibrate over the span you
intend to use wherever accuracy matters.

## Handing the instrument back

Driving the H4 with `scan` freezes its display — that is what makes `scan` fast
— so anything that stops driving it must `resume`, or the instrument sits there
looking hung.

A `close()` in a `finally` block is not enough. **SIGTERM terminates CPython
without running `finally`**, and `pkill`, `kill` and a supervisor shutting a job
down all send exactly that. So every open instrument is in a registry, and
`atexit` plus SIGTERM/SIGINT/SIGHUP handlers resume them all on the way out,
re-raising so the exit status still reads *killed by signal* rather than a quiet
success.

The handlers are installed only over *default* handlers and only from the main
thread: a library has no business stamping on an application's signal handling.

If you copy the file into something with its own process management, that is the
part to read before trusting it.

## As a command

```bash
python vna.py                                        # list ports, mark likely ones
python vna.py --sweep out.s2p --start 130e6 --stop 165e6 --points 401 --average 2
python vna.py --probe-cal                            # what does this firmware expose?
python vna.py --sweep raw.s2p --start 130e6 --stop 165e6 --no-cal   # correction off
```

Port autodetection returns only a port whose USB identity looks like a NanoVNA.
Falling back to "whatever serial port exists" is worse than useless: with the
instrument unplugged it picked `/dev/cu.Bluetooth-Incoming-Port`, spent the probe
timeout on it, and reported that Bluetooth did not answer as a NanoVNA — which
sends you hunting for a driver problem instead of a USB cable.
