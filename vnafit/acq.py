"""Background sweep acquisition.

Lifted, with light edits, from the author's earlier helical-filter tuning
tool.  The details that
look fussy are all bugs that happened on the bench:

  * the IF bandwidth is applied HERE, not from the GUI thread -- `cmd()` takes
    the port lock, and calling it from the UI blocks the whole window for a
    sweep;
  * a minimum interval, because a replay source returns instantly and would
    otherwise spin flat out and starve the GIL;
  * averaging tolerates a short sweep mid-average (a dropped line) by
    truncating both to the common length instead of raising;
  * errors are held rather than thrown, so a knocked cable pauses the display
    instead of killing the thread.
"""
import threading
import time

import numpy as np


class Acq(threading.Thread):
    MIN_INTERVAL = 0.15

    def __init__(self, dev, cfg):
        super().__init__(daemon=True)
        self.dev, self.cfg = dev, cfg
        self.lock = threading.Lock()
        self.data = self.err = None
        self.partial = None
        self.pcount = 0
        self.prog = (0, 1, 0, 1)
        self.stop_flag = False
        self.count = 0
        self.ifbw_applied = None

    def run(self):
        while not self.stop_flag:
            t0 = time.perf_counter()
            c = dict(self.cfg)
            try:
                want = c.get("ifbw")
                if want != self.ifbw_applied:
                    if want is not None and hasattr(self.dev, "set_bandwidth"):
                        self.dev.set_bandwidth(want)
                    self.ifbw_applied = want

                acc = None
                navg = max(1, int(c.get("average", 1)))
                for a_i in range(navg):
                    def seg(i, nseg, f, s11, s21, _a=a_i, _n=navg):
                        with self.lock:
                            self.partial = (f, s11, s21)
                            self.prog = (i, nseg, _a + 1, _n)
                            self.pcount += 1

                    f, s11, s21 = self.dev.scan_hires(
                        c["start"], c["stop"], c.get("segments", 1),
                        c.get("points", 401), on_segment=seg)
                    if acc is not None and len(f) != len(acc[0]):
                        k = min(len(f), len(acc[0]))
                        acc = (acc[0][:k], acc[1][:k], acc[2][:k])
                        f, s11, s21 = f[:k], s11[:k], s21[:k]
                    acc = ((f, s11, s21) if acc is None
                           else (f, acc[1] + s11, acc[2] + s21))
                with self.lock:
                    self.data = (acc[0], acc[1] / navg, acc[2] / navg)
                    self.count += 1
                    self.err = None
            except Exception as e:                  # noqa: BLE001 - shown in the UI
                with self.lock:
                    self.err = str(e)
                time.sleep(1.0)
            rest = self.MIN_INTERVAL - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def latest(self):
        with self.lock:
            return self.data, self.err, self.count

    def latest_partial(self):
        with self.lock:
            return self.partial, self.pcount, self.prog

    def stop(self):
        self.stop_flag = True


class FileSource:
    """A Touchstone file pretending to be an instrument.

    Duck-types the driver's `scan_hires`, so the entire live path -- thread,
    plotting, fitting, the lot -- can be exercised with no hardware on the
    bench.  `Replay` in vna.py does the same for the older format.
    """

    def __init__(self, ts, jitter_db=0.0, seed=0):
        self.ts = ts
        self.jitter_db = float(jitter_db)
        self._rng = np.random.default_rng(seed)
        self.portname = ts.path or "<file>"
        self._info = f"replay {self.portname}"

    def scan_hires(self, start, stop, segments=1, points=401, on_segment=None,
                   **kw):
        m = (self.ts.f >= start) & (self.ts.f <= stop)
        if not m.any():
            raise IOError(f"the file covers {self.ts.f[0]/1e6:.3f}-"
                          f"{self.ts.f[-1]/1e6:.3f} MHz, not "
                          f"{start/1e6:.3f}-{stop/1e6:.3f} MHz")
        f, s11, s21 = self.ts.f[m], self.ts.s11[m], self.ts.s21[m]
        if self.jitter_db:
            g = 10 ** (self._rng.normal(scale=self.jitter_db, size=len(f)) / 20)
            s21 = s21 * g
        if on_segment:
            on_segment(1, 1, f, s11, s21)
        return f, s11, s21

    def scan(self, start, stop, points=401):
        return self.scan_hires(start, stop, 1, points)

    def close(self):
        pass
