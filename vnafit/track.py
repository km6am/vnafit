"""Live tuning readout: sweep, fit, print, repeat.

Built for the normal case, which is an INCOMPLETE model.  Nobody types a netlist
that matches their hardware to a tenth of a dB, and the tool is not useful only
once they do.  So:

  * the headline numbers are model-free -- peak, f0, 3 dB bandwidth, insertion
    loss, return loss.  Those come straight off the trace and are right even
    when every model is wrong.
  * a fit is layered on top for k and Qe, and reported with its residual so you
    can see how much to trust it.
  * a netlist, if supplied, is drawn against the measurement and scored.  It is
    never required.
  * a failed fit prints a dash and the loop keeps going.  Something is always
    on screen while you have both hands on a trimmer.

Every sweep is appended to a CSV, so the tuning session is reviewable
afterwards -- which is where you see whether a knob moved what you thought.
"""
import json
import os
import time

import numpy as np

from . import coupled as M
from . import mna, touchstone


def _bw(f, d, level):
    inside = np.flatnonzero(d >= d.max() - level)
    if len(inside) < 2 or inside[0] == 0 or inside[-1] == len(f) - 1:
        return None
    return float(f[inside[-1]] - f[inside[0]])


def measure_row(f, s11, s21, poles, qu, resonator="lumped", seed=None):
    """Model-free numbers first, then a fit if it will converge."""
    f = np.asarray(f, float)
    d = M.db(s21)
    i = int(np.argmax(d))
    row = dict(t=time.time(), peak_db=float(d[i]), f_peak=float(f[i]),
               bw3=_bw(f, d, 3.0), bw20=_bw(f, d, 20.0),
               rl_db=float(-20 * np.log10(max(np.abs(s11).min(), 1e-9))),
               s11_at_peak=float(np.abs(s11[i])))
    try:
        sd = seed or dict(f0s=[f[i]] * poles, ks=[0.02] * (poles - 1),
                          Qe1=60.0, Qen=60.0)
        r = M.fit(f, s11, s21, poles, qu, seed=sd, resonator=resonator)
        row.update(fit_rms=float(r["rms"]), k=float(np.mean(r["ks"])),
                   qe1=float(r["Qe1"]), qen=float(r["Qen"]),
                   f0=float(np.mean(r["f0s"])),
                   detune=float(max(r["f0s"]) - min(r["f0s"])))
        row["_fit"] = r
    except Exception as e:                                   # noqa: BLE001
        row["fit_error"] = str(e)
    return row


def netlist_score(built, f, s21, floor_dbc=-70.0):
    """How well a supplied netlist matches, split where it matters.

    Passband and skirts separately: every model agrees in the passband, so a
    single number hides the half that is actually telling you something.
    """
    S = built.solve(np.asarray(f, float))
    dm, dn = M.db(s21), M.db(S[:, 1, 0])
    f = np.asarray(f, float)
    pk = f[int(np.argmax(dm))]
    live = dm > floor_dbc + 6.0
    near = live & (np.abs(f - pk) < 0.03 * pk)
    far = live & ~near
    g = lambda m: float(np.sqrt(np.mean((dn[m] - dm[m]) ** 2))) if m.any() else float("nan")
    return dict(model=S, all=g(live), passband=g(near), skirts=g(far))


HEADER = ("   n   f_peak     IL     3dB BW    RL     k        Qe1/Qen   fit    model p/s")


def format_row(n, row, score=None):
    def q(key, fmt, scale=1.0):
        v = row.get(key)
        return (fmt % (v * scale)) if v is not None and np.isfinite(v) else "    -"
    line = ("%4d %9.4f %+7.2f %9.4f %6.1f  %s %s/%s %s"
            % (n, row["f_peak"] / 1e6, row["peak_db"],
               (row["bw3"] or np.nan) / 1e6, row["rl_db"],
               q("k", "%.5f"), q("qe1", "%5.0f"), q("qen", "%-5.0f"),
               q("fit_rms", "%5.2f")))
    if score:
        line += "  %5.2f/%-5.2f" % (score["passband"], score["skirts"])
    return line


def run(dev, cfg, seconds, poles, qu, netlist=None, resonator="lumped",
        out_dir=None, every=1.0, on_row=None, verbose=True):
    built = mna.build(netlist) if netlist is not None else None
    rows, t0, n = [], time.time(), 0
    csv = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv = open(os.path.join(out_dir, "track.csv"), "w")
        csv.write("n,t,f_peak,peak_db,bw3,bw20,rl_db,f0,k,qe1,qen,detune,"
                  "fit_rms,model_pass,model_skirt\n")
    if verbose:
        print(HEADER, flush=True)
    while time.time() - t0 < seconds:
        tick = time.time()
        try:
            f, s11, s21 = dev.scan_hires(cfg["start"], cfg["stop"],
                                         cfg.get("segments", 1),
                                         cfg.get("points", 401))
        except Exception as e:                               # noqa: BLE001
            if verbose:
                print("  sweep failed: %s" % e, flush=True)
            time.sleep(1.0)
            continue
        n += 1
        row = measure_row(f, s11, s21, poles, qu, resonator)
        score = netlist_score(built, f, s21) if built is not None else None
        row["sweep"] = (f, s11, s21)
        rows.append(row)
        if verbose:
            print(format_row(n, row, score), flush=True)
        if csv:
            csv.write("%d,%.3f,%.6g,%.4f,%.6g,%.6g,%.3f,%.6g,%.6g,%.4g,%.4g,"
                      "%.6g,%.4g,%.4g,%.4g\n"
                      % (n, row["t"], row["f_peak"], row["peak_db"],
                         row.get("bw3") or np.nan, row.get("bw20") or np.nan,
                         row["rl_db"], row.get("f0", np.nan), row.get("k", np.nan),
                         row.get("qe1", np.nan), row.get("qen", np.nan),
                         row.get("detune", np.nan), row.get("fit_rms", np.nan),
                         (score or {}).get("passband", np.nan),
                         (score or {}).get("skirts", np.nan)))
            csv.flush()
        if out_dir and n % 5 == 1:
            S = np.zeros((len(f), 2, 2), complex)
            S[:, 0, 0], S[:, 1, 0] = s11, s21
            touchstone.save(os.path.join(out_dir, "latest.s2p"), f, S,
                            comments=["vnafit track, sweep %d" % n])
        if on_row:
            on_row(n, row, score)
        rest = every - (time.time() - tick)
        if rest > 0:
            time.sleep(rest)
    if csv:
        csv.close()
    return rows
