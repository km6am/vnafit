"""What a sweep could determine about a netlist, before any sweep exists.

Which parameters a 2-port measurement can pin down is a property of the CIRCUIT,
not of the data.  So it is computable up front, and the tool should compute it
rather than let someone discover it after an afternoon of fitting -- which is
how this project learned it the first time.

The method: perturb each parameter in LOG space, build the Jacobian of
[S21 in dB, |S11|] on a dense noise-free grid, and take its SVD.  A null
direction is a way of moving the parameters that changes nothing at all, so the
data can never separate them.  In log space those directions read as power laws,
which is why the report can say "only al * turns^2 is determined" rather than
handing back an eigenvector.

Two things decide whether this works, and both were got wrong first:

**Derived parameters must be allowed to recompute.**  Overriding every parameter
-- including one defined as `{al*turns*turns}` -- pins the very dependency being
perturbed, and `al` and `turns` then look individually free when the truth is
that only their product matters.  Only literals are perturbed.

**The step size decides the answer.**  At h=1e-2 an exactly degenerate pair
showed a relative singular value of 1.2e-2, big enough to read as a real if weak
direction.  It was truncation error: it fell linearly with h and went to zero.  A
fixed step with a fixed threshold reports the wrong rank, so a singular value is
called zero only when it ALSO shrinks with h -- and *how fast* it shrinks is the
test, not merely that it does.  Truncation error of a central difference is
O(h^p) with p >= 1, so a spurious direction falls by at least the step ratio.
Asking only for "5x smaller over a 100x step change" passed a direction that had
fallen by 5x, i.e. p = 0.35: a real, weak direction whose coarse-h value was
contaminated.  It was reported as a null while the independent random-restart
rank count said it was not, and the two numbers sat four lines apart disagreeing.

**A null direction is a subspace, not a list of vectors.**  The SVD returns an
arbitrary orthonormal basis of it, and reading those rows one at a time reports
degeneracies nobody can act on: a netlist with an `al*turns^2` gauge and an
internal-node impedance gauge came back as two dense vectors touching four
parameters each, so neither could be named and six of nine parameters were
thrown away.  Rotating the same subspace to the sparsest basis it admits
recovers `al^-2 * turns` and `L2/C2 vs cm` separately.  The subspace was right
all along; only the basis was wrong.
"""
import numpy as np

from .coupled import db
from .units import Expr

VERDICTS = ("determined", "degenerate", "gauge")


def independent(nl):
    """Parameters that are literals, and so may be perturbed independently."""
    return [n for n, v in nl.params.items() if not isinstance(v, Expr)]


def element_params(nl, targets=None):
    """Element name -> the parameters (from `targets`) that drive it.

    Resolved transitively through the `.param` graph, but STOPPING at any name
    in `targets`.  That last part matters once degenerate parameters have been
    collapsed: after `al` and `turns` are replaced by `lsh`, expanding all the
    way to the leaves reports `al, turns` again -- the very names the report
    just said not to think in -- and `lsh` appears to drive nothing.
    `targets` defaults to the independent parameters.
    """
    targets = set(targets) if targets else set(independent(nl))

    def expand(name, seen=None):
        seen = seen or set()
        if name in targets or name in seen:
            return {name} if name in targets else set()
        raw = nl.params.get(name)
        if not isinstance(raw, Expr):
            return set()
        seen = seen | {name}
        out = set()
        for x in raw.names:
            out |= expand(x, seen)
        return out

    out = {}
    for e in nl.elements:
        got = set()
        for v in e.args.values():
            if isinstance(v, Expr):
                for x in v.names:
                    got |= expand(x)
        out[e.name] = got
    return out


def _response(nl, over, f):
    from . import mna
    S = mna.build(nl, over).solve(f)
    return np.concatenate([db(S[:, 1, 0]), np.abs(S[:, 0, 0]) * 20])


def jacobian(nl, names, f, base=None, h=1e-5):
    """d[response] / d ln(theta), central differences in log space."""
    base = base or {n: nl.resolve_params()[n] for n in names}
    cols = []
    for n in names:
        a = dict(base)
        a[n] = base[n] * np.exp(+h)
        b = dict(base)
        b[n] = base[n] * np.exp(-h)
        cols.append((_response(nl, a, f) - _response(nl, b, f)) / (2 * h))
    return np.array(cols).T


def sparse_null_basis(N, tol=0.12, max_support=5, max_combos=50000):
    """Rotate a null subspace into the sparsest basis it admits.

    `N` is (m, p): m orthonormal rows spanning the directions the data cannot
    see.  Any combination of them is equally null, so the SVD's own rows carry
    no meaning -- and a dense row cannot be named, which is the whole point of
    the report.  So: search supports smallest-first, and for each candidate
    support S ask how much of a null vector confined to S leaks outside it.
    `leak` is the smallest singular value of N restricted to the complement of
    S, i.e. exactly the residual of the best confined combination.

    Falls back to the input rows for whatever it cannot sparsify, so this can
    only improve the report, never lose a direction.
    """
    import itertools
    N = np.atleast_2d(np.asarray(N, float))
    m, p = N.shape
    if m <= 1:
        return N
    found, combos = [], 0
    for size in range(1, min(max_support, p) + 1):
        if len(found) == m or combos > max_combos:
            break
        cands = []
        for S in itertools.combinations(range(p), size):
            combos += 1
            if combos > max_combos:
                break
            comp = [j for j in range(p) if j not in S]
            if not comp:
                continue
            M = N[:, comp]
            w, V = np.linalg.eigh(M @ M.T)
            leak = float(np.sqrt(max(w[0], 0.0)))
            if leak < tol:
                cands.append((leak, S, V[:, 0]))
        for leak, S, c in sorted(cands, key=lambda t: (t[0], t[1])):
            if len(found) == m:
                break
            v = c @ N
            keep = np.zeros(p)
            keep[list(S)] = v[list(S)]           # drop the leak, keep the shape
            n = np.linalg.norm(keep)
            if n < 1e-12:
                continue
            keep /= n
            # Independence by residual, not by matrix_rank: two vectors on the
            # SAME support that differ only by numerical noise still have a
            # second singular value of ~2e-3, which clears any absolute rank
            # tolerance.  That accepted the al/turns direction twice and
            # reported it twice, with a different member pinned each time.
            r = keep - sum(float(keep @ v) * v for v in found)
            if np.linalg.norm(r) < 0.2:
                continue                          # already spanned
            found.append(keep)
    for row in N:                                 # complete the basis
        if len(found) == m:
            break
        r = row - sum(float(row @ v) * v for v in found)
        if np.linalg.norm(r) > 0.2:
            found.append(r / np.linalg.norm(r))
    return np.array(found) if found else N


class Structure:
    """The result of the analysis, and the verdicts drawn from it."""

    skipped = 0                # restart trials discarded as out of window
    rank_dispute = ""          # set when the two rank counts disagree

    def __init__(self, nl, f, names, J, s, Vt, null, ranks, p0):
        self.nl, self.f, self.names = nl, f, names
        self.J, self.s, self.Vt = J, s, Vt
        self.null, self.ranks, self.p0 = null, ranks, p0
        self.nullvecs = (sparse_null_basis(Vt[null]) if len(null)
                         else np.zeros((0, len(names))))
        self.epar = element_params(nl, names)
        self.verdict = self._verdicts()

    rank = property(lambda self: len(self.names) - len(self.null))

    def _verdicts(self):
        out = {n: ("determined", "") for n in self.names}
        letters = iter("ABCDEFGH")
        for v in self.nullvecs:
            # Any appreciable component counts.  A 0.5 cut called one end of a
            # degeneracy degenerate and left the other looking merely weak, when
            # (2, -1) normalises to (0.894, 0.447) and both ends are the same
            # fact.
            idx = [i for i, c in enumerate(v) if abs(c) > 0.15]
            if len(idx) == 1:
                out[self.names[idx[0]]] = ("gauge", "")
            else:
                g = next(letters)
                for i in idx:
                    out[self.names[i]] = ("degenerate", g)
        return out

    def drives(self, param):
        return sorted(e for e, ps in self.epar.items() if param in ps)

    def peak_sensitivity(self, param):
        i = self.names.index(param)
        return float(np.abs(self.J[:len(self.f), i]).max())

    def sensitivity(self, param):
        i = self.names.index(param)
        return np.abs(self.J[:len(self.f), i])

    SOFT = 1e-3              # relative singular value below which a direction
                             # is soft enough to need gauge-fixing, not fitting

    def soft_directions(self, thresh=None):
        """Directions the data can barely see, short of not seeing them at all.

        A null direction is caught by the rank test.  This catches the ones just
        above it, and they do the real damage: on the TinyFilter the impedance
        scale -- `l` and `qr` up, every capacitor down -- sits at a relative
        singular value of 1.8e-4, and the fitter walks a very long way along it.
        Measured: pin `l` anywhere from 12 to 80 nH, refit the rest, and the
        residual moves by 0.04 dB.  Left free it parked at 3.6 nH with 300 pF
        capacitors and an excellent-looking fit.

        The Cramer-Rao bound does NOT catch this and is not being used wrongly:
        it says `l` is known to 2.6%, which is the right answer to "how well is
        it known if the model is exact and the noise is 0.05 dB".  The model is
        wrong by 0.33 dB here, and inside that the scale is free.
        """
        thresh = self.SOFT if thresh is None else thresh
        rel = self.s / self.s[0] if self.s[0] > 0 else self.s
        return [self.Vt[i] for i in range(len(self.names))
                if i not in self.null and rel[i] < thresh]

    def fittable(self):
        """The parameters worth turning loose."""
        return [n for n in self.names if self.verdict[n][0] == "determined"]

    MEASURABLE = 0.30            # fractional CRB above which it is not worth it

    def measurable(self, sigma_db=0.05, limit=None):
        """Determined AND resolvable at this trace noise.

        Structural determinacy is necessary, not sufficient, and the gap between
        them is not academic: an LTspice netlist of the symmetric TinyFilter
        comes out rank 12 of 12 -- every parameter structurally determined --
        with best-case errors of 967% on `l1` and `l3`.  They are separable in
        principle and unmeasurable in practice, because with only S11 and S21
        the two ends of a symmetric filter are nearly indistinguishable.

        Calling both of those "identifiable" in the same column is what made the
        report misleading, so the two are now separate questions.
        """
        limit = self.MEASURABLE if limit is None else limit
        prec = self.precision(sigma_db)
        return [n for n in self.names
                if self.verdict[n][0] == "determined" and prec[n] < limit]

    def groups(self):
        """Degeneracy group letter -> the parameters in it."""
        out = {}
        for n, (st, g) in self.verdict.items():
            if st == "degenerate":
                out.setdefault(g, []).append(n)
        return {g: sorted(v) for g, v in out.items()}

    def precision(self, sigma_db=0.05, sigma_s11=0.01):
        """Cramer-Rao bound on the fractional error of each parameter.

        A BOUND, and it is labelled as one wherever it is shown.  It assumes the
        model is exactly right, so it says what the sweep could do at best -- not
        what you will get.  Whether the model is right is a question for the
        residual plot; compressing that into a scalar correction here was tried
        and abandoned, because model error biases the answer as well as widening
        it and no inflation factor fixes a bias.
        """
        J = self.J.copy()
        nf = len(self.f)
        J[:nf] /= sigma_db
        J[nf:] /= (sigma_s11 * 20)
        _U, s, Vt = np.linalg.svd(J, full_matrices=False)
        safe = np.where(s > s[0] * 1e-9, s, np.inf)
        cov = (Vt.T * (1.0 / safe) ** 2) @ Vt
        return {n: float(np.sqrt(max(cov[i, i], 0.0)))
                for i, n in enumerate(self.names)}

    def report(self, sigma_db=0.05):
        prec = self.precision(sigma_db)
        extra = (f", {self.skipped} discarded as out of window"
                 if self.skipped else "")
        out = [f"{self.rank} of {len(self.names)} parameters are determinable "
               f"(rank {sorted(set(self.ranks))} at "
               f"{len(self.ranks)} random parameter points{extra})",
               f"{'parameter':<10}{'value':>12}  {'drives':<20}"
               f"{'peak dB/e-fold':>15}{'best case':>12}  verdict"]
        order = sorted(self.names, key=lambda n: -self.peak_sensitivity(n))
        for n in order:
            st, g = self.verdict[n]
            if st == "gauge":
                got, v = "     -", "no effect at all"
            elif st == "degenerate":
                got, v = "     -", f"degenerate, group {g}"
            else:
                e = prec[n]
                got = f"{100*e:11.4f}%" if e < 10 else "       >10x"
                v = ("identifiable" if e < self.MEASURABLE else
                     "determined, but NOT measurable at this noise/span")
            out.append(f"{n:<10}{self.p0[n]:12.4g}  "
                       f"{','.join(self.drives(n)) or '-':<20}"
                       f"{self.peak_sensitivity(n):15.2f}{got:>12}  {v}")
        out.append("")
        for v in self.nullvecs:
            who, why = describe_null(self.names, v)
            out.append(f"  {who}: {why}")
        if getattr(self, "rank_dispute", ""):
            out.append(f"  NOTE: {self.rank_dispute}")
        out.append("")
        out.append("  'best case' is the Cramer-Rao bound at the stated trace "
                   "noise.  It assumes the model is")
        out.append("  exactly right and says nothing about how far a wrong model "
                   "has pushed the answer off")
        out.append("  centre -- that is a question for the residual plot.")
        return "\n".join(out)


def invariant_for(nl, members, v, names, eps=0.05, tol=1e-6):
    """The derived .param that stays put along a null direction, if there is one.

    When `al` and `turns` are inseparable, listing them both as parameters
    contradicts the finding.  What is actually determined is their invariant,
    and the netlist usually already has a name for it -- `lsh = al*turns^2`.
    Found by moving along the null direction and seeing which derived parameter
    does not budge, rather than by pattern-matching the expression, because the
    numerical check is the claim being made.

    `tol` was 1e-9, which is exact-arithmetic thinking: the null direction is
    numerical, so a real invariant drifts a little along it.  On the al/turns
    case `lsh` drifts 7e-8 along its own null direction and 3.3e-2 along the
    other one -- nearly six decades apart -- so 1e-6 separates them with room
    to spare, and 1e-9 rejected the true invariant along with the false one.
    """
    idx = {n: i for i, n in enumerate(names)}
    move = {m: nl.resolve_params()[m] * np.exp(eps * v[idx[m]]) for m in members}
    before, after = nl.resolve_params(), nl.resolve_params(move)
    best = None
    for n, raw in nl.params.items():
        if not isinstance(raw, Expr) or n in members:
            continue
        if set(raw.names) - set(members):
            continue                       # must depend only on this group
        if abs(after[n] / before[n] - 1) < tol:
            best = n
    return best


def suggest_coordinates(st):
    """The coordinate set worth fitting.

    Three ways to deal with a direction the data cannot see, in order of how
    much they keep:

    1. one parameter -- a pure gauge; drop it, nothing is lost;
    2. a group the netlist already has a name for (`lsh = al*turns^2`) --
       substitute the invariant, and every determined direction survives;
    3. a group with no name -- **pin one member and free the rest**.  Dropping
       the whole group was the first behaviour and it was badly wrong: a rank-7
       netlist came back proposing that one parameter be fitted, because two
       null directions between them touched six names.  A degeneracy of size m
       costs exactly one coordinate, not m.  Which member gets pinned is
       arbitrary by construction -- that is what a gauge means -- so the note
       says so, and says pinning it is a choice rather than a measurement.
    """
    out_drop, subs, note = set(), [], []
    renamed = {}                     # member -> the invariant it became
    for v in st.nullvecs:
        idx = [i for i, c in enumerate(v) if abs(c) > 0.15]
        members = [renamed.get(st.names[i], st.names[i]) for i in idx]
        members = list(dict.fromkeys(members))   # a group may share a member
        if len(members) == 1:
            out_drop.add(members[0])
            note.append(f"{members[0]} dropped -- it is a gauge and does nothing")
            continue
        raw = [st.names[i] for i in idx]
        inv = invariant_for(st.nl, raw, v, st.names)
        if inv:
            out_drop.update(raw)
            subs.append(inv)
            renamed.update({m: inv for m in raw})
            note.append(f"{' & '.join(members)} replaced by {inv}, which is the "
                        f"combination that is actually determined")
            continue
        free = sorted((abs(v[i]), st.names[i]) for i in idx
                      if st.names[i] not in out_drop)
        if not free:
            note.append(f"{' & '.join(members)} -- already fixed by an earlier "
                        f"substitution")
            continue
        pick = free[-1][1]
        out_drop.add(pick)
        rest = [m for m in members if m != pick]
        note.append(f"{' & '.join(members)} are not separable and the netlist "
                    f"has no .param for their invariant, so {pick} is pinned at "
                    f"its stated value and {', '.join(rest)} stay free -- any "
                    f"one of them would have done, which is what makes this a "
                    f"gauge choice and not a measurement")
    out = [n for n in st.names if n not in out_drop] + subs
    return out, note


def _in_window(nl, base, f, margin=0.05):
    """Is the response still inside the analysis window at these parameters?

    The random-restart rank check perturbs the parameters and re-ranks, to catch
    a degeneracy that only exists at the nominal point.  Perturbed too hard it
    catches something else entirely: the passband walks off the edge of the
    window, every parameter loses leverage in band, and the rank drops for a
    reason that has nothing to do with the circuit.  Observed exactly that --
    rank [4, 5, 6] on a filter whose true rank is 6, with every low trial's peak
    sitting on a window boundary.
    """
    from . import mna
    S = mna.build(nl, base).solve(f)
    i = int(np.argmax(np.abs(S[:, 1, 0])))
    n = len(f)
    return int(margin * n) < i < int((1 - margin) * n)


def analyse(nl, f=None, trials=5, h=1e-5, h_coarse=1e-3, names=None,
            jitter=0.10):
    if f is None:
        lo, hi = (nl.ac[2], nl.ac[3]) if nl.ac else (1e6, 1e8)
        f = np.logspace(np.log10(lo), np.log10(hi), 400)
    f = np.asarray(f, float)
    names = list(names) if names else independent(nl)
    if not names:
        raise ValueError("this netlist has no .param lines, so there is nothing "
                         "to analyse.  Move the values you want to reason about "
                         "into .param and refer to them from the elements.")
    resolved = nl.resolve_params()
    missing = [n for n in names if n not in resolved]
    if missing:
        raise ValueError(f"no such .param: {', '.join(missing)}")
    p0 = {n: resolved[n] for n in names}
    J = jacobian(nl, names, f, p0, h)
    Jc = jacobian(nl, names, f, p0, h_coarse)
    _U, s, Vt = np.linalg.svd(J, full_matrices=False)
    sc = np.linalg.svd(Jc, compute_uv=False)
    rel = s / s[0] if s[0] > 0 else s
    relc = sc / sc[0] if sc[0] > 0 else sc

    # A direction that is only truncation error falls as O(h^p), p >= 1, so
    # over a 100x change in step it must shrink by at least 100x.  The old test
    # asked for 5x and let through a direction that had fallen by 5.0 -- p =
    # 0.35, a real if weak direction -- while the restart rank count below
    # disagreed with it in the same breath.
    ratio = np.log(np.maximum(relc, 1e-300) / np.maximum(rel, 1e-300))
    order = ratio / np.log(h_coarse / h)
    null = []
    for i in range(len(names)):
        if rel[i] < 1e-6 or (rel[i] < 1e-2 and order[i] >= 1.0):
            null.append(i)

    ranks, skipped = [], 0
    t = 0
    while len(ranks) < trials and t < trials * 6:
        rng = np.random.default_rng(t)
        t += 1
        base = {n: v * np.exp(jitter * rng.normal()) for n, v in p0.items()}
        try:
            if not _in_window(nl, base, f):
                skipped += 1
                continue
            sv = np.linalg.svd(jacobian(nl, names, f, base, h), compute_uv=False)
        except Exception:                                    # noqa: BLE001
            skipped += 1
            continue
        ranks.append(int((sv / sv[0] > 1e-6).sum()) if sv[0] > 0 else 0)

    st = Structure(nl, f, names, J, s, Vt, null, ranks, p0)
    st.skipped = skipped
    # Two independent counts of the same thing: the h-scaling test above, and
    # the rank of the Jacobian at jittered parameter points.  They used to be
    # allowed to disagree silently.  Say so instead -- a disagreement means the
    # netlist has a direction sitting right on the edge of measurable, which is
    # a fact about the filter and not a detail of the numerics.
    st.rank_dispute = ""
    if ranks:
        by_restart = len(names) - int(np.median(ranks))
        if by_restart != len(null):
            st.rank_dispute = (
                f"the step-size test finds {len(null)} null direction(s) but "
                f"the rank at jittered points implies {by_restart}; one "
                f"direction is borderline, so read the weakest verdict as "
                f"'barely measurable' rather than as either extreme")
    return st


def _integerish(v, tol=0.08):
    v = np.asarray(v, float)
    nz = np.abs(v[np.abs(v) > 1e-9])
    if not len(nz):
        return None
    v = v / nz.min()
    for scale in (1, 2, 3, 4):
        w = v * scale
        if np.all(np.abs(w - np.round(w)) < tol):
            return np.round(w).astype(int)
    return None


def describe_null(names, v, tol=0.06):
    """Turn a null direction into a sentence someone can act on.

    A null direction says which way you may move freely.  What a person can use
    is the INVARIANT -- the combination that stays put -- so for a two-parameter
    degeneracy the invariant is reported: "only al * turns^2 is determined" is
    actionable where "the free direction is al^2 / turns" is not.
    """
    idx = [i for i, c in enumerate(v) if abs(c) > tol]
    if len(idx) == 1:
        return names[idx[0]], ("changes nothing measurable -- it is a gauge, "
                               "cancelled elsewhere in the netlist")
    if len(idx) == 2:
        i, j = idx
        w = _integerish([-v[j], v[i]])
        if w is not None and w[0] < 0:
            w = -w                    # same invariant, readable way round
        if w is not None:
            terms = []
            for k, e in zip((i, j), w):
                if e == 0:
                    continue
                t = names[k] + ("" if abs(e) == 1 else f"^{abs(e)}")
                terms.append(("1/" + t) if e < 0 else t)
            return (f"{names[i]} & {names[j]}",
                    f"not separable -- only {' * '.join(terms)} is determined")
    return (", ".join(names[i] for i in idx), "not separable from each other")


def analyse_reduced(nl, f=None, **kw):
    """Analyse, then re-analyse in the coordinates that are actually determined.

    Returns (structure_in_good_coordinates, notes).  Reporting `al` and `turns`
    as two parameters when only `al*turns^2` is determined contradicts the very
    finding being reported, so the card is built on the reduced set and the
    substitution is explained.
    """
    first = analyse(nl, f, **kw)
    coords, notes = suggest_coordinates(first)
    if sorted(coords) == sorted(first.names):
        return first, notes
    if not coords:
        return first, notes + ["nothing is determinable in this netlist"]
    return analyse(nl, f, names=coords, **kw), notes
