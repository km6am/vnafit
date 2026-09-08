"""Vectorized modified nodal analysis -> S-parameters.

The whole frequency sweep is one `np.linalg.solve` on a stacked
`(nfreq, dim, dim)` complex matrix.  Two structural choices make that work, and
both are load-bearing:

**Affine assembly.**  For R/L/C/K/transformer/port networks the system matrix is

    A(w) = G + 1j*w*W

with `G` and `W` frequency-independent.  So the element loop runs once per
parameter vector, not once per frequency, and the sweep is a single broadcast.
Only TLIN breaks the affine form; those elements are stamped per frequency into
a separate, usually empty, list.

**Norton ports.**  A port is a current injection plus a conductance between its
nodes -- not a group-2 voltage source.  Purely nodal (no extra unknown), it
works for a floating differential port, it keeps `A` complex-*symmetric*, and,
the point, `A` is then IDENTICAL for every excitation.  One factorisation
serves all ports; only the right-hand side changes.

Scaling: values are normalised at build time to the reference impedance and the
sweep centre (`L^ = w_ref*L/Z0`, `C^ = w_ref*C*Z0`, `R^ = R/Z0`), with the
branch-current unknowns carried as `Z0*i` and the KCL rows multiplied by `Z0`.
Every passband matrix entry is then O(1) and the unit-mismatch part of the
condition number is gone, at zero per-evaluation cost.  What remains is
physics: near a high-Q resonance the matrix is genuinely near-singular, with
`cond ~ Q_L`.
"""
import numpy as np

from .netlist import GND, NetlistError
from .units import resolve

Z0_REF = 50.0


class Built:
    """A netlist with its parameters resolved, ready to sweep.

    Holds the assembled `G`/`W` and the node map.  Rebuilt per parameter
    vector during a fit; swept many times per build during a live overlay.
    """

    __slots__ = ("G", "W", "B", "dim", "nodes", "ports", "tlins", "w_ref",
                 "params", "netlist", "_branch_of")

    def __init__(self, G, W, B, nodes, ports, tlins, w_ref, params, netlist,
                 branch_of):
        self.G, self.W, self.B = G, W, B
        self.dim = G.shape[0]
        self.nodes, self.ports, self.tlins = nodes, ports, tlins
        self.w_ref, self.params, self.netlist = w_ref, params, netlist
        self._branch_of = branch_of

    # ------------------------------------------------------------------ solve
    def matrix(self, f):
        """The stacked system matrix A(f), shape (nfreq, dim, dim)."""
        f = np.atleast_1d(np.asarray(f, float))
        if np.any(f <= 0):
            raise ValueError("frequencies must be > 0; this is an AC solver and "
                             "an inductor branch is singular at DC")
        wn = 2 * np.pi * f / self.w_ref                 # normalised omega
        A = self.G[None] + 1j * wn[:, None, None] * self.W[None]
        for stamp in self.tlins:
            stamp(A, f, self.w_ref)
        return A

    def solve(self, f):
        """S-parameters over `f`.  Returns (nfreq, nports, nports) complex,
        indexed `S[:, j, k]` = wave out of port j for a wave into port k."""
        f = np.atleast_1d(np.asarray(f, float))
        A = self.matrix(f)
        B = np.broadcast_to(self.B[None], (len(f),) + self.B.shape)
        V = np.linalg.solve(A, B)                       # (nf, dim, nports)

        z0 = np.array([p.z0 for p in self.ports], float)
        np_, nn = zip(*[(self.nodes[p.pos], self.nodes[p.neg]) for p in self.ports])
        Vp = self._pick(V, np_) - self._pick(V, nn)     # (nf, nports, nports)

        # S_jk = V_j * sqrt(Z0k/Z0j); the bare V_j is only right for equal Z0.
        S = Vp * np.sqrt(np.outer(1.0 / z0, z0))[None]
        S[:, np.arange(len(z0)), np.arange(len(z0))] -= 1.0
        return S

    def _pick(self, V, idx):
        """Rows of V for node indices, with -1 meaning ground (zero volts)."""
        out = np.zeros((V.shape[0], len(idx), V.shape[2]), complex)
        for j, i in enumerate(idx):
            if i >= 0:
                out[:, j, :] = V[:, i, :]
        return out

    def s2p(self, f):
        """Convenience for the 2-port case: (s11, s21, s12, s22)."""
        S = self.solve(f)
        if S.shape[1] < 2:
            raise ValueError("s2p() needs a 2-port netlist")
        return S[:, 0, 0], S[:, 1, 0], S[:, 0, 1], S[:, 1, 1]

    def cond(self, f):
        """Condition number of A over the sweep -- a diagnostic, not a guard."""
        return np.linalg.cond(self.matrix(f))


# --------------------------------------------------------------------- build
def build(nl, overrides=None, w_ref=None, z0_ref=Z0_REF):
    """Resolve parameters and assemble the MNA matrices.

    `w_ref` defaults to the geometric centre of the netlist's `.ac` sweep, or
    to 2*pi*1e8 if it declares none.  It affects conditioning only, never the
    answer.
    """
    params = nl.resolve_params(overrides)
    if w_ref is None:
        if nl.ac:
            w_ref = 2 * np.pi * float(np.sqrt(nl.ac[2] * nl.ac[3]))
        else:
            w_ref = 2 * np.pi * 1e8

    nodes = _map_nodes(nl)
    n_node = max(nodes.values()) + 1 if nodes else 0

    # Group-2 unknowns: every inductor and every ideal transformer.  An L is
    # ALWAYS group 2 -- the 1/(jwL) admittance stamp blows up toward DC, while
    # the branch form degrades gracefully to a short, and K needs the branch
    # current to couple to anyway.
    branch_of, n_br = {}, 0
    for e in nl.elements:
        if e.kind in ("L", "XFMR"):
            branch_of[e.name.lower()] = n_node + n_br
            n_br += 1
    dim = n_node + n_br
    if dim == 0:
        raise NetlistError("netlist has no unknowns")

    G = np.zeros((dim, dim), complex)
    W = np.zeros((dim, dim), complex)
    B = np.zeros((dim, len(nl.ports)), complex)
    tlins = []

    by_name = {e.name.lower(): e for e in nl.elements}
    for e in nl.elements:
        if e.kind == "R":
            _stamp_r(G, nodes, e, params, z0_ref)
        elif e.kind == "C":
            _stamp_c(W, nodes, e, params, w_ref, z0_ref)
        elif e.kind == "L":
            _stamp_l(G, W, nodes, branch_of, e, params, w_ref, z0_ref)
        elif e.kind == "XFMR":
            _stamp_xfmr(G, nodes, branch_of, e, params)
        elif e.kind == "TLIN":
            tlins.append(_tlin_stamper(nodes, e, params, z0_ref))
        elif e.kind != "K":
            raise NetlistError(f"no stamp for element kind {e.kind!r}")

    _stamp_k(W, branch_of, by_name, nl, params, w_ref, z0_ref)

    for j, p in enumerate(nl.ports):
        _stamp_port(G, B, nodes, p, j, z0_ref)

    _check_galvanic(nl, nodes, branch_of)
    return Built(G, W, B, nodes, nl.ports, tlins, w_ref, params, nl, branch_of)


def _map_nodes(nl):
    """Node name -> row index; ground maps to -1 and gets no row."""
    seen = []
    for e in nl.elements:
        seen.extend(e.nodes)
    for p in nl.ports:
        seen.extend((p.pos, p.neg))
    out, k = {}, 0
    for n in seen:
        if n in out:
            continue
        if n in GND:
            out[n] = -1
        else:
            out[n] = k
            k += 1
    for g in GND:
        out.setdefault(g, -1)
    return out


def _add(M, i, j, v):
    """Stamp with the ground row/column dropped rather than pinned."""
    if i >= 0 and j >= 0:
        M[i, j] += v


def _pair(M, p, n, y):
    _add(M, p, p, y)
    _add(M, n, n, y)
    _add(M, p, n, -y)
    _add(M, n, p, -y)


def _stamp_r(G, nodes, e, params, z0):
    r = resolve(e.args["value"], params) / z0
    if r == 0:
        raise NetlistError(f"{e.name}: zero resistance (use a wire, i.e. the same "
                           f"node name, rather than R=0)", e.lineno)
    _pair(G, nodes[e.nodes[0]], nodes[e.nodes[1]], 1.0 / r)


def _stamp_c(W, nodes, e, params, w_ref, z0):
    c = resolve(e.args["value"], params) * w_ref * z0
    # Loss: a capacitor's "Q" is 1/tand.  Both land as a complex capacitance
    # C* = C(1 - j/Q), which is exactly the form that keeps A affine in w.
    q = _loss_q(e, params)
    if q is not None:
        c = c * (1 - 1j / q)
    _pair(W, nodes[e.nodes[0]], nodes[e.nodes[1]], c)


def _stamp_l(G, W, nodes, branch_of, e, params, w_ref, z0):
    ln = resolve(e.args["value"], params) * w_ref / z0
    if ln <= 0:
        raise NetlistError(f"{e.name}: inductance must be positive", e.lineno)
    q = _loss_q(e, params)
    if q is not None:
        # Constant-Q loss: -(R + jwL) = -jw*L(1 - j/Q).  Chosen because it is
        # the unique loss law that preserves the affine form -- NOT because it
        # is the physics.  Skin effect gives R ~ sqrt(f); over a filter's span
        # loss only moves the peak ~0.1-0.2 dB and is invisible in the skirts,
        # so the two are indistinguishable here.  Note it is non-causal
        # (violates Kramers-Kronig): fine for magnitude work, invalid for any
        # future time-domain or group-delay-integral feature.
        ln = ln * (1 - 1j / q)
    p, n, x = nodes[e.nodes[0]], nodes[e.nodes[1]], branch_of[e.name.lower()]
    _add(G, p, x, 1.0)
    _add(G, n, x, -1.0)
    _add(G, x, p, 1.0)
    _add(G, x, n, -1.0)
    W[x, x] += -ln


def _loss_q(e, params):
    if "q" in e.args:
        q = resolve(e.args["q"], params)
    elif "tand" in e.args:
        td = resolve(e.args["tand"], params)
        q = 1.0 / td if td else None
    else:
        return None
    if q is not None and q <= 0:
        raise NetlistError(f"{e.name}: Q must be positive", e.lineno)
    return q


def _stamp_k(W, branch_of, by_name, nl, params, w_ref, z0):
    """Mutual inductance, stamped as the full inductance matrix.

    Handles any number of mutually coupled inductors, and rejects the
    physically impossible ones.  Pairwise |k| < 1 does NOT guarantee a passive
    network: k12 = k13 = +0.9 with k23 = -0.9 gives det(Lmat) = -2.89, an
    energy-generating circuit that otherwise returns |S21| > 1 with no error at
    all.  Positive-definiteness of Lmat is the real condition.
    """
    ks = [e for e in nl.elements if e.kind == "K"]
    if not ks:
        return
    # Group coupled inductors into connected sets; check each set on its own.
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in ks:
        union(e.args["a"].lower(), e.args["b"].lower())

    groups = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)

    kmap = {}
    for e in ks:
        a, b = e.args["a"].lower(), e.args["b"].lower()
        kv = resolve(e.args["k"], params)
        if abs(kv) >= 1.0:
            raise NetlistError(f"{e.name}: |k| must be below 1, got {kv}", e.lineno)
        kmap[frozenset((a, b))] = kv

    for members in groups.values():
        members.sort()
        Ls = [resolve(by_name[m].args["value"], params) * w_ref / z0 for m in members]
        m = len(members)
        Lmat = np.zeros((m, m))
        for i in range(m):
            Lmat[i, i] = Ls[i]
        for i in range(m):
            for j in range(i + 1, m):
                kv = kmap.get(frozenset((members[i], members[j])), 0.0)
                Lmat[i, j] = Lmat[j, i] = kv * np.sqrt(Ls[i] * Ls[j])
        ev = np.linalg.eigvalsh(Lmat)
        if ev.min() <= 0:
            raise NetlistError(
                "coupling matrix for (" + ", ".join(members) + ") is not positive "
                f"definite (min eigenvalue {ev.min():.4g}); those k values describe "
                "an energy-generating network, not a filter")
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue        # the diagonal is already stamped by _stamp_l
                xi, xj = branch_of[members[i]], branch_of[members[j]]
                W[xi, xj] += -Lmat[i, j]


def _stamp_xfmr(G, nodes, branch_of, e, params):
    """Ideal transformer: V(p1,n1) = n * V(p2,n2), I2 = -n * I1.

    Frequency-independent, DC-safe, and symmetric.  One branch current.
    """
    n = resolve(e.args["n"], params)
    if n == 0:
        raise NetlistError(f"{e.name}: turns ratio cannot be zero", e.lineno)
    p1, n1, p2, n2 = [nodes[x] for x in e.nodes]
    x = branch_of[e.name.lower()]
    for node, sign in ((p1, 1.0), (n1, -1.0), (p2, -n), (n2, n)):
        _add(G, node, x, sign)
        _add(G, x, node, sign)


def _tlin_stamper(nodes, e, params, z0):
    """Exact Pi equivalent of a uniform line, stamped per frequency.

    NOT the Z- or Y-parameter form: both diverge as gamma*l -> 0, where the
    true network limit is a plain short.  The Pi form degrades gracefully.
    """
    zc = resolve(e.args["z0"], params) / z0
    if "td" in e.args:
        td = resolve(e.args["td"], params)
    else:
        vf = resolve(e.args.get("vf", 1.0), params)
        td = resolve(e.args["len"], params) / (vf * 299792458.0)
    q = _loss_q(e, params)
    p1, n1, p2, n2 = [nodes[x] for x in e.nodes]

    def stamp(A, f, w_ref):
        beta_l = 2 * np.pi * f * td
        gl = 1j * beta_l
        if q is not None:
            gl = gl * (1 + 1.0 / (2 * q))       # mild uniform loss
        ys = 1.0 / (zc * np.sinh(gl))
        yp = np.tanh(gl / 2.0) / zc
        for i, j, v in ((p1, p2, -ys), (p2, p1, -ys)):
            if i >= 0 and j >= 0:
                A[:, i, j] += v
        for i, v in ((p1, ys + yp), (p2, ys + yp)):
            if i >= 0:
                A[:, i, i] += v
        # The return conductors are treated as the reference for this element;
        # a fully floating line needs the 4-port form, which is not implemented.
        for i in (n1, n2):
            if i >= 0:
                raise NetlistError(f"{e.name}: transmission line return nodes must "
                                   f"be ground in this version")
    return stamp


def _stamp_port(G, B, nodes, p, j, z0):
    """Norton port: conductance 1/Z0 between the nodes, current 2/Z0 injected.

    In normalised units (KCL rows times Z0, unknown currents times Z0) that is
    a conductance of Z0/Z0port and an injection of 2*Z0/Z0port.
    """
    g = z0 / p.z0
    a, b = nodes[p.pos], nodes[p.neg]
    _pair(G, a, b, g)
    if a >= 0:
        B[a, j] += 2.0 * g
    if b >= 0:
        B[b, j] -= 2.0 * g


def _check_galvanic(nl, nodes, branch_of):
    """Reject a subnetwork joined to the reference by NOTHING at all.

    Such a network is a genuine nullspace -- a transformer secondary with no
    other connection, or two resonators coupled only by K.  The usual fix, a
    GMIN conductance to ground, is refused here on purpose: a 1e-12 entry in a
    matrix whose other entries are O(1) destroys exactly the conditioning the
    scaling just bought.  A named error beats a LinAlgError at some arbitrary
    frequency.

    CAPACITORS COUNT.  They did not at first, on the reasoning that a capacitor
    is an open circuit at DC -- but this is an AC solver that refuses f <= 0, and
    at every frequency it does evaluate, a capacitor is a finite non-zero
    admittance like any other.  Excluding them rejected the crystal ladder, one
    of the most common RF filter topologies there is: series L-C mesh arms with
    capacitive shunt coupling have no DC path to ground anywhere in the middle
    of the filter.  Checked before changing it -- the 4-pole example solves to
    |S^H S - I| = 2.4e-13, i.e. exactly unitary, with cond(A) ~ 1e9 which is
    the loaded Q of a 1 mH / 0.25 pF resonator and not a numerical problem.
    Only K stays excluded: a magnetically coupled island really is singular.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    REF = "\0ref"
    find(REF)
    # Anything with a finite non-zero admittance at f > 0: R, L, C, a line's
    # series path, transformer windings, and the port conductance.  NOT K.
    for e in nl.elements:
        if e.kind in ("R", "L", "C", "TLIN"):
            union(e.nodes[0], e.nodes[1])
        elif e.kind == "XFMR":
            union(e.nodes[0], e.nodes[1])
            union(e.nodes[2], e.nodes[3])
    for p in nl.ports:
        union(p.pos, p.neg)
    for g in GND:
        if g in nodes:
            union(g, REF)

    orphan = sorted({n for n in nodes if nodes[n] >= 0 and find(n) != find(REF)})
    if orphan:
        raise NetlistError(
            "nothing connects node(s) " + ", ".join(orphan) +
            " to the rest of the circuit at any frequency.  A magnetic coupling "
            "is not a connection: K needs a return path as well.")


def sweep(nl, f, overrides=None, **kw):
    """Parse-free convenience: build then solve."""
    return build(nl, overrides, **kw).solve(f)
