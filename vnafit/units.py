"""SPICE value/expression parsing.

Two jobs, kept together because they share the suffix table:

  value("100n")        -> 1e-07          a bare SPICE number
  Expr("2*pi*f0*L").eval({...})          an arithmetic expression over .param

The expression evaluator is an AST walker, not `eval`.  A netlist is a file the
user may have been handed by someone else, and `eval` on it is a shell.
"""
import ast
import math
import operator
import re

# SPICE suffixes.  The awkward ones, in the order they must be tested:
# "Meg" is mega and "m" is milli, so the match has to be longest-first and
# case-insensitive -- SPICE is case-insensitive everywhere, which is why
# "M" means MILLI, not mega.  That trips people up; it is not a bug here.
SUFFIX = [
    ("meg", 1e6), ("mil", 25.4e-6),
    ("t", 1e12), ("g", 1e9), ("k", 1e3),
    ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15),
]

_NUM = re.compile(r"""^\s*
    ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)   # mantissa
    \s*([A-Za-z]*)\s*$""", re.X)


def value(tok):
    """'100n' -> 1e-7, '1Meg' -> 1e6, '4.7k' -> 4700.0, '1e-9' -> 1e-9.

    Trailing letters beyond the suffix are ignored the way SPICE ignores them
    ('100nF', '1kOhm', '10MegHz'), so a netlist copied out of a datasheet works.
    """
    m = _NUM.match(str(tok))
    if not m:
        raise ValueError(f"not a number: {tok!r}")
    mant, suf = m.group(1), m.group(2).lower()
    if not suf:
        return float(mant)
    for name, mul in SUFFIX:
        if suf.startswith(name):
            return float(mant) * mul
    # No recognised suffix: it is a unit, not a multiplier ("50ohm", "1Hz").
    return float(mant)


_BINOP = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow,
          ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Deliberately small.  Anything not here is a name error, not a silent import.
FUNCS = {
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan": math.atan,
    "abs": abs, "min": min, "max": max, "pow": math.pow,
}
CONSTS = {"pi": math.pi, "e": math.e}


class Expr:
    """A parsed arithmetic expression over netlist parameters.

    Parsed once at load time and evaluated per fit iteration, so the AST walk
    is not on the hot path in any meaningful way.
    """

    __slots__ = ("src", "_tree", "names")

    def __init__(self, src):
        self.src = str(src).strip()
        try:
            self._tree = ast.parse(self.src, mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"bad expression {self.src!r}: {e}") from None
        self.names = self._collect(self._tree)

    @staticmethod
    def _collect(node):
        out = set()
        for n in ast.walk(node):
            nid = n.id.lower() if isinstance(n, ast.Name) else None
            if nid and nid not in CONSTS and nid not in FUNCS:
                out.add(nid)
        return out

    def eval(self, params):
        return self._ev(self._tree, params)

    def _ev(self, n, p):
        if isinstance(n, ast.Constant):
            if not isinstance(n.value, (int, float)):
                raise ValueError(f"non-numeric constant in {self.src!r}")
            return float(n.value)
        if isinstance(n, ast.Name):
            # SPICE is case-insensitive, so `{L}` must find `.param l=...`.
            # Parameter names are stored lowercased; fold the lookup to match.
            nid = n.id.lower()
            if nid in p:
                return float(p[nid])
            if nid in CONSTS:
                return CONSTS[nid]
            raise ValueError(f"undefined parameter {n.id!r} in {self.src!r}")
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOP:
            try:
                return _BINOP[type(n.op)](self._ev(n.left, p), self._ev(n.right, p))
            except ZeroDivisionError:
                raise ValueError(f"division by zero in {{{self.src}}}") from None
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY:
            return _UNARY[type(n.op)](self._ev(n.operand, p))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            fn = n.func.id.lower()
            if fn not in FUNCS:
                raise ValueError(f"unknown function {n.func.id!r} in {self.src!r}")
            args = [self._ev(a, p) for a in n.args]
            try:
                return FUNCS[fn](*args)
            except ValueError as e:
                # math.sqrt(-1) says "math domain error", which tells whoever is
                # turning a slider precisely nothing.  Name the expression, the
                # argument, and the parameters that could have caused it.
                used = ", ".join(f"{k}={p[k]:g}" for k in sorted(self.names)
                                 if k in p) or "no parameters"
                raise ValueError(
                    f"{fn}({', '.join(f'{a:g}' for a in args)}) is undefined in "
                    f"{{{self.src}}} -- with {used}.  A negative or zero value "
                    f"has reached it; check those parameters.") from None
            except ZeroDivisionError:
                raise ValueError(f"division by zero in {{{self.src}}}") from None
        raise ValueError(f"unsupported syntax in expression {self.src!r}")

    def __repr__(self):
        return f"Expr({self.src!r})"


def numeric(tok):
    """Parse a netlist field into either a float or an Expr.

    Braces are SPICE's expression marker: `{2*L1}`.  A bare token that parses
    as a number is a number; anything else is treated as an expression, so
    `L1*2` works without braces too.
    """
    s = str(tok).strip()
    if s.startswith("{") and s.endswith("}"):
        return Expr(s[1:-1])
    try:
        return value(s)
    except ValueError:
        return Expr(s)


def resolve(v, params):
    """A float stays a float; an Expr is evaluated against `params`."""
    return v.eval(params) if isinstance(v, Expr) else float(v)
