# Installing

The dependency that decides everything here is **tkinter**, which is not
pip-installable.

## The launcher does not trust `python`

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
