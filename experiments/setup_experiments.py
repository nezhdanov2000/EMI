"""
One-command environment for the experiments of the paper.

    python experiments/setup_experiments.py            # creates .venv-exp
    python experiments/setup_experiments.py --no-venv  # into the current interpreter

Installs requirements-experiments.txt, then `rulelist` and `pysubgroup`
without their stale pins (rulelist is also patched by
experiments/patch_rulelist.py), then runs the import checks. Every version
is pinned; nothing is upgraded silently.

`pysubgroup` 0.9.0 declares numpy<2.0.0, which contradicts the pinned numpy,
so with it in the requirements file `pip install -r` refused the whole file
on a clean machine. It is installed with --no-deps instead (its one missing
dependency, statsmodels, is pinned in the requirements file); its only use,
experiments/subgroup_demo/cv_pysubgroup.py, reproduces the committed rows of
experiments/results/pysubgroup_titanic.csv under the pinned numpy.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements-experiments.txt"
RULELIST = "rulelist==0.2.0"
PYSUBGROUP = "pysubgroup==0.9.0"


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-venv", action="store_true", help="install into the running interpreter")
    ap.add_argument("--venv", type=Path, default=ROOT / ".venv-exp")
    args = ap.parse_args()

    if args.no_venv:
        py = sys.executable
    else:
        if not args.venv.exists():
            venv.create(args.venv, with_pip=True)
        bin_dir = "Scripts" if sys.platform == "win32" else "bin"
        py = str(args.venv / bin_dir / ("python.exe" if sys.platform == "win32" else "python"))

    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", str(REQ)])
    run([py, "-m", "pip", "install", "--no-deps", RULELIST])
    run([py, "-m", "pip", "install", "--no-deps", PYSUBGROUP])
    run([py, str(ROOT / "experiments" / "patch_rulelist.py")])
    run([py, "-m", "pip", "install", "-e", str(ROOT)])
    run([py, "-c", "import sklearn, scipy, matplotlib, pysubgroup, pydl85, rulelist, vsf; print('experiments environment ok')"])
    print("\nRun the suite with:", py, "-m pytest -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
