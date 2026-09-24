"""
Make the `rulelist` package (SSD++, Proenca et al. 2022; PyPI 0.2.0) import
and run under numpy >= 2 and pandas >= 2.

The package pins scipy ~= 1.5 and numpy 1.17, which do not build on Python
3.12, so it is installed without its pins:

    pip install --no-deps rulelist
    pip install gmpy2 numba
    python experiments/patch_rulelist.py

and then patched in place: `DataFrame.iteritems` -> `items`, `np.NINF` ->
`-np.inf` (and the other names numpy 2 removed), `ndarray.unique` ->
`pd.unique`. Nothing in the search or the MDL score is touched; the patch is
idempotent and prints what it changed. `experiments/baselines.select_ssdpp`
uses the package through its public `RuleList` and the fitted `_rulelist`.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPLACEMENTS = [
    (r"\.iteritems\(\)", ".items()"),
    (r"np\.NINF", "-np.inf"),
    (r"np\.PINF", "np.inf"),
    (r"np\.Inf\b", "np.inf"),
    (r"np\.NaN\b", "np.nan"),
    (r"np\.Infinity", "np.inf"),
    (r"np\.float_\b", "np.float64"),
    (r"np\.bool8\b", "np.bool_"),
    (r"np\.product\(", "np.prod("),
    (r"self\.categories = self\.values\.unique\(\)", "self.categories = pd.unique(self.values)"),
]


def main() -> int:
    spec = importlib.util.find_spec("rulelist")
    if spec is None or spec.origin is None:
        print("rulelist is not installed: pip install --no-deps rulelist gmpy2 numba", file=sys.stderr)
        return 1
    root = Path(spec.origin).parent
    changed = 0
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = text
        for pat, rep in REPLACEMENTS:
            new = re.sub(pat, rep, new)
        if "pd.unique(self.values)" in new and not re.search(r"^\s*import pandas as pd", new, re.M):
            new = "import pandas as pd\n" + new
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
            print(f"patched {path.relative_to(root)}")
    print(f"{changed} file(s) changed")
    from rulelist import RuleList  # noqa: F401  (import check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
