"""
The price of the family certificate, computed rather than measured.

Everything here follows from the threshold of the product,

    k*(n, tau, level) = min { k : k >= ceil(n tau) + 1,
                              P[Bin(n, tau) >= k] <= level (1 - 1e-9) }

(`vsf.centers.min_successes_to_certify_heterogeneous`; PMD 4.14,
Proposition 3), re-implemented with the standard library only so that it
runs where numpy is absent. `--check` compares it against the n_min values
that `n_min_table.py` computed with scipy and stops on any mismatch.

Outputs (all under `paper/`):

* `tables/certificate_price.tex` - the smallest certifiable cell n_min for
  the family certificate at the family sizes T of `n_min_table.csv`, next
  to the split certificate of Webb (2007) as implemented in
  `vsf.selective` (half A proposes, half B tests at alpha / T_B): a split
  certificate needs n_min(alpha / T_B) rows IN HALF B, hence about twice
  that in the table, for T_B candidates. Columns: pure cell and a cell
  whose observed purity is five points above tau.
* `fig_certificate_power.dat` - P[certified] as a function of the cell size
  n for a cell whose rows have success probability p, at tau = 0.7 and 0.9,
  p = tau + 0.05 and p = 1, T = 3 102 (titanic) and T = 303 620 (chess).
  Read by `fig_certificate_power.tex` (pgfplots).
* `tables/certificate_n80.tex` - n_80, the smallest n at which
  P[certified] >= 0.8, for the same curves.

Under the theorem the certificate is valid for heterogeneous rows; the
power curves take homogeneous rows (every p_i = p), which is the case the
threshold is tight for.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FAMILY_TAIL_MARGIN = 1e-9
ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[1]


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_sf(n: int, k: int, p: float) -> float:
    """P[Bin(n, p) >= k], summed in log space from the upper end."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [_log_binom_pmf(n, j, p) for j in range(k, n + 1)]
    top = max(logs)
    if top == -math.inf:
        return 0.0
    return math.exp(top) * sum(math.exp(v - top) for v in logs)


def _guard(n: int, tau: float) -> int:
    """ceil(n tau) + 1 in exact integer arithmetic from tau's binary expansion."""
    num, den = float(tau).as_integer_ratio()
    return -((-n * num) // den) + 1


def k_star(n: int, tau: float, level: float) -> int:
    """
    The product's threshold; n + 1 means the cell can never be certified.
    One pass from k = n downwards, accumulating log P[Bin(n, tau) >= k]
    with the pmf ratio pmf(k-1) / pmf(k) = k (1 - tau) / ((n - k + 1) tau),
    so the cost is O(n) per call and nothing underflows.
    """
    if n <= 0:
        return 1
    log_level = math.log(level * (1.0 - FAMILY_TAIL_MARGIN))
    guard = _guard(n, tau)
    if guard > n:
        return n + 1
    log_ratio = math.log1p(-tau) - math.log(tau)
    log_pmf = n * math.log(tau)              # k = n
    log_sf = log_pmf
    if log_sf > log_level:
        return n + 1
    k = n
    while k > guard:
        # move to k - 1
        log_pmf += math.log(k) - math.log(n - k + 1) + log_ratio
        new_sf = _logaddexp(log_sf, log_pmf)
        if new_sf > log_level:
            return k
        log_sf = new_sf
        k -= 1
    return guard


def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _k_observed(n: int, purity: float) -> int:
    """ceil(purity * n) in exact arithmetic (n for a pure cell), as `n_min_table.py` counts it."""
    if purity >= 1.0:
        return n
    num, den = float(purity).as_integer_ratio()
    return -((-n * num) // den)


def n_min(tau: float, level: float, purity: float, n_max: int = 20000, window: int = 300) -> Optional[int]:
    """
    Smallest n from which on every cell of n rows with observed count
    ceil(purity * n) reaches k*(n, tau, level) - the definition of
    `n_min_table.py`. For a pure cell the first passing n is the answer
    (tau^n is monotone); otherwise the rounding of ceil(purity * n) can
    make a few n just above the first passing one fail again, so the scan
    continues for `window` more rows (at most n_max) and returns the last
    failure plus one; the margin ceil(purity n) - k*(n) grows like sqrt(n),
    so failures far above the first pass do not occur. `--check` verifies this against the
    scipy scan to 20 000 rows of `n_min_table.py`. None when nothing
    passes up to n_max.
    """
    first: Optional[int] = None
    last_bad = 0
    n = 1
    while n <= n_max:
        if _k_observed(n, purity) >= k_star(n, tau, level):
            if first is None:
                first = n
                if purity >= 1.0:
                    return n
        else:
            last_bad = n
        if first is not None and n >= min(n_max, first + window):
            break
        n += 1
    return None if first is None else last_bad + 1


def p_certified(n: int, tau: float, level: float, p: float) -> float:
    """P[Bin(n, p) >= k*(n, tau, level)] for homogeneous rows of success probability p."""
    return binom_sf(n, k_star(n, tau, level), p)


def n_power(tau: float, level: float, p: float, power: float = 0.8, n_max: int = 20000) -> Optional[int]:
    for n in range(1, n_max + 1):
        if p_certified(n, tau, level, p) >= power:
            return n
    return None


# ---------------------------------------------------------------------------

def check_against_table(path: Path) -> None:
    """Every (tau, T) row of n_min_table.csv must reproduce n_min_pure and n_min_mid."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad: List[str] = []
    for r in rows:
        level = ALPHA / int(r["T_distinct"])
        tau = float(r["tau"])
        pure = n_min(tau, level, 1.0)
        mid = n_min(tau, level, float(r["purity_mid"]))
        if pure != int(r["n_min_pure"]) or mid != int(r["n_min_mid"]):
            bad.append(f"{r['source']} N={r['N']} M={r['M']} tau={tau}: "
                       f"pure {pure} vs {r['n_min_pure']}, mid {mid} vs {r['n_min_mid']}")
    if bad:
        raise SystemExit("n_min mismatch against n_min_table.csv:\n" + "\n".join(bad))
    print(f"check: {len(rows)} rows of {path.name} reproduced")


def _source_label(r: Dict[str, str]) -> str:
    name = "synthetic" if r["source"].startswith("synthetic") else r["source"].replace("_", r"\_")
    n_str = f"{int(r['N']):,}".replace(",", r"\,")
    return rf"{name} ($N={n_str}$, $M={r['M']}$)"


def price_table(sources: Sequence[Dict[str, str]], split_candidates: Sequence[int]) -> str:
    """
    Smallest certifiable cell, pure / observed purity tau + 0.05, at
    tau = 0.7 and 0.9: the family certificate at the listed T, and the
    split certificate for T_B candidates (rows in the whole table, i.e.
    twice the rows half B needs).
    """
    taus = (0.7, 0.9)
    n_split = len(split_candidates)
    lines = [r"\begin{tabular}{lr" + "rr" * (1 + n_split) + "}", r"\toprule",
             r"& & \multicolumn{2}{c}{family} " + " ".join(
                 rf"& \multicolumn{{2}}{{c}}{{split, $T_B={tb}$}}" for tb in split_candidates) + r"\\",
             r"data & $T$ " + " ".join(rf"& $\tau={t}$" for _ in range(1 + n_split) for t in taus) + r"\\",
             r"\midrule"]
    for r in sources:
        T = int(r["T_distinct"])
        cells = [_source_label(r), f"{T:,}".replace(",", r"\,")]
        for t in taus:
            lvl = ALPHA / T
            cells.append(f"{n_min(t, lvl, 1.0)} / {n_min(t, lvl, t + 0.05)}")
        for tb in split_candidates:
            for t in taus:
                lvl = ALPHA / tb
                cells.append(f"{2 * n_min(t, lvl, 1.0)} / {2 * n_min(t, lvl, t + 0.05)}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def power_curves(T_values: Sequence[Tuple[str, int]], taus: Sequence[float],
                 n_grid: Iterable[int]) -> Tuple[str, str]:
    """(.dat for pgfplots, n_80 table)."""
    ns = list(n_grid)
    cols: List[Tuple[str, float, float, int, str]] = []   # (name, tau, p, T, label)
    for name, T in T_values:
        for tau in taus:
            for p in (tau + 0.05, 1.0):
                cols.append((f"{name}_t{int(round(tau * 100))}_p{int(round(p * 100))}", tau, p, T, name))
    header = "n " + " ".join(c[0] for c in cols)
    rows = [header]
    for n in ns:
        vals = [f"{p_certified(n, tau, ALPHA / T, p):.4f}" for _, tau, p, T, _ in cols]
        rows.append(f"{n} " + " ".join(vals))
    dat = "\n".join(rows) + "\n"

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"$T$ & $\tau$ & $p=\tau+0.05$ & $p=1$ & \multicolumn{2}{l}{$n_{\min}$ (pure / $\tau+0.05$)}\\",
             r"\midrule"]
    for name, T in T_values:
        for tau in taus:
            lvl = ALPHA / T
            n80_mid = n_power(tau, lvl, tau + 0.05)
            n80_pure = n_power(tau, lvl, 1.0)
            lines.append(f"{T:,} ({name}) & {tau} & {n80_mid} & {n80_pure} & "
                         f"{n_min(tau, lvl, 1.0)} & {n_min(tau, lvl, tau + 0.05)}\\\\".replace(",", r"\,"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return dat, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-min-table", type=Path, default=ROOT / "experiments" / "results" / "n_min_table.csv")
    ap.add_argument("--paper", type=Path, default=ROOT / "paper")
    ap.add_argument("--check", action="store_true", help="reproduce n_min_table.csv first")
    args = ap.parse_args()

    if args.check:
        check_against_table(args.n_min_table)

    with args.n_min_table.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # one row per (source, N, M): the T does not depend on tau
    seen: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in rows:
        seen.setdefault((r["source"], r["N"], r["M"]), r)
    # the real tables and the synthetic family at N = 5000, M = 5..30: T grows, N does not
    keep = [r for r in seen.values()
            if not r["source"].startswith("synthetic") or (r["N"] == "5000" and r["M"] in ("5", "10", "20", "30"))]
    sources = sorted(keep, key=lambda r: int(r["T_distinct"]))

    tables = args.paper / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    price = price_table(sources, split_candidates=(10, 50))
    (tables / "certificate_price.tex").write_text(price, encoding="utf-8")
    print(price)

    T_values = [("titanic", 3102), ("chess", 303620)]
    n_grid = sorted(set(list(range(10, 200, 5)) + list(range(200, 1000, 20)) + list(range(1000, 3001, 100))))
    dat, n80 = power_curves(T_values, (0.7, 0.9), n_grid)
    (args.paper / "fig_certificate_power.dat").write_text(dat, encoding="utf-8")
    (tables / "certificate_n80.tex").write_text(n80, encoding="utf-8")
    print(n80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
