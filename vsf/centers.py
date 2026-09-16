"""
VSF Certified Discrete Centers: the decision-theoretic reporting layer.

Why this module exists
-----------------------------------------------------------------------
The product answers one narrow question: "which cells of the displayed
lattice are almost purely the target value, and how much of the target do
they account for". Every number it reports is a count ratio with an exact
binomial confidence bound, and the search ranks candidates by the same
quantities (`coverage_score`). An association statistic would answer a
different question - "is there a relationship" - and on a rare target the
two have OPPOSITE answers.

Worked example, reproducible from the UCI Adult / Census Income dataset
(N = 32 561, Z = [occupation == "Armed-Forces"], 9 positives, prevalence
0.0276 %). The branch workclass + sex + income is strongly associated with
the target by any association measure, and yet:

  * the highest cell purity over all 32 occupied cells is 2.42 % (8 of 330);
  * the Bayes rule under 0-1 loss never predicts the target class, so
    Goodman-Kruskal lambda is exactly 0;
  * 8 of the 9 positives sit in ONE cell of 330 samples.

No colour threshold can make that display green: the honest report is
zero certified centres, coverage 0, and a best cell purity of 2.42 % -
which is what this module reports, unconditionally.

What a "discrete centre" is
-----------------------------------------------------------------------
Let cell c hold n_c objects of which k_c carry the target value, and let
pi_c = P(Z = 1 | cell c). Two selection rules are offered, chosen by
`CenterSpec.rule`; both feed the identical report, so every downstream
consumer is written once.

`rule="purity"` (DEFAULT). Cell c is a centre iff

    k_c / n_c >= tau     and     n_c >= min_samples,

with tau in (0, 1] and min_samples defaulting to 1. This is the product's own
definition: the user sets tau directly, the green cells on screen and the
`coverage` in the panel are the same object by construction, and tau = 1.0 is
a legitimate request ("cells that are entirely the target value") because it
is a statement about the observed table rather than an inference about pi_c.

The cost is stated rather than hidden: with min_samples = 1, a cell holding a
single target-value object has an observed purity of 1.0, is a centre, and
contributes to coverage. Its Clopper-Pearson interval is [alpha, 1], i.e. the
data are equally consistent with pi_c = 0.05. That interval is computed and
reported for every cell under either rule, so the weakness is visible on the
cell rather than argued about in the abstract; `min_samples` exists to be
exposed to the user, and this module does not pick a value for them.

`rule="certified"`. Cell c is a centre iff the one-sided exact binomial test
of

    H_0: pi_c <= tau     against     H_1: pi_c > tau

is rejected at level alpha / C, where C is the number of occupied cells --
equivalently, iff the Clopper-Pearson lower bound on pi_c at simultaneous
level 1 - alpha reaches tau. `certified_centers` uses the first formulation
(an exact binomial tail in log-factorials) and `clopper_pearson_lower`
computes the second; `tests/test_centers.py` pins that the two agree cell by
cell, because the display shows the bound and the metric counts the
rejections, and they must not be able to disagree.

Bonferroni across the C occupied cells is deliberate: the cells are chosen by
looking at their own purity, so an uncorrected per-cell bound has no
simultaneous coverage and a 4-D grid with 3 000 cells certifies ~150
pure-noise cells at alpha = 0.05. The cell counts are multinomial and hence
negatively associated, so the union bound is conservative, which is the
correct direction for a certificate.

Under this rule a purity of exactly 100 % is NOT certifiable at any finite n,
since H_0: pi <= 1 can never be rejected; `tau = 1` is therefore rejected by
`CenterSpec` rather than clamped. A cell with k = n = 1 has a Clopper-Pearson
lower bound of alpha itself and so is never selected.

Which to use: `"purity"` is what the interface shows and what a user's
threshold means; `"certified"` is what a claim about the underlying
population needs. `crossvalidated_coverage` is the bridge -- it is defined
identically under both rules, and under `"purity"` it is the thing that
exposes a coverage built out of one-object cells, because such cells do not
reproduce on a held-out fold.

The three reported numbers
-----------------------------------------------------------------------
`n_centers` (K)
    How many certified centres the branch produces. Fewer is better at equal
    coverage: the stated goal is a SMALL number of nearly pure cells.
`coverage` (recall)
    Share of all target-value samples that lie inside certified centres.
    This is the headline "efficiency" number. It is 0 - not 41 % - for the
    Armed-Forces branch above.
`purity_pooled` (precision)
    Share of the samples inside certified centres that carry the target
    value, with its own simultaneous bound.

`mass` (what fraction of the population one must inspect) and `lift`
(precision over prevalence) are carried alongside because on a rare target
the pair (coverage, mass) is the operationally meaningful summary and a
single scalar cannot be.

In-sample vs out-of-sample
-----------------------------------------------------------------------
`coverage` as defined above is an IN-SAMPLE quantity computed on cells
selected by their own contents. The Bonferroni-corrected certificate makes
each individual centre's purity claim honest, but it does not make the
resulting coverage an unbiased estimate of the coverage the same rule would
achieve on new data. Two corrections are provided and both are required for
a publishable claim:

  * `crossvalidated_coverage` - centres are certified on a training fold and
    coverage/purity are measured on the held-out fold, over repeated
    stratified K-fold splits, with the Nadeau & Bengio (2003) corrected
    variance for the non-independence of the folds. This is the number that
    answers "how many features do I need" (`select_dimensionality`), because
    it is the only one of the three that can DECREASE when a branch is
    over-resolved.
  * `coverage_null` / `coverage_p_value` - the permutation distribution of
    the coverage statistic itself, drawn exactly from the multivariate
    hypergeometric law of the cell counts given both margins. Because
    the branch reported by `vsf.avr.discover_branches` is an argmax over a
    candidate family, `familywise_max_coverage_null` gives the
    look-elsewhere-corrected version, which is the one a paper must quote.

References
-----------------------------------------------------------------------
Clopper, C. J. & Pearson, E. S. (1934). Biometrika 26, 404-413.
Wilson, E. B. (1927). JASA 22, 209-212.
Nadeau, C. & Bengio, Y. (2003). Machine Learning 52, 239-281.
Vinh, N. X., Epps, J. & Bailey, J. (2010). JMLR 11, 2837-2854.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from statistics import NormalDist
from typing import Callable, Final, Hashable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BoundMethod",
    "CenterRule",
    "CVCoverage",
    "Center",
    "CenterReport",
    "CenterSpec",
    "FamilywiseCoverageNull",
    "FAMILY_TAIL_MARGIN",
    "MIN_POSITIVES_FOR_CV",
    "Multiplicity",
    "PairedGain",
    "binarize_target",
    "center_report",
    "certified_centers",
    "clopper_pearson_lower",
    "min_successes_to_select",
    "select_centers",
    "clopper_pearson_upper",
    "coverage_null",
    "coverage_score",
    "crossvalidated_coverage",
    "familywise_max_coverage_null",
    "min_successes_to_certify",
    "min_successes_to_certify_heterogeneous",
    "paired_gain",
    "purity_bounds",
    "select_dimensionality",
    "stratified_repeated_kfold",
    "summarize_cv",
    "wilson_lower",
    "wilson_upper",
]

_EPS: Final[float] = 1e-12
_TINY: Final[float] = 1e-300

BoundMethod = Literal["clopper-pearson", "wilson"]
CenterRule = Literal["purity", "certified"]
Multiplicity = Literal["bonferroni", "none", "family"]

#: Relative margin by which a family certificate's computed binomial tail
#: must clear the per-cell level. The tail is evaluated in floating point
#: (log-gamma table, reverse cumulative sum); its relative error is below
#: 1e-11 for the cell sizes this package handles, so the margin makes a
#: rounding error push a decision towards "not certified", never the other way.
FAMILY_TAIL_MARGIN: Final[float] = 1e-9

#: Below this many target-value samples, no out-of-sample coverage statement
#: is attempted and `CenterReport.coverage_cv` is None.
#:
#: Derivation, not a convention: with `n_pos` positives and K folds, a single
#: held-out fold contains n_pos / K positives, and the fold-level coverage is
#: a proportion over that many Bernoulli draws, so its standard error is at
#: least 1 / (2 sqrt(n_pos / K)). Pooling over R x K splits divides that by
#: at most sqrt(R K) BEFORE the Nadeau-Bengio inflation factor
#: sqrt(1/(RK) + 1/(K-1)), which for K = 5, R = 5 is 0.53 - i.e. the pooled
#: standard error cannot go below ~0.5 / sqrt(n_pos). Requiring it to be at
#: most 0.10 (a coverage quoted to the nearest 10 points) gives n_pos >= 25.
MIN_POSITIVES_FOR_CV: Final[int] = 25


# --------------------------------------------------------------------------
# Special functions (no scipy: `vsf` depends on numpy and pandas only)
# --------------------------------------------------------------------------
def _log_beta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b), elementwise."""
    lg = np.frompyfunc(lgamma, 1, 1)
    return (lg(a) + lg(b) - lg(a + b)).astype(np.float64)


def _betacf(a: np.ndarray, b: np.ndarray, x: np.ndarray,
            max_iter: int = 400, eps: float = 3e-16, check_every: int = 8) -> np.ndarray:
    """
    Continued-fraction expansion of the incomplete beta function, evaluated by
    the modified Lentz algorithm (Press et al., Numerical Recipes 3rd ed.,
    Sec. 6.4), vectorised over the leading array shape.

    Converges only for x < (a + 1) / (a + b + 2); `_betainc` applies the
    symmetry transform before calling this, and does not rely on the caller
    to have done so.

    Active-set evaluation: every `check_every` iterations the elements whose
    last multiplicative correction `delta` is within `eps` of 1 are written to
    the output and REMOVED from the working arrays, so the loop's cost is
    driven by the sum of per-element iteration counts rather than by the
    slowest element times the whole array. On a real lattice the median cell
    converges in 2-5 iterations while a handful of large cells (n >= 1000,
    x near the mean) need hundreds; without compaction every cell paid for
    those hundreds. Measured on a 41 188-row branch: 5.8x on
    `purity_bounds`, with the returned bounds unchanged. A converged element
    stops being multiplied by further `delta` factors that are all within
    `eps` (i.e. within about one ulp) of 1, so the value it is left with can
    differ from the fully-iterated one by at most that rounding.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape[0] == 1:
        # A single element pays ~30 us of ufunc dispatch per iteration in the
        # array loop below and nothing for its vectorisation; the same IEEE
        # arithmetic in plain Python floats is two orders of magnitude
        # cheaper and produces the same doubles.
        return np.array(
            [_betacf_scalar(float(a[0]), float(b[0]), float(x[0]), max_iter, eps, check_every)],
            dtype=np.float64,
        )
    out = np.empty(x.shape, dtype=np.float64)
    active = np.arange(x.shape[0], dtype=np.int64)
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _TINY, _TINY, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, max_iter + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _TINY, _TINY, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _TINY, _TINY, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if m % check_every == 0:
            converged = np.abs(delta - 1.0) < eps
            if np.any(converged):
                out[active[converged]] = h[converged]
                keep = ~converged
                if not np.any(keep):
                    return out
                active = active[keep]
                a, b, x = a[keep], b[keep], x[keep]
                qab, qap, qam = qab[keep], qap[keep], qam[keep]
                c, d, h = c[keep], d[keep], h[keep]
    out[active] = h
    return out


def _betacf_scalar(a: float, b: float, x: float, max_iter: int, eps: float, check_every: int) -> float:
    """`_betacf` for one element, operation for operation, in Python floats."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if m % check_every == 0 and abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Regularised incomplete beta I_x(a, b), vectorised. Relative accuracy
    ~1e-14 over the range this module uses (a, b positive integers up to N,
    x in (0, 1)); pinned against exact binomial tails in the test suite.
    """
    a_arr, b_arr, x_arr = (
        np.array(v, dtype=np.float64) for v in np.broadcast_arrays(a, b, x)
    )
    out = np.zeros(x_arr.shape, dtype=np.float64)
    out[x_arr >= 1.0] = 1.0
    interior = (x_arr > 0.0) & (x_arr < 1.0)
    if not np.any(interior):
        return out
    ai, bi, xi = a_arr[interior], b_arr[interior], x_arr[interior]
    front = np.exp(ai * np.log(xi) + bi * np.log1p(-xi) - _log_beta(ai, bi))
    swap = xi >= (ai + 1.0) / (ai + bi + 2.0)
    res = np.empty_like(xi)
    keep = ~swap
    if np.any(keep):
        res[keep] = front[keep] * _betacf(ai[keep], bi[keep], xi[keep]) / ai[keep]
    if np.any(swap):
        res[swap] = 1.0 - front[swap] * _betacf(
            bi[swap], ai[swap], 1.0 - xi[swap]
        ) / bi[swap]
    out[interior] = np.clip(res, 0.0, 1.0)
    return out


def _beta_quantile(p: float, a: np.ndarray, b: np.ndarray, n_bisect: int = 40) -> np.ndarray:
    """
    Inverse of `_betainc` in x, by bisection on [0, 1].

    Bisection rather than Newton: I_x is monotone in x, so `n_bisect`
    halvings give a bracket of width 2^-n_bisect, and the iteration cannot
    diverge on the extreme shapes this module generates (a = 1, b = N - 1 for
    a singleton cell in a 30 000-row dataset), where a Newton step from a bad
    start does.

    40 halvings give an absolute error below 1e-12 on a quantity that is a
    probability, displayed to one decimal place of a percent and compared
    against a user-set threshold. A selection decision could only flip if a
    cell's bound sat within 1e-12 of tau, which is not reachable from integer
    counts at any n this package supports. This is a deliberate trade: each
    halving costs a full continued-fraction evaluation over every distinct
    (k, n) pair on the lattice, and 80 of them dominated the cost of building
    a payload.
    """
    lo = np.zeros(a.shape, dtype=np.float64)
    hi = np.ones(a.shape, dtype=np.float64)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        above = _betainc(a, b, mid) > p
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Binomial proportion bounds
# --------------------------------------------------------------------------
def clopper_pearson_lower(k: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """
    One-sided Clopper-Pearson lower bound: the largest p with
    P(Bin(n, p) >= k) <= alpha, i.e. the alpha-quantile of Beta(k, n - k + 1).
    Exactly 0 when k = 0. Guaranteed coverage >= 1 - alpha for every p and n.

    Preferred over Wilson for anything that is CERTIFIED and published:
    Wilson's coverage oscillates around 1 - alpha and dips below it (Brown,
    Cai & DasGupta, Statist. Sci. 16 (2001), 101-133), which is not
    acceptable for a claim of the form "this cell is at least tau pure".
    `wilson_lower` remains available for interactive display, where mean
    coverage is the relevant property and the bound is evaluated per frame.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k_arr, n_arr = (np.array(v, dtype=np.int64) for v in np.broadcast_arrays(k, n))
    if np.any(k_arr < 0) or np.any(n_arr < 0) or np.any(k_arr > n_arr):
        raise ValueError("require 0 <= k <= n elementwise")
    out = np.zeros(k_arr.shape, dtype=np.float64)
    # Closed form for the fully pure cell: I_x(n, 1) = x^n, so the alpha
    # quantile is alpha^(1/n) exactly. Worth special-casing rather than
    # inverting: on a sparse lattice a large share of the occupied cells are
    # all-positive or all-negative, and those are also the shapes on which
    # the continued fraction takes the most iterations (b = 1 forces the
    # symmetry branch with x near 1).
    full = (k_arr == n_arr) & (n_arr > 0)
    if np.any(full):
        out[full] = np.power(alpha, 1.0 / n_arr[full].astype(np.float64))
    mask = (k_arr > 0) & ~full
    if np.any(mask):
        a = k_arr[mask].astype(np.float64)
        b = (n_arr[mask] - k_arr[mask] + 1).astype(np.float64)
        out[mask] = _beta_quantile(alpha, a, b)
    return out


def clopper_pearson_upper(k: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """
    One-sided Clopper-Pearson upper bound: the (1 - alpha)-quantile of
    Beta(k + 1, n - k). Exactly 1 when k = n.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k_arr, n_arr = (np.array(v, dtype=np.int64) for v in np.broadcast_arrays(k, n))
    if np.any(k_arr < 0) or np.any(n_arr < 0) or np.any(k_arr > n_arr):
        raise ValueError("require 0 <= k <= n elementwise")
    out = np.ones(k_arr.shape, dtype=np.float64)
    # Mirror of the lower bound's closed form: for k = 0, the (1 - alpha)
    # quantile of Beta(1, n) is 1 - alpha^(1/n) exactly.
    empty = (k_arr == 0) & (n_arr > 0)
    if np.any(empty):
        out[empty] = 1.0 - np.power(alpha, 1.0 / n_arr[empty].astype(np.float64))
    mask = (k_arr < n_arr) & ~empty
    if np.any(mask):
        a = (k_arr[mask] + 1).astype(np.float64)
        b = (n_arr[mask] - k_arr[mask]).astype(np.float64)
        out[mask] = _beta_quantile(1.0 - alpha, a, b)
    return out


def _z_score(alpha: float) -> float:
    """Upper alpha-quantile of the standard normal (stdlib, no scipy)."""
    return float(NormalDist().inv_cdf(1.0 - alpha))


def wilson_lower(k: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """One-sided Wilson score lower bound. Closed form; display use only."""
    k_arr, n_arr = (np.array(v, dtype=np.float64) for v in np.broadcast_arrays(k, n))
    out = np.zeros(k_arr.shape, dtype=np.float64)
    mask = n_arr > 0
    if not np.any(mask):
        return out
    z = _z_score(alpha)
    nn = n_arr[mask]
    ph = k_arr[mask] / nn
    denom = 1.0 + z * z / nn
    centre = (ph + z * z / (2.0 * nn)) / denom
    half = z * np.sqrt(ph * (1.0 - ph) / nn + z * z / (4.0 * nn * nn)) / denom
    out[mask] = np.clip(centre - half, 0.0, 1.0)
    return out


def wilson_upper(k: np.ndarray | int, n: np.ndarray | int, alpha: float) -> np.ndarray:
    """One-sided Wilson score upper bound. Closed form; display use only."""
    k_arr, n_arr = (np.array(v, dtype=np.float64) for v in np.broadcast_arrays(k, n))
    out = np.ones(k_arr.shape, dtype=np.float64)
    mask = n_arr > 0
    if not np.any(mask):
        return out
    z = _z_score(alpha)
    nn = n_arr[mask]
    ph = k_arr[mask] / nn
    denom = 1.0 + z * z / nn
    centre = (ph + z * z / (2.0 * nn)) / denom
    half = z * np.sqrt(ph * (1.0 - ph) / nn + z * z / (4.0 * nn * nn)) / denom
    out[mask] = np.clip(centre + half, 0.0, 1.0)
    return out


def purity_bounds(
    k: np.ndarray, n: np.ndarray, alpha: float, method: BoundMethod = "clopper-pearson"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (lower, upper) simultaneous-ready bounds on the per-cell purity.

    The Clopper-Pearson path deduplicates on (k, n) before inverting anything.
    A displayed lattice has thousands of cells and only hundreds of distinct
    (k, n) pairs -- most cells are small and many are identical -- while each
    distinct pair costs an 80-step bisection over a continued fraction. The
    renderer calls this once per view dimensionality per branch, plus once per
    4-D slice, so the saving is measured in seconds per interaction, not in
    microseconds.
    """
    if method == "wilson":
        return wilson_lower(k, n, alpha), wilson_upper(k, n, alpha)
    if method != "clopper-pearson":
        raise ValueError(f"unknown bound method: {method!r}")
    k_arr, n_arr = (np.array(v, dtype=np.int64) for v in np.broadcast_arrays(k, n))
    shape = k_arr.shape
    flat = np.stack([k_arr.ravel(), n_arr.ravel()], axis=1)
    if flat.shape[0] == 0:
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty.copy()
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    lower, upper = _cached_clopper_pearson(uniq[:, 0], uniq[:, 1], alpha)
    inverse = inverse.ravel()
    return lower[inverse].reshape(shape), upper[inverse].reshape(shape)


#: Process-wide memo of Clopper-Pearson (lower, upper) pairs, keyed by alpha
#: and then by (k, n). The bounds are a pure function of three numbers, and
#: one analysis asks for the same (k, n, alpha) many times over: the renderer
#: recomputes the bounds per view dimensionality, per 4-D slice and per
#: branch, and the branches' prefix views share most of their cells.
#: Measured on the 41 188-row bank-marketing branch set: 7 843 cell bounds
#: requested, 1 681 distinct. Bounded by `_CP_CACHE_MAX_ENTRIES`; when
#: exceeded the whole cache is dropped rather than evicted piecemeal.
_CP_CACHE: dict[float, dict[tuple[int, int], tuple[float, float]]] = {}
_CP_CACHE_MAX_ENTRIES: Final[int] = 2_000_000
_cp_cache_size = 0


def _cached_clopper_pearson(
    k: np.ndarray, n: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """(lower, upper) Clopper-Pearson bounds for distinct (k, n) rows, memoised."""
    global _cp_cache_size
    k_arr = np.asarray(k, dtype=np.int64).ravel()
    n_arr = np.asarray(n, dtype=np.int64).ravel()
    table = _CP_CACHE.setdefault(float(alpha), {})
    lower = np.empty(k_arr.shape, dtype=np.float64)
    upper = np.empty(k_arr.shape, dtype=np.float64)
    missing: list[int] = []
    for i, (kk, nn) in enumerate(zip(k_arr.tolist(), n_arr.tolist())):
        hit = table.get((kk, nn))
        if hit is None:
            missing.append(i)
        else:
            lower[i], upper[i] = hit
    if missing:
        idx = np.asarray(missing, dtype=np.int64)
        lo = clopper_pearson_lower(k_arr[idx], n_arr[idx], alpha)
        up = clopper_pearson_upper(k_arr[idx], n_arr[idx], alpha)
        lower[idx] = lo
        upper[idx] = up
        if _cp_cache_size + idx.shape[0] > _CP_CACHE_MAX_ENTRIES:
            _CP_CACHE.clear()
            _cp_cache_size = 0
            table = _CP_CACHE.setdefault(float(alpha), {})
        for kk, nn, l, u in zip(
            k_arr[idx].tolist(), n_arr[idx].tolist(), lo.tolist(), up.tolist()
        ):
            table[(kk, nn)] = (l, u)
        _cp_cache_size += idx.shape[0]
    return lower, upper


# --------------------------------------------------------------------------
# Certification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CenterSpec:
    """
    What counts as a discrete centre. One object, passed unchanged from the
    search through the report to the renderer, so the number the panel prints
    and the colour the cell is drawn in can never be produced by different
    rules.

    rule
        `"purity"` (default): a cell is a centre iff its OBSERVED share of the
        target value reaches `tau`, i.e. k_c / n_c >= tau, and n_c >=
        `min_samples`. This is the product's own definition -- the green cells
        on screen and the coverage in the panel are the same object, and the
        user moves `tau` directly.

        `"certified"`: a cell is a centre iff the one-sided exact binomial
        test of H_0: pi_c <= tau is rejected at level alpha / C, equivalently
        iff the Clopper-Pearson lower bound at that level reaches tau. This is
        the conservative variant; it never calls a cell a centre on the
        strength of a handful of samples, at the cost of never reaching
        tau = 1 (see below) and of losing power on a fine grid.

        Both rules produce the same `CenterReport` fields, so `coverage`,
        `K` and the cross-validated estimate are comparable in kind but NOT
        in value between them. The bounds are computed and reported under
        either rule; under `"purity"` they are information attached to a cell,
        not the gate.

    tau
        Purity floor. Under `"purity"`, `tau` may be exactly 1.0 -- "cells
        that are entirely the target value" is a well-posed request about the
        observed table, and it is the natural thing a user asks for. Under
        `"certified"`, `tau = 1` is rejected rather than clamped: H_0: pi <= 1
        is never rejectable, so a request to CERTIFY exact purity has no valid
        answer and must not silently become a request to certify 0.999.

    min_samples
        Minimum cell occupancy for a cell to be eligible at all. Default 1,
        i.e. no restriction: a cell holding a single target-value object is a
        centre under `rule="purity"`, and its coverage contribution is
        counted. That is deliberate and is the caller's decision to make --
        the parameter exists so it can be exposed to the user rather than
        being fixed here. Under `rule="certified"` it is redundant, since the
        bound already excludes small cells.

    alpha
        SIMULTANEOUS error rate over all occupied cells of the branch, when
        `multiplicity = "bonferroni"` (the default). Under `"certified"` it
        gates selection; under `"purity"` it sets the width of the interval
        reported beside each cell.

    method
        `"clopper-pearson"` (exact, guaranteed coverage) for anything
        reported; `"wilson"` only where mean coverage suffices.

    multiplicity
        `"bonferroni"` divides alpha by the number of occupied cells C of the
        partition at hand. That is the correct level for ONE partition fixed
        in advance, and too liberal for a partition a search chose (Section
        4.5 of the specification: on pure noise a reported branch carries a
        false certificate in 68 % of runs).
        `"family"` divides alpha by `family_tests`, the number of cells of
        EVERY partition the search can report or display
        (`vsf.avr.family_cell_count`), so a certificate holds for whichever
        schema the search picks (Section 4.14). Requires
        `rule="certified"` and `method="clopper-pearson"`, and selects with
        `min_successes_to_certify_heterogeneous`, whose threshold is valid
        for a cell of rows with different success probabilities. The search
        entry points of `vsf.avr`, `vsf.redundancy` and `vsf.selective`
        fill `family_tests` themselves (`vsf.avr.resolve_center_spec`); a
        partition-level function given an unresolved family spec raises.
        `"none"` reproduces the naive per-cell bound and exists so the test
        suite can demonstrate the false-certification rate it produces
        (~alpha * C centres on pure noise); it must not be used for a report
        under `rule="certified"`.

    family_tests
        T of `multiplicity="family"`; None until resolved. Set it only from
        `vsf.avr.family_cell_count` on the same data, feature set, maximum
        dimensionality and pruning as the search it certifies: a smaller T
        voids the guarantee.
    """

    tau: float = 0.90
    alpha: float = 0.05
    rule: CenterRule = "purity"
    min_samples: int = 1
    method: BoundMethod = "clopper-pearson"
    multiplicity: Multiplicity = "bonferroni"
    family_tests: Optional[int] = None

    def __post_init__(self) -> None:
        if self.rule not in ("purity", "certified"):
            raise ValueError(f"unknown centre rule: {self.rule!r}")
        if self.rule == "certified":
            if not (0.0 < self.tau < 1.0):
                raise ValueError(
                    f"tau must be in (0, 1) under rule='certified', got {self.tau}: "
                    "a purity of exactly 1.0 is not certifiable at any finite "
                    "sample size. Use rule='purity' to select cells whose "
                    "OBSERVED purity is 1.0."
                )
        elif not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        if self.method not in ("clopper-pearson", "wilson"):
            raise ValueError(f"unknown bound method: {self.method!r}")
        if self.multiplicity not in ("bonferroni", "none", "family"):
            raise ValueError(f"unknown multiplicity policy: {self.multiplicity!r}")
        if self.multiplicity == "family":
            if self.rule != "certified":
                raise ValueError(
                    "multiplicity='family' is a certificate and needs rule='certified'; "
                    "rule='purity' selects cells by their observed share and has no level to correct"
                )
            if self.method != "clopper-pearson":
                raise ValueError(
                    "multiplicity='family' needs method='clopper-pearson': the certificate "
                    "is an exact binomial test"
                )
            if self.family_tests is not None and int(self.family_tests) < 1:
                raise ValueError(f"family_tests must be >= 1, got {self.family_tests}")
        elif self.family_tests is not None:
            raise ValueError("family_tests is only meaningful with multiplicity='family'")

    @property
    def is_resolved(self) -> bool:
        """False only for a family spec whose family size is still unknown."""
        return self.multiplicity != "family" or self.family_tests is not None

    def effective_alpha(self, n_occupied_cells: int) -> float:
        """Per-cell level of the reported bounds, after the multiplicity policy."""
        if self.multiplicity == "none":
            return self.alpha
        if self.multiplicity == "family":
            if self.family_tests is None:
                raise ValueError(
                    "unresolved multiplicity='family': the family size is unknown. "
                    "Pass the spec through a search entry point (vsf.discover_branches, "
                    "compute_landscape, ...) or vsf.avr.resolve_center_spec first."
                )
            return self.alpha / int(self.family_tests)
        return self.alpha / max(1, int(n_occupied_cells))


def min_successes_to_certify(
    n_values: np.ndarray | Sequence[int], tau: float, alpha_eff: float
) -> np.ndarray:
    """
    For each cell size n, the smallest k such that the one-sided exact
    binomial test rejects H_0: pi <= tau at level `alpha_eff`, i.e. the
    smallest k with P(Bin(n, tau) >= k) <= alpha_eff. Returns n + 1 for a
    cell size that can never be certified at this (tau, alpha_eff) - e.g.
    n = 1 at tau = 0.9, alpha_eff = 0.05, where even k = n gives a tail of
    0.9 > 0.05.

    Exists as a separate entry point from `clopper_pearson_lower` because the
    permutation nulls need the threshold as a function of n ONLY (the cell
    sizes are fixed under permutation, the counts are not), which turns each
    replicate's certification into one integer comparison instead of B x C
    beta-quantile inversions. `tests/test_centers.py` pins that it agrees
    exactly with the bound-based route.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not (0.0 < alpha_eff < 1.0):
        raise ValueError(f"alpha_eff must be in (0, 1), got {alpha_eff}")
    n_arr = np.asarray(n_values, dtype=np.int64)
    out = np.zeros(n_arr.shape, dtype=np.int64)
    uniq = np.unique(n_arr[n_arr > 0])
    if uniq.size == 0:
        out[n_arr <= 0] = 1
        return out
    memo = _CERTIFY_MEMO.setdefault((float(tau), float(alpha_eff)), {})
    missing = [n for n in uniq.tolist() if n not in memo]
    if missing:
        lg = _lgamma_table(int(max(missing)) + 1)
        log_tau = float(np.log(tau))
        log_1mtau = float(np.log1p(-tau))
        for n in missing:
            k = np.arange(n + 1, dtype=np.float64)
            k_int = np.arange(n + 1, dtype=np.int64)
            # lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1), read from
            # the shared table: the same `math.lgamma` doubles the previous
            # per-element list comprehensions produced, without 2(n + 1)
            # Python calls per distinct cell size.
            log_choose = lg[n + 1] - lg[k_int + 1] - lg[n - k_int + 1]
            pmf = np.exp(log_choose + k * log_tau + (n - k) * log_1mtau)
            survival = np.cumsum(pmf[::-1])[::-1]
            hit = np.nonzero(survival <= alpha_eff)[0]
            memo[int(n)] = int(hit[0]) if hit.size > 0 else int(n) + 1
        if len(_CERTIFY_MEMO) > 4096:  # bound the number of (tau, alpha) keys
            _CERTIFY_MEMO.clear()
            _CERTIFY_MEMO[(float(tau), float(alpha_eff))] = memo
    lookup = np.fromiter((memo[n] for n in uniq.tolist()), dtype=np.int64, count=uniq.size)
    positive = n_arr > 0
    out[positive] = lookup[np.searchsorted(uniq, n_arr[positive])]
    out[~positive] = 1  # an empty cell holds no successes and never certifies
    return out


def min_successes_to_certify_heterogeneous(
    n_values: np.ndarray | Sequence[int], tau: float, alpha_eff: float
) -> np.ndarray:
    """
    Certification threshold that is valid when the rows of a cell have
    DIFFERENT success probabilities - the situation of every cell once the
    features are conditioned on. Returns, per cell size n, the smallest k
    with

        k >= ceil(n * tau) + 1                                  (i)
        P(Bin(n, tau) >= k) <= alpha_eff * (1 - FAMILY_TAIL_MARGIN)   (ii)

    (n + 1 when no k <= n satisfies both).

    Proposition. Let S be a sum of n independent Bernoulli(p_i) with mean
    pbar = sum(p_i) / n <= tau. Then P(S >= k) <= alpha_eff.
    Proof. By (i), k - 1 >= n tau >= n pbar, so Hoeffding (1956, Theorem 4:
    P(S <= c) >= P(Bin(n, pbar) <= c) for n pbar <= c <= n) with c = k - 1
    gives P(S >= k) <= P(Bin(n, pbar) >= k). The binomial upper tail is
    non-decreasing in the success probability, so this is at most
    P(Bin(n, tau) >= k) <= alpha_eff by (ii). QED

    Why (i) is needed and not decorative: without it the binomial test can
    be anti-conservative for heterogeneous rows. With n = 3, tau = 0.01 the
    binomial threshold at level 0.05 is k = 1 (tail 0.0297), while rows with
    p = (0.03, 0, 0) have mean 0.01 and P(S >= 1) = 0.03 > 0.0297: by AM-GM
    the probability of at least one success is SMALLEST when the p_i are
    equal.

    Simultaneity over tau. If the cell is certified at some tau' >= pbar,
    then k >= ceil(n tau') + 1 >= ceil(n pbar) + 1 and
    P(Bin(n, pbar) >= k) <= P(Bin(n, tau') >= k) <= alpha_eff, so k is at
    least this function's threshold at tau = pbar; the event "certified at
    some tau >= pbar" is therefore contained in one event of probability at
    most alpha_eff, whatever floors the user tries.

    Condition (i) is evaluated in exact rational arithmetic on the binary
    value of `tau`.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not (0.0 < alpha_eff < 1.0):
        raise ValueError(f"alpha_eff must be in (0, 1), got {alpha_eff}")
    n_arr = np.asarray(n_values, dtype=np.int64)
    base = min_successes_to_certify(n_arr, tau, alpha_eff * (1.0 - FAMILY_TAIL_MARGIN))
    num, den = float(tau).as_integer_ratio()
    guard = np.ones(n_arr.shape, dtype=np.int64)
    positive = n_arr > 0
    if np.any(positive):
        uniq = np.unique(n_arr[positive])
        # ceil(n * num / den) + 1 with Python integers: no rounding, no overflow.
        g = np.fromiter(
            (-((-int(n) * num) // den) + 1 for n in uniq.tolist()),
            dtype=np.int64, count=uniq.size,
        )
        guard[positive] = g[np.searchsorted(uniq, n_arr[positive])]
    out = np.maximum(base, guard)
    return np.where(positive, np.minimum(out, n_arr + 1), out)


#: `min_successes_to_certify` thresholds, keyed by (tau, alpha_eff) and then
#: by cell size. The threshold is a pure function of those three numbers,
#: and the search asks for it once per candidate at the same alpha_eff for
#: every candidate with the same occupied-cell count, while cross-validation
#: asks for it once per fold.
_CERTIFY_MEMO: dict[tuple[float, float], dict[int, int]] = {}

_LGAMMA_TABLE: np.ndarray = np.zeros(0, dtype=np.float64)


def _lgamma_table(n_max: int) -> np.ndarray:
    """`math.lgamma(i)` for i = 0 .. n_max + 1 (entry 0 is unused, set to inf)."""
    global _LGAMMA_TABLE
    if _LGAMMA_TABLE.shape[0] < n_max + 2:
        size = max(n_max + 2, 2 * _LGAMMA_TABLE.shape[0], 1024)
        table = np.empty(size, dtype=np.float64)
        table[0] = np.inf
        table[1:] = np.fromiter(
            (lgamma(float(i)) for i in range(1, size)), dtype=np.float64, count=size - 1
        )
        _LGAMMA_TABLE = table
    return _LGAMMA_TABLE


def min_successes_to_select(
    n_values: np.ndarray | Sequence[int], spec: CenterSpec, alpha_eff: float
) -> np.ndarray:
    """
    For each cell size n, the smallest count of target-value objects that
    makes the cell a centre under `spec`. Returns n + 1 for a size that can
    never qualify.

    Under `rule="purity"` this is simply ceil(tau * n), with n + 1 for cells
    below `min_samples`. Under `rule="certified"` it is the exact binomial
    threshold of `min_successes_to_certify`.

    Exists as a separate entry point because the permutation nulls need the
    threshold as a function of n ONLY -- the cell sizes are fixed under
    permutation, the counts are not -- which turns each replicate's selection
    into one integer comparison instead of a per-cell recomputation.
    """
    n_arr = np.asarray(n_values, dtype=np.int64)
    if spec.rule == "certified" and spec.multiplicity == "family":
        out = min_successes_to_certify_heterogeneous(n_arr, spec.tau, alpha_eff)
    elif spec.rule == "certified":
        out = min_successes_to_certify(n_arr, spec.tau, alpha_eff)
    else:
        # ceil(tau * n) with a tolerance, so that tau = 0.9 and n = 10 gives 9
        # rather than 10 through binary-floating-point drift in 0.9 * 10.
        out = np.ceil(spec.tau * n_arr.astype(np.float64) - 1e-9).astype(np.int64)
        out = np.maximum(out, 1)
    out = np.where(n_arr >= spec.min_samples, out, n_arr + 1)
    out = np.where(n_arr > 0, out, 1)
    return out.astype(np.int64)


def select_centers(
    k: np.ndarray, n: np.ndarray, spec: CenterSpec, n_occupied_cells: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Boolean mask of discrete centres under `spec`, plus the per-cell level of
    the reported bounds.

    This is the single definition of "is this cell a centre" used by the
    report, the search objective, the cross-validation, the permutation nulls
    and the renderer. Under `rule="purity"` it is exactly
    `k / n >= tau and n >= min_samples`; under `rule="certified"` it defers to
    `certified_centers`.

    `n_occupied_cells` defaults to the number of entries with n > 0, which is
    the family the Bonferroni correction must cover: the analyst looks at
    every drawn cell and picks the green ones, so the family is every cell
    that could have been drawn green, not the ones that were.
    """
    k_arr = np.asarray(k, dtype=np.int64)
    n_arr = np.asarray(n, dtype=np.int64)
    if k_arr.shape != n_arr.shape:
        raise ValueError(f"shape mismatch: {k_arr.shape} vs {n_arr.shape}")
    if spec.rule == "certified":
        mask, alpha_eff = certified_centers(k_arr, n_arr, spec, n_occupied_cells)
        # `min_samples` is redundant under this rule but not ignored: if a
        # caller sets both, both apply, so the two knobs never silently
        # disagree about which cells are eligible.
        return mask & (n_arr >= spec.min_samples), alpha_eff
    occupied = (
        int(np.count_nonzero(n_arr > 0)) if n_occupied_cells is None else int(n_occupied_cells)
    )
    alpha_eff = spec.effective_alpha(occupied)
    if occupied == 0:
        return np.zeros(k_arr.shape, dtype=bool), alpha_eff
    k_min = min_successes_to_select(n_arr, spec, alpha_eff)
    return (k_arr >= k_min) & (n_arr > 0), alpha_eff


def certified_centers(
    k: np.ndarray, n: np.ndarray, spec: CenterSpec, n_occupied_cells: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Boolean mask of certified centres, plus the per-cell level actually used.

    `n_occupied_cells` defaults to the number of entries with n > 0, which is
    the family the Bonferroni correction must cover: the analyst looks at
    every drawn cell and picks the green ones, so the family is every cell
    that could have been drawn green, not the ones that were.
    """
    k_arr = np.asarray(k, dtype=np.int64)
    n_arr = np.asarray(n, dtype=np.int64)
    if k_arr.shape != n_arr.shape:
        raise ValueError(f"shape mismatch: {k_arr.shape} vs {n_arr.shape}")
    occupied = int(np.count_nonzero(n_arr > 0)) if n_occupied_cells is None else int(n_occupied_cells)
    alpha_eff = spec.effective_alpha(occupied)
    if occupied == 0:
        return np.zeros(k_arr.shape, dtype=bool), alpha_eff
    if spec.method == "clopper-pearson":
        # Exact-tail route. Identical result to `clopper_pearson_lower(...) >=
        # tau` by the binomial/beta duality, and ~50x cheaper: the threshold
        # depends on the cell SIZE only, so it is tabulated once per distinct
        # n instead of inverting a beta quantile per cell. This function sits
        # inside the cross-validation and permutation loops, where the
        # difference is seconds per branch, not milliseconds.
        k_min = np.full(k_arr.shape, np.iinfo(np.int64).max, dtype=np.int64)
        positive = n_arr > 0
        threshold = (
            min_successes_to_certify_heterogeneous if spec.multiplicity == "family"
            else min_successes_to_certify
        )
        k_min[positive] = threshold(n_arr[positive], spec.tau, alpha_eff)
        return (k_arr >= k_min) & positive, alpha_eff
    lower, _ = purity_bounds(k_arr, n_arr, alpha_eff, spec.method)
    return (lower >= spec.tau) & (n_arr > 0), alpha_eff


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Center:
    """One certified cell."""

    cell: int
    n: int
    k: int
    purity: float
    purity_lower: float
    purity_upper: float
    lift: float


@dataclass(frozen=True)
class CVCoverage:
    """
    Out-of-sample coverage over repeated stratified K-fold splits.

    `se` is the Nadeau & Bengio (2003) corrected standard error,

        se^2 = (1 / (R K) + n_test / n_train) * s^2,

    which accounts for the training sets of different folds overlapping. The
    naive s / sqrt(R K) understates the spread by a factor of ~2.3 at K = 5,
    R = 5 and produces significant "improvements" from d to d + 1 that do not
    replicate.

    `per_split` is a tuple, not an ndarray, and deliberately so: this object
    is a field of `vsf.avr.BranchResult`, whose generated `__eq__` compares
    fields with `==`. A numpy array there makes every equality test on a
    BranchResult raise "truth value of an array is ambiguous", which is
    exactly what the determinism tests in `tests/test_avr.py` do.
    """

    per_split: Tuple[float, ...]
    mean: float
    se: float
    purity_mean: float
    n_splits: int
    n_repeats: int
    test_train_ratio: float

    @property
    def n_observations(self) -> int:
        return len(self.per_split)


@dataclass(frozen=True)
class PairedGain:
    """Result of `paired_gain`: a difference of two `CVCoverage` estimates."""

    difference: float
    se: float
    t_statistic: float
    n_observations: int


@dataclass(frozen=True)
class CenterReport:
    """
    Everything the display and the paper are allowed to say about one branch's
    centres. Field order mirrors the reporting order: what was certified,
    how much of the target it captures, how clean it is, and how much of that
    survives out of sample and against the null.
    """

    spec: CenterSpec
    alpha_effective: float
    n_samples: int
    n_positive: int
    prevalence: float
    n_cells_occupied: int
    centers: Tuple[Center, ...]
    n_centers: int
    coverage: float
    coverage_lower: float
    mass: float
    purity_pooled: float
    purity_pooled_lower: float
    lift: float
    max_purity_point: float
    max_purity_lower: float
    coverage_cv: Optional[CVCoverage] = None
    coverage_null_mean: Optional[float] = None
    coverage_p_value: Optional[float] = None
    undetermined_reason: Optional[str] = None

    @property
    def is_undetermined(self) -> bool:
        """
        True when the data cannot support an out-of-sample statement at all
        (too few positives). The UI must render this state as "undetermined"
        rather than as a percentage - a coverage of 0 % measured on 9
        positives and a coverage of 0 % measured on 900 are different claims.
        """
        return self.undetermined_reason is not None


def binarize_target(
    Z: np.ndarray | Sequence[object], positive_value: object
) -> np.ndarray:
    """
    0/1 indicator of `Z == positive_value`, as int8.

    Centres are defined against ONE target value. A multi-class target has no
    single "purity" - `vsf.vis` previously took the share of the
    highest-index class, which is the criterion value only by accident of
    `np.unique` ordering. Callers must state which value they mean.
    """
    arr = np.asarray(Z).ravel()
    return (arr == positive_value).astype(np.int8)


def _cell_counts(
    z_binary: np.ndarray, cell_codes: np.ndarray, n_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(k per cell, n per cell) for a 0/1 target and dense cell codes."""
    n_per_cell = np.bincount(cell_codes, minlength=n_cells).astype(np.int64)
    k_per_cell = np.bincount(
        cell_codes, weights=z_binary.astype(np.float64), minlength=n_cells
    )
    return np.rint(k_per_cell).astype(np.int64), n_per_cell


def _coverage_from_counts(
    k: np.ndarray, n: np.ndarray, mask: np.ndarray, n_positive: int, n_samples: int
) -> Tuple[float, float, float]:
    """(coverage, mass, pooled purity) for a given centre mask."""
    if n_positive <= 0 or n_samples <= 0:
        return 0.0, 0.0, 0.0
    k_sel = int(k[mask].sum())
    n_sel = int(n[mask].sum())
    coverage = k_sel / n_positive
    mass = n_sel / n_samples
    purity = (k_sel / n_sel) if n_sel > 0 else 0.0
    return coverage, mass, purity


def center_summary(
    z_binary: np.ndarray,
    cell_codes: np.ndarray,
    n_cells: Optional[int] = None,
    spec: CenterSpec = CenterSpec(),
) -> Tuple[float, int, float]:
    """
    (coverage, n_centers, pooled purity) of one (binary target, cell
    partition) pair - the three in-sample scalars of `center_report`, from
    the same `_cell_counts` / `select_centers` / `_coverage_from_counts`
    chain, without the per-cell confidence bounds, the centre tuple, the
    cross-validation or the permutation null. For callers that only need
    the scalars (`vsf.avr`'s per-branch prefix series) the bounds were the
    whole cost: 40 continued-fraction bisections per distinct cell shape,
    discarded on return.
    """
    z = np.asarray(z_binary).ravel().astype(np.int8)
    codes = np.asarray(cell_codes).ravel().astype(np.int64)
    if z.shape[0] != codes.shape[0]:
        raise ValueError(f"sample count mismatch: {z.shape[0]} vs {codes.shape[0]}")
    total_cells = int(codes.max()) + 1 if codes.size > 0 else 0
    if n_cells is None:
        n_cells = total_cells
    elif n_cells < total_cells:
        raise ValueError(f"n_cells={n_cells} is smaller than the observed code max")
    n_samples = int(z.shape[0])
    n_positive = int(z.sum())
    k_cell, n_cell = _cell_counts(z, codes, n_cells)
    occupied = int(np.count_nonzero(n_cell > 0))
    mask, _ = select_centers(k_cell, n_cell, spec, occupied)
    coverage, _, purity_pooled = _coverage_from_counts(
        k_cell, n_cell, mask, n_positive, n_samples
    )
    return coverage, int(mask.sum()), purity_pooled


def center_report(
    z_binary: np.ndarray,
    cell_codes: np.ndarray,
    n_cells: Optional[int] = None,
    spec: CenterSpec = CenterSpec(),
    n_splits: int = 5,
    n_repeats: int = 5,
    n_permutations: int = 0,
    random_state: Optional[int] = 0,
    cell_bounds: bool = True,
) -> CenterReport:
    """
    Full centre report for one (binary target, cell partition) pair.

    `z_binary` must be a 0/1 indicator - use `binarize_target`. `cell_codes`
    must be dense codes 0..C-1, as produced by `vsf.metrics.cell_codes`, over
    the SAME rows as `z_binary`.

    `n_splits`/`n_repeats` drive the out-of-sample estimate; set
    `n_repeats = 0` to skip it (the in-sample fields are always computed).
    `n_permutations > 0` attaches the coverage null and its uncorrected
    p-value; for a discovered branch use `familywise_max_coverage_null`
    instead - a branch selected as an argmax over the whole candidate family
    needs a family-aware null (look-elsewhere effect).

    `cell_bounds=False` skips the per-cell Clopper-Pearson intervals, which
    are the dominant cost of a report on a fine lattice and which a bulk
    consumer (the Global Pattern Scan) never reads. Every selection decision,
    coverage, K, purity, the two pooled lower bounds, the cross-validated
    estimate and the permutation null are unaffected - certification never
    goes through the bounds (see `certified_centers`). What changes: each
    `Center.purity_lower` / `purity_upper` reads 0.0 / 1.0,
    `max_purity_lower` is 0.0, and `centers` is ordered by descending point
    purity (then cell index) instead of by descending lower bound.
    """
    z = np.asarray(z_binary).ravel().astype(np.int8)
    codes = np.asarray(cell_codes).ravel().astype(np.int64)
    if z.shape[0] != codes.shape[0]:
        raise ValueError(f"sample count mismatch: {z.shape[0]} vs {codes.shape[0]}")
    if z.size and not np.isin(np.unique(z), (0, 1)).all():
        raise ValueError("z_binary must contain only 0 and 1; use binarize_target")
    total_cells = int(codes.max()) + 1 if codes.size > 0 else 0
    if n_cells is None:
        n_cells = total_cells
    elif n_cells < total_cells:
        raise ValueError(f"n_cells={n_cells} is smaller than the observed code max")

    n_samples = int(z.shape[0])
    n_positive = int(z.sum())
    prevalence = n_positive / n_samples if n_samples > 0 else 0.0
    k_cell, n_cell = _cell_counts(z, codes, n_cells)
    occupied = int(np.count_nonzero(n_cell > 0))

    mask, alpha_eff = select_centers(k_cell, n_cell, spec, occupied)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity_point = np.where(n_cell > 0, k_cell / np.maximum(n_cell, 1), 0.0)
    if cell_bounds:
        lower, upper = purity_bounds(k_cell, n_cell, alpha_eff, spec.method)
        order = np.argsort(-(lower * mask))
    else:
        lower = np.zeros(k_cell.shape, dtype=np.float64)
        upper = np.ones(k_cell.shape, dtype=np.float64)
        order = np.argsort(-(purity_point * mask), kind="stable")

    coverage, mass, purity_pooled = _coverage_from_counts(
        k_cell, n_cell, mask, n_positive, n_samples
    )
    k_sel = int(k_cell[mask].sum())
    n_sel = int(n_cell[mask].sum())
    coverage_lower = (
        float(_cached_clopper_pearson([k_sel], [n_positive], spec.alpha)[0][0])
        if n_positive > 0 else 0.0
    )
    purity_pooled_lower = (
        float(_cached_clopper_pearson([k_sel], [n_sel], spec.alpha)[0][0])
        if n_sel > 0 else 0.0
    )

    centers = tuple(
        Center(
            cell=int(c),
            n=int(n_cell[c]),
            k=int(k_cell[c]),
            purity=float(purity_point[c]),
            purity_lower=float(lower[c]),
            purity_upper=float(upper[c]),
            lift=float(purity_point[c] / prevalence) if prevalence > _EPS else 0.0,
        )
        for c in order.tolist()
        if mask[c]
    )

    cv: Optional[CVCoverage] = None
    reason: Optional[str] = None
    if n_positive < MIN_POSITIVES_FOR_CV:
        reason = (
            f"only {n_positive} samples carry the target value; "
            f"{MIN_POSITIVES_FOR_CV} are required before an out-of-sample "
            "coverage can be estimated to better than +/-10 percentage points"
        )
    elif n_repeats > 0:
        cv = crossvalidated_coverage(
            z, codes, n_cells, spec, n_splits=n_splits,
            n_repeats=n_repeats, random_state=random_state,
        )

    null_mean: Optional[float] = None
    p_value: Optional[float] = None
    if n_permutations > 0:
        null_scores = coverage_null(n_cell, n_positive, spec, n_permutations, random_state)
        null_mean = float(null_scores.mean()) if null_scores.size else 0.0
        exceed = int(np.count_nonzero(null_scores >= coverage - _EPS))
        p_value = float((1 + exceed) / (n_permutations + 1))

    return CenterReport(
        spec=spec,
        alpha_effective=alpha_eff,
        n_samples=n_samples,
        n_positive=n_positive,
        prevalence=prevalence,
        n_cells_occupied=occupied,
        centers=centers,
        n_centers=int(mask.sum()),
        coverage=coverage,
        coverage_lower=coverage_lower,
        mass=mass,
        purity_pooled=purity_pooled,
        purity_pooled_lower=purity_pooled_lower,
        lift=float(purity_pooled / prevalence) if prevalence > _EPS else 0.0,
        max_purity_point=float(purity_point.max()) if purity_point.size else 0.0,
        max_purity_lower=float(lower.max()) if (lower.size and cell_bounds) else 0.0,
        coverage_cv=cv,
        coverage_null_mean=null_mean,
        coverage_p_value=p_value,
        undetermined_reason=reason,
    )


# --------------------------------------------------------------------------
# Out-of-sample estimate
# --------------------------------------------------------------------------
def stratified_repeated_kfold(
    z_binary: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Repeated stratified K-fold index splits, balanced on the 0/1 target.

    Deterministic given `random_state`, and a pure function of
    (`z_binary`, `n_splits`, `n_repeats`, `random_state`) alone - NOT of the
    cell partition. That is what licenses `paired_gain`: two branches scored
    through this function with the same arguments are evaluated on
    byte-identical folds, so their per-split coverages are paired
    observations and their difference has the smaller, paired variance.

    Round-robin assignment within each class (rather than contiguous chunks
    of a shuffled array) keeps the per-fold positive count within 1 of
    n_positive / K even when n_positive is small, which contiguous slicing
    does not.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    z = np.asarray(z_binary).ravel()
    n = int(z.shape[0])
    rng = np.random.default_rng(random_state)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_repeats):
        fold_id = np.empty(n, dtype=np.int64)
        for label in (0, 1):
            idx = np.nonzero(z == label)[0]
            rng.shuffle(idx)
            fold_id[idx] = np.arange(idx.shape[0], dtype=np.int64) % n_splits
        for k in range(n_splits):
            test = np.nonzero(fold_id == k)[0]
            train = np.nonzero(fold_id != k)[0]
            splits.append((train, test))
    return splits


def crossvalidated_coverage(
    z_binary: np.ndarray,
    cell_codes: np.ndarray,
    n_cells: int,
    spec: CenterSpec = CenterSpec(),
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: Optional[int] = 0,
) -> CVCoverage:
    """
    Coverage of centres certified on a training fold, measured on the
    held-out fold.

    This is the only coverage number that can go DOWN when a branch adds an
    axis, and therefore the only one that can answer "how many features do I
    need". The in-sample `CenterReport.coverage` tends to grow with the
    number of cells: a finer partition always has more chances to isolate a
    pure cell. The certificate bounds
    the per-cell FALSE-purity rate, not the optimism of selecting cells by
    their own contents.

    A cell unseen in the training fold is not a centre, by construction: the
    rule being evaluated is "certify on what you have, then apply". Test
    folds contribute a coverage of 0 when they contain no positives, which
    `MIN_POSITIVES_FOR_CV` makes rare and `CenterReport` refuses outright
    below that count.
    """
    z = np.asarray(z_binary).ravel().astype(np.int8)
    codes = np.asarray(cell_codes).ravel().astype(np.int64)
    splits = stratified_repeated_kfold(z, n_splits, n_repeats, random_state)
    coverages = np.zeros(len(splits), dtype=np.float64)
    purities = np.full(len(splits), np.nan, dtype=np.float64)
    for i, (train, test) in enumerate(splits):
        k_tr, n_tr = _cell_counts(z[train], codes[train], n_cells)
        mask_tr, _ = select_centers(k_tr, n_tr, spec)
        k_te, n_te = _cell_counts(z[test], codes[test], n_cells)
        pos_te = int(k_te.sum())
        k_sel = int(k_te[mask_tr].sum())
        n_sel = int(n_te[mask_tr].sum())
        coverages[i] = (k_sel / pos_te) if pos_te > 0 else 0.0
        if n_sel > 0:
            purities[i] = k_sel / n_sel
    return summarize_cv(coverages, purities, n_splits, n_repeats)


def summarize_cv(
    coverages: np.ndarray, purities: np.ndarray, n_splits: int, n_repeats: int
) -> CVCoverage:
    """
    `CVCoverage` from per-split coverages and pooled purities (NaN where a
    split selected nothing), with the Nadeau-Bengio corrected standard
    error. Shared by `crossvalidated_coverage` and
    `vsf.selective.nested_crossvalidation`, so the two estimates are
    summarised identically and can be compared with `paired_gain`.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    coverages = np.asarray(coverages, dtype=np.float64).ravel()
    purities = np.asarray(purities, dtype=np.float64).ravel()
    ratio = 1.0 / (n_splits - 1)
    variance = float(coverages.var(ddof=1)) if coverages.size > 1 else 0.0
    se = float(np.sqrt(max(0.0, (1.0 / max(1, coverages.size) + ratio) * variance)))
    return CVCoverage(
        per_split=tuple(float(v) for v in coverages.tolist()),
        mean=float(coverages.mean()) if coverages.size else 0.0,
        se=se,
        purity_mean=float(np.nanmean(purities)) if np.any(np.isfinite(purities)) else 0.0,
        n_splits=n_splits,
        n_repeats=n_repeats,
        test_train_ratio=ratio,
    )


def paired_gain(higher: CVCoverage, lower: CVCoverage) -> PairedGain:
    """
    Paired Nadeau-Bengio corrected test of `higher.mean - lower.mean`.

    Requires both `CVCoverage` objects to have been produced by
    `crossvalidated_coverage` with identical (`z_binary`, `n_splits`,
    `n_repeats`, `random_state`), so that entry i of each `per_split` refers
    to the same fold; this is checked only by length, so the caller is
    responsible for not mixing seeds.
    """
    a = np.asarray(higher.per_split, dtype=np.float64)
    b = np.asarray(lower.per_split, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"unpaired CV results: {a.shape} vs {b.shape}")
    diff = a - b
    n_obs = int(diff.size)
    variance = float(diff.var(ddof=1)) if n_obs > 1 else 0.0
    se = float(np.sqrt(max(0.0, (1.0 / n_obs + higher.test_train_ratio) * variance)))
    mean_diff = float(diff.mean()) if n_obs else 0.0
    t = mean_diff / se if se > _EPS else (0.0 if abs(mean_diff) <= _EPS else np.inf)
    return PairedGain(difference=mean_diff, se=se, t_statistic=float(t), n_observations=n_obs)


def select_dimensionality(
    cv_by_d: dict[int, CVCoverage], t_threshold: float = 2.0
) -> Optional[int]:
    """
    "How many characteristics does it take to describe the target value?"

    Forward-stopping rule on the out-of-sample coverage curve. Scanning d in
    increasing order, an incumbent is held; d replaces it when the PAIRED
    difference of their per-fold coverages exceeds `t_threshold` corrected
    standard errors (`paired_gain`), and the first incumbent must clear zero
    by the same margin. The answer is the last d that was accepted.

    Semantics, stated because the alternative is a plausible misreading: this
    is the SMALLEST SUFFICIENT dimensionality, not the first useful one. If
    d = 3 genuinely beats d = 2, then two characteristics do not describe the
    target and the answer is 3; if d = 4 does not beat d = 3, the fourth axis
    is not needed and the answer stays 3.

    The scan does NOT stop at the first non-significant step. Coverage is not
    submodular in the feature set: a pair of axes can isolate a pure cell
    that neither axis produces alone, so a flat step from d to d + 1 does not
    license skipping d + 2. Every d up to `max(cv_by_d)` is compared against
    the standing incumbent.

    Returns None when no dimensionality achieves coverage distinguishable
    from zero. That is a result - "this target has no certifiable centres at
    any dimensionality the display can show" - and must be reported as such,
    never silently rounded down to d = 1.

    `t_threshold = 2.0` is a two-sigma convention on a statistic whose
    reference distribution is approximately t with (R K - 1) degrees of
    freedom (Nadeau & Bengio, 2003, Sec. 4); at R = K = 5 the exact 0.975
    quantile is 2.06, so the default is marginally liberal and is stated as a
    convention rather than derived.
    """
    if not cv_by_d:
        return None
    accepted: Optional[int] = None
    incumbent: Optional[CVCoverage] = None
    for d in sorted(cv_by_d):
        cv = cv_by_d[d]
        if cv.mean <= 0.0:
            continue
        if incumbent is None:
            # Nothing to beat yet: the first candidate must separate from
            # zero, otherwise a dimensionality that landed above 0 on fold
            # noise alone becomes the answer.
            if cv.se <= _EPS or cv.mean / cv.se > t_threshold:
                accepted, incumbent = d, cv
            continue
        if paired_gain(cv, incumbent).t_statistic > t_threshold:
            accepted, incumbent = d, cv
    return accepted


# --------------------------------------------------------------------------
# Permutation nulls for the coverage statistic
# --------------------------------------------------------------------------
def coverage_null(
    n_per_cell: np.ndarray,
    n_positive: int,
    spec: CenterSpec = CenterSpec(),
    n_permutations: int = 999,
    random_state: Optional[int] = 0,
) -> np.ndarray:
    """
    Distribution of `CenterReport.coverage` under the null that the target is
    independent of the cell partition, holding BOTH margins fixed.

    Conditional on the cell sizes and the total number of positives, the
    permutation null of the per-cell positive counts is exactly the
    multivariate hypergeometric law - the same null underlying Fisher's exact
    test. Sampling it
    directly is exact, not an approximation of permutation, and costs O(C)
    per replicate instead of O(N).

    Certification inside the loop uses `min_successes_to_certify`, which
    depends only on the cell sizes; those are fixed under permutation, so the
    thresholds are computed once and every replicate is a single integer
    comparison.
    """
    n_cell = np.asarray(n_per_cell, dtype=np.int64).ravel()
    occupied_mask = n_cell > 0
    occupied = int(np.count_nonzero(occupied_mask))
    if n_permutations <= 0 or occupied == 0 or n_positive <= 0:
        return np.zeros(max(0, n_permutations), dtype=np.float64)
    sizes = n_cell[occupied_mask]
    if int(sizes.sum()) < n_positive:
        raise ValueError("n_positive exceeds the total number of samples in the cells")
    alpha_eff = spec.effective_alpha(occupied)
    k_min = min_successes_to_select(sizes, spec, alpha_eff)
    rng = np.random.default_rng(random_state)
    draws = rng.multivariate_hypergeometric(sizes, int(n_positive), size=int(n_permutations))
    certified = draws >= k_min[None, :]
    return np.sum(np.where(certified, draws, 0), axis=1) / float(n_positive)


@dataclass(frozen=True)
class FamilywiseCoverageNull:
    """
    Distribution of the MAXIMUM certified coverage attained anywhere in a
    candidate family under one shared permutation of the target.

    The shared permutation matters and is why this cannot be assembled from
    per-candidate `coverage_null` draws: candidates are highly dependent
    (they are partitions of the same rows by overlapping feature sets), and
    maximising over independently drawn nulls would overstate the null
    maximum and make the resulting p-value conservative to the point of
    uselessness.
    """

    max_scores: np.ndarray
    n_permutations: int
    n_candidates: int

    def p_value(self, observed: float) -> float:
        """Look-elsewhere-corrected p-value, floored at 1 / (B + 1)."""
        if self.n_permutations <= 0:
            return 1.0
        exceed = int(np.count_nonzero(self.max_scores >= observed - _EPS))
        return float((1 + exceed) / (self.n_permutations + 1))


def familywise_max_coverage_null(
    z_binary: np.ndarray,
    candidates: Callable[[], Iterable[Tuple[Hashable, np.ndarray, int]]],
    spec: CenterSpec = CenterSpec(),
    n_permutations: int = 999,
    random_state: Optional[int] = 0,
    block_size: int = 128,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FamilywiseCoverageNull:
    """
    Permutation null of `max_S coverage(Z_pi; X_S)` over the whole candidate
    family - the null of the statistic `vsf.avr.discover_branches` actually
    reports when it is run with `objective="coverage"`.

    `candidates` is a CALLABLE returning a fresh iterable of
    `(key, cell_codes, n_cells)` triples, matching
    `vsf.metrics.familywise_max_null`'s contract, so a caller may regenerate
    codes lazily rather than holding every candidate's (N,) code vector.

    Cost: O(B * sum_S N) gather-and-bincount passes. Not affordable on an
    interactive path for
    wide datasets; that is why it is opt-in and why any published coverage
    must nevertheless set it.
    """
    z = np.asarray(z_binary).ravel().astype(np.int8)
    n_samples = int(z.shape[0])
    n_positive = int(z.sum())
    if n_permutations <= 0 or n_positive <= 0:
        return FamilywiseCoverageNull(np.empty(0, dtype=np.float64), 0, 0)

    rng = np.random.default_rng(random_state)
    max_scores = np.full(n_permutations, -np.inf, dtype=np.float64)
    thresholds: dict[Hashable, np.ndarray] = {}
    n_candidates = 0
    done = 0
    while done < n_permutations:
        size = int(min(block_size, n_permutations - done))
        perms = np.argsort(rng.random((size, n_samples)), axis=1)
        zp = z[perms].astype(np.float64)
        sl = slice(done, done + size)
        n_candidates = 0
        for key, codes, n_cells in candidates():
            n_candidates += 1
            codes = np.asarray(codes, dtype=np.int64).ravel()
            if key not in thresholds:
                n_cell = np.bincount(codes, minlength=n_cells).astype(np.int64)
                occupied = int(np.count_nonzero(n_cell > 0))
                alpha_eff = spec.effective_alpha(occupied)
                k_min = np.full(n_cell.shape, np.iinfo(np.int64).max, dtype=np.int64)
                if occupied > 0:
                    k_min[n_cell > 0] = min_successes_to_select(
                        n_cell[n_cell > 0], spec, alpha_eff
                    )
                thresholds[key] = k_min
            k_min = thresholds[key]
            offsets = (np.arange(size, dtype=np.int64) * n_cells)[:, None]
            flat = (offsets + codes[None, :]).ravel()
            counts = np.bincount(
                flat, weights=zp.ravel(), minlength=size * n_cells
            ).reshape(size, n_cells)
            k_draw = np.rint(counts).astype(np.int64)
            certified = k_draw >= k_min[None, :]
            scores = np.sum(np.where(certified, k_draw, 0), axis=1) / float(n_positive)
            np.maximum(max_scores[sl], scores, out=max_scores[sl])
        done += size
        if progress is not None:
            progress(done, n_permutations)
    return FamilywiseCoverageNull(max_scores, n_permutations, n_candidates)


# --------------------------------------------------------------------------
# Search objective
# --------------------------------------------------------------------------
def coverage_score(
    z_binary: np.ndarray,
    cell_codes: np.ndarray,
    n_cells: int,
    spec: CenterSpec = CenterSpec(),
) -> Tuple[float, float, float, float]:
    """
    Ranking key for `vsf.avr.discover_branches` (v2.3: the only ranking
    there is, not an `objective=` choice -- see `vsf.avr`'s module
    docstring): `(coverage, -n_centers, -mass, max_purity_lower)`, compared
    lexicographically by `max`.

    Rationale for the tie-breaks, in the stated product order: capture as
    much of the target as possible; among equal captures prefer FEWER
    certified centres, because the deliverable is a small set of cells an
    analyst can actually read; among equal centre counts prefer the one
    covering less of the population, i.e. the more concentrated finding.

    The fourth component is not cosmetic. When NO candidate certifies
    anything - the common case on a rare target, and precisely the case this
    module exists to report honestly - the first three components are
    identically (0, 0, 0) for every candidate, and an argmax over them
    returns whichever combination the enumeration happened to reach first.
    That is a non-deterministic-looking, meaningless branch. Breaking the tie
    on the highest per-cell lower bound instead returns the branch that came
    CLOSEST to certifying, which is the only informative thing left to say,
    and is fully determined by the data.

    This is the in-sample statistic and is used ONLY to order candidates.
    Whether the winning branch's coverage is real is decided by
    `familywise_max_coverage_null`, and how much of it survives on new data
    by `crossvalidated_coverage`. Ranking by an in-sample optimum and
    validating it with an out-of-sample estimate are different jobs; the
    search must not be given the CV estimate as its objective, or the folds
    stop being held out.
    """
    z = np.asarray(z_binary).ravel().astype(np.int8)
    codes = np.asarray(cell_codes).ravel().astype(np.int64)
    n_positive = int(z.sum())
    n_samples = int(z.shape[0])
    k_cell, n_cell = _cell_counts(z, codes, n_cells)
    mask, alpha_eff = select_centers(k_cell, n_cell, spec)
    coverage, mass, _ = _coverage_from_counts(k_cell, n_cell, mask, n_positive, n_samples)
    # Wilson, not Clopper-Pearson, for the tie-break ONLY. This key is
    # evaluated once per candidate subset inside the exhaustive search
    # (`vsf.avr._exhaustive_search` computes the same tuple from a fused
    # contingency table and only evaluates this fourth component on ties) --
    # C(M,1) + ... + C(M,4) times per target, and once per (column, value)
    # pair in a Global Pattern Scan -- and the Clopper-Pearson bound costs an
    # 80-step bisection over a continued fraction per cell, which measured
    # ~5x on the scan's total runtime. Wilson is closed-form, monotone in k
    # for fixed n in the same direction, and is used here only to order
    # candidates that are otherwise exactly tied; every REPORTED bound stays
    # Clopper-Pearson (`center_report`, and the renderer's cell intervals).
    best_lower = float(wilson_lower(k_cell, n_cell, alpha_eff).max()) if n_cell.size else 0.0
    return coverage, -float(mask.sum()), -mass, best_lower
