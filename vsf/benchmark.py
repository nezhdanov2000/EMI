"""
VSF Benchmark Module: Synthetic Dataset Generator & Evaluation Protocol

v2.0 "Clean Core": `evaluate_vsf_accuracy` evaluates `vsf.avr.discover_branches`
/ `BranchEngine` (Independent Branch Discovery), not the v1.0 `AVREngine`'s
greedy, permutation-gated, single-`d_star` selection (see
Project_Master_Document.md Section 0 for the full removal list). This module
was rewritten, not patched, because its two v1.0 metrics no longer have a
referent:

  - `d_star_accuracy` asked "did the engine autonomously stop at the right
    dimensionality?" — meaningless now, since `discover_branches` never
    "stops" at one d; it always returns one independent branch per
    dimensionality in [1, max_d] (bounded only by feature count).
  - `mean_vir` reported a confidence score derived from permutation-test
    p-values — permutation testing was removed entirely (see `vsf.avr`'s
    module docstring), so there is no VIR to average.

What replaces them: for a ground-truth dataset whose `d_true` informative
features are genuinely necessary at exactly `d_true` (see
`generate_synthetic_dataset`'s docstring), the branch at
`expected_d = min(engine.max_d, d_true)` is the one that SHOULD recover
those features (fully, when `d_true <= engine.max_d`; only partially, by
construction, when `d_true` exceeds the engine's dimensionality ceiling —
that degradation is expected and is not scored as a failure). This module
reports `feature_recall_at_expected_d` (does that branch's
`selected_features` recover the true informative features?) and that same
branch's mean MI/NMI, instead of d_star_accuracy/mean_vir.
"""

import warnings

import numpy as np

from .avr import BranchEngine

# Used by evaluate_vsf_accuracy to keep the per-class sample count in
# generate_synthetic_dataset's 2**d_true-class target from collapsing into
# statistical noise at large d_true (see that function's docstring).
_DEFAULT_MIN_SAMPLES_PER_CLASS = 20


def generate_synthetic_dataset(
    n_samples: int = 1000,
    d_true: int = 3,
    n_noise_features: int = 5,
    noise_level: float = 0.2,
    feature_error_rate: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """
    Generates a synthetic dataset with a genuinely necessary ground-truth
    dimensionality d_true.

    Z is determined by d_true INDEPENDENT uniform binary factors, and each of
    the d_true informative features is a noisy channel for exactly one
    factor. Because the factors are mutually independent, I(Z; feature_j) <=
    H(factor_j) = 1 bit for any single feature while H(Z) = d_true bits
    overall — no single feature, nor any strict subset of fewer than d_true
    features, can approach full NMI(Z; features) this way.

    Structure:
    - d_true independent Bernoulli(0.5) factors b_1..b_d_true
    - Z = integer encoding of (b_1, ..., b_d_true) in [0, 2**d_true)
    - Feature j < d_true: with probability (1 - feature_error_rate) reveals
      b_j (as 0.0/1.0 plus small Gaussian jitter); otherwise an independent
      random bit is substituted, so the feature is not a perfect channel
    - n_noise_features additional columns are independent standard Gaussian
      noise, uninformative about Z

    Args:
        n_samples: Number of rows to generate. Because Z has 2**d_true
            distinct classes, callers requesting a large d_true should scale
            n_samples accordingly (see `evaluate_vsf_accuracy`, which does
            this automatically) — with too few samples per class, entropy
            estimates for both Z and the informative features become
            unreliable regardless of how correct the underlying algorithm is.
        feature_error_rate: Probability that an informative feature's bit is
            replaced by independent noise rather than revealing its factor.
            Lower values make individual features more individually
            predictive; higher values push more of the necessary signal into
            needing multiple features together.

    Returns:
        (X, Z, true_feature_indices, feature_names)
    """
    if d_true < 1:
        raise ValueError("d_true must be >= 1")
    if not (0.0 <= feature_error_rate < 0.5):
        raise ValueError("feature_error_rate must be in [0, 0.5) to keep each factor informative")

    rng = np.random.default_rng(random_state)

    factors = rng.integers(0, 2, size=(n_samples, d_true))  # independent bits, one per true dimension
    powers_of_two = 1 << np.arange(d_true)
    Z = (factors * powers_of_two).sum(axis=1).astype(int)  # 0 .. 2**d_true - 1, one class per bit pattern

    X_list = []
    feature_names = []
    true_indices = list(range(d_true))

    # Generate d_true informative features, each a noisy channel for exactly
    # one independent factor of Z.
    for j in range(d_true):
        b_j = factors[:, j].astype(float)
        reveals_true_bit = rng.random(n_samples) >= feature_error_rate
        corrupted_bit = rng.integers(0, 2, size=n_samples).astype(float)
        proxy = np.where(reveals_true_bit, b_j, corrupted_bit)
        jitter = rng.normal(0, noise_level, size=n_samples)
        X_list.append(proxy + jitter)
        feature_names.append(f"Informative_{j+1}")

    # Generate noise features
    for k in range(n_noise_features):
        noise_col = rng.normal(0, 1.0, size=n_samples)
        X_list.append(noise_col)
        feature_names.append(f"Noise_{k+1}")

    X = np.column_stack(X_list)
    return X, Z, true_indices, feature_names


def evaluate_vsf_accuracy(
    engine: BranchEngine,
    n_runs: int = 10,
    d_true_list: list[int] = [1, 2, 3, 4, 6],
    n_samples: int = 1000,
    random_state: int = 42,
    min_samples_per_class: int = _DEFAULT_MIN_SAMPLES_PER_CLASS,
) -> dict[str, float | dict]:
    """
    Evaluates Independent Branch Discovery accuracy across multiple
    synthetic benchmark ground truths.

    `d_true_list` defaults to `[1, 2, 3, 4, 6]` deliberately: `4` matches
    `vsf.avr.MAX_BRANCH_D`, so most entries exercise the case where the
    engine's dimensionality ceiling is sufficient to fully recover the
    ground truth (`expected_d == d_true`), while `6` deliberately exceeds it
    (`expected_d == engine.max_d < d_true`) to document, rather than hide,
    how recall degrades once the ground truth needs more dimensions than the
    engine can search — see this module's docstring.

    Since `generate_synthetic_dataset` encodes Z over 2**d_true classes, the
    requested `n_samples` is auto-scaled per d_true to
    `max(n_samples, min_samples_per_class * 2**d_true)` so every class keeps
    a statistically workable number of observations; a UserWarning is raised
    whenever this scaling kicks in, naming the effective N used, so a caller
    is never silently handed a benchmark run on an underpowered sample. This
    means wall-clock cost grows with the larger d_true values in
    `d_true_list` — pass a smaller d_true_list or raise
    min_samples_per_class's tolerance by lowering it (at the cost of noisier
    per-class estimates) if that cost is prohibitive for a given use. Cost
    also grows with `engine.max_d` itself, since `BranchEngine.fit` performs
    an honest full enumeration per dimensionality (Project_Master_Document.md
    Section 4.6) over `d_true + 5` columns (the informative features plus
    the fixed 5 noise columns) on every one of `len(d_true_list) * n_runs`
    synthetic fits.

    Returns:
        Summary dict containing, per d_true and overall, the feature recall
        and mean MI/NMI of the branch at `expected_d = min(engine.max_d,
        d_true)` — see this module's docstring for why this replaces v1.0's
        d_star_accuracy/mean_vir.
    """
    results = {}
    total_recalls = []
    total_nmis = []

    for d_true in d_true_list:
        expected_d = min(engine.max_d, d_true)
        recalls = []
        mis = []
        nmis = []

        effective_n_samples = max(n_samples, min_samples_per_class * (2 ** d_true))
        if effective_n_samples > n_samples:
            warnings.warn(
                f"evaluate_vsf_accuracy: d_true={d_true} implies 2**{d_true} = "
                f"{2 ** d_true} target classes; n_samples={n_samples} would average "
                f"under {min_samples_per_class} samples/class. Using "
                f"n_samples={effective_n_samples} for this d_true instead so entropy "
                "estimates stay meaningful.",
                stacklevel=2,
            )

        for run_idx in range(n_runs):
            seed = random_state + run_idx * 100 + d_true
            X, Z, true_idx, names = generate_synthetic_dataset(
                n_samples=effective_n_samples,
                d_true=d_true,
                n_noise_features=5,
                random_state=seed,
            )

            branches = engine.fit(X, Z, feature_names=names)
            branch = branches.get(expected_d)
            if branch is None:
                # Only possible if d_true + 5 noise columns < expected_d,
                # which cannot happen for this generator's fixed 5 noise
                # columns and expected_d <= engine.max_d <= MAX_BRANCH_D=4 —
                # guarded rather than silently skipped, so a future change
                # to either constant fails loudly instead of quietly
                # under-counting n_runs.
                raise RuntimeError(
                    f"discover_branches returned no branch at expected_d={expected_d} "
                    f"for d_true={d_true} (available: {sorted(branches.keys())}) — "
                    "the synthetic dataset has fewer columns than expected_d."
                )

            recalled_true = set(branch.selected_features).intersection(set(true_idx))
            recall = len(recalled_true) / max(1, len(true_idx))
            recalls.append(recall)
            total_recalls.append(recall)
            mis.append(branch.mi)
            nmis.append(branch.nmi)
            total_nmis.append(branch.nmi)

        results[f"d_true_{d_true}"] = {
            "expected_d": expected_d,
            "feature_recall_at_expected_d": float(np.mean(recalls)),
            "mean_mi_at_expected_d": float(np.mean(mis)),
            "mean_nmi_at_expected_d": float(np.mean(nmis)),
            "effective_n_samples": effective_n_samples,
        }

    overall_recall = float(np.mean(total_recalls))
    overall_nmi = float(np.mean(total_nmis))

    return {
        "overall_feature_recall": overall_recall,
        "overall_mean_nmi": overall_nmi,
        "detailed_results": results,
    }
