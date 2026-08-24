"""
VSF Benchmark Module: Synthetic Dataset Generator & Evaluation Protocol
Implements ground truth synthetic dataset generation (Section 7.1) and quantitative evaluation.
"""

import warnings

import numpy as np

from .avr import AVREngine, AVRResult

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

    This replaces an earlier version of this generator whose d_true
    "informative" features were three recycled deterministic transforms of
    the same scalar Z (cycling on j % 3): a single feature already reached
    NMI(Z; feature) ~= 0.81 regardless of d_true, so `d_true` did not
    represent a genuinely necessary dimensionality and
    `evaluate_vsf_accuracy`'s d_star_accuracy metric scored a correctly
    minimal feature selection as a near-total failure (0% accuracy at
    d_true=5 and d_true=7, with mean_vir ~= 0.99 in the same runs). See the
    project code review for the reproduction.

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
            predictive (and easier for Phase 1's marginal filter to catch);
            higher values push more of the necessary signal into needing
            multiple features together.

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
    engine: AVREngine,
    n_runs: int = 10,
    d_true_list: list[int] = [2, 3, 5, 7, 9],
    n_samples: int = 1000,
    random_state: int = 42,
    min_samples_per_class: int = _DEFAULT_MIN_SAMPLES_PER_CLASS,
) -> dict[str, float | dict]:
    """
    Evaluates VSF engine accuracy across multiple synthetic benchmark ground truths.

    Since `generate_synthetic_dataset` encodes Z over 2**d_true classes, the
    requested `n_samples` is auto-scaled per d_true to
    `max(n_samples, min_samples_per_class * 2**d_true)` so every class keeps
    a statistically workable number of observations; a UserWarning is raised
    whenever this scaling kicks in, naming the effective N used, so a caller
    is never silently handed a benchmark run on an underpowered sample. This
    means wall-clock cost grows with the larger d_true values in
    `d_true_list` (2**9 * 20 = 10,240 rows for d_true=9 at the default
    min_samples_per_class) — pass a smaller d_true_list or raise
    min_samples_per_class's tolerance by lowering it (at the cost of noisier
    per-class estimates) if that cost is prohibitive for a given use.

    Returns:
        Summary dict containing d* accuracy, feature recall@d*, and mean VIR.
    """
    results = {}
    total_d_correct = 0
    total_recalls = []

    for d_true in d_true_list:
        d_correct_count = 0
        recalls = []
        virs = []

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

            res: AVRResult = engine.fit(X, Z, feature_names=names)

            # Expected predicted dimension d_expected = min(max_d, d_true)
            expected_d = min(engine.max_d, d_true)
            if res.d_star == expected_d:
                d_correct_count += 1
                total_d_correct += 1

            # Compute recall of true informative features
            recalled_true = set(res.selected_features).intersection(set(true_idx))
            recall = len(recalled_true) / max(1, len(true_idx))
            recalls.append(recall)
            total_recalls.append(recall)
            virs.append(res.vir)

        results[f"d_true_{d_true}"] = {
            "d_star_accuracy": d_correct_count / n_runs,
            "mean_recall": float(np.mean(recalls)),
            "mean_vir": float(np.mean(virs)),
            "effective_n_samples": effective_n_samples,
        }

    total_experiments = len(d_true_list) * n_runs
    overall_accuracy = total_d_correct / total_experiments
    overall_recall = float(np.mean(total_recalls))

    return {
        "overall_d_star_accuracy": overall_accuracy,
        "overall_feature_recall": overall_recall,
        "detailed_results": results,
    }
