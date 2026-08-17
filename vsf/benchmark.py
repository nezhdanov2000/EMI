"""
VSF Benchmark Module: Synthetic Dataset Generator & Evaluation Protocol
Implements ground truth synthetic dataset generation (Section 7.1) and quantitative evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from .avr import AVREngine, AVRResult


def generate_synthetic_dataset(
    n_samples: int = 1000,
    d_true: int = 3,
    n_noise_features: int = 5,
    noise_level: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Generates a synthetic dataset with known ground truth dimensionality d_true.
    
    Structure:
    - Z ~ Uniform discrete classes (e.g. 4 clusters)
    - First d_true features are strongly correlated with Z (nonlinear non-Gaussian signals)
    - n_noise_features are independent random noise
    
    Returns:
        (X, Z, true_feature_indices, feature_names)
    """
    rng = np.random.default_rng(random_state)
    
    # Target Z: 4 categorical clusters
    Z = rng.choice([0, 1, 2, 3], size=n_samples)
    
    X_list = []
    feature_names = []
    true_indices = list(range(d_true))
    
    # Generate d_true informative features
    for j in range(d_true):
        # Create non-linear functions of Z
        if j % 3 == 0:
            signal = np.sin(Z * np.pi / 2.0) + (Z ** 2) * 0.5
        elif j % 3 == 1:
            signal = np.cos(Z * np.pi) - Z * 1.5
        else:
            signal = (Z % 2) * 3.0 + np.exp(Z * 0.3)
            
        noise = rng.normal(0, noise_level, size=n_samples)
        feature_col = signal + noise
        X_list.append(feature_col)
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
    d_true_list: List[int] = [2, 3, 5, 7, 9],
    n_samples: int = 1000,
    random_state: int = 42,
) -> Dict[str, Union[float, Dict]]:
    """
    Evaluates VSF engine accuracy across multiple synthetic benchmark ground truths.
    
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
        
        for run_idx in range(n_runs):
            seed = random_state + run_idx * 100 + d_true
            X, Z, true_idx, names = generate_synthetic_dataset(
                n_samples=n_samples,
                d_true=d_true,
                n_noise_features=5,
                random_state=seed,
            )
            
            res: AVRResult = engine.fit(X, Z, feature_names=names)
            
            # Expected predicted dimension d_expected = min(7, d_true)
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
        }
        
    total_experiments = len(d_true_list) * n_runs
    overall_accuracy = total_d_correct / total_experiments
    overall_recall = float(np.mean(total_recalls))
    
    return {
        "overall_d_star_accuracy": overall_accuracy,
        "overall_feature_recall": overall_recall,
        "detailed_results": results,
    }
