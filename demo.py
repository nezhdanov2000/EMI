"""
VSF Python Library Demonstration & Benchmark Runner
"""

import vsf


def main():
    print("=" * 70)
    print(f"Visual Sufficiency Framework (VSF) v{vsf.__version__} — Python Library Demo")
    print("=" * 70)

    # 1. Generate Synthetic Dataset (d_true = 3 informative, 5 noise features)
    print("\n[1] Generating Synthetic Dataset (d_true = 3, n_noise = 5, n_samples = 1000)...")
    X, Z, true_indices, feature_names = vsf.generate_synthetic_dataset(
        n_samples=1000,
        d_true=3,
        n_noise_features=5,
        noise_level=0.15,
        random_state=42,
    )
    print(f"    Features matrix shape: {X.shape}")
    print(f"    Target Z shape: {Z.shape}")
    print(f"    Informative features: {[feature_names[i] for i in true_indices]}")

    # 2. Run AVR Engine
    print("\n[2] Running Adaptive Visual Routing (AVR Engine)...")
    engine = vsf.AVREngine(
        alpha=0.01,
        vir_threshold=0.85,
        max_d=7,
        n_permutations=500,
        random_state=42,
    )
    res = engine.fit(X, Z, feature_names=feature_names, offline_brute_force=True)

    print("\n" + "-" * 70)
    print("AVR RESULT SUMMARY:")
    print("-" * 70)
    print(f"  • Predicted Optimal Dimension (d*): {res.d_star}")
    print(f"  • Selected Feature Indices:         {res.selected_features}")
    print(f"  • Selected Feature Names:           {res.selected_feature_names}")
    print(f"  • Routing Scenario:                 {res.scenario.value}")
    print(f"  • Visual Information Ratio (VIR):   {res.vir * 100:.2f}%")
    print(f"  • Target Projection Loss (L_target):{res.l_target * 100:.2f}%")
    print(f"  • Feature Projection Loss (L_feat): {res.l_feat * 100:.2f}%")
    print(f"  • Full Feature Set NMI:             {res.nmi_full:.4f}")
    if res.submodularity_ratio is not None:
        print(f"  • Weak Submodularity Ratio (g):    {res.submodularity_ratio * 100:.2f}% of OPT")
    print(f"  • XAI Message:                      {res.xai_message}")
    print("-" * 70)

    # 3. Quantitative Evaluation Benchmark
    print("\n[3] Running Evaluation Benchmark across multiple d_true ground truths...")
    eval_res = vsf.evaluate_vsf_accuracy(
        engine=engine,
        n_runs=5,
        d_true_list=[2, 3, 5],
        n_samples=500,
        random_state=42,
    )

    print(f"  • Overall d* Prediction Accuracy:   {eval_res['overall_d_star_accuracy'] * 100:.1f}%")
    print(f"  • Overall Feature Recall@d*:        {eval_res['overall_feature_recall'] * 100:.1f}%")
    for key, val in eval_res["detailed_results"].items():
        print(
            f"    - {key}: Accuracy = {val['d_star_accuracy']*100:.0f}%, "
            f"Recall = {val['mean_recall']*100:.1f}%, Mean VIR = {val['mean_vir']*100:.1f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
