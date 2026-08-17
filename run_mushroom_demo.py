"""
VSF Analysis on UCI Mushroom Dataset (100% Categorical Labels)
"""

import pandas as pd
import numpy as np
import vsf


def main():
    print("=" * 75)
    print(" VSF Analysis: UCI Mushroom Dataset (Categorical Labels Only)")
    print("=" * 75)

    # 1. Load dataset
    csv_path = "data/mushrooms.csv"
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} total columns.")

    # Target variable: 'class' (e = edible, p = poisonous)
    Z = df["class"].values
    X_df = df.drop(columns=["class"])

    feature_names = list(X_df.columns)
    X = X_df.values

    print(f"Target Z: 'class' ({len(np.unique(Z))} unique categories: {np.unique(Z)})")
    print(f"Features X: {len(feature_names)} categorical attributes (100% labels).")

    # 2. Run AVR Engine
    print("\nRunning Adaptive Visual Routing (AVR Engine)...")
    engine = vsf.AVREngine(
        alpha=0.01,
        vir_threshold=0.85,
        max_d=7,
        n_permutations=500,
        random_state=42,
    )

    res = engine.fit(X, Z, feature_names=feature_names, offline_brute_force=True)

    print("\n" + "=" * 75)
    print(" VSF ANALYSIS RESULT ON CATEGORICAL MUSHROOM DATASET")
    print("=" * 75)
    print(f" • Recommended Optimal Dimension d*: {res.d_star}")
    print(f" • Selected Feature Names:           {res.selected_feature_names}")
    print(f" • Routing Scenario:                 {res.scenario.value}")
    print(f" • Visual Information Ratio (VIR):   {res.vir * 100:.2f}%")
    print(f" • Target Projection Loss (L_target):{res.l_target * 100:.2f}%")
    print(f" • Feature Projection Loss (L_feat): {res.l_feat * 100:.2f}%")
    print(f" • Full Feature Set NMI:             {res.nmi_full:.4f}")
    if res.submodularity_ratio is not None:
        print(f" • Submodularity Ratio vs Global OPT:{res.submodularity_ratio * 100:.2f}%")
    print(f" • XAI Message:                      {res.xai_message}")
    print("=" * 75)

    # 3. Print Individual Feature NMI Ranking
    print("\nTop 10 Categorical Features Ranked by Marginal NMI with Target Z:")
    print("-" * 60)
    scores = []
    for col_name in feature_names:
        nmi_val = vsf.normalized_mutual_information(Z, X_df[col_name].values)
        scores.append((col_name, nmi_val))

    scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (fname, score) in enumerate(scores[:10], start=1):
        selected_mark = " [SELECTED BY VSF]" if fname in res.selected_feature_names else ""
        print(f"  {rank:2d}. {fname:<28} NMI = {score:.4f}{selected_mark}")
    print("-" * 60)


if __name__ == "__main__":
    main()
