# Visual Sufficiency Framework (VSF): Information-Theoretic Adaptive Dimensionality Selection for Multimodal Data Visualization

**Formal Algorithm Specification and Mathematical Framework**

---

## Abstract

We present the Visual Sufficiency Framework (VSF) — an information-theoretic framework for adaptive visualization dimensionality selection. VSF formally answers the foundational question: *"How many and which visual channels are necessary and sufficient to faithfully represent data structure?"*. The framework introduces three key theoretical contributions: (1) a formal **Visual Sufficiency Criterion** based on statistically significant gains in Mutual Information (permutation test, $\alpha = 0.01$); (2) **Perceptually-Matched Discretization** (PMD) — formulating optimal quantization within Rate-Distortion Theory with an explicit perceptual distortion metric; (3) a scalable feature selection architecture: exact exhaustive search for $M \leq 20$ and a greedy forward selection algorithm with weak submodularity guarantees (Krause et al., 2008) for $M > 20$. The system dynamically routes data into one of four rendering scenarios (2D/3D → 7D → Warning → Block), where each transition is driven by rigorous statistical hypothesis testing rather than arbitrary heuristics.

---

## 1. Problem Formulation

### 1.1. Core Concept and Intuition

> *"Rather than forcing the analyst to blindly toggle dozens of axes in search of patterns, our framework enables the user to specify a concept of interest (Target $Z$). The system then mathematically guarantees the selection of a minimal high-dimensional space (from 2D to 7D) where clusters of this concept are visually separable, or explicitly issues a diagnostic warning if the data exhibits chaos/noise."*

In mathematical terms, "visual separability of concept clusters" is formalized through **Normalized Mutual Information ($NMI$)**: knowing the selected visual axes $S^*$ maximally reduces uncertainty (entropy) regarding the target concept $Z$.

### 1.2. Formal Definition

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, z_i)\}_{i=1}^{N}$, where $\mathbf{x}_i \in \mathbb{R}^M$ is a feature vector, $z_i$ is the target variable, and $M$ is the full dimensionality.

A visual display system provides up to $d_{max} = 7$ independent encoding channels:

$$\mathcal{V} = \{v_1, v_2, ..., v_7\} = \{X, Y, Z, R, G, B, T\}$$

**Objective:** Identify an optimal subset $S^* \subseteq \{1, ..., M\}$, $|S^*| = d^*$, where $d^* \leq d_{max}$, such that:
1. $d^*$ represents the **minimum sufficient dimensionality** for visualization;
2. $S^*$ constitutes the **optimal feature subset** for that dimensionality;
3. Every transition between rendering scenarios is **statistically grounded**.

### 1.3. Visual Channels: The Perceptual-Display Limit

The constraint $d_{max} = 7$ is a **Perceptual-Display Limit** derived from empirical psychophysics and hardware display realities. A standard monitor physically outputs 2 spatial coordinates + 3 color channels (RGB), with animation multiplexing time. For *analytical visualization*, we operate over 7 **functionally independent** encoding channels:

$$\mathcal{V} = \{\text{Position}_{X,Y},\ \text{Depth}_Z,\ \text{Hue},\ \text{Saturation},\ \text{Lightness},\ \text{Motion/Time}\}$$

The bound $d_{max} = 7$ is supported by visual perception literature (Munzner, 2014; Ware, 2004): human analysts reliably track **no more than 7–10 independent simultaneous encodings** in a single view (Healey & Enns, 2012). Selecting $d_{max} = 7$ provides a **conservative upper ceiling** spanning all perceptually distinguishable analytical channels. 3D visualization is strictly adaptive: the system engages 3D (or higher dimensions with color/time) only when 2D is mathematically insufficient to explain the target structure.

### 1.4. Two-Phase Workflow

VSF operates in two synchronized phases, enabling exploratory analysis and hypothesis validation:

1. **Unsupervised Discovery Phase:** The system analyzes the dataset without a predefined target ($Z$ is unassigned), uncovering natural high-dimensional manifold geometry and intrinsic clusters.
2. **Target-Conditioned Alignment Phase:** The user selects a specific target variable $Z$.
   - The prior plot is decommissioned.
   - VSF re-executes, searching exclusively for feature subsets ($X_S$) that statistically predict the new target $Z$.
   - **Dynamic Re-rendering:** The visualization re-renders with the minimal sufficient dimensionality ($1D \le d^* \le 7D$).
   - **Target-Conditioned Ordering:** Discrete axis category ticks are automatically re-sorted (Section 9.1) such that categories associated with identical target outcomes are spatially collocated.
   - **Analytical Insight:** The analyst contrasts the target class distribution against spatial clusters, immediately assessing feature predictive power.

---

## 2. Perceptually-Matched Discretization (PMD)

### 2.1. Theoretical Foundation: Optimal Quantization

**Key Observation:** Display outputs are *discrete by definition*. A 1920×1080 display resolves ~$2 \times 10^6$ distinct spatial locations. Human categorical color perception is bounded at ~5–12 distinct bins in analytical contexts (Ware, 2004; Borland & Taylor, 2007). Temporal frames at 30 fps provide ~100–300 distinguishable states.

Quantizing (binning) continuous feature $X_j$ into $k$ bins is an inherently **lossy operation** (Rate-Distortion Theory, Shannon, 1959). PMD finds the minimal $k$ where structural distortion remains below an allowable threshold $\epsilon_v$, determined by the Just Noticeable Difference (JND) of visual channel $v$.

### 2.2. Formalization: Distortion Function & Rate-Distortion Objective

**Definition 1 (Distortion Function).** For feature $X_j$ discretized into $k$ bins:

$$\mathcal{D}_j(k) = 1 - NMI(\tilde{X}_j^{(k)};\ X_j^{cont})$$

where $\tilde{X}_j^{(k)}$ is the $k$-bin discretized feature, and $X_j^{cont}$ is the continuous original. $\mathcal{D}_j(k) = 0$ signifies complete structural retention; $\mathcal{D}_j(k) \to 1$ denotes total information loss.

**Definition 2 (Channel Capacity).** For visual channel $v$, perceptual capacity $L_v$ is defined as the maximum number of distinguishable levels (Miller, 1956; Ware, 2004; Munzner, 2014).

| Channel $v$ | Type | $L_v$ (Levels) | $C_v = \log_2 L_v$ (Bits) |
|---|---|:---:|:---:|
| Position X | Spatial | ~200–500 | ~8–9 |
| Position Y | Spatial | ~200–500 | ~8–9 |
| Depth Z | Spatial | ~10–20 | ~3.5–4.5 |
| Color Hue | Chromatic | ~5–12 | ~2.3–3.6 |
| Color Saturation | Chromatic | ~3–7 | ~1.6–2.8 |
| Color Lightness | Chromatic | ~5–9 | ~2.3–3.2 |
| Motion/Time | Temporal | ~50–200 | ~5.6–7.6 |

### 2.3. Proposition 1: Optimal PMD Binning

> **Proposition 1 (Perceptually-Matched Discretization).** Let feature $X_j$ be mapped to visual channel $v$ with perceptual capacity $L_v$. The optimal bin count $k_j^*$ solves:
> 
> $$k_j^* = \min_k \; k \quad \text{s.t.} \quad \mathcal{D}_j(k) \leq \epsilon_v, \quad k \leq L_v$$
> 
> where the perceptual distortion tolerance is $\epsilon_v = \dfrac{1}{L_v + 1}$.

In practice, $k_j^* = \min\left(L_v,\ k_{MDL}(X_j, Z)\right)$, where $k_{MDL}$ is determined via Minimum Description Length Principle (MDLP, Fayyad & Irani, 1993).

**Grid Capacity Limit:** When computing joint entropy $H(\tilde{X}_S)$ in high-dimensional subsets ($d \ge 4$), the total grid hypervolume $\prod_{j \in S} k_j$ must not exceed $N / 10$ (where $N$ is sample size). When exceeded, the engine adaptively coarsens bins to prevent small-sample bias (Miller-Madow bias) and contingency table sparsity collapse.

### 2.4. Human-in-the-Loop Discretization Modes

1. **User Authority Priority:** Domain experts may specify or override category intervals. The system verifies $k_{user} \leq L_v$ and warns if channel capacity is exceeded.
2. **Recommendation Engine:** When automated binning is requested, the system computes recommended partitions via MDLP or Freedman-Diaconis rules before execution.

---

## 3. Mathematical Core: Discrete Mutual Information

### 3.1. Fundamental Equations

Following discretization, all features $\tilde{X}_j$ and target $\tilde{Z}$ take discrete values evaluated over contingency tables.

**Shannon Entropy:**

$$H(\tilde{X}) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

**Joint Entropy:**

$$H(\tilde{Z}, \tilde{X}_S) = -\sum_{z, x_S} p(z, x_S) \log_2 p(z, x_S)$$

**Mutual Information of Subset $S$ with Target $Z$:**

$$I(\tilde{Z}; \tilde{X}_S) = H(\tilde{Z}) + H(\tilde{X}_S) - H(\tilde{Z}, \tilde{X}_S)$$

### 3.2. Normalized Mutual Information (NMI)

To compare subsets across varying dimensionalities, we employ Normalized Mutual Information:

$$NMI(\tilde{Z}; \tilde{X}_S) = \frac{I(\tilde{Z}; \tilde{X}_S)}{\min\left(H(\tilde{Z}), H(\tilde{X}_S)\right)}$$

where $NMI \in [0, 1]$ (0 = statistical independence, 1 = deterministic functional dependency). Normalizing by $\min(H(\tilde{Z}), H(\tilde{X}_S))$ provides a conservative upper-bound, making visual sufficiency criteria rigorous.

---

## 4. Adaptive Visual Routing (AVR) Algorithm

### 4.1. Overview

The Adaptive Visual Routing (AVR) algorithm selects optimal dimensionality $d^*$ and feature subset $S^*$ via greedy forward selection coupled with statistical permutation testing at every step.

### 4.2. Statistical Significance Testing (Permutation Tests)

**Definition 3 (Marginal Permutation Test).** To test $H_0: I(\tilde{Z}; \tilde{X}_j) = 0$ (feature has no association with target):

1. Compute observed mutual information $I_{obs} = I(\tilde{Z}; \tilde{X}_j)$.
2. Generate $B = 1000$ random permutations of target vector $\tilde{Z}$.
3. For each permutation $\pi_b$, compute $I_{\pi_b} = I(\pi_b(\tilde{Z});\ \tilde{X}_j)$.
4. Compute empirical p-value:

$$p = \frac{1 + \sum_{b=1}^{B} \mathbb{1}[I_{\pi_b} \geq I_{obs}]}{1 + B}$$

5. Feature is deemed **statistically significant** if $p < \alpha$ (with $\alpha = 0.01$).

**Definition 4 (Conditional Permutation Test).** To test $H_0: \Delta I(j \mid S) = 0$ when appending feature $X_j$ to current set $S$:

1. Compute observed marginal gain $\Delta I_{obs}(j \mid S) = I(\tilde{Z}; \tilde{X}_{S \cup \{j\}}) - I(\tilde{Z}; \tilde{X}_S)$.
2. For each iteration $b \in [1, B]$, permute $\tilde{Z}$ **within strata** defined by unique configurations of $\tilde{X}_S$, computing $\Delta I_{\pi_b}$.
3. Calculate conditional p-value. Stratification preserves joint $(\tilde{X}_S, \tilde{Z})$ covariance, avoiding conditional bias.

### 4.3. Marginal Visual Information Gain (MVIG)

**Definition 5 (MVIG).**

$$\Delta I(j \mid S) = I(\tilde{Z}; \tilde{X}_{S \cup \{j\}}) - I(\tilde{Z}; \tilde{X}_S)$$

$$\rho(j \mid S) = \frac{\Delta I(j \mid S)}{I(\tilde{Z}; \tilde{X}_S)}$$

### 4.4. Formal AVR Algorithm Specification

```
Algorithm: Adaptive Visual Routing (AVR)
────────────────────────────────────────
Input:  Dataset D, Target Z, Significance level α = 0.01
Output: Optimal dimensionality d*, Feature subset S*, Scenario Σ

PHASE 1: NOISE FILTERING
────────────────────────
1. Discretize features (PMD, Section 2).
2. For each Xⱼ, j = 1..M:
     Compute I(Z̃; X̃ⱼ)
     Run marginal permutation test (Definition 3)
3. Filter significant features: F ← {j : pⱼ < α}
4. IF |F| = 0:
     RETURN d* = 0, S* = ∅, Σ = SCENARIO_D (Chaos / Block)

PHASE 2: GREEDY FORWARD SELECTION
─────────────────────────────────
5. Sort F in descending order of I(Z̃; X̃ⱼ).
6. S ← {arg max_{j∈F} I(Z̃; X̃ⱼ)}     // Best 1D predictor
7. FOR d = 2 TO min(7, |F|):
     a. j* ← arg max_{j∈F\S} ΔI(j | S)
     b. Run conditional permutation test for ΔI(j* | S):
        - Permute Z̃ within strata of X̃_S (B iterations)
        - Compute p-value for marginal gain
     c. IF p ≥ α:
          BREAK   // Additional axes are not statistically significant
     d. S ← S ∪ {j*}
8. d* ← |S|,  S* ← S

PHASE 3: SCENARIO ROUTING & LOSS METRICS
───────────────────────────────────────
9. Compute Projection Losses:
     𝓛_{target} = 1 − NMI(Z̃; X̃_{S*})       // Target unexplained variance
     𝓛_{feat}   = 1 − VIR                  // Unrendered significant feature info
10. Compute Visual Information Ratio (VIR):
     VIR = I(Z̃; X̃_{S*}) / I(Z̃; X̃_{F})    // F = all significant features
11. Compute Full NMI:
     NMI_full = NMI(Z̃; X̃_{F})

12. ROUTING LOGIC:
    IF d* = 0                    → Σ = SCENARIO_D (Chaos / Block)
    IF d* ≤ 3                    → Σ = SCENARIO_A (Minimalist)
    IF d* ∈ [4, 7] and VIR ≥ 0.85 → Σ = SCENARIO_B (Full Load)
    IF d* = 7 and VIR < 0.85      → Σ = SCENARIO_C (Warning: >7D)

13. RETURN d*, S*, Σ
```

---

## 5. Theoretical Guarantees

### 5.1. Selection Strategy: Runtime Engine vs. Gold Standard

**Submodularity Note:** Mutual Information $f(S) = I(\tilde{Z}; \tilde{X}_S)$ is **not submodular in general** due to potential feature synergy (Krause & Guestrin, 2005).

**Runtime Engine:** The AVR algorithm employs greedy forward selection across all dimensions $M$, delivering sub-second response times in interactive environments.

**Offline Benchmark:** For $M \leq 20$, the system provides an exhaustive brute-force search over all $\sum_{d=1}^{7} \binom{M}{d} \leq 77,520$ combinations as a gold-standard benchmark.

Under **weak submodularity** with parameter $\gamma \in (0, 1]$ (Krause et al., 2008; Elenberg et al., 2018):

$$f(S_{greedy}) \geq \left(1 - e^{-\gamma}\right) \cdot f(S^*_{OPT})$$

### 5.2. Computational Complexity

| $M$ (Features) | Strategy | Combinations | Runtime ($N \leq 10^4$) |
|:---:|---|:---:|:---:|
| ≤ 15 | Greedy (Runtime) / Exhaustive (Offline) | ≤ 6,435 | < 1 sec |
| 16–20 | Greedy (Runtime) / Exhaustive (Offline) | ≤ 77,520 | ~ 1–5 sec |
| 21–50 | Greedy + Weak Submodularity | $O(M \cdot 7)$ | < 1 sec |
| > 50 | Greedy + Pre-filtered MI | $O(M \cdot 7)$ | < 1 sec |

---

## 6. Four Rendering Scenarios

### Scenario A: Minimalist (2D/3D)
* **Trigger:** $d^* \leq 3$ (permutation test halted selection at 2nd or 3rd axis).
* **Interpretation:** 95%+ of predictable variance is captured by 2–3 features. Additional channels yield no significant gain.
* **Action:** Render spatial coordinates ($X, Y$ or $X, Y, Z$).

### Scenario B: Full Load (4D–7D)
* **Trigger:** $d^* \in [4, 7]$ and $VIR \geq 0.85$.
* **Interpretation:** Multidimensional structure. Each axis provides statistically significant predictive gain, capturing $\ge 85\%$ of total mutual information.
* **Action:** Engage spatial + color + (optionally) temporal encoding channels.

### Scenario C: Warning (>7D)
* **Trigger:** $d^* = 7$ and $VIR < 0.85$.
* **Interpretation:** Top 7 visual axes lose >15% of significant target information due to display channel saturation.
* **Action:** Display optimal 7D projection + mandatory XAI diagnostic banner highlighting $\mathcal{L}_{feat} = 1 - VIR$.
* **XAI Message:** *"Feature Projection Loss = {𝓛_feat·100}%. Current visualization is incomplete: {M−7} significant features are unrendered."*

### Scenario D: Chaos (Block / Abort)
* **Trigger:** $|F| = 0$ (no feature passes marginal permutation test).
* **Interpretation:** No statistically significant association with target variable.
* **Action:** Block visualization to prevent spurious pattern synthesis (apophenia).
* **XAI Message:** *"No statistically significant structure detected in data (p > {α} for all features). Visualization aborted to prevent misleading interpretations."*

---

## 7. Evaluation & Verification Protocol

### 7.1. Quantitative Benchmark (Synthetic & Benchmark Data)
* **Ground Truth Benchmark:** Synthetic datasets with parameterized ground-truth dimensionality $d_{true} \in [2, 10]$ and controlled MI structures.
* **Evaluation Metrics:**
  * **Accuracy of $d^*$:** Exact match rate between $d^*$ and $d_{true}$.
  * **Feature Recall@$d^*$:** Proportion of true causal features selected.
  * **VIR Consistency:** Empirical correlation between predicted VIR and true joint MI.

### 7.2. Controlled User Study ($n \ge 40$ participants)
* **Conditions:**
  * **(A) VSF** — Adaptive dimensionality (our system).
  * **(B) Fixed-7D** — All 7 channels active regardless of data structure.
  * **(C) Expert Baseline** — Manual axis selection by senior data scientists.
  * **(D) Rank-by-Feature + mRMR (d=7)** — Fixed-threshold ranking baseline.
* **Metrics:** Task completion accuracy, time-to-insight, NASA-TLX cognitive load score, and user confidence.

---

## 8. Summary of Scientific Contributions

| # | Contribution | Type | Novelty |
|---|---|---|---|
| C1 | Visual Sufficiency Criterion — formal permutation stopping rule for MVIG | Theoretical | First application to adaptive visual dimensionality selection |
| C2 | Perceptually-Matched Discretization (PMD) — channel capacity binning | Theoretical | New bridge between Rate-Distortion Theory and Visualization |
| C3 | Adaptive Visual Routing (AVR) — 2D→7D routing engine across 4 scenarios | Algorithmic | First engine with information-theoretic guarantees |
| C4 | Projection Loss ($\mathcal{L}$) & VIR — standardized visual loss metrics | Metric | Unifies SDR with visual analytics |
| C5 | XAI Chaos Blocking — automated refusal to render spurious noise | Systemic | First visual framework that explicitly refuses misleading plots |

---

## 9. Open Challenges & Architectural Solutions

### 9.1. Cluster-Preserving Categorical Ordering & Diagnostic Modes
* **Challenge:** Categorical variables have arbitrary default orderings (e.g. alphabetical), which may visually shatter coherent clusters.
* **Dual-Mode Solution:**
  1. **Cluster Mode (Default):** 1D Spectral Ordering on $p(\tilde{Z} \mid X_j)$ to group categories with similar outcome distributions.
  2. **Target-Conditioned Trend Mode:** Sorted by conditional expectation $\mathbb{E}[Z \mid X_j = c]$.
* **Diagnostic Order Discrepancy (Kendall's $\tau$):** Measures divergence between cluster and trend orderings to automatically flag multimodal or non-linear effects in XAI tooltips.

### 9.2. Discrete Overplotting & Sparsity-Preserving Support
* **Challenge:** Overplotting hides multi-sample densities in discrete grid coordinates, while isolated outliers risk visual disappearance.
* **Solution 1 (Occupancy Shading):** Subtle background cell shading for any cell with $N_{cell} \ge 1$, preserving empty cells ($N_{cell} = 0$) as transparent.
* **Solution 2 (Sub-linear Glyph Scaling):** Glyphs scale by $\sqrt{\log(1 + N_{cell})}$, preventing screen occlusion while maintaining perceptual distinguishability.
* **Solution 3 (Linked Contingency Views):** Synchronized 2D cross-tabular slices provide immediate occlusion-free drill-downs.

---

## References

* Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal.*
* Miller, G. A. (1956). The Magical Number Seven, Plus or Minus Two. *Psychological Review.*
* Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica.*
* Fayyad, U. & Irani, K. (1993). Multi-Interval Discretization of Continuous-Valued Attributes (MDLP). *IJCAI.*
* Peng, H. et al. (2005). Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy (mRMR). *IEEE TPAMI.*
* Krause, A. & Guestrin, C. (2005). Near-optimal sensor placements in Gaussian processes. *ICML.*
* Krause, A. et al. (2008). Robust Submodular Observation Selection. *JMLR.*
* Wilkinson, L. et al. (2005). Graph-Theoretic Scagnostics. *IEEE InfoVis.*
* Seo, J. & Shneiderman, B. (2005). Rank-by-Feature Framework. *IEEE InfoVis.*
* Good, P. (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses. *Springer.*
* Ware, C. (2004). Information Visualization: Perception for Design. *Morgan Kaufmann.*
* Munzner, T. (2014). Visualization Analysis and Design. *CRC Press.*
* Healey, C. G. & Enns, J. T. (2012). Attention and Visual Memory in Visualization. *IEEE TVCG.*
* Runge, J. et al. (2018). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances.*
* Elenberg, E. R. et al. (2018). Restricted Strong Convexity Implies Weak Submodularity. *Annals of Statistics.*
* Wongsuphasawat, K. et al. (2017). Voyager 2: Augmenting Visual Analysis with Partial View Specifications. *ACM CHI.*
* Borland, D. & Taylor, R. (2007). Rainbow Color Map (Still) Considered Harmful. *IEEE CG&A.*
