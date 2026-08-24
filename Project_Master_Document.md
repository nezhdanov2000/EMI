# Visual Sufficiency Framework (VSF): Information-Theoretic Adaptive Dimensionality Selection for Multimodal Data Visualization

**Formal Algorithm and Mathematical Specification**

---

## Abstract

We present the Visual Sufficiency Framework (VSF) — an information-theoretic framework for adaptive visualization dimensionality selection. VSF formally addresses the fundamental question: *"How many and which visual channels are necessary and sufficient to faithfully represent the underlying data structure?"*. The framework introduces three primary theoretical contributions: (1) a formal **Visual Sufficiency Criterion** based on statistically significant gains in Mutual Information via conditional permutation testing ($\alpha = 0.01$, with Benjamini-Hochberg FDR control $q \le 0.05$); (2) **Perceptually-Matched Discretization** (PMD) — framing optimal feature quantization under Rate-Distortion Theory constrained by perceptual channel capacities; (3) a scalable feature selection engine: exhaustive search for $M \leq 20$ and greedy forward selection backed by weak submodularity guarantees (Krause et al., 2008) for $M > 20$. The system adaptively routes data into one of four distinct rendering scenarios (2D/3D → 7D → Warning → Block), where every transition is governed by rigorous statistical hypothesis testing rather than heuristic thresholds.

---

## 1. Problem Formulation

### 1.1. Conceptual and Intuitive Definition

> *"Instead of forcing analysts to blindly iterate over dozens of visual axes searching for patterns, our framework allows the user to specify an analytical concept of interest (Target $Z$), after which the system mathematically determines the minimal necessary multidimensional subspace (from 2D to 7D) where clusters of this concept become visually separable, or issues an explicit warning in the presence of noise/chaos."*

In information-theoretic terms, "visual separability of concept clusters" is formalized via **Normalized Mutual Information ($NMI$)**: observing the selected coordinate axes $S^*$ maximally reduces uncertainty (entropy) regarding the target concept $Z$.

### 1.2. Formal Definition

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, z_i)\}_{i=1}^{N}$, where $\mathbf{x}_i \in \mathbb{R}^M$ is a feature vector, $z_i$ is a target variable, and $M$ is the ambient dimensionality.

The visual display interface provides $d_{max} = 7$ functionally independent encoding channels:

$$\mathcal{V} = \{v_1, v_2, ..., v_7\} = \{X, Y, Z, R, G, B, T\}$$

**Objective:** Find an optimal subset $S^* \subseteq \{1, ..., M\}$, $|S^*| = d^*$, where $d^* \leq d_{max}$, such that:
1. $d^*$ represents the **minimal sufficient dimensionality** for visual interpretation;
2. $S^*$ represents the **optimal feature subset** for this dimensionality;
3. Every transition between rendering scenarios is **statistically grounded**.

### 1.3. Visual Channels: The Perceptual-Display Compromise

The constraint $d_{max} = 7$ represents a **Perceptual-Display Compromise** rather than a rigid hardware limitation or arbitrary cognitive rule. A standard monitor physically provides 2 spatial coordinates + 3 color channels (RGB), while animation provides temporal multiplexing over these channels. However, for analytical visualization purposes, we operate with 7 **functionally independent** visual encoding channels:

$$\mathcal{V} = \{\text{Position}_{X,Y},\ \text{Depth}_Z,\ \text{Hue},\ \text{Saturation},\ \text{Lightness},\ \text{Motion/Time}\}$$

The choice of $d_{max} = 7$ is well-grounded in perceptual visualization theory (Munzner, 2014; Ware, 2004): empirical studies establish that human observers can simultaneously distinguish **no more than 7–10 independent visual encodings** in a single view (Healey & Enns, 2012). Setting $d_{max} = 7$ provides a **conservative upper bound** covering the full spectrum of perceptually distinguishable channels for analytical tasks. Crucially, 3D rendering is not mandatory: the system employs 3D (or higher dimensions using color and time) only when 2D (or 1D) is mathematically insufficient to capture the data structure.

### 1.4. Two-Phase Analytical Workflow

Working with VSF is conceptually structured into two phases, ensuring a seamless transition from autonomous pattern discovery to rigorous hypothesis verification:

1. **Auto-Discovery & Propositional Mining:**
   - The algorithm does not depend on a pre-assigned binary target variable $Z$. It conducts an exhaustive scan across **all dataset features**, including multi-class categorical and continuous attributes.
   - Any attribute (e.g., "cap color") is automatically binarized into atomic predicates via **One-vs-Rest decomposition** (e.g., *"brown cap" vs "all others"*).
   - The engine evaluates candidate subsets, identifying multidimensional coordinate bases with maximum $NMI$ and $VIR$ (e.g., capturing 90.7%+ correlation between odor and edibility, as well as complex synergistic interactions) to compile a **Dataset Insight Catalog**.
   - This phase simultaneously establishes the foundation for **Information-Theoretic Data Compression**: identifying a minimal informative basis $S^*$ that preserves key structure without redundant dimensions.
2. **Target-Conditioned Hypothesis Alignment:** The user selects a specific target concept $Z$ (from the generated insight catalog or any arbitrary column/predicate).
   - The previous visual canvas is reset.
   - The VSF engine executes, selecting only features ($X_S$) that provide statistically significant predictive power for the chosen concept $Z$.
   - **Dynamic Reconstruction:** The plot is re-rendered using the minimal necessary dimensionality (from 1D to 7D).
   - **Target-Conditioned Ordering:** Discrete axis categories are automatically reordered (Section 9.1) so that categories predicting the same target outcome $Z$ are spatially grouped together.
   - **Insight Delivery:** The user visually compares the target color coding against spatial clusters, immediately assessing the explanatory power of the selected subspace.

---

## 2. Perceptually-Matched Discretization (PMD)

### 2.1. Theoretical Basis: The Optimal Quantization Problem

**Key Observation:** Visual output on any digital display is *discrete by definition*. A standard 1920×1080 screen distinguishes $\approx 2 \times 10^6$ spatial positions. Human color discrimination is bounded at $\approx 5\text{--}12$ categorical gradations in analytical tasks (Ware, 2004; Borland & Taylor, 2007). A 30 fps temporal axis provides $\approx 100\text{--}300$ discernible states.

Quantizing (binning) a continuous feature $X_j$ into $k$ bins is inherently a **lossy operation** (Rate-Distortion Theory, Shannon, 1959). The PMD objective is to determine the minimum $k$ such that information distortion does not exceed a permissible threshold $\epsilon$, defined by the channel's perceptual Just Noticeable Difference (JND).

### 2.2. Formalization: Distortion Function and Rate-Distortion Formulation

**Definition 1 (Distortion Function).** We define the perceptual distortion function for feature $X_j$ quantized into $k$ bins as:

$$\mathcal{D}_j(k) = 1 - NMI(\tilde{X}_j^{(k)};\ X_j^{cont})$$

where $\tilde{X}_j^{(k)}$ is the discretized representation with $k$ bins, and $X_j^{cont}$ is the continuous original. $\mathcal{D}_j(k) = 0$ signifies complete structural preservation; $\mathcal{D}_j(k) \to 1$ denotes total information loss.

**Definition 2 (Channel Capacity).** For visual channel $v$, perceptual channel capacity (in discrete levels) is defined as $L_v$ — the maximum number of perceptually distinguishable levels (Miller, 1956; Ware, 2004; Munzner, 2014).

| Channel $v$ | Type | $L_v$ (Levels) | $C_v = \log_2 L_v$ (Bits) |
|---|---|:---:|:---:|
| Position X | Spatial | ~200–500 | ~8–9 |
| Position Y | Spatial | ~200–500 | ~8–9 |
| Depth Z | Spatial | ~10–20 | ~3.5–4.5 |
| Color Hue | Chromatic | ~5–12 | ~2.3–3.6 |
| Color Saturation | Chromatic | ~3–7 | ~1.6–2.8 |
| Color Lightness | Chromatic | ~5–9 | ~2.3–3.2 |
| Motion/Time | Temporal | ~50–200 | ~5.6–7.6 |

### 2.3. Proposition 1: Optimal Binning (PMD)

> **Proposition 1 (Perceptually-Matched Discretization).** Let feature $X_j$ be mapped to visual channel $v$ with perceptual capacity $L_v$. The optimal bin count $k_j^*$ is the solution to:
> 
> $$k_j^* = \min_k \; k \quad \text{s.t.} \quad \mathcal{D}_j(k) \leq \epsilon_v, \quad k \leq L_v$$
> 
> where the perceptual distortion threshold is given by $\epsilon_v = \dfrac{1}{L_v + 1}$. This formulation defines the minimum structural loss fraction corresponding to one indistinguishable level of channel $v$.

In practice, $k_j^*$ is determined as: $k_j^* = \min\left(L_v,\ k_{MDL}(X_j, Z)\right)$, where $k_{MDL}$ is obtained via the Minimum Description Length Principle (Fayyad & Irani, 1993).

**Selection of $\mathcal{D}_j = 1 - NMI$:** NMI is chosen as the measure of *structural* distortion because it captures dependency degradation between discretized and continuous variables without assuming metric linearity (unlike Mean Squared Error).

**Grid Capacity Limit:** When computing joint entropy $H(\tilde{X}_S)$ for multidimensional subsets ($d \ge 4$), the total number of hypervolume cells $\prod_{j \in S} k_j$ must not exceed $N / 10$ (where $N$ is sample size). When this threshold is exceeded, the algorithm adaptively merges bins for MI computation to prevent Miller-Madow sample bias and contingency table sparsity.

### 2.4. Discretization Modes and Human-in-the-Loop Authority

**Core VSF Principle:** *The algorithm must not autonomously impose subjective semantic categories on target concepts $Z$ or features $X_j$. Semantic grouping is a domain-specific user responsibility.*

1. **Human-in-the-Loop Priority:** The user specifies or confirms category intervals based on domain expertise. For continuous target $Z$, the system requests explicit grouping rules. The system verifies that $k_{user} \leq L_v$ and warns if channel limits are exceeded.
2. **Recommendation Engine:** When automated guidance is requested, the system suggests mathematically sound partitions via MDLP (Fayyad & Irani, 1993) or the Freedman-Diaconis rule, requiring user confirmation before proceeding.

### 2.5. Information-Theoretic Data Compression & Distillation

**Theoretical Justification:** When VSF identifies a subset $S^* \subseteq \{1, ..., M\}$ of dimensionality $d^* \ll M$ such that $NMI(\tilde{Z}; \tilde{X}_{S^*}) \approx 1$ (or $VIR \approx 1.0$), the remaining $M - d^*$ features contribute negligible additional statistical information regarding $Z$:

$$I(\tilde{Z}; \tilde{X}_{\setminus S^*} \mid \tilde{X}_{S^*}) \to 0$$

**Information Distillation Protocol:**
1. The original feature matrix $\mathbf{X} \in \mathbb{R}^{N \times M}$ is projected onto the compact basis $\mathbf{X}_{S^*} \in \mathbb{R}^{N \times d^*}$, preserving $(1 - \mathcal{L}_{target}) \cdot 100\%$ of target structure.
2. Under Rate-Distortion Theory (Shannon, 1959) and the Minimum Description Length Principle (MDL, Rissanen, 1978), $S^*$ represents a **Minimal Sufficient Basis**, eliminating redundant entropy and noise without sacrificing structural relationships.
3. The compression ratio is defined as:
   $$C_{ratio} = \frac{d^*}{M} \quad \text{subject to} \quad \mathcal{L}_{feat} \leq \epsilon$$

---

## 3. Mathematical Core: Discrete Mutual Information

### 3.1. Fundamental Definitions

Following discretization, all features $\tilde{X}_j$ and target $\tilde{Z}$ take discrete values. Computations are executed via Contingency Tables.

**Shannon Entropy:**

$$H(\tilde{X}) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

**Joint Entropy:**

$$H(\tilde{Z}, \tilde{X}_S) = -\sum_{z, x_S} p(z, x_S) \log_2 p(z, x_S)$$

**Mutual Information of Feature Subset $S$ with Target $Z$:**

$$I(\tilde{Z}; \tilde{X}_S) = H(\tilde{Z}) + H(\tilde{X}_S) - H(\tilde{Z}, \tilde{X}_S)$$

### 3.2. Normalized Mutual Information (NMI)

To compare subsets across varying dimensionalities, normalized mutual information is utilized:

$$NMI(\tilde{Z}; \tilde{X}_S) = \frac{I(\tilde{Z}; \tilde{X}_S)}{\min\left(H(\tilde{Z}), H(\tilde{X}_S)\right)}$$

where $NMI \in [0, 1]$ (0 = statistical independence, 1 = deterministic dependence).

**Normalization Rationale:** Normalizing by $\min(H(\tilde{Z}), H(\tilde{X}_S))$ provides a conservative upper bound on NMI, rendering the Visual Sufficiency Criterion strict and guarding against premature stopping.

---

## 4. Adaptive Visual Routing (AVR) Algorithm

### 4.1. Overview

The Adaptive Visual Routing (AVR) algorithm identifies the optimal dimensionality $d^*$ and feature subset $S^*$ via greedy forward selection coupled with statistical hypothesis testing at each step.

### 4.2. Statistical Significance Testing for MI (Permutation Testing)

**Definition 3 (Marginal Test Significance).** To test $H_0: I(\tilde{Z}; \tilde{X}_j) = 0$ (feature $j$ is uninformative regarding target $Z$), a **permutation test permuting target vector $\tilde{Z}$** is executed (Good, 2005; Runge et al., 2018):

1. Compute observed value $I_{obs} = I(\tilde{Z}; \tilde{X}_j)$.
2. Generate $B = 1000$ random permutations of **target vector $\tilde{Z}$**.
3. For each permutation $\pi_b$, compute $I_{\pi_b} = I(\pi_b(\tilde{Z});\ \tilde{X}_j)$.
4. Compute empirical p-value with add-1 smoothing:

$$p = \frac{1 + \sum_{b=1}^{B} \mathbb{1}[I_{\pi_b} \geq I_{obs}]}{1 + B}$$

5. Feature $j$ is deemed **significant** if it satisfies the Benjamini-Hochberg FDR control threshold ($q \le 0.05$).

**Definition 4 (Conditional Permutation Test for Marginal Gain).** To evaluate $H_0: \Delta I(j \mid S) = 0$ upon appending feature $X_j$ to already selected set $S$, a **stratified conditional permutation test** is employed (Runge et al., 2018):

1. Compute observed gain $\Delta I_{obs}(j \mid S) = I(\tilde{Z}; \tilde{X}_{S \cup \{j\}}) - I(\tilde{Z}; \tilde{X}_S)$.
2. Across $B$ iterations: permute $\tilde{Z}$ **within strata** defined by distinct values of $\tilde{X}_S$ (singleton strata are held fixed to preserve exchangeability), and compute $\Delta I_{\pi_b}$.
3. The empirical p-value is computed using the smoothed formula. Stratification on $\tilde{X}_S$ preserves the joint distribution $P(\tilde{X}_S, \tilde{Z})$, eliminating null bias.

### 4.3. Marginal Visual Information Gain (MVIG)

**Definition 5 (MVIG).** When adding feature $X_j$ to subset $S$:

$$\Delta I(j \mid S) = I(\tilde{Z}; \tilde{X}_{S \cup \{j\}}) - I(\tilde{Z}; \tilde{X}_S)$$

**Definition 6 (Relative MVIG).**

$$\rho(j \mid S) = \frac{\Delta I(j \mid S)}{I(\tilde{Z}; \tilde{X}_S)}$$

### 4.4. Formal AVR Algorithm Specification

```
Algorithm: Adaptive Visual Routing (AVR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  Dataset D, Target Z, Significance level α = 0.01, FDR target q = 0.05
Output: Optimal dimensionality d*, Feature subset S*, Scenario Σ

PHASE 1: FILTERING (Marginal Noise Filtering & FDR Control)
────────────────────────────────────────────────────────────
1.  Discretize all features (PMD, Section 2).
2.  FOR each feature Xⱼ, j = 1..M:
      Compute I(Z̃; X̃ⱼ)
      Execute marginal permutation test (Definition 3) to obtain pⱼ
3.  Apply Benjamini-Hochberg FDR correction:
      Sort p-values: p(1) <= p(2) <= ... <= p(M)
      Find k_max = max { k : p(k) <= (k / M) * q }
      F <- { j : pⱼ <= p(k_max) }
4.  IF |F| = 0:
      RETURN d* = 0, S* = ∅, Σ = SCENARIO_D (Chaos / Block)

PHASE 2: GREEDY FORWARD SELECTION
──────────────────────────────────
5.  Sort F in descending order of I(Z̃; X̃ⱼ).
6.  S <- {arg max_{j ∈ F} I(Z̃; X̃ⱼ)}     // Best primary axis
7.  FOR d = 2 TO min(7, |F|):
      a. j* <- arg max_{j ∈ F \ S} ΔI(j | S)
      b. Execute conditional permutation test for ΔI(j* | S):
         - Permute Z̃ within strata of X̃_S (B iterations)
         - Compute empirical p-value for marginal gain (Definition 4)
      c. IF p >= α:
           BREAK   // Additional dimensions lack statistical significance
      d. S <- S ∪ {j*}
8.  d* <- |S|,  S* <- S

PHASE 3: SCENARIO ROUTING
──────────────────────────
9.  Compute Projection Losses:
      𝓛_{target} = 1 − NMI(Z̃; X̃_{S*})       // Unexplained target variance
      𝓛_{feat}   = 1 − VIR                  // Uncaptured feature information
10. Compute Visual Information Ratio (VIR):
      VIR = I(Z̃; X̃_{S*}) / I(Z̃; X̃_{F})    // F = all Phase 1 significant features
11. Routing Decision:
    IF d* = 0                          -> Σ = SCENARIO_D (Chaos / Block)
    IF d* <= 3                         -> Σ = SCENARIO_A (Minimalist)
    IF d* in [4, 7) and VIR >= 0.85    -> Σ = SCENARIO_B (Full Load)
    IF d* in [4, 7) and VIR < 0.85     -> Σ = SCENARIO_B_INCOMPLETE (Full Load - Incomplete Plateau)
    IF d* = 7 and VIR < 0.85           -> Σ = SCENARIO_C (Warning: >7D)

12. RETURN d*, S*, Σ
```

### 4.5. Universal Propositional Screening & Auto-Discovery

To discover global dependencies autonomously without requiring manual target selection, VSF integrates an end-to-end hypothesis mining engine:

```
                               RAW DATASET (M features)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          Categorical Features                         Continuous Features
                    │                                           │
                    ▼                                           ▼
          One-vs-Rest Binarization                    PMD Quantization + One-vs-Rest
      Z_{k,c} = I[X_k = c]                        Z_{k,b} = I[X_k ∈ Bin_b]
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                    Support Threshold: Support(Z) ≥ 3% (or N_pos ≥ 30)
                                          │
                                          ▼
                    STAGE 1: Fast Matrix Pre-Screening
                    Vectorized I(Z_i; X_j) computation in O(M · K) [~50 ms]
                    Top Candidate Selection (I ≥ I_threshold)
                                          │
                                          ▼
                    STAGE 2: Full AVR Cycle + FDR Control
                    - Conditional permutation test (p < 0.01)
                    - Benjamini-Hochberg correction (FDR q ≤ 0.05)
                    - Synergistic basis search S* (1D to 7D)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          INSIGHTS CATALOG                            GLOBAL DATA DISTILLATION
    Ranked registry of dependencies               Minimal sufficient basis S*_global
    (Click -> instant 7D projection)              (Eliminates uninformative features)
```

#### 4.5.1. Atomic Propositionalization
For each dataset feature $X_k$ ($k=1..M$):
1. **Binary Features:** directly utilized as atomic target concept $Z = X_k$.
2. **Multi-Class Categorical Features:** decomposed into One-vs-Rest predicates:
   $$Z_{k, c} = \mathbb{I}[X_k = c], \quad \forall c \in \text{Categories}(X_k)$$
3. **Continuous Features:** quantized via PMD (Section 2.3) into interval predicates $Z_{k, b} = \mathbb{I}[X_k \in \text{Bin}_b]$.

#### 4.5.2. Statistical Support Threshold
To guard against division by near-zero entropy and spurious $NMI = 1.0$ scores on micro-samples, candidate predicates must satisfy minimum support:
$$\text{Support}(Z_{k,c}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}[x_{i,k} = c] \ge \theta_{supp} \quad (\theta_{supp} = 0.03 \text{ or } N_{pos} \ge 30)$$

#### 4.5.3. Two-Stage High-Performance Pipeline
* **Stage 1 (Fast Vectorized Filter):** Vectorized calculation of pairwise mutual information $I(Z_{k,c}; X_j)$ over contingency matrices in $O(M \cdot K_{total})$, completing within 10–50 ms.
* **Stage 2 (Full AVR Execution with FDR Control):** Top-ranked candidates undergo full AVR forward selection (Section 4.4) with stratified permutation testing.
* **Multiple Testing Protection:** Benjamini-Hochberg FDR control bounds false discoveries at $q \le 0.05$:
  $$p_{(i)} \le \frac{i}{K_{tests}} \cdot q$$

---

## 5. Theoretical Guarantees

### 5.1. Selection Strategy & Submodularity

**Key Note on Submodularity:** Mutual Information $f(S) = I(\tilde{Z}; \tilde{X}_S)$ is **not submodular in the general case** (Krause & Guestrin, 2005) due to potential synergy between variables.

**Runtime Engine:** AVR employs **Greedy Forward Selection** for all dimensionalities $M$, delivering sub-second response times (< 1 s) in interactive applications.

**Gold Standard Benchmark:** For research evaluation when $M \leq 20$, the system supports **exhaustive search** over all $\sum_{d=1}^{7} \binom{M}{d} \leq 77\,520$ combinations as an exact baseline.

For greedy selection under weak submodularity with parameter $\gamma \in (0, 1]$ (Krause et al., 2008; Elenberg et al., 2018):

$$f(S_{greedy}) \geq \left(1 - e^{-\gamma}\right) \cdot f(S^*_{OPT})$$

Under full conditional independence (Naive Bayes regime), $\gamma \to 1$, achieving the classic $(1 - 1/e) \approx 63\%$ approximation bound.

### 5.2. Computational Complexity & Scalability

| $M$ (Features) | Strategy | Combinations | Runtime ($N \leq 10^4$) |
|:---:|---|:---:|:---:|
| $\le 15$ | Greedy (Runtime) / Exhaustive (Offline) | $\le 6\,435$ | < 1 s |
| 16–20 | Greedy (Runtime) / Exhaustive (Offline) | $\le 77\,520$ | ~1–5 s |
| 21–50 | Greedy + Weak Submodularity | $O(M \cdot 7)$ | < 1 s |
| > 50 | Greedy + Fast MI Pre-Filter | $O(M \cdot 7)$ | < 1 s |

---

## 6. Four Rendering Scenarios

### Scenario A: Minimalist (2D/3D)
* **Trigger:** $d^* \leq 3$ (permutation test terminates selection at 2nd or 3rd axis).
* **Interpretation:** Over 95% of data structure is captured in 2–3 dimensions. Additional visual axes provide no statistically significant information.
* **Action:** Render clean spatial channels ($X, Y$ or $X, Y, Z$).

### Scenario B: Full Load (4D–7D)
* **Trigger:** $d^* \in [4, 7]$ and $VIR \geq 0.85$.
* **Interpretation:** Structure is multidimensional. Every axis yields statistically significant improvements, capturing $\ge 85\%$ of available information.
* **Action:** Deploy spatial coordinates + color encodings + temporal slicing.
* **Plateau Subcase ($d^* \in [4, 7)$ and $VIR < 0.85$):** Greedy selection stopped before 7D because no remaining feature provided statistically significant marginal gain ($p \ge \alpha$), rather than display channel exhaustion. Labeled as Scenario B with XAI tag *"Full Load - Incomplete Plateau"*.

### Scenario C: Warning (>7D)
* **Trigger:** $d^* = 7$ and $VIR < 0.85$.
* **Interpretation:** Even 7 visual channels lose >15% of information. Full structure exceeds display bandwidth.
* **Action:** Render best 7D projection + display XAI warning specifying $\mathcal{L}_{feat} = 1 - VIR$.
* **XAI Message:** *«Feature Projection Loss = {𝓛_feat·100}%. Current visualization is incomplete. {M−7} significant features omitted.»*

### Scenario D: Chaos / Block
* **Trigger:** $|F| = 0$ (no feature passes marginal significance test).
* **Interpretation:** No feature subset exhibits statistically significant correlation with target $Z$.
* **Action:** Rendering blocked to prevent false pattern perception.
* **XAI Message:** *«No statistically significant structure detected in data (p > {α} for all features). Visualization blocked to prevent spurious pattern interpretation.»*

---

## 7. Evaluation & Verification Plan

### 7.1. Quantitative Benchmark (Synthetic Ground Truth)
1. **Benchmark Datasets:** Generated with known true dimensionality $d_{true} \in [2, 10]$ and verified orthogonal informative features.
2. **Metrics:**
   * **$d^*$ Accuracy:** Exact match rate between predicted $d^*$ and ground-truth $d_{true}$.
   * **Feature Recall@$d^*$:** Proportion of true informative features recovered.
   * **VIR Calibration:** Correlation between estimated VIR and true mutual information $I_{true}$.

### 7.2. Controlled User Study ($n = 40$ Participants)
* **Design:** Four conditions evaluated across exploratory analytics tasks:
  * **(A) VSF** — Adaptive dimensionality routing (our system).
  * **(B) Fixed-7D** — All 7 channels active unconditionally (baseline).
  * **(C) Expert Baseline** — Manual axis selection by senior data scientists.
  * **(D) Rank-by-Feature + mRMR (d=7)** — Fixed rank-based baseline.
* **Measured Variables:** Task accuracy, completion time, NASA-TLX cognitive load, decision confidence.

### 7.3. Competitive Analysis & Related Work

| System / Method | Selection & Scoring | Target-Conditioned? | Adaptive $d^*$ | Perceptual Binning (PMD) | Statistical Stop Rule | Visual Unit |
|---|---|:---:|:---:|:---:|:---:|---|
| **Voyager 2** (2017) | Compass perceptual heuristics | ❌ Partial | ❌ No (Fixed 2D) | ❌ No | ❌ No | Charts |
| **Scagnostics** (2005) | MST / Alpha graph metrics | ❌ No | ❌ No (Fixed 2D) | ❌ No | ❌ No | Points |
| **mRMR** (2005) | Mutual Information balance | ✔️ Yes | ❌ No (Manual $K$) | ❌ No | ❌ No | Features |
| **Rank-by-Feature** (2005) | 1D/2D statistical criteria | ✔️ Yes | ❌ No (2D SPLOM) | ❌ No | ❌ No | Points |
| **Jeon et al.** (TVCG 2025) | Structural metrics (Pds, Mnc) | ❌ No (Unsupervised) | ❌ No (2D DR tuning) | ❌ No | ❌ No | Points |
| **VSF (Ours)** | **MI + PMD + Permutation Test** | **✔️ Yes (Full)** | **✔️ Yes ($1\text{D}\to 7\text{D}$)** | **✔️ Yes (Rate-Distortion)** | **✔️ Yes ($\alpha = 0.01$, FDR)** | **Discrete Centers (Macro-States)** |

---

## 8. Summary of Scientific Contributions

| # | Contribution | Type | Scientific Novelty |
|---|---|---|---|
| C1 | Visual Sufficiency Criterion — formal statistical stopping rule based on conditional permutation testing | Theoretical | First application of formal hypothesis testing to visualization dimensionality selection |
| C2 | Perceptually-Matched Discretization (PMD) — quantization grounded in visual channel bandwidth | Theoretical | Novel bridge connecting Rate-Distortion Theory to Information Visualization |
| C3 | Adaptive Visual Routing (AVR) — complete 2D→7D routing pipeline across four formal scenarios | Algorithmic | First framework to adaptively select visual dimensionality with MI guarantees |
| C4 | Projection Loss ($\mathcal{L}$) & VIR — standardized metrics for visual information loss | Metric | Unifies statistical dimensionality reduction with visual perception bounds |
| C5 | XAI Chaos Guard — principled rendering refusal in the absence of statistical signal | System | Only visual analytics system designed to explicitly refuse visualization of pure noise |
| C6 | Universal Propositional Auto-Discovery & Compression — dataset-wide dependency mining and distillation | Applied | Autonomous extraction of Pareto-optimal coordinate bases with FDR verification |

---

## 9. Architectural Extensions & Interaction Mechanics

### 9.1. Cluster-Preserving Categorical Ordering & Diagnostic Modes

While Mutual Information $I(\tilde{Z}; \tilde{X}_S)$ is invariant to category permutations, **human visual perception relies heavily on spatial proximity (Gestalt Principle)**.

* **Dual-Mode Axis Serialization:**
  * **Cluster Mode (Default):** 1D Spectral Ordering on $P(\tilde{Z} \mid X_j)$ groups categories sharing identical target response distributions.
  * **Impact / Trend Mode (Option):** Target-Conditioned sorting ordered by conditional expectation $\mathbb{E}[Z \mid X_j = c]$.

* **XAI Discrepancy Diagnostics:** Disagreement between cluster order ($\pi_{cluster}$) and impact order ($\pi_{impact}$) is quantified via **Kendall's rank correlation coefficient ($\tau$)**:
  $$\tau(X_j) = \frac{C - D}{\frac{1}{2} K_j (K_j - 1)}$$
  High agreement ($\tau \approx 1$) indicates monotonic structure; low agreement ($\tau \ll 1$) alerts the analyst to multimodality or complex subgroup interactions.

### 9.2. Discrete Center Rendering: Macro-State Paradigm, Mass & Purity Encoding

#### 9.2.1. Fundamental Visual Unit: Discrete Centers
VSF does not render raw sample rows as individual points. The fundamental visual unit is the **Discrete Center** $\mathbf{c} = (\tilde{x}_{j_1}, \dots, \tilde{x}_{j_{d^*}})$ — a macro-state representing an occupied cell in the contingency table.

#### 9.2.2. Mass Encoding via Area Scaling
Each center aggregates $N_{cell} \ge 1$ observations. Sphere radii follow strict area scaling calibrated against the 1D maximum density to guarantee scale constancy across dimensional transitions without inter-cell collision:

$$\text{Radius}(\mathbf{c}) = R_{\max} \cdot \sqrt{\frac{N_{cell}(\mathbf{c})}{\max_{c'} N_{1D}(c')}}, \quad 2 R_{\max} = 1.0$$

#### 9.2.3. Purity Semantic Color Scale
Center purity reflects target concentration $\text{Purity}(\mathbf{c}) = \frac{N_{cell, z=1}(\mathbf{c})}{N_{cell}(\mathbf{c})} \in [0, 1]$. A discrete 5-zone diverging scale ensures pre-attentive discrimination:
* 🟢 **Target (Pure, 85%–100%):** Strong green (`#53ea4c`).
* 🟡 **High (70%–85%):** Yellow (`#ffeb3b`).
* 🟤 **Dirty / Ambiguous Zone (30%–70%):** Dark brown (`#57463a`) — explicit metaphor for mixed, unseparated classes.
* 🟠 **Low (15%–30%):** Orange (`#ff9800`).
* 🔴 **Alternative (0%–15%):** Strong red (`#f44336`).

#### 9.2.4. Interactive 4D Conditioning Slices
For $d^* \ge 4$, spatial axes $X, Y, Z$ represent the top 3 features, while the **4th feature** ($X_{s_4}$) is controlled via an interactive discrete stepper below the plot. Each slice represents the conditional phase space:

$$\text{Slice}(c) = \{(x_1, x_2, x_3) \mid X_{s_4} = c\}, \quad c \in \text{Dom}(X_{s_4})$$

Discrete stepping eliminates change blindness while maintaining a fixed coordinate scaffold.

#### 9.2.5. Interactive Dirty Center Decomposition (Conjunctive Filter Mining)
For ambiguous centers ($\text{Purity} \in [0.30, 0.70]$), the analyst can inspect candidate conjunctive splits $\phi_k = \bigwedge_{t=1}^k (X_{a_t} = v_t)$ using unused features $X \notin S^*$.

Local NMI is scored with canonical Miller-Madow bias correction:

$$I_{corrected} = I_{plugin} + \frac{(k_Z - 1) + (k_\phi - 1) - (k_{Z,\phi} - 1)}{2 N_\mathbf{c} \ln 2}$$

Candidate filters require support $N_\phi \ge \min(30, \lceil 0.03 \cdot N_\mathbf{c} \rceil)$ and provide one-click feature injection back into the AVR engine.

---

## Key References

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
* Jeon, H., Park, J., Lee, S., Kim, D. H., Shin, S., & Seo, J. (2025). Dataset-Adaptive Dimensionality Reduction. *IEEE TVCG*. DOI: 10.1109/TVCG.2025.3634784.
* Sharpee, T., Rust, N. C., & Bialek, W. (2004). Analyzing neural responses to natural signals: maximally informative dimensions. *Neural Computation*, 16(2), 223-250.
* Cleveland, W. S. (1993). Visualizing Data. *Hobart Press.*
* Heer, J. & Robertson, G. (2007). Animated Transitions in Statistical Data Graphics. *IEEE TVCG.*
