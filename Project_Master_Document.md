# Visual Sufficiency Framework (VSF): Independent-Branch Mutual-Information Visualization

**Formal description of the algorithm and mathematical apparatus**

**Version 2.0 â€” "Clean Core".** This version is a deliberate simplification of the project relative to version 1.0 (preserved in the repository history). Section 0 below documents exactly what changed and why, before proceeding to the formal description.

---

## 0. Revision Note: what changed and why

Version 1.0 of this document described a system with seven-channel ($d_{max}=7$) visualization, greedy feature selection with permutation tests of significance and FDR correction at each step, four rendering scenarios (A/B/C/D), an end-to-end Auto-Discovery engine across all dataset columns (Section 4.5 v1.0), interactive mining of conjunctive filters for "dirty centers" (Section 9.2.8 v1.0), and also â€” outside this document, in the project code â€” a composite AND-filter for the target, a Graph Inference / Knowledge Base engine, and multi-scenario dashboard export.

By direct decision of the project owner (2026-08-28), the product scope was reduced to a **single simple core**: for a user-selected target variable $Z$ (one column, optionally â€” one criterion-value), the system finds **up to 4 independent branches** â€” one for each dimensionality $d \in \{1,2,3,4\}$ â€” and visualizes them as grids of discrete centers (circles), with "collapse/split" animation between dimensions. Everything that does not serve this single purpose was removed, rather than left "just in case" behind a flag.

**Explicitly removed from the core (was in v1.0 or in adjacent code modules not described in this document):**

| Removed | Was | Reason for removal |
|---|---|---|
| Rendering scenarios A/B/C/D and routing by $d^*$ | Section 6 v1.0 | Built around permutation stopping of greedy selection, which no longer exists â€” there is no single "optimal" $d^*$, there are 4 parallel branches |
| Greedy forward selection + permutation significance tests + FDR (Benjamini-Hochberg) | Sections 4.2, 4.4 v1.0 | Replaced by honest exhaustive search with ranking by "raw" MI â€” see Section 4.5 below, where the cost of this decision is explicitly stated |
| Weak submodularity guarantee of the greedy algorithm (Krause et al., 2008) | Section 5.1 v1.0 | Inapplicable: the algorithm is no longer greedy |
| Universal Propositional Screening / Auto-Discovery across all columns, global distillation basis | Section 4.5 v1.0 | A separate task "find interesting Z across the whole dataset", not part of the defined core "given Z â†’ show 4 branches". **This decision was made by me by analogy with other removals (significance testing, which Auto-Discovery entirely relied on), and not directly confirmed by the user** â€” see the explicit assumption note below |
| Interactive mining of "dirty centers" (Conjunctive Filter Explorer) | Section 9.2.8 v1.0 | Explicitly named by the user among the items to remove |
| Composite AND-filter of the target (`composite_target` in `/api/analyze`) | Code (`server.py`, `app.js`), not described in this document | Explicitly named by the user among the items to remove |
| Graph Inference / Knowledge Base (`graph.html`, `graph_reasoning.js`) | Code, separate HTML | Explicitly named by the user among the items to remove |
| Multi-scenario dashboard export (`export_full_dashboard(targets=...)`) | Code (`vsf/dashboard.py`) | Explicitly named by the user among the items to remove |
| $d_{max} = 7$ and Hue/Saturation/Lightness channels | Sections 1.3, 2.2 v1.0 | Replaced by $d_{max} = 4$ (3 spatial axes + 1 temporal/frame axis) â€” this is the actual state of affairs already in the current visualization implementation, not a new restriction |

**Explicitly preserved without substantive changes (moved here, often with cosmetic editing):** perceptually-aligned discretization (PMD), MI/NMI mathematical core, the Discrete Centers paradigm with mass encoding via area, 4-zone (was 5-zone) purity semantics, film strip metaphor for 4D, animated transitions between dimensions, canonical target-conditioned axis sorting and Object Constancy, diagnostics of the divergence between cluster/trend ordering of categories.

> **Assumption requiring confirmation:** the removal of Auto-Discovery/Universal Propositional Screening (Section 4.5 v1.0) and the associated "Insights catalog across all columns" â€” is my decision by analogy, and not directly requested by the user. Functionally, this means that the user must **explicitly specify the target column $Z$** (and, optionally, a criterion-value) themselves â€” the system no longer scans the entire dataset for "interesting" targets automatically. If this is a misinterpretation â€” this is the only point where it's worth going back and clarifying before starting implementation.

---

## 1. Problem Formulation

### 1.1. Concept and intuitive definition

> *"Given a dataset and a target variable (or criterion-value within it). The system finds up to 4 best independent sets of features â€” one for each dimensionality from 1 to 4 â€” and shows where the target class occurs frequently, where it is rare, and where the situation is ambiguous, with circles within a grid of categories."*

The key word is **independent**: the best set of features for 3D does not have to contain the best set for 2D. Mutual information is not submodular in the general case (Krause & Guestrin, 2005) â€” feature synergy means that the pair $\{A, B\}$ can beat the pair $\{C, D\}$ in explaining $Z$, even if $A$ alone is stronger than $C$. Hence â€” 4 parallel "development branches", rather than a single chain of "adding one best feature at a time".

### 1.2. Formal definition

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, z_i)\}_{i=1}^{N}$, where $\mathbf{x}_i \in \mathbb{R}^M$ is a feature vector, $z_i$ is the target variable (one dataset column; optionally binarizable by a criterion-value, see Section 1.4), $M$ is the full dimensionality.

**Task:** for each dimensionality $d \in \{1, 2, 3, 4\}$ independently find

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; I(\tilde{Z}; \tilde{X}_S)$$

The result is up to 4 **branches** $(d, S^*_d, I(\tilde{Z}; \tilde{X}_{S^*_d}))$, $d = 1, \dots, \min(4, M)$. No single "optimal $d^*$" is computed or selected by the algorithm â€” the choice of dimensionality is given to the user (see Section 6).

### 1.3. Visual channels

The display provides **4 independent encoding channels**, without claiming the 7-channel perceptual limit of version 1.0 (this is the actual state of affairs already in the current rendering implementation, not a new restriction):

$$\mathcal{V} = \{X, Y, Z_{depth}, T\}$$

The first three channels are the spatial axes of the 3D cube; the fourth is not an independent visual axis, but a **temporal/frame channel** (see Section 5.5): the values of the 4th feature turn into a stack of parallel frames ("film strip"), between which the user switches manually (stepper) or starts auto-play (play/pause/speed).

Circle color and size in this version are **not independent feature selection channels**, but the encoding of aggregate statistics of the already selected branch (the proportion of the target class and the number of objects in the cell, respectively, see Section 5.3) â€” they do not participate in the search for $S^*_d$.

### 1.4. Target variable and criterion

The user sets the target variable in one of two ways, identical to the already existing `target`/`criterion` semantics:

1. **Raw multiclass target:** $Z = X_{target}$ as is (for example, the entire `class` column with its original values).
2. **Binary criterion:** a specific value $v$ of one column, $Z = \mathbb{I}[X_{target} = v]$ (for example, `class = "p"`).

A composite AND-criterion across several columns (was in v1.0 as `composite_target`) is not included in the core â€” see Section 0.

---

## 2. Perceptually-Aligned Discretization (PMD)

### 2.1. Theoretical justification: optimal quantization problem

The visual output of any display is discrete by design: a circle occupies a finite area of a grid cell, the number of cells is limited. Quantization (binning) of a continuous feature $X_j$ into $k$ bins is a lossy operation (Rate-Distortion Theory, Shannon, 1959). PMD formulates this problem as finding the minimum $k$ at which the distortion of the data structure does not exceed an acceptable threshold, determined by the perceptual sensitivity of the channel.

### 2.2. Formalization: distortion function

**Definition 1 (Distortion Function).**

$$\mathcal{D}_j(k) = 1 - NMI(\tilde{X}_j^{(k)};\ X_j^{cont})$$

where $\tilde{X}_j^{(k)}$ is the discretized version of the feature in $k$ bins, $X_j^{cont}$ is the continuous original.

| Channel $v$ | Type | $L_v$ (levels) | Role in core |
|-----------|-----|:---:|---|
| Position X / Y / Depth Z | Spatial | ~10â€“20 (limited by grid size, not screen) | Branch axis |
| Time/Frame | Temporal | ~number of unique values of the 4th feature | 4th element of branch when $d=4$ |

Difference from version 1.0: there $L_v$ was calibrated for 7 perceptual channels (including Hue/Saturation/Lightness as independent selection axes). Here discretization is needed **only** to turn a continuous feature into a finite number of axis interval-categories, not to calibrate for 7 visual channels.

### 2.3. Proposition 1: Optimal binning (PMD)

> Let feature $X_j$ be quantized into $k$ intervals for mapping onto the axis grid. The optimal number of intervals $k_j^*$ is the solution to the problem:
>
> $$k_j^* = \min_k \; k \quad \text{s.t.} \quad \mathcal{D}_j(k) \leq \epsilon, \quad k \leq L_{grid}$$
>
> where $\epsilon$ is the acceptable structure loss fraction, $L_{grid}$ is the practical limit on the number of divisions of one axis (limited by grid size and minimum statistical support of the cell, not color/brightness perception).

In practice, $k_j^*$ is found via MDL (Fayyad & Irani, 1993), as in version 1.0: $k_j^* = \min\left(L_{grid},\ k_{MDL}(X_j, Z)\right)$.

### 2.4. Grid Capacity Limit

When computing joint entropy for a subset of features $S$ ($|S| \leq 4$), the total number of cells $\prod_{j \in S} k_j$ must not exceed $N / 10$ (where $N$ is the number of rows). This limit is **especially important in the new version**: the exhaustive search (Section 4) computes MI for each of the potentially millions of feature combinations (Section 4.6), and without adaptive bin merging, a portion of these combinations would yield degenerate, statistically unreliable MI estimates simply due to contingency table sparsity â€” which would distort branch ranking, not just visualization. If the threshold is exceeded, the bins are adaptively merged before calculating MI for this specific combination.

### 2.5. Human-in-the-Loop

As in version 1.0: the algorithm does not assign semantic categories to the target variable autonomously. The user sets or confirms the target column and (optionally) the criterion-value; for continuous features, the system suggests an MDL-recommended split, without applying it without confirmation.

---

## 3. Mathematical Core: Discrete Mutual Information

This section has not changed essentially relative to version 1.0 â€” the foundation remains the same.

### 3.1. Basic definitions

**Shannon Entropy:**

$$H(\tilde{X}) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

**Joint Entropy:**

$$H(\tilde{Z}, \tilde{X}_S) = -\sum_{z, x_S} p(z, x_S) \log_2 p(z, x_S)$$

**Mutual Information of subset $S$ with target $Z$:**

$$I(\tilde{Z}; \tilde{X}_S) = H(\tilde{Z}) + H(\tilde{X}_S) - H(\tilde{Z}, \tilde{X}_S)$$

### 3.2. Normalized Mutual Information (NMI)

$$NMI(\tilde{Z}; \tilde{X}_S) = \frac{I(\tilde{Z}; \tilde{X}_S)}{\min\left(H(\tilde{Z}), H(\tilde{X}_S)\right)}$$

$NMI \in [0, 1]$. Used for display (in branch HUD metrics) and for discretization (Section 2), but **not** for ranking branches of the same dimensionality â€” see the explicit disclaimer in Section 4.3.

---

## 4. Independent Branch Discovery Algorithm

### 4.1. Overview

This is the section that completely replaces the Adaptive Visual Routing (AVR) algorithm of version 1.0. Instead of a single greedy forward selection with a permutation stopping criterion â€” **4 independent exact searches**, one for each dimensionality $d \in \{1,2,3,4\}$, each an honest exhaustive search.

### 4.2. Formal definition of a branch

**Definition (Branch).** For dimensionality $d$, branch $S^*_d$ is

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; I(\tilde{Z}; \tilde{X}_S)$$

Ranking within a single dimensionality is done by **raw (unnormalized) mutual information** $I$, not by $NMI$ â€” a direct requirement of the product specification (see the explicit compromise disclaimer in Section 4.5). For $d > M$, branch $S^*_d$ is undefined (not enough features in the dataset).

### 4.3. Algorithm

```
Algorithm: Independent Branch Discovery (IBD)
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
Input:   Dataset D, target Z (column + optional criterion)
Output:  Up to 4 branches {(d, S*_d, I_d)}, d = 1..min(4, M)

1. Discretize all features (PMD, Section 2).
2. Prepare ZÌƒ (raw multiclass target or binary criterion).
3. FOR d = 1 TO min(4, M):
      a. FOR EACH combination S âŠ† {1,...,M}, |S| = d:
           Compute XÌƒ_S (joint discrete code of features S)
           Compute I(ZÌƒ; XÌƒ_S)
      b. S*_d â† arg max over all iterated S
      c. Save (d, S*_d, I(ZÌƒ; XÌƒ_{S*_d})) as branch d
4. RETURN all found branches
```

No stopping condition, significance threshold, or VIR threshold â€” the searches for different $d$ are completely independent and are **always** executed in full (see Section 4.6 regarding the cost of this decision).

### 4.4. Why branches do not have to be nested

Mutual information $f(S) = I(\tilde{Z}; \tilde{X}_S)$ is not submodular in the general case (Krause & Guestrin, 2005): with feature synergy, $f$ can be supermodular, meaning $f(S \cup \{a, b\}) - f(S)$ can exceed $\big(f(S \cup \{a\}) - f(S)\big) + \big(f(S \cup \{b\}) - f(S)\big)$. Practical consequence: a pair of features, neither of which individually belongs to the best single axis $S^*_1$, can jointly form a highly superior $S^*_2$ â€” version 1.0's greedy algorithm, growing $S^*_1 \to S^*_2 \to S^*_3$ one feature at a time, **cannot find** such a pair in principle, since it never re-evaluates an already selected feature. Exhaustive search for each $d$ separately is the only exact way to guarantee finding the true $S^*_d$ under possible synergy.

### 4.5. Methodological Trade-off: abandoning statistical significance control

**Directly and without softening (Zero False Optimism):** version 1.0 accompanied each selection step with a permutation significance test ($\alpha=0.01$) and controlled multiple testing with Benjamini-Hochberg FDR correction ($q \le 0.05$) â€” precisely to distinguish real structure from random coincidence. Version 2.0 **completely removes this control** by direct product decision: branches are ranked solely by the raw value $I(\tilde{Z}; \tilde{X}_S)$, without a single statistical test.

This brings back exactly the risk that significance control was designed to eliminate: the **multiple comparisons problem** (look-elsewhere effect). Exhaustive search for $d=4$ on a dataset with $M=22$ features (like mushrooms) evaluates $I$ for 7,315 different combinations; for each of them, in the absence of a true relationship with $Z$, there is a non-zero probability of accidentally getting a high observed $I$ just due to noise in the finite sample â€” and the more combinations iterated, the higher the chance that at least one of them will "win" purely by chance, especially on small $N$ or for features with a large number of categories (more categories â†’ more degrees of freedom for the MI estimate â†’ systematic positive bias on small samples, the same effect that the plugin MI estimator without Miller-Madow correction is always upwardly biased). No FDR correction or other adjustment for this effect is applied in version 2.0 â€” the winning branch is not accompanied by a claim of statistical significance, only the number $I$.

Additionally: ranking by **raw** $I$, rather than $NMI$, does not correct for the different number of degrees of freedom among different combinations of the same dimensionality $d$ (a combination of high-cardinality features has structurally more "places" where random noise can leak than a combination of binary features) â€” this is the same class of bias that was previously solved by normalizing by $\min(H(\tilde{Z}), H(\tilde{X}_S))$.

This is an accepted, not a hidden trade-off. The product gains simplicity and predictability ("we rank by MI, period") at the cost that the winning branch could technically turn out to be an artifact of overfitting on a specific sample, especially with large $M$ (many candidates) and small $N$ (few rows per combination). **Possible future mitigation** (not implemented now, only noted as an option): rank by $NMI$ instead of $I$, or add a single optional "reliability badge" based on sample size in the branch cells (similar to the ðŸŸ¢/ðŸŸ¡/ðŸ”´ indicator of version 1.0) without the full permutation infrastructure.

### 4.6. Computational complexity: the cost of "always honest exhaustive search"

By direct product decision, the exhaustive search **does not** have a pre-filter, a feature count limit, and does not switch to an approximate (greedy/beam) search at any $M$. The number of combinations to evaluate for all 4 branches:

$$\text{Total}(M) = \sum_{d=1}^{\min(4,M)} \binom{M}{d}$$

| $M$ (features) | $\binom{M}{1}$ | $\binom{M}{2}$ | $\binom{M}{3}$ | $\binom{M}{4}$ | Total combinations |
|:---:|---:|---:|---:|---:|---:|
| 10 | 10 | 45 | 120 | 210 | 385 |
| 22 (mushrooms) | 22 | 231 | 1,540 | 7,315 | 9,108 |
| 30 | 30 | 435 | 4,060 | 27,405 | 31,930 |
| 50 | 50 | 1,225 | 19,600 | 230,300 | 251,175 |
| 100 | 100 | 4,950 | 161,700 | 3,921,225 | 4,087,975 |

Each $I(\tilde{Z}; \tilde{X}_S)$ estimation via contingency table costs $O(N)$ (vectorized construction of the joint feature code + frequency counting), where $N$ is the number of rows. The total cost is $O\big(N \cdot \text{Total}(M)\big)$. For $M=100$, $N=10^4$, this is on the order of $4 \times 10^{10}$ elementary operations â€” in practice, minutes-hours even with a fully vectorized NumPy implementation, let alone a naive Python loop over combinations.

**This is an honestly documented consequence of the explicit decision of the product owner ("always honest exhaustive search"), not a forgotten edge case.** Datasets of the mushrooms scale ($M \approx 22$) remain completely comfortable (9,108 combinations, fractions of a second to seconds on vectorized NumPy). Datasets with $M \gtrsim 50-100$ features will visually "hang" for tens of minutes with an honest implementation â€” this is not a reason to silently substitute the algorithm with an approximate one, but a reason to inform the user of the expected time before launching (a UI engineering task, not the subject of this document) and, separately, during implementation, to invest in accelerating the honest search itself (batch vectorization over many combinations simultaneously, reusing partial contingency tables between combinations with a common prefix, parallelization) â€” that is, make the search faster, not incomplete.

### 4.7. Global Pattern Scan: a dataset-wide reformulation of the same algorithm (only `vsf.serve()`)

Upon explicit user request (after resolving the open question in Section 10, item 3 of the version prior to this revision), the live application was augmented with the ability to apply Section 4.3 not to a single pre-selected target $Z$, but to **every** pair (column, observed value) in the dataset, taken as a One-vs-Rest criterion â€” the same form as a regular manual criterion in Section 1.4.

**This is NOT a reincarnation of Auto-Discovery from version 1.0.** The only overlap is that both ideas iterate through all dataset columns. The differences are fundamental:

* v1.0 Auto-Discovery replaced ordinary target selection and ran by default, with a permutation significance test and greedy selection of a single $d^*$ (all this was removed, see Section 0). Global Pattern Scan is a strictly optional, button-triggered operation that **only filters** the already existing flat Target Selector by NMI threshold; it does not replace or pre-select a target for the user, and introduces no statistical significance check â€” it uses the same "raw MI/NMI without proof claim" principle (Section 4.5), just applied to many targets in a row rather than one.
* For each pair (column $c$, value $v$), $\tilde Z = \mathbb{1}[c = v]$ is built, and over all other columns as features, **exactly the same** Independent Branch Discovery (Section 4.3) is run as for a single manual analysis â€” not an approximation, not a cheaper variant. From the up to 4 branches found, the maximum $\mathrm{NMI}$ is taken; the pair is kept if this maximum is strictly greater than the user-specified threshold.
* **The cost scales linearly** with the number of (column, value) pairs â€” with $M$ columns and an average of $\bar k$ observed values per column, this is $\approx M \cdot \bar k$ full runs of Section 4.3, meaning the cost of Section 4.6 multiplied by $M \cdot \bar k$. On the mushrooms dataset ($M=22$, 119 (column, value) pairs), this is empirically â‰ˆ16 s per pair Ã— 119 â‰ˆ 32 minutes of honest search â€” measured, not estimated. Just as in Section 4.6, no pre-filter, approximation, or shortcut (such as exploiting MI symmetry of binary columns, which gives only ~5% savings on this dataset) is deliberately applied â€” the price of the honest search is considered the acceptable price of honesty, not a reason to simplify the algorithm.
* Since the operation is magnitudes more expensive than a single analysis and unambiguously exceeds any reasonable synchronous HTTP request timeout, it is executed in a background thread on the server (`vsf/server.py`) with a client-polled progress status and cooperative cancellation between (not within) calls to Section 4.3 â€” an engineering implementation, not a separate algorithm.
* **Scope â€” only `vsf.serve()`.** The static export (`export_full_dashboard`, Section 1 `UI_Functional_Spec.md`) does not have a server to run a search in the browser, so this feature is not added there.

Full UI/API specification (endpoints, progress format, behavior on page reload) is in `UI_Functional_Spec.md`, Section 2.1.

---

## 5. Visualization: Discrete Centers

This section consolidates and brings to the forefront material that in version 1.0 was an "open problem" in Section 9.2 â€” in version 2.0, it is the central, rather than peripheral, part of the specification, since the entire product boils down to this visualization plus the algorithm of Section 4.

### 5.1. Fundamental visual unit: Discrete Center

VSF does not visualize individual dataset rows as individual points. The visual unit is a **Discrete Center**: a unique combination of discretized feature values from branch $S^*_d$, representing a macrostate of the phase space.

$$\mathbf{c} = (\tilde{x}_{j_1}, \dots, \tilde{x}_{j_d}) \in \prod_{j \in S^*_d} \text{Bins}(X_j)$$

The number of actually populated centers is typically $|\mathcal{C}_{occ}| \ll \prod_{j} k_j$ â€” not every category combination occurs in the data.

### 5.2. Center mass encoding: circle size (Mass Encoding via Area Scaling)

Each discrete center aggregates $N_{cell} \ge 1$ objects. Scaling by area (Tufte, 1983) minimizes the "Lie Factor":

$$\text{Radius}(\mathbf{c}) = R_{\max} \cdot \sqrt{\frac{N_{cell}(\mathbf{c})}{\max_{c'} N_{1D}(c')}}$$

where the normalizing denominator is the maximum density in the 1D projection of the same branch, $2R_{\max} = 1.0$ (the side of the elementary grid cell) â€” the maximum circle exactly fills the interval/cell, never overlapping neighbors.

### 5.3. Semantic purity encoding: circle color (Purity Color Scale)

The purity of center $\mathbf{c}$ is the proportion of the target class among its objects:

$$\text{Purity}(\mathbf{c}) = \frac{N_{cell,\, z=\text{target}}(\mathbf{c})}{N_{cell}(\mathbf{c})} \in [0, 1]$$

**Discrete 4-zone color scale** (version 2.0 â€” replaces the 5-zone scale of version 1.0, boundaries are set explicitly and exhaustively, without overlaps):

| Zone | $\text{Purity}$ Range | Color | Semantics |
|------|:---:|---------|-----------|
| **Target** | $(0.85,\ 1.0]$ | ðŸŸ¢ Green | Center is pure: target class confidently dominates |
| **High** | $(0.75,\ 0.85]$ | ðŸŸ¡ Yellow | Target class prevails, but with noticeable admixture |
| **Muddy Zone** | $[0.25,\ 0.75]$ | ðŸŸ¤ Brown | Uncertainty: classes are physically mixed, the chosen branch does not separate them |
| **Alternative** | $[0.0,\ 0.25)$ | ðŸ”´ Red | Target class is practically absent |

Boundaries are given as a partition of $[0,1]$ into 4 non-overlapping, exhaustive intervals (half-open/closed exactly as indicated in the table, so each purity value falls into exactly one zone). Sharp transitions (not a gradient) â€” as in version 1.0, for instant categorical distinctness (Healey & Enns, 2012).

### 5.4. Grid axes: from 1D to 3D

Adding a feature turns an axis (1D) into a grid (2D), then into a cube (3D). When $d < 3$, the missing spatial axes collapse (similar to the `dim >= 2`/`dim >= 3` grid-building branches already in the current implementation): 1D view is a line of intervals, 2D is a flat grid, 3D is a full cube of discrete centers.

### 5.5. 4D: film strip metaphor (Film Strip)

When a 4-dimensional branch $S^*_4 = \{j_1,j_2,j_3,j_4\}$ is selected, the first three features are mapped to the spatial axes of the 3D cube, and the fourth is implemented via an **interactive stepper** (slice tabs) below the chart â€” not a separate spatial axis:

$$\text{Slice}(c) = \{(x_1, x_2, x_3) \mid X_{j_4} = c\}, \quad c \in \text{Dom}(X_{j_4})$$

A discrete stepper (not a continuous auto-animation by default) was chosen for resilience to Change Blindness and to maintain spatial axis constancy between slices â€” the reasoning is the same as in version 1.0. Optional auto-play (play/pause/speed) is available on top of the discrete slices, not replacing them.

The "All" tab â€” marginalization over the 4th dimension (aggregates all slices into one 3D view), allows comparing the overall picture with a single slice.

### 5.6. "Collapse/Split" animation between dimensions

**Important clarification regarding version 1.0, necessary due to branch independence (Section 4.4):** the dimensional transition animation in version 2.0 acts **within a single already selected branch**, not between independently found $S^*_d$ of different dimensions. These are two different interactions that must not be confused:

1. **Branch selection:** the user chooses which of the up to 4 independently found branches to view â€” for example, the top 3D branch $S^*_3 = \{A, B, C\}$. Switching between branches of different $d$ (for example, from $S^*_3$ to an independently found $S^*_2$, which could consist of entirely different features $\{D, E\}$) is an **instant scene rebuild** with new axes, not a collapse animation: the axes physically change, Object Constancy is inapplicable and is not claimed here.
2. **Dimensionality collapse within a branch:** after a branch is selected (e.g., $S^*_3 = \{A,B,C\}$), the user can visually "collapse" it to 2D (marginalization over $C$: aggregate over $\{A,B\}$) or to 1D (marginalization over $B,C$: aggregate over $\{A\}$), and "split" it back. **This** is the animation preserving Object Constancy, exactly as described below â€” it folds/unfolds the axes of the same branch, never swapping features.

Formally, for case (2), when removing axis $C$ from an already selected 3D branch $\{A,B,C\}$:

* **Collapse (3D â†’ 2D):** discrete centers do not disappear, but fall onto the $(A,B)$ plane. Centers with the same $(A,B)$ but different $C$ smoothly **merge**: masses (radii) are summed, colors are interpolated into a mixed brown ("mud") â€” visually demonstrating why dropping the axis loses separability.
* **Split (2D â†’ 3D):** the reverse process. A large muddy center in 2D splits along the returning $C$ axis into several purer centers â€” a visual "aha moment": adding a feature untangles the mixture.
* **Film Strip (3D â†” 4D):** transition to a slice â€” scenes slide in/out using a rewinding metaphor, as in Section 5.5.

All transitions preserve the relative sizes of the circles (Object Constancy, Heer & Robertson, 2007) â€” an invariant kept from version 1.0, crucial for the interpretability of the animation.

### 5.7. Canonical axis sorting and Object Constancy

Categorical axes do not have a natural metric order. The divisions of each axis are ordered by the conditional mean of the target variable:

$$\text{Score}(c) = \mathbb{E}[Z_{\text{canonical}} \mid X_j = c], \quad c \in \text{Dom}(X_j)$$

The geometric structure (order of axis divisions) is calculated once for the selected target with deterministic collision resolution (tie-break by cluster volume, then lexically) and remains fixed when switching the criterion **within the same target column** (e.g., `class = "p"` â†” `class = "e"`) â€” centers do not make parasitic movements, only color and size update. When changing the target column itself (switching the branch to a different target), the basis is recalculated.

---

## 6. Categorical Axis Order Diagnostics (Cluster-Preserving Categorical Ordering)

This section is fundamentally unchanged from version 1.0.

The system provides two complementary modes for placing labels on axes:

| Mode | Purpose | Method | Question |
|-------|-----------|-------|--------|
| **Cluster** (default) | Groups of categories with similar behavior | 1D Spectral Ordering / Correspondence Analysis on $p(\tilde{Z} \mid X_j)$ | "Which categories behave the same?" |
| **By impact (trend)** | Global impact gradient | Target-Conditioned Sort by $\mathbb{E}[Z \mid X_j=c]$ | "Which categories shift $Z$ more strongly?" |

The divergence is measured by Kendall's $\tau$ coefficient between the cluster ($\pi_{cluster}$) and trend ($\pi_{impact}$) orders:

$$\tau(X_j) = \frac{C - D}{\frac{1}{2} K_j (K_j - 1)}$$

Significant divergence ($\tau \ll 1$) signals non-linearity/multi-modality: the averaged estimate $\mathbb{E}[Z \mid X_j]$ is deceptive; there are hidden subgroups within categories.

---

## 7. Comparison with Existing Approaches

### 7.1. Direct conceptual competitor: Jeon et al. (IEEE TVCG 2025)

*"Dataset-Adaptive Dimensionality Reduction"* (Jeon et al., *IEEE TVCG*, 2025, DOI: 10.1109/TVCG.2025.3634784) adapts a **projection algorithm** (t-SNE/UMAP/PCA and their hyperparameters) for a fixed 2D scatterplot unconditionally (unsupervised), minimizing geometric distance distortions. VSF is a target-conditioned selection of the **meaningful composition of axes** $S^*_d \subseteq \{1,\dots,M\}$ for an explicitly given $Z$, on a discrete information-theoretic basis (MI), not continuous geometry. The approaches are orthogonal and do not compete directly.

### 7.2. Classical approaches

* **mRMR (Peng et al., 2005):** balance of relevance $I(X_j;Z)$ and redundancy. VSF version 2.0, unlike mRMR, does not solve the problem of selecting a single $K$ â€” instead, it explicitly materializes all 4 dimensions in parallel, removing the need for the user to choose $K$ at the cost of abandoning a statistical stopping criterion (see Section 4.5).
* **Scagnostics (Wilkinson et al., 2005):** 9 graph metrics for ready-made 2D projections (passive diagnostics). VSF is an active synthesizer, deriving axes prior to visualization.
* **Maximally Informative Dimensions (Sharpee et al., 2004):** continuous gradient optimization of projection by MI. VSF replaces it with discrete exact search, which for $d \le 4$ guarantees a global optimum (not local, unlike gradient descent), at the cost of exponential $M$ complexity (Section 4.6).

### 7.3. Summary comparison matrix

| System | Selection Method | Target-Conditioned? | Branches by Dim? | PMD | Statistical Criterion | Unit |
|---|---|:---:|:---:|:---:|:---:|---|
| **Voyager 2** (2017) | Perceptual rules | âŒ Partially | âŒ | âŒ | âŒ | Charts |
| **Scagnostics** (2005) | Graph metrics | âŒ | âŒ | âŒ | âŒ | Points |
| **mRMR** (2005) | MI, manual $K$ | âœ”ï¸ | âŒ | âŒ | âŒ | Features |
| **Jeon et al.** (2025) | Structural complexity | âŒ | âŒ | âŒ | âŒ | Points |
| **VSF v1.0** | Greedy + permutation test + FDR | âœ”ï¸ | âŒ (single $d^*$) | âœ”ï¸ | âœ”ï¸ | Discrete centers |
| **VSF v2.0 (current)** | Honest exhaustive search by MI, 4 independent branches | âœ”ï¸ | âœ”ï¸ (1Dâ€“4D parallel) | âœ”ï¸ | âŒ (deliberately removed, Sec. 4.5) | Discrete centers |

The loss explicitly reflected in the table compared to v1.0 â€” the lack of a statistical criterion â€” is not an accidental omission, it is Section 4.5.

---

## 8. Summary of Contributions (version 2.0)

| # | Contribution | Type |
|---|-------|-----|
| C1 | Independent Branch Discovery â€” exact (not approximate) search for up to 4 independent MI-optimal feature sets, one per dimensionality, without assuming nestedness | Algorithmic |
| C2 | Explicit formalization of MI non-monotonicity/supermodularity as the reason why a single greedy path fundamentally cannot find what independent searches find (Section 4.4) | Theoretical |
| C3 | Perceptually-Matched Discretization (PMD), carried over from v1.0 without essential changes | Theoretical |
| C4 | 4-zone (simplified from 5-zone) purity semantics of a discrete center with exact non-overlapping boundaries | Metric / UX |
| C5 | Collapse/split animation as explicitly demarcated from switching between independent branches (Section 5.6) â€” removes the ambiguity inherited from v1.0 where branches were nested by design | System / UX |
| C6 | Explicitly documented, accepted (not hidden) methodological trade-off of abandoning statistical significance control and its consequences (Section 4.5) | Methodological |

---

## 9. Evaluation Plan

This section is essentially preserved from version 1.0, adjusted for the removal of VIR-specific metrics.

### 9.1. Quantitative evaluation (synthetic data)

1. **Ground Truth Benchmark:** datasets with a known true structure of feature synergy (e.g., XOR-like dependencies, where no single feature is informative on its own, but a pair completely determines $Z$) â€” specifically to verify that Independent Branch Discovery finds such pairs, while the hypothetical v1.0 greedy algorithm does not.
2. **Metrics:** match of the found $S^*_d$ with the true synergistic combination; execution time of the exhaustive search as a function of $M$ (verifying the honesty of Section 4.6 estimates on real hardware).

### 9.2. Controlled User Study (minimum $n = 40$ participants)

Task: participants solve analytical tasks (finding clusters, outliers, separating rules) comparing VSF v2.0 (4 independent branches) against (a) VSF v1.0 (single $d^*$, if still available for comparison) and (b) expert manual axis selection. Hypothesis: parallel presentation of 4 branches does not increase cognitive load compared to a single v1.0 branch, while more frequently leading the user to discover synergistic feature combinations.

---

## 10. Open Problems and Limitations (version 2.0)

1. **Multiple comparisons effect without correction** â€” a direct consequence of Section 4.5, not resolved, accepted consciously.
2. **Scalability of exhaustive search** â€” datasets with $M \gtrsim 50$â€“100 features require tens of minutes for an honest search of all $\binom{M}{\le 4}$ combinations (Section 4.6); neither pre-filtering nor approximate search is allowed by the current specification.
3. **The fate of Auto-Discovery / cross-column insights catalog â€” resolved.** The user is still required to specify the target column themselves by default â€” there is no automatic scan on load. However, upon explicit user request, an optional, button-triggered **Global Pattern Scan** feature was added (only `vsf.serve()`, Section 4.7), which iterates through all (column, value) pairs in the dataset using the same honest Section 4.3 exhaustive search and **filters** (does not replace or substitute) the flat Target Selector by NMI threshold. Significance is still not tested â€” this is not a return of the permutation test or greedy $d^*$ of version 1.0, but a dataset-wide reformulation of the very same algorithm with an honestly disclosed (not approximated) cost, moved to a background thread with progress and cancellation due to its multiplicative cost.
4. **Ranking by raw $I$, rather than $NMI$** â€” consciously accepted per the direct problem formulation (Sections 4.2, 4.5), but does not correct for different category cardinalities among combinations of the same dimensionality; documented as a possible future refinement that does not block current implementation.

---

## Key References

* Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal.*
* Miller, G. A. (1956). The Magical Number Seven, Plus or Minus Two. *Psychological Review.*
* Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica.*
* Fayyad, U. & Irani, K. (1993). Multi-Interval Discretization of Continuous-Valued Attributes (MDLP). *IJCAI.*
* Peng, H. et al. (2005). Feature Selection Based on Mutual Information: mRMR. *IEEE TPAMI.*
* Krause, A. & Guestrin, C. (2005). Near-optimal sensor placements in Gaussian processes. *ICML.* â€” justification for MI non-submodularity (Section 4.4), no longer as a guarantee of the greedy algorithm
* Wilkinson, L. et al. (2005). Graph-Theoretic Scagnostics. *IEEE InfoVis.*
* Seo, J. & Shneiderman, B. (2005). Rank-by-Feature Framework. *IEEE InfoVis.*
* Ware, C. (2004). Information Visualization: Perception for Design. *Morgan Kaufmann.*
* Munzner, T. (2014). Visualization Analysis and Design. *CRC Press.*
* Healey, C. G. & Enns, J. T. (2012). Attention and Visual Memory in Visualization. *IEEE TVCG.*
* Wongsuphasawat, K. et al. (2017). Voyager 2: Augmenting Visual Analysis with Partial View Specifications. *ACM CHI.*
* Jeon, H., Park, J., Lee, S., Kim, D. H., Shin, S., & Seo, J. (2025). Dataset-Adaptive Dimensionality Reduction. *IEEE TVCG*. DOI: 10.1109/TVCG.2025.3634784.
* Sharpee, T., Rust, N. C., & Bialek, W. (2004). Analyzing neural responses to natural signals: maximally informative dimensions. *Neural Computation*, 16(2), 223-250.
* Cleveland, W. S. (1993). Visualizing Data. *Hobart Press.*
* Heer, J. & Robertson, G. (2007). Animated Transitions in Statistical Data Graphics. *IEEE TVCG.*
* Tufte, E. R. (1983). The Visual Display of Quantitative Information. *Graphics Press.*

**Removed from the version 1.0 references list as no longer cited in the text:** Good (2005), Runge et al. (2018), Krause et al. (2008), Elenberg et al. (2018), Borland & Taylor (2007) â€” all were specific to permutation testing, FDR, or the weak submodularity guarantee of the greedy algorithm, none of which are part of the core anymore.
