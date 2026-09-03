# Visual Sufficiency Framework (VSF): Independent-Branch Mutual-Information Visualization

**Formal description of the algorithm and mathematical apparatus**

**Version 2.2 — "Certified Centers".** Version 2.0 was a deliberate simplification of the project relative to version 1.0 (preserved in the repository history); Section 0 documents that reduction. Version 2.1 left the product scope of 2.0 untouched and corrected its *estimator*: the plug-in mutual-information estimator 2.0 reported and ranked by is positively biased by an amount that grows with the cardinality of the joint support, which made both the ranking and the reported percentage unsound. Section 0-bis documents that correction, with the control experiment that forced it. Version 2.2 corrects the *reported quantity*: a bias-corrected association statistic is the right answer to "is there a relationship" and the wrong answer to "which cells are almost purely the target value", which is what this product exists to show. Section 0-ter documents that correction, with the case that forced it; Section 3.3 gives the replacement.

---

## 0. Revision Note: what changed and why

Version 1.0 of this document described a system with seven-channel ($d_{max}=7$) visualization, greedy feature selection with permutation tests of significance and FDR correction at each step, four rendering scenarios (A/B/C/D), an end-to-end Auto-Discovery engine across all dataset columns (Section 4.5 v1.0), interactive mining of conjunctive filters for "dirty centers" (Section 9.2.8 v1.0), and also — outside this document, in the project code — a composite AND-filter for the target, a Graph Inference / Knowledge Base engine, and multi-scenario dashboard export.

By direct decision of the project owner (2026-08-28), the product scope was reduced to a **single simple core**: for a user-selected target variable $Z$ (one column, optionally — one criterion-value), the system finds **up to 4 independent branches** — one for each dimensionality $d \in \{1,2,3,4\}$ — and visualizes them as grids of discrete centers (circles), with "collapse/split" animation between dimensions. Everything that does not serve this single purpose was removed, rather than left "just in case" behind a flag.

**Explicitly removed from the core (was in v1.0 or in adjacent code modules not described in this document):**

| Removed | Was | Reason for removal |
|---|---|---|
| Rendering scenarios A/B/C/D and routing by $d^*$ | Section 6 v1.0 | Built around permutation stopping of greedy selection, which no longer exists — there is no single "optimal" $d^*$, there are 4 parallel branches |
| Greedy forward selection + permutation significance tests + FDR (Benjamini-Hochberg) | Sections 4.2, 4.4 v1.0 | Replaced by honest exhaustive search with ranking by "raw" MI — see Section 4.5 below, where the cost of this decision is explicitly stated |
| Weak submodularity guarantee of the greedy algorithm (Krause et al., 2008) | Section 5.1 v1.0 | Inapplicable: the algorithm is no longer greedy |
| Universal Propositional Screening / Auto-Discovery across all columns, global distillation basis | Section 4.5 v1.0 | A separate task "find interesting Z across the whole dataset", not part of the defined core "given Z → show 4 branches". **This decision was made by me by analogy with other removals (significance testing, which Auto-Discovery entirely relied on), and not directly confirmed by the user** — see the explicit assumption note below |
| Interactive mining of "dirty centers" (Conjunctive Filter Explorer) | Section 9.2.8 v1.0 | Explicitly named by the user among the items to remove |
| Composite AND-filter of the target (`composite_target` in `/api/analyze`) | Code (`server.py`, `app.js`), not described in this document | Explicitly named by the user among the items to remove |
| Graph Inference / Knowledge Base (`graph.html`, `graph_reasoning.js`) | Code, separate HTML | Explicitly named by the user among the items to remove |
| Multi-scenario dashboard export (`export_full_dashboard(targets=...)`) | Code (`vsf/dashboard.py`) | Explicitly named by the user among the items to remove |
| $d_{max} = 7$ and Hue/Saturation/Lightness channels | Sections 1.3, 2.2 v1.0 | Replaced by $d_{max} = 4$ (3 spatial axes + 1 temporal/frame axis) — this is the actual state of affairs already in the current visualization implementation, not a new restriction |

**Explicitly preserved without substantive changes (moved here, often with cosmetic editing):** perceptually-aligned discretization (PMD), the MI mathematical core (its *estimator* was subsequently corrected in v2.1 and $NMI$ removed — see Section 0-bis), the Discrete Centers paradigm with mass encoding via area, 4-zone (was 5-zone) purity semantics, film strip metaphor for 4D, animated transitions between dimensions, canonical target-conditioned axis sorting and Object Constancy, diagnostics of the divergence between cluster/trend ordering of categories.

> **Assumption requiring confirmation:** the removal of Auto-Discovery/Universal Propositional Screening (Section 4.5 v1.0) and the associated "Insights catalog across all columns" — is my decision by analogy, and not directly requested by the user. Functionally, this means that the user must **explicitly specify the target column $Z$** (and, optionally, a criterion-value) themselves — the system no longer scans the entire dataset for "interesting" targets automatically. If this is a misinterpretation — this is the only point where it's worth going back and clarifying before starting implementation.

---

---

## 0-bis. Revision 2.1 "Corrected Core": the estimator, not the product

Version 2.0's simplification (Section 0) removed the statistical machinery but kept the *plug-in* (maximum-likelihood) mutual-information estimator and treated its output as if it were the quantity it estimates. That does not hold. Under exact independence the plug-in estimator's expectation is approximately

$$\mathbb{E}_0[\hat I] \;\approx\; \frac{(R-1)(C-1)}{2 N \ln 2}, \qquad R = |\mathrm{supp}(\tilde Z)|,\; C = |\mathrm{supp}(\tilde X_S)|$$

(Miller & Madow, 1955), and Section 2.3's Grid Capacity Limit permits $C$ up to $N/10$. On the reference Adult dataset ($N = 32{,}561$) that puts the **noise floor of a 4D branch at $\approx 0.08$ bits** — an order of magnitude above the "$<0.01$ bits is noise" reading that an absolute MI in bits invites.

Two v2.0 decisions do not survive that fact.

**(a) Ranking by raw $I$ selects on cardinality, not on association.** The bias term grows with $C$, so $\arg\max_S \hat I$ is pulled toward whichever combination has the most cells. On a control dataset in which *every* feature is generated independently of the target ($N = 32{,}561$; feature cardinalities $2, 3, 5, 10, 50, 200$), v2.0's search returns:

| $d$ | selected | $\hat I$ (bits) | $\hat I / H(\tilde Z)$ |
|:---:|---|---:|---:|
| 1 | `k=200` | 0.0047 | 0.6 % |
| 2 | `k=50`, `k=200` | 0.0677 | 8.6 % |
| 3 | `k=3`, `k=5`, `k=200` | 0.0778 | 9.8 % |

The true mutual information is exactly zero at every row. This is a defect of **selection**, not of display, and no change to a normalization denominator affects it.

**(b) $NMI_{\min}$ is not a reportable percentage.** For a rare target class $H(\tilde Z) \to 0$ while the numerator keeps its full positive bias, so the ratio saturates. Measured on a 7-in-32,561 target class at the capacity ceiling, an independently generated feature grid scores $NMI_{\min} = 66\,\%$. Section 3.2 gives the full comparison, including why replacing $\min$ with a geometric mean does not repair it.

**What v2.1 changes.**

| Changed | v2.0 | v2.1 |
|---|---|---|
| Branch ranking (Section 4.2) | $\arg\max \hat I$ | $\arg\max \hat I_{\mathrm{adj}} = \hat I - \mathbb{E}_0[\hat I]$, with $\mathbb{E}_0$ the **exact** permutation expectation (Vinh, Epps & Bailey, 2010) |
| Reported percentage (Section 3.2) | $NMI_{\min}$ | $U_{\mathrm{adj}}$, the bias-corrected uncertainty coefficient |
| Significance (Section 4.5) | none | permutation $p$-value per branch; optional look-elsewhere-corrected $p$-value over the whole $C(M,1{..}4)$ family |
| Global Pattern Scan filter (Section 4.7) | $\max NMI >$ threshold | $\max U_{\mathrm{adj}} >$ threshold **and** Benjamini–Hochberg FDR control across every scanned pair |
| Distortion $\mathcal{D}_j$ (Section 2.2) | $1 - NMI_{\min}$, non-monotone in $k$ | $1 - I/H(X^{cont})$, monotone non-increasing in $k$ |
| Per-class reporting | none | exact decomposition $I = \sum_z p(z)\, D_{KL}(p(x|z)\,\|\,p(x))$ |

**What v2.1 does not change.** The product remains "given $Z$, show up to 4 independent branches". The search remains exhaustive and never approximated. No greedy selection, no scenario routing, no $d^*$ returns. The corrections are confined to the estimator and to the claims attached to its output.

**What v2.1 still does not claim.** Every estimate is in-sample: a significant $\hat I_{\mathrm{adj}}$ establishes that an association exists in the observed contingency table, not that it generalizes. Out-of-sample predictability requires a held-out split, which is deliberately outside this specification.

## 0-ter. Revision 2.2 "Certified Centers": the reported quantity, not the estimator

Version 2.1 corrected the *estimator* and left the *reported quantity* mismatched to the product. $U_{\mathrm{adj}}$ answers "is there an association between $\tilde Z$ and the cell partition". Sections 1.1 and 5.1 state that the deliverable is narrower: **a small number of cells that are almost purely the target value.** On a rare target those two questions have opposite answers, and v2.1 printed the answer to the first one in the position reserved for the second.

**The case that forced the revision.** Fully reproducible from `data/adult_census.csv` ($N = 32{,}561$), with $\tilde Z = \mathbb{1}[\texttt{occupation} = \texttt{Armed-Forces}]$, $N_+ = 9$, prevalence $0.0276\,\%$:

| $d$ | branch | $\hat I$ | $\mathbb{E}_0[\hat I]$ | $U_{\mathrm{adj}}$ | max cell purity | certified centres $K$ | coverage |
|:---:|---|---:|---:|---:|---:|---:|---:|
| 3 | `workclass + sex + income` | 0.0017 | 0.0003 | **41.3 %** | 2.42 % (8/330) | **0** | **0 %** |
| 4 | `workclass + relationship + race + sex` | 0.0020 | 0.0007 | **44.4 %** | 100 % (1/1) | **0** | **0 %** |

$U_{\mathrm{adj}} = 41.3\,\%$ is arithmetically correct and operationally empty. $H(\tilde Z) = 0.00367$ bits, so the branch resolves $0.0015$ bits of an already almost-zero uncertainty. In the same branch: the highest purity among all 32 occupied cells is $2.42\,\%$; eight of the nine positives sit in that single cell of 330 samples; and the Bayes rule under 0–1 loss never predicts the target class, so Goodman–Kruskal $\lambda = 0$ **exactly**. No choice of colour boundary can make that display green, and the panel's headline said 41.3 %.

The 4D row is the same failure from the other side: its "max cell purity" of 100 % is one cell holding one person. A point purity of $1/1$ was drawn as a fully saturated green centre, and the interface's answer was a *"Noise Reduction (minimum samples)"* slider — a hand-set integer with no stated error rate, hiding a defect of the estimator behind a display filter.

**Three defects, all present simultaneously in v2.1.**

1. **The headline measured the wrong thing.** $U_{\mathrm{adj}}$ is a share of $H(\tilde Z)$; both its numerator and its denominator collapse toward zero for a rare target, and their ratio carries no information about whether any cell is usable.
2. **The panel's $N$ was not the statistics' $N$.** `prepare_visualization_payload` subsamples to `max_display_samples = 10 000` rows for rendering and reported that count as "Total Samples", while `vsf.avr` computed every metric on all 32 561. Worse, cell purities were computed on the subsample: three of the nine Armed-Forces positives are absent from it entirely, so the purity drawn on screen was a different quantity from the one behind the reported number. Neither figure was wrong in isolation; their combination was not reproducible.
3. **Colour encoded a point estimate.** The v2.0/v2.1 bands $0.25 / 0.75 / 0.85$ (Section 5.3) had no derivation and could not distinguish $1/1$ from $300/300$.

**What v2.2 changes.**

| Changed | v2.1 | v2.2 |
|---|---|---|
| Reported headline (Section 3.3) | $U_{\mathrm{adj}}$ | **(coverage, $K$, pooled purity)**; $U_{\mathrm{adj}}$ demoted to a labelled diagnostic |
| Definition of a green centre (Section 5.3) | point purity $> 0.85$, fixed | point purity $\ge \tau$, **with $\tau$ set by the user** ($\tau = 1$ permitted); a strict variant certifies it against an exact binomial test |
| Colour boundaries (Section 5.3) | $0.25 / 0.75 / 0.85$, underived, 4 zones including yellow | **two user-movable boundaries, 3 zones, no yellow**; the upper one is $\tau$ itself, so what is green and what is counted are the same set |
| Outlier handling | "minimum samples" display filter | `min_samples`, part of the reported definition, default 1 (no restriction), owned by the user |
| $N$ used for cell statistics | render subsample | **all rows**; the subsample now affects only which spheres are drawn |
| Search objective (Section 4.3) | $\hat I_{\mathrm{adj}}$ | **coverage by default** (`objective="auto"`), falling back to $\hat I_{\mathrm{adj}}$ only when the target has no declared positive value |
| Global Pattern Scan filter (Section 4.7) | $\max U_{\mathrm{adj}} >$ threshold | $\max$ **coverage** $>$ threshold, with the same BH control |
| Out-of-sample claim | none | cross-validated coverage with Nadeau–Bengio corrected variance (Section 3.3) |
| Dimensionality question | unanswered | **smallest sufficient $d$** by out-of-sample coverage (Definitions 8, 12) |

**Why the colour boundaries are the user's and not the framework's.** The first v2.2 draft derived them from the confidence bound: a cell was green iff its simultaneous lower bound reached $\tau$. That is the stronger statistical object and it is retained as `rule="certified"`. It is not the default, for a reason that is a product argument rather than a statistical one: the boundary the analyst moves *is* the definition of the finding they are looking for, and a boundary computed for them out of $(k_\mathbf{c}, n_\mathbf{c}, \alpha, C)$ cannot be moved. Under either rule the green cells on screen and the `coverage` in the panel are the same set by construction; under the default rule the user also controls where the line falls. Both are exposed; the default is the one that answers the question the user is asking.

**What v2.2 does not change.** The product remains "given $Z$, show up to 4 independent branches". The search remains exhaustive and never approximated. $\hat I_{\mathrm{adj}}$ and $U_{\mathrm{adj}}$ are not removed and Section 3.2 stands unchanged: they remain the correct answer to "does an association exist between the target and this partition", which is a real question and the only one available when the target has no declared positive value. What changed is that it is no longer mistaken for the product's own question, and no longer ranks the search when the product's own question is answerable.

**What v2.2 still does not claim.** Two exposures remain, and both are measured rather than argued away.

*Selection optimism.* Centres are chosen by inspecting their own contents, so the in-sample coverage is optimistic even when every individual cell is honest. On a constructed case with true purities straddling $\tau$ (40 cells of 300 objects, $\pi(\mathbf{c}) \sim U[0.85, 0.95]$): in-sample $58.9\,\%$, cross-validated $54.7 \pm 2.5\,\%$. Any coverage quoted in a publication must be the cross-validated one, with the look-elsewhere-corrected $p$-value of Definition 11.

*The price of `min_samples = 1`.* With one-object cells eligible, a few of them come up entirely target-valued by chance. Measured on pure noise at the worst grid density this specification permits (3 000 cells over 32 561 rows, target independent of every cell, $\tau = 0.90$): $K = 2$, coverage $0.09\,\%$, permutation $p = 0.23$. The price is real, it is small, and the permutation test does not mistake it for a finding. Both remedies drive it to exactly zero — `min_samples = 20`, or `rule="certified"` — and the specification reports the number rather than choosing for the user.

## 1. Problem Formulation

### 1.1. Concept and intuitive definition

> *"Given a dataset and a target variable (or criterion-value within it). The system finds up to 4 best independent sets of features — one for each dimensionality from 1 to 4 — and shows where the target class occurs frequently, where it is rare, and where the situation is ambiguous, with circles within a grid of categories."*

The key word is **independent**: the best set of features for 3D does not have to contain the best set for 2D. Mutual information is not submodular in the general case (Krause & Guestrin, 2005) — feature synergy means that the pair $\{A, B\}$ can beat the pair $\{C, D\}$ in explaining $Z$, even if $A$ alone is stronger than $C$. Hence — 4 parallel "development branches", rather than a single chain of "adding one best feature at a time".

### 1.2. Formal definition

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, z_i)\}_{i=1}^{N}$, where $\mathbf{x}_i \in \mathbb{R}^M$ is a feature vector, $z_i$ is the target variable (one dataset column; optionally binarizable by a criterion-value, see Section 1.4), $M$ is the full dimensionality.

**Task:** for each dimensionality $d \in \{1, 2, 3, 4\}$ independently find

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; I(\tilde{Z}; \tilde{X}_S)$$

The result is up to 4 **branches** $(d, S^*_d, I(\tilde{Z}; \tilde{X}_{S^*_d}))$, $d = 1, \dots, \min(4, M)$. No single "optimal $d^*$" is computed or selected by the algorithm — the choice of dimensionality is given to the user (see Section 6).

### 1.3. Visual channels

The display provides **4 independent encoding channels**, without claiming the 7-channel perceptual limit of version 1.0 (this is the actual state of affairs already in the current rendering implementation, not a new restriction):

$$\mathcal{V} = \{X, Y, Z_{depth}, T\}$$

The first three channels are the spatial axes of the 3D cube; the fourth is not an independent visual axis, but a **temporal/frame channel** (see Section 5.5): the values of the 4th feature turn into a stack of parallel frames ("film strip"), between which the user switches manually (stepper) or starts auto-play (play/pause/speed).

Circle color and size in this version are **not independent feature selection channels**, but the encoding of aggregate statistics of the already selected branch (the proportion of the target class and the number of objects in the cell, respectively, see Section 5.3) — they do not participate in the search for $S^*_d$.

### 1.4. Target variable and criterion

The user sets the target variable in one of two ways, identical to the already existing `target`/`criterion` semantics:

1. **Raw multiclass target:** $Z = X_{target}$ as is (for example, the entire `class` column with its original values).
2. **Binary criterion:** a specific value $v$ of one column, $Z = \mathbb{I}[X_{target} = v]$ (for example, `class = "p"`).

A composite AND-criterion across several columns (was in v1.0 as `composite_target`) is not included in the core — see Section 0.

---

## 2. Perceptually-Aligned Discretization (PMD)

### 2.1. Theoretical justification: optimal quantization problem

The visual output of any display is discrete by design: a circle occupies a finite area of a grid cell, the number of cells is limited. Quantization (binning) of a continuous feature $X_j$ into $k$ bins is a lossy operation (Rate-Distortion Theory, Shannon, 1959). PMD formulates this problem as finding the minimum $k$ at which the distortion of the data structure does not exceed an acceptable threshold, determined by the perceptual sensitivity of the channel.

### 2.2. Formalization: distortion function

**Definition 1 (Distortion Function).**

$$\mathcal{D}_j(k) = 1 - \frac{I\big(\tilde{X}_j^{(k)};\ X_j^{cont}\big)}{H\big(X_j^{cont}\big)}$$

where $\tilde{X}_j^{(k)}$ is the discretized version of the feature in $k$ bins and $X_j^{cont}$ is the continuous original (in practice, a fine reference quantization of at most 200 levels). $\mathcal{D}_j(k) \in [0,1]$, is non-increasing in $k$, and is $0$ exactly when the coarse code is lossless with respect to the reference.

> **Changed in v2.1.** The definition was previously $1 - NMI_{\min}(\tilde{X}_j^{(k)};\ X_j^{cont})$, which is not a rate–distortion quantity: $\tilde{X}_j^{(k)}$ is (up to bin-edge ties) a deterministic function of $X_j^{cont}$, so $I = H(\tilde{X}_j^{(k)})$ and the ratio collapses to $\approx 1$ whenever $H(\tilde{X}_j^{(k)}) \le H(X_j^{cont})$ — leaving $\mathcal{D}_j$ driven entirely by residual tie noise. Measured consequence: for a standard normal feature the old formula returned $\mathcal{D}_j = 0.034$ at $k = 12$ but $\mathcal{D}_j = 0.121$ at $k = 101$ — *more* bins scoring as *more* distortion, inverting the monotonicity Section 2.3 relies on when it trades $L_v$ against $\mathcal{D}_j$. Normalizing by the information actually available to be preserved, $H(X_j^{cont})$, restores it.

| Channel $v$ | Type | $L_v$ (levels) | Role in core |
|-----------|-----|:---:|---|
| Position X / Y / Depth Z | Spatial | ~10–20 (limited by grid size, not screen) | Branch axis |
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

When computing joint entropy for a subset of features $S$ ($|S| \leq 4$), the total number of cells $\prod_{j \in S} k_j$ must not exceed $N / 10$ (where $N$ is the number of rows). This limit is **especially important in the new version**: the exhaustive search (Section 4) computes MI for each of the potentially millions of feature combinations (Section 4.6), and without adaptive bin merging, a portion of these combinations would yield degenerate, statistically unreliable MI estimates simply due to contingency table sparsity — which would distort branch ranking, not just visualization. If the threshold is exceeded, the bins are adaptively merged before calculating MI for this specific combination.

### 2.5. Human-in-the-Loop

As in version 1.0: the algorithm does not assign semantic categories to the target variable autonomously. The user sets or confirms the target column and (optionally) the criterion-value; for continuous features, the system suggests an MDL-recommended split, without applying it without confirmation.

---

## 3. Mathematical Core: Discrete Mutual Information

This section has not changed essentially relative to version 1.0 — the foundation remains the same.

### 3.1. Basic definitions

**Shannon Entropy:**

$$H(\tilde{X}) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

**Joint Entropy:**

$$H(\tilde{Z}, \tilde{X}_S) = -\sum_{z, x_S} p(z, x_S) \log_2 p(z, x_S)$$

**Mutual Information of subset $S$ with target $Z$:**

$$I(\tilde{Z}; \tilde{X}_S) = H(\tilde{Z}) + H(\tilde{X}_S) - H(\tilde{Z}, \tilde{X}_S)$$

### 3.2. Reported association metrics

$\hat I$ above is the plug-in estimator. It is biased upward, by an amount that grows with the cardinality of the joint support, so neither $\hat I$ nor any ratio built from it may be reported or compared across feature subsets without correction (Section 0-bis).

**Definition 2 (Null expectation).** Conditional on both margins of the contingency table, the permutation null of the table is the multiple hypergeometric distribution. Its induced expectation of the plug-in estimator,

$$\mathbb{E}_0[\hat I] = \sum_{i}\sum_{j}\ \sum_{n_{ij}=\max(1,\,a_i+b_j-N)}^{\min(a_i,\,b_j)} \frac{n_{ij}}{N}\log_2\!\frac{N\,n_{ij}}{a_i b_j}\cdot \frac{a_i!\,b_j!\,(N-a_i)!\,(N-b_j)!}{N!\,n_{ij}!\,(a_i-n_{ij})!\,(b_j-n_{ij})!\,(N-a_i-b_j+n_{ij})!}$$

is available in closed form (Vinh, Epps & Bailey, JMLR 11 (2010), 2837–2854, eq. 24a) and is computed exactly — not by Monte Carlo — in $O\big(\sum_{ij}\min(a_i,b_j)\big)$. It depends only on the margins, which permutation preserves.

**Definition 3 (Adjusted mutual information, in bits).**

$$\hat I_{\mathrm{adj}}(\tilde Z; \tilde X_S) = \hat I(\tilde Z; \tilde X_S) - \mathbb{E}_0\big[\hat I(\tilde Z; \tilde X_S)\big]$$

This is the ranking statistic (Section 4.2). Because $\mathbb{E}_0$ absorbs the cardinality-dependent bias, $\hat I_{\mathrm{adj}}$ is comparable across candidate subsets of unequal cell count and across dimensionalities; raw $\hat I$ is not.

**Definition 4 (Adjusted uncertainty coefficient).**

$$U_{\mathrm{adj}}(\tilde Z \mid \tilde X_S) = \frac{\hat I - \mathbb{E}_0[\hat I]}{H(\tilde Z) - \mathbb{E}_0[\hat I]} \in [0, 1]$$

This is the reported percentage. It is **directional**: the denominator is the target's entropy, not a symmetric combination of $H(\tilde Z)$ and $H(\tilde X_S)$, because the question a percentage is asked to answer here is "what share of the target did these axes explain", not "how similar are these two partitions".

#### Why not $NMI$

Three normalizations were evaluated on a deliberately adversarial case: $N = 32{,}561$, a target class of 7 members, and a feature grid at the Grid Capacity Limit ($C = 3{,}256$ cells). Two inputs are compared — one with no relationship at all, one in which the target is a deterministic function of the cell index.

| | $\hat I$ | $\mathbb{E}_0[\hat I]$ | $NMI_{\min}$ | $NMI_{\mathrm{geo}}$ | $U_{\mathrm{adj}}$ | $p$ |
|---|---:|---:|---:|---:|---:|---:|
| independent | 0.0019 | 0.0019 | **66 %** | 1.0 % | 2 %\* | 0.34 |
| deterministic | 0.0029 | 0.0019 | 100 % | **1.6 %** | 100 % | 0.005 |
| real signal, $p(z)=0.24$ | 0.616 | 0.004 | 78 % | 25 % | 78 % | 0.005 |
| independent, $p(z)=0.24$ | 0.081 | 0.081 | 10 % | 2.8 % | 0 % | 0.62 |

* $NMI_{\min} = \hat I / \min(H(\tilde Z), H(\tilde X_S))$ — the v2.0 definition. For a micro-class $\min(\cdot)$ always selects the near-zero $H(\tilde Z)$, and the biased numerator saturates against it. Reads 66 % on data with no relationship. Rejected.
* $NMI_{\mathrm{geo}} = \hat I / \sqrt{H(\tilde Z) H(\tilde X_S)}$ — the standard clustering-comparison normalization (Strehl & Ghosh, 2002). It does not inflate, and that is precisely the trap: it also does not *respond*. Noise reads 1.0 % and a deterministic relation reads 1.6 %, so no threshold can be placed between them. Dividing every number by the same large constant is not a bias correction. Rejected.
* $U_{\mathrm{adj}}$ — 2 % and 100 % on the same two inputs, a separation of a factor of 50 where $NMI_{\mathrm{geo}}$ manages 1.6. Adopted.

\* Not 0 %: see the residual limitation below. On a *balanced* target the same measurement gives 0.06 %. The 2 % is the price of a 7-member positive class, not of the metric.

The general lesson, and the reason the geometric mean is the wrong repair: the defect lives in the **estimator**, so it must be removed from the numerator. Changing the denominator can only rescale it.

**Honest residual limitation.** For a micro-class the adjusted denominator $H(\tilde Z) - \mathbb{E}_0$ is itself small, so a chance fluctuation in the numerator still moves $U_{\mathrm{adj}}$ by a few percent under the null (measured: $\approx 2\,\%$ at $N_{+} = 7$). For target classes with fewer than $\sim 50$ members the **$p$-value, not the effect size, is what settles the question**. This is a power limitation of the data, not of the metric, and it is why Section 4.5 makes the permutation test mandatory rather than optional.

#### Per-class decomposition

A single scalar cannot say whether a headline number is carried by the class the user cares about or entirely by the majority class. The exact additive decomposition (DeWeese & Meister, 1999)

$$I(\tilde Z; \tilde X_S) = \sum_z p(z)\, D_{KL}\big(p(x_S \mid z)\ \big\|\ p(x_S)\big)$$

is reported alongside, per class: prevalence, contribution in bits, and share of the total. Because the terms are weighted by $p(z)$, a class of 0.1 % prevalence contributes under 1 % of the total regardless of how predictable it is — which is the precise, quantitative form of the observation that a global MI answers a different question from "can I identify members of *this* class".

---

### 3.3. Reported deliverable metrics: certified discrete centers

Section 3.2's quantities answer *"is $\tilde Z$ associated with the partition induced by $\tilde X_S$"*. This section defines what the product actually reports, which is the answer to a different question: *"which cells of that partition are almost purely the target value, and how much of the target do they account for"*. Section 0-ter gives the case in which the two answers diverge completely.

Throughout, one target **value** $z^\ast$ is fixed and $\tilde Z = \mathbb{1}[Z = z^\ast]$ — the One-vs-Rest criterion of Section 1.4. A purity has no meaning without a designated positive value; for a $K$-valued target with no criterion, this whole section is undefined and the implementation reports it as such rather than defaulting to an arbitrary class.

#### Definitions

**Definition 5 (Cell purity).** For an occupied cell $\mathbf{c}$ holding $n_\mathbf{c}$ objects of which $k_\mathbf{c}$ carry $z^\ast$,

$$\pi(\mathbf{c}) = \Pr\big[Z = z^\ast \mid \tilde X_S \in \mathbf{c}\big], \qquad \hat\pi(\mathbf{c}) = k_\mathbf{c} / n_\mathbf{c}.$$

**Definition 6 (Discrete center).** Fix a purity floor $\tau$, a minimum occupancy $m \ge 1$, and a selection rule. Let $C$ be the number of *occupied* cells of the branch.

*Rule P (purity — the default).* Cell $\mathbf{c}$ is a center iff

$$\hat\pi(\mathbf{c}) = \frac{k_\mathbf{c}}{n_\mathbf{c}} \;\ge\; \tau \qquad\text{and}\qquad n_\mathbf{c} \ge m, \qquad \tau \in (0, 1].$$

This is a statement about the observed table, so $\tau = 1$ is admissible and means "cells that are entirely the target value" — the request a user actually makes. $\tau$ and $m$ are the user's parameters, exposed directly in the interface, and $\tau$ is simultaneously the green colour boundary (Section 5.3), so the set drawn green and the set entering Definition 7 are identical by construction rather than by convention.

*Rule C (certified).* Cell $\mathbf{c}$ is a center iff the one-sided exact binomial test of

$$H_0: \pi(\mathbf{c}) \le \tau \qquad\text{against}\qquad H_1: \pi(\mathbf{c}) > \tau$$

rejects at level $\alpha / C$, i.e. iff

$$\Pr\big[\mathrm{Bin}(n_\mathbf{c}, \tau) \ge k_\mathbf{c}\big] \;\le\; \alpha / C
\qquad\Longleftrightarrow\qquad
L_{\mathrm{CP}}\big(k_\mathbf{c}, n_\mathbf{c};\, \alpha/C\big) \;\ge\; \tau,$$

where $L_{\mathrm{CP}}(k,n;\alpha) = \mathrm{Beta}^{-1}\big(\alpha;\, k,\, n-k+1\big)$ is the Clopper–Pearson lower confidence bound ($L_{\mathrm{CP}} = 0$ for $k = 0$). The equivalence is the standard binomial–beta duality; the implementation evaluates the tail for the decision and the quantile for the displayed bound, and the test suite pins that the two agree cell by cell.

Both rules feed the identical Definition 7, so every downstream quantity — coverage, $K$, the cross-validated estimate, the permutation nulls, the reported dimensionality — is written once and is comparable *in kind* between rules, though not in value. **Under either rule the interval $[L_{\mathrm{CP}}, U_{\mathrm{CP}}]$ is computed and displayed for every cell.** Under Rule C it decides; under Rule P it is information attached to the cell, which is what makes Rule P's weakness visible on the object rather than arguable in the abstract: a cell that is green on one object shows the interval $[\alpha/C,\, 1]$.

**Proposition 2 (Simultaneous validity of Rule C).** Let $\mathcal{N} = \{\mathbf{c} : \pi(\mathbf{c}) \le \tau\}$ be the set of cells that do *not* deserve certification. Under Rule C,

$$\Pr\big[\exists\, \mathbf{c} \in \mathcal{N} \text{ selected}\big] \;\le\; \sum_{\mathbf{c}\in\mathcal{N}} \frac{\alpha}{C} \;\le\; \alpha .$$

*Proof.* Each test has exact level $\le \alpha/C$ under its own $H_0$ (Clopper–Pearson is conservative, never anti-conservative); a union bound over $|\mathcal{N}| \le C$ tests gives the claim. $\square$

Bonferroni is used rather than a sharper simultaneous procedure for two reasons. First, the cell counts are multinomial and hence negatively associated, so the union bound is conservative — the correct direction for a certificate. Second, the family is not the selected cells but *every occupied cell*: the analyst looks at the whole lattice and reads the green ones, so the multiplicity is $C$ regardless of how many turn green.

**Corollary 1 (100 % purity: observable, not certifiable).** Under Rule P, $\tau = 1$ selects exactly the cells with $k_\mathbf{c} = n_\mathbf{c}$. Under Rule C no cell is ever selected at $\tau = 1$ at any sample size, since $H_0: \pi \le 1$ can never be rejected; `CenterSpec` therefore rejects $\tau = 1$ under Rule C rather than clamping it, so that a request to *prove* exact purity fails loudly instead of silently answering a different question. The distinction is the whole difference between the two rules stated in one line: Rule P describes the table, Rule C infers about the population.

**Corollary 2 (what $m = 1$ costs, and what removes it).** Under Rule P with $m = 1$, a cell holding a single target-value object has $\hat\pi = 1$, is a center, and contributes $1/N_+$ to coverage. Its interval is $[\alpha/C, 1]$. Measured on pure noise at the density ceiling of Section 2.4 (3 000 cells, $N = 32{,}561$, target independent of every cell, $\tau = 0.90$): $K = 2$, coverage $0.09\,\%$, permutation $p = 0.23$ — small, and not significant. Setting $m = 20$, or switching to Rule C, gives $K = 0$ exactly. Under Rule C the parameter is redundant in any case: $L_{\mathrm{CP}}(1,1;\alpha) = \alpha$, and the smallest fully pure cell selectable at $\tau = 0.90$, $\alpha = 0.05$, $C = 1$ is $n = 29$, since $\alpha^{1/n} \ge \tau \iff n \ge \ln\alpha / \ln\tau = 28.4$.

**Definition 7 (Deliverable triple).** Let $\mathcal{G}$ be the set of centers selected by Definition 6, $N_+ = \sum_\mathbf{c} k_\mathbf{c}$ the total number of target-value objects, and $N$ the number of rows.

$$K = |\mathcal{G}|, \qquad
\mathrm{Coverage} = \frac{\sum_{\mathbf{c}\in\mathcal{G}} k_\mathbf{c}}{N_+}, \qquad
\mathrm{Purity}_{\mathrm{pooled}} = \frac{\sum_{\mathbf{c}\in\mathcal{G}} k_\mathbf{c}}{\sum_{\mathbf{c}\in\mathcal{G}} n_\mathbf{c}} .$$

$\mathrm{Coverage}$ is recall and is the **headline**: the share of the target value that the display localizes. $\mathrm{Purity}_{\mathrm{pooled}}$ is precision. $K$ is reported because the product goal is a *small* set of readable cells, so at equal coverage fewer centers is strictly better. Two further quantities are carried because on a rare target no single scalar suffices: $\mathrm{Mass} = \sum_{\mathcal{G}} n_\mathbf{c} / N$, the fraction of the population one must inspect, and $\mathrm{Lift} = \mathrm{Purity}_{\mathrm{pooled}} / (N_+/N)$, the enrichment over the base rate.

Both $\mathrm{Coverage}$ and $\mathrm{Purity}_{\mathrm{pooled}}$ are reported with their own Clopper–Pearson lower bounds at level $\alpha$ (unadjusted: each is a single pre-specified ratio, not a maximum over cells).

#### Out-of-sample coverage

**Definition 8 (Cross-validated coverage).** Over $R$ repetitions of a stratified $K$-fold split of the rows, centers are selected on the training fold by Definition 6 (with $C$ recomputed on that fold, and a cell unseen in training treated as not a center) and $\mathrm{Coverage}$ is evaluated on the held-out fold. $\mathrm{Coverage}_{\mathrm{CV}}$ is the mean of the $RK$ fold values; its standard error is the Nadeau & Bengio (2003) corrected

$$\mathrm{se}^2 = \Big(\tfrac{1}{RK} + \tfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\Big)\, s^2, \qquad \tfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}} = \tfrac{1}{K-1},$$

which accounts for the overlapping training sets. At $R = K = 5$ the naive $s/\sqrt{RK}$ understates the spread by a factor of $\approx 2.3$ and manufactures "improvements" from $d$ to $d+1$ that do not replicate.

**Why this is required and not optional.** Under Rule P nothing at all controls selection optimism; under Rule C, Proposition 2 bounds the rate at which an *individual* cell is falsely certified and says nothing about a coverage computed over cells chosen by inspecting their own contents. Measured on a constructed case (40 cells of 300 objects, true purities drawn uniformly on $[0.85, 0.95]$, i.e. straddling $\tau = 0.90$): in-sample coverage $58.9\,\%$, cross-validated $54.7 \pm 2.5\,\%$. The in-sample figure is not wrong — it describes the observed table exactly — but it is not the figure a reader will reproduce. Cross-validation is also what exposes a coverage built out of one-object cells, since such cells do not recur on a held-out fold.

**Definition 9 (Power floor).** With $N_+$ target-value objects and $K$ folds, a held-out fold carries $N_+/K$ of them, so a fold-level coverage has standard error at least $1/(2\sqrt{N_+/K})$; pooling over $RK$ splits and applying the Nadeau–Bengio factor $\sqrt{1/(RK) + 1/(K-1)} = 0.53$ at $R = K = 5$ leaves a pooled standard error no smaller than $\approx 0.5/\sqrt{N_+}$. Requiring it to be at most $0.10$ gives $N_+ \ge 25$. Below that threshold the implementation reports **"undetermined"** rather than a number: a coverage of $0\,\%$ measured on 9 objects and one measured on 900 are different claims and must not be rendered identically.

#### Null distribution and multiplicity

**Definition 10 (Coverage null).** Conditional on the cell sizes $\{n_\mathbf{c}\}$ and on $N_+$, the permutation null of the per-cell counts $\{k_\mathbf{c}\}$ is exactly the multivariate hypergeometric law — the same null as in Definition 2. Coverage is evaluated on draws from it, giving an exact permutation $p$-value at $O(C)$ per replicate rather than $O(N)$. Since the selection threshold depends only on $n_\mathbf{c}$ — it is $\lceil \tau n_\mathbf{c} \rceil$ under Rule P and the exact binomial threshold under Rule C — and permutation preserves the cell sizes, it is tabulated once and every replicate reduces to one integer comparison per cell.

**Definition 11 (Familywise coverage null).** A branch reported by Section 4.3 is an $\arg\max$ over $\sum_{d\le 4} \binom{M}{d}$ candidates, so Definition 10's $p$-value is anti-conservative for it. The look-elsewhere-corrected null is the distribution of $\max_S \mathrm{Coverage}(\tilde Z_\pi; \tilde X_S)$ under a **single shared** permutation $\pi$ applied to every candidate. The permutation must be shared: the candidates are partitions of the same rows by overlapping feature sets and are strongly dependent, so maximizing over independently drawn per-candidate nulls would overstate the null maximum. This is the $p$-value a publication quotes.

#### Sufficient dimensionality

**Definition 12 (Smallest sufficient $d$).** Scanning $d = 1,2,3,4$ in increasing order and holding an incumbent, $d$ replaces the incumbent when the **paired** difference of their per-fold coverages exceeds $t^\ast$ Nadeau–Bengio corrected standard errors; the first incumbent must clear zero by the same margin. The answer is the last accepted $d$, and **None** when no $d$ achieves a coverage distinguishable from zero.

Three properties are deliberate.

* It is the smallest *sufficient* dimensionality, not the first useful one: if $d=3$ genuinely beats $d=2$, then two characteristics do not describe the target and the answer is 3.
* The scan does not stop at the first flat step. Coverage is not submodular in the feature set, for the same reason mutual information is not (Krause & Guestrin, 2005; Section 4.4), so a flat step from $d$ to $d+1$ does not license skipping $d+2$.
* **None** is a result, not a failure, and is reported as such. Rounding it down to $d = 1$ would reintroduce exactly the defect of Section 0-ter.

The pairing is valid because the fold assignment is a function of $(\tilde Z, K, R, \text{seed})$ alone and not of the cell partition, so all four branches are evaluated on byte-identical folds.

$t^\ast = 2$ is a two-sigma convention on a statistic whose reference distribution is approximately $t$ with $RK-1$ degrees of freedom (Nadeau & Bengio, 2003, Sec. 4); at $R = K = 5$ the exact $0.975$ quantile is $2.06$, so the default is marginally liberal and is stated as a convention rather than derived.

#### Worked contrast

All rows from `data/adult_census.csv`, Rule P, $m = 1$, features = every column except the target's own, search objective `"coverage"` (the default):

| criterion | $\tau$ | $d$ | branch | $U_{\mathrm{adj}}$ | $K$ | Coverage | $\mathrm{Coverage}_{\mathrm{CV}}$ | $p$ | $d^\ast$ |
|---|:---:|:---:|---|---:|---:|---:|---:|---:|:---:|
| `occupation = Armed-Forces` ($N_+ = 9$) | 0.90 | 3 | `workclass+relationship+race` | 41.2 % | 1 | 11.1 % | undetermined | 0.006 | none |
| `income = >50K` ($N_+ = 7841$) | 0.90 | 2 | `workclass+education` | 13.6 % | 2 | 1.2 % | $1.2 \pm 0.1\,\%$ | 0.001 | 3 |
| | 0.90 | 3 | `workclass+education+occupation` | 18.6 % | 55 | 2.4 % | $1.9 \pm 0.1\,\%$ | 0.001 | 3 |
| | 0.75 | 2 | `education+marital_status` | 30.8 % | 6 | 17.5 % | $17.4 \pm 0.6\,\%$ | 0.001 | 3 |
| | 0.75 | 3 | `education+marital_status+occupation` | 33.9 % | 46 | 24.6 % | $24.8 \pm 0.7\,\%$ | 0.001 | 3 |
| `relationship = Husband` ($N_+ = 13193$) | 0.90 | 2 | `marital_status+sex` | 96.5 % | 2 | 100.0 % | $100.0 \pm 0.0\,\%$ | 0.001 | 2 |
| | 1.00 | 4 | `workclass+education+marital_status+sex` | 96.8 % | 75 | 19.2 % | $22.6 \pm 1.6\,\%$ | 0.001 | 4 |

Four things this table is here to make unavoidable.

1. **`Armed-Forces` at $\tau = 0.90$ is what $m = 1$ buys.** One center, holding one of the nine target objects — coverage $11.1\,\%$ that is one row of the dataset. $N_+ = 9$ is below the power floor of Definition 9, so there is no out-of-sample estimate to contradict it and $d^\ast$ is correctly **none**. Compare the v2.1 headline for the same target: $U_{\mathrm{adj}} = 41.3\,\%$.
2. **Coverage and $K$ are in tension and the specification does not resolve it silently.** For `income = >50K` at $\tau = 0.90$, going from $d = 2$ to $d = 3$ raises coverage from $1.2\,\%$ to $2.4\,\%$ and raises the number of centers from 2 to 55. The product rule is explicit that coverage dominates and that $K$ only breaks ties (Section 4.3), so $d = 3$ wins — but the panel shows both numbers, and 55 cells is not a readable finding whatever the coverage says. A user who wants the 2-center answer lowers $d$ or raises $\tau$; the framework will not make that trade for them.
3. **$\tau$ is the dominant parameter, not a cosmetic one.** The same target and the same branch family move from $2.4\,\%$ coverage at $\tau = 0.90$ to $24.6\,\%$ at $\tau = 0.75$. A coverage quoted without its $\tau$ is meaningless.
4. **$\tau = 1$ is a real setting with a real price.** `Husband` is captured essentially completely by two cells at $\tau = 0.90$. Demanding *entirely* pure cells costs 80 points of coverage and 73 additional centers, and the cross-validated figure ($22.6\,\%$) exceeds the in-sample one, which is the signature of a rule so strict that fold-to-fold variation in cell contents dominates the selection.

---

## 4. Independent Branch Discovery Algorithm

### 4.1. Overview

This is the section that completely replaces the Adaptive Visual Routing (AVR) algorithm of version 1.0. Instead of a single greedy forward selection with a permutation stopping criterion — **4 independent exact searches**, one for each dimensionality $d \in \{1,2,3,4\}$, each an honest exhaustive search.

### 4.2. Formal definition of a branch

**Definition (Branch).** For dimensionality $d$, branch $S^*_d$ is

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; \hat I_{\mathrm{adj}}(\tilde{Z}; \tilde{X}_S), \qquad \hat I_{\mathrm{adj}} = \hat I - \mathbb{E}_0[\hat I]$$

Ranking within a single dimensionality is by the **bias-corrected** mutual information of Section 3.2, Definition 3 — not by raw $\hat I$ (v2.0) and not by any $NMI$. Raw $\hat I$ cannot rank candidates of unequal cell count, because its bias grows with that count; the control experiment in Section 0-bis shows the resulting selection failure explicitly. $NMI$ is rejected for the separate reasons given in Section 3.2. For $d > M$, branch $S^*_d$ is undefined (not enough features in the dataset).

Note that the correction can reorder branches across $d$, and does. On the Adult dataset the raw estimator reports $\hat I = 0.2867$ at $d=3$ and $\hat I = 0.2948$ at $d=4$, suggesting the 4th axis adds information; after correction, $\hat I_{\mathrm{adj}} = 0.2631$ at $d=3$ and $0.2606$ at $d=4$ — the apparent gain is entirely bias ($\mathbb{E}_0$ rises from $0.0237$ to $0.0342$ bits), and the 4D branch carries no more association than the 3D one.

### 4.3. Algorithm

```
Algorithm: Independent Branch Discovery (IBD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:   Dataset D, target Z (column + optional criterion)
Output:  Up to 4 branches {(d, S*_d, I_d)}, d = 1..min(4, M)

1. Discretize all features (PMD, Section 2).
2. Prepare Z̃ (raw multiclass target or binary criterion).
3. FOR d = 1 TO min(4, M):
      a. FOR EACH combination S ⊆ {1,...,M}, |S| = d:
           Compute X̃_S (joint discrete code of features S)
           Compute I(Z̃; X̃_S)
      b. S*_d ← arg max over all iterated S
      c. Save (d, S*_d, I(Z̃; X̃_{S*_d})) as branch d
4. RETURN all found branches
```

No stopping condition, significance threshold, or VIR threshold — the searches for different $d$ are completely independent and are **always** executed in full (see Section 4.6 regarding the cost of this decision).

### 4.4. Why branches do not have to be nested

Mutual information $f(S) = I(\tilde{Z}; \tilde{X}_S)$ is not submodular in the general case (Krause & Guestrin, 2005): with feature synergy, $f$ can be supermodular, meaning $f(S \cup \{a, b\}) - f(S)$ can exceed $\big(f(S \cup \{a\}) - f(S)\big) + \big(f(S \cup \{b\}) - f(S)\big)$. Practical consequence: a pair of features, neither of which individually belongs to the best single axis $S^*_1$, can jointly form a highly superior $S^*_2$ — version 1.0's greedy algorithm, growing $S^*_1 \to S^*_2 \to S^*_3$ one feature at a time, **cannot find** such a pair in principle, since it never re-evaluates an already selected feature. Exhaustive search for each $d$ separately is the only exact way to guarantee finding the true $S^*_d$ under possible synergy.

### 4.5. Statistical control: what is now guaranteed, and what still is not

**Directly and without softening (Zero False Optimism).** Version 1.0 gated each greedy selection step with a permutation test ($\alpha = 0.01$) under Benjamini–Hochberg FDR control ($q \le 0.05$). Version 2.0 removed that control entirely and ranked by raw $\hat I$ with no test of any kind, documenting the resulting look-elsewhere exposure as an accepted product trade-off. Section 0-bis shows that trade-off was not survivable: on a control dataset with no signal anywhere, the v2.0 search returned a 3D branch at $\hat I = 0.078$ bits and $9.8\,\%$ normalized. Version 2.1 restores statistical control, but at a different place in the pipeline than v1.0 put it.

**The distinction that matters.** v1.0 used significance tests to *drive selection* — the greedy chain stopped when the next step failed its test, so the returned feature set was a function of the test. v2.1 does not. Selection remains a deterministic exhaustive $\arg\max$ over $\hat I_{\mathrm{adj}}$; the tests below only attach a *claim* to an already-selected branch. Nothing about the search depends on a random seed.

**Three distinct sources of optimism, handled separately.**

1. **Estimator bias.** Removed analytically by subtracting the exact null expectation (Section 3.2, Definition 2). This is not a test and costs nothing statistically: it is a correction to a point estimate, applied to every candidate before ranking.

2. **Sampling variability of a pre-specified subset.** Addressed by a permutation $p$-value with $B = 999$ replicates. Reported as `p_value`; its resolution floor is $1/(B+1)$. Two exactly equivalent routes to the same null are available — drawing the table directly from the multiple hypergeometric law conditional on both margins ($O(RC)$ per replicate, never touching the $N$ samples), or explicitly relabelling ($O(N)$) — and the implementation dispatches on $5RC \le N$, which is where the measured crossover sits. The direct draw is up to $80\times$ faster on a sparse grid and marginally *slower* at the Grid Capacity Limit, so neither is uniformly preferable; see `vsf.metrics._sample_null_mi` for the measured table.

3. **The look-elsewhere effect of the search itself.** `p_value` is *not* valid for a branch that was chosen as the maximum of $\sum_d \binom{M}{d}$ candidates. The corrected statistic is the permutation distribution of

$$T = \max_{S:\,|S| \le d_{\max}} \hat I_{\mathrm{adj}}(\tilde Z_\pi; \tilde X_S)$$

   under a **common** permutation $\pi$ applied across the whole family (a per-candidate independent shuffle would be conservative but wrong, since candidate scores are positively correlated). Reported as `p_value_familywise`. The gap is not academic: on the control dataset above, the $d=3$ winner has an uncorrected $p = 0.036$ — nominally significant at the conventional 0.05 — and a familywise $p = 0.200$.

**Defaults and their justification.** `p_value` is computed by default (it costs $\approx 0.25$ s per branch at $N = 3\times10^4$, $B = 999$). `p_value_familywise` is **off** by default, because it costs $B$ times the entire search and cannot be afforded on an interactive click; the interactive HUD therefore labels the $p$-value it displays as uncorrected. **Any published result must set it.** Measured on the reference dataset ($M = 7$, 98 candidates, $N = 32{,}561$): 6–7 s at $B = 199$, $\approx 32$ s at $B = 999$.

**What is still not controlled.**

* **No FDR correction within a single search.** v2.1 reports the *maximum* per $d$ and tests that maximum; it does not attempt to enumerate which of the $\binom{M}{d}$ candidates are individually significant. That is a different question from the one the product asks, and the familywise statistic is the correct one for the question it does ask. Multiplicity across *targets* — where it genuinely accumulates — is controlled by the Global Pattern Scan (Section 4.7).
* **No out-of-sample validation.** In-sample, bias-corrected, and significant means: this association is present in the observed contingency table and is not an artifact of estimator bias or of the search. It does not mean the branch will hold on new data. No claim of predictive generalization is made anywhere in this specification.
* **No correction for the discretization search.** The PMD bin counts (Section 2) are chosen from the same data. That exposure is small relative to the subset search but is not zero, and is not corrected.

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

Each $I(\tilde{Z}; \tilde{X}_S)$ estimation via contingency table costs $O(N)$ (vectorized construction of the joint feature code + frequency counting), where $N$ is the number of rows. The total cost is $O\big(N \cdot \text{Total}(M)\big)$. For $M=100$, $N=10^4$, this is on the order of $4 \times 10^{10}$ elementary operations — in practice, minutes-hours even with a fully vectorized NumPy implementation, let alone a naive Python loop over combinations.

**Cost of the statistical layer (v2.1).** Three components sit on top of the search above.

| Component | Cost | Default |
|---|---|---|
| Exact $\mathbb{E}_0[\hat I]$ per candidate | $O\big(\sum_{ij}\min(a_i,b_j)\big)$, measured 2.1 ms median for a $2 \times 1379$ table, growing linearly in $R$ (16 ms at $R=20$) — $\approx 0.2$ s for the whole $M=7$ reference search | on |
| Permutation $p$-value, winners only | $4 \times B$ table builds; 0.25 s per branch at $B=999$, $N = 3\times10^4$ | on ($B = 999$) |
| Familywise max-null | $B \times \text{Total}(M)$ table builds; 6–7 s at $B = 199$ for 98 candidates | **off** |

The permutation null is drawn by whichever of two exactly equivalent routes is cheaper for the table at hand (Section 4.5, item 2): direct multiple-hypergeometric sampling, $O(RC)$ per replicate, or explicit relabelling, $O(N)$. Measured at $N = 32{,}561$, $R = 2$, $B = 999$, the direct draw is $80\times$ faster at $C = 20$, $5.4\times$ at $C = 500$, $2.1\times$ at $C = 1379$ and $0.9\times$ — i.e. slower — at $C = 3256$; the dispatch threshold $5RC \le N$ sits at that crossover. Since typical branches are far below the Grid Capacity ceiling, most tables take the fast route. The bias correction adds no replicates at all: $\mathbb{E}_0$ is closed-form.

**This is an honestly documented consequence of the explicit decision of the product owner ("always honest exhaustive search"), not a forgotten edge case.** Datasets of the mushrooms scale ($M \approx 22$) remain completely comfortable (9,108 combinations, fractions of a second to seconds on vectorized NumPy). Datasets with $M \gtrsim 50-100$ features will visually "hang" for tens of minutes with an honest implementation — this is not a reason to silently substitute the algorithm with an approximate one, but a reason to inform the user of the expected time before launching (a UI engineering task, not the subject of this document) and, separately, during implementation, to invest in accelerating the honest search itself (batch vectorization over many combinations simultaneously, reusing partial contingency tables between combinations with a common prefix, parallelization) — that is, make the search faster, not incomplete.

### 4.7. Global Pattern Scan: a dataset-wide reformulation of the same algorithm (only `vsf.serve()`)

Upon explicit user request (after resolving the open question in Section 10, item 3 of the version prior to this revision), the live application was augmented with the ability to apply Section 4.3 not to a single pre-selected target $Z$, but to **every** pair (column, observed value) in the dataset, taken as a One-vs-Rest criterion — the same form as a regular manual criterion in Section 1.4.

**This is NOT a reincarnation of Auto-Discovery from version 1.0.** The only overlap is that both ideas iterate through all dataset columns. The differences are fundamental:

* v1.0 Auto-Discovery replaced ordinary target selection and ran by default, with a permutation significance test *gating greedy selection* of a single $d^*$ (all this was removed, see Section 0). Global Pattern Scan is a strictly optional, button-triggered operation that **only filters** the already existing flat Target Selector; it does not replace or pre-select a target for the user, and its significance test does not touch selection — selection remains the deterministic exhaustive search of Section 4.3.
* For each pair (column $c$, value $v$), $\tilde Z = \mathbb{1}[c = v]$ is built, and over all other columns as features, **exactly the same** Independent Branch Discovery (Section 4.3) is run as for a single manual analysis — not an approximation, not a cheaper variant. From the up to 4 branches found, the maximum **coverage** (Definition 7) is taken; ties break toward fewer centers and then toward the lower dimensionality.
* A pair is kept only if **both** conditions hold: (i) that maximum strictly exceeds the user-specified threshold, and (ii) the pair survives Benjamini–Hochberg FDR control at rate $q$ applied across **every** pair scanned in the run.

    **v2.2 changed condition (i) from $U_{\mathrm{adj}}$ to coverage, and the search objective with it** (`objective="coverage"`, Section 4.3). The scan's product is a list of (column, value) pairs worth *displaying as discrete centers*, and $U_{\mathrm{adj}}$ does not measure that: on `data/adult_census.csv`, `income = ">50K"` reaches $U_{\mathrm{adj}} = 34.0\,\%$ with a coverage of $2.4\,\%$ at $\tau = 0.90$, and `occupation = "Armed-Forces"` reaches $44.4\,\%$ with a coverage of $11.1\,\%$ carried by a single object. A $U_{\mathrm{adj}}$-filtered scan ranks both near the top; a coverage-filtered scan at any usable threshold drops both, and that is the correct behaviour for the question being asked. Filtering on one statistic while ranking by another would also be incoherent, which is why the objective moves with the filter. $U_{\mathrm{adj}}$ remains recorded per pair as a diagnostic, so the divergence between "an association exists" and "a center exists" stays visible in the results instead of being resolved silently. The FDR procedure correspondingly uses each pair's **coverage** $p$-value (Definitions 10–11).

    Both are load-bearing, and this is the largest correctness change v2.1 makes to this feature.

    Condition (i) used $U_{\mathrm{adj}}$ in v2.1, not v2.0's $\max NMI$. A scan over all (column, value) pairs manufactures rare One-vs-Rest criteria in bulk — that is what it is *for* — and $NMI_{\min}$ saturates toward 100 % on exactly those (Section 3.2). The v2.0 scan therefore selected preferentially for the artifact it was most exposed to. That correction stands; v2.2 goes one step further and replaces the association statistic with the deliverable statistic, per the paragraph above. The threshold's *scale* changed twice as a result: a coverage threshold is a share of the criterion's own occurrences, not a share of its entropy, and the two are not comparable numbers.

    Condition (ii) exists because a sweep of $k$ targets at a nominal $\alpha$ produces $\approx \alpha k$ spurious "patterns" by construction; on a dataset with a hundred (column, value) pairs that is several false discoveries per scan, guaranteed. BH is applied to the whole scanned family *before* the effect-size filter — filtering first and correcting afterwards is itself a selection effect and voids the guarantee. Setting `n_permutations_familywise > 0` additionally upgrades each pair's $p$-value to the look-elsewhere-corrected one of Section 4.5, covering both levels of multiplicity (across targets, and within each target's own subset search) at $B$ times the per-pair cost; without it, FDR is controlled across targets only and the per-pair $p$-values remain anti-conservative.
* **The cost scales linearly** with the number of (column, value) pairs — with $M$ columns and an average of $\bar k$ observed values per column, this is $\approx M \cdot \bar k$ full runs of Section 4.3, meaning the cost of Section 4.6 multiplied by $M \cdot \bar k$. On the mushrooms dataset ($M=22$, 119 (column, value) pairs), this is empirically ≈16 s per pair × 119 ≈ 32 minutes of honest search — measured, not estimated. Just as in Section 4.6, no pre-filter, approximation, or shortcut (such as exploiting MI symmetry of binary columns, which gives only ~5% savings on this dataset) is deliberately applied — the price of the honest search is considered the acceptable price of honesty, not a reason to simplify the algorithm.
* Since the operation is magnitudes more expensive than a single analysis and unambiguously exceeds any reasonable synchronous HTTP request timeout, it is executed in a background thread on the server (`vsf/server.py`) with a client-polled progress status and cooperative cancellation between (not within) calls to Section 4.3 — an engineering implementation, not a separate algorithm.
* **Scope — only `vsf.serve()`.** The static export (`export_full_dashboard`, Section 1 `UI_Functional_Spec.md`) does not have a server to run a search in the browser, so this feature is not added there.

Full UI/API specification (endpoints, progress format, behavior on page reload) is in `UI_Functional_Spec.md`, Section 2.1.

---

## 5. Visualization: Discrete Centers

This section consolidates and brings to the forefront material that in version 1.0 was an "open problem" in Section 9.2 — in version 2.0, it is the central, rather than peripheral, part of the specification, since the entire product boils down to this visualization plus the algorithm of Section 4.

### 5.1. Fundamental visual unit: Discrete Center

VSF does not visualize individual dataset rows as individual points. The visual unit is a **Discrete Center**: a unique combination of discretized feature values from branch $S^*_d$, representing a macrostate of the phase space.

$$\mathbf{c} = (\tilde{x}_{j_1}, \dots, \tilde{x}_{j_d}) \in \prod_{j \in S^*_d} \text{Bins}(X_j)$$

The number of actually populated centers is typically $|\mathcal{C}_{occ}| \ll \prod_{j} k_j$ — not every category combination occurs in the data.

### 5.2. Center mass encoding: circle size (Mass Encoding via Area Scaling)

Each discrete center aggregates $N_{cell} \ge 1$ objects. Scaling by area (Tufte, 1983) minimizes the "Lie Factor":

$$\text{Radius}(\mathbf{c}) = R_{\max} \cdot \sqrt{\frac{N_{cell}(\mathbf{c})}{\max_{c'} N_{1D}(c')}}$$

where the normalizing denominator is the maximum density in the 1D projection of the same branch, $2R_{\max} = 1.0$ (the side of the elementary grid cell) — the maximum circle exactly fills the interval/cell, never overlapping neighbors.

### 5.3. Semantic purity encoding: circle color (User-Bounded Purity Scale)

The purity of center $\mathbf{c}$ is Definition 5, computed on **all $N$ rows** — not on the render subsample. The renderer may draw a sample of the objects as individual spheres (`max_display_samples`), but every number and every colour attached to a cell is computed on the full table. In v2.1 these were the same field and were not the same $N$; see Section 0-ter, defect 2.

**Discrete 3-zone colour scale (version 2.2).** Version 2.0/2.1 banded the point purity at $0.25 / 0.75 / 0.85$ and added a yellow zone between the last two. Those three numbers had no derivation, and the scale could not distinguish $\hat\pi = 1/1$ from $\hat\pi = 300/300$; the *"Noise Reduction (minimum samples)"* control existed to hide that case, which made the drawn colour depend on a hand-set integer sitting outside the reported definition.

Version 2.2 keeps the scale on the point purity, removes yellow, and hands **both** boundaries to the user:

| Zone | Condition | Colour | Semantics |
|------|---|---------|-----------|
| **Discrete center** | $\hat\pi(\mathbf{c}) \ge \tau$ and $n_\mathbf{c} \ge m$ (Rule P), or Definition 6 Rule C | 🟢 Green | The deliverable. **Exactly the cells Definition 7 counts.** |
| **Mixed** | $\beta \le \hat\pi(\mathbf{c}) < \tau$ | 🟤 Brown | Enriched but below the user's threshold. |
| **Low** | $\hat\pi(\mathbf{c}) < \beta$ | 🔴 Red | Target value largely absent. |

The two boundaries are **not the same kind of control**, and the interface separates them:

* $\tau$ — the green boundary — *is* Definition 6's purity floor. Moving it changes the certified set, the coverage, the center count and the cross-validated figure, so it triggers a recomputation and forms part of the analysis cache key. $\tau = 1$ is permitted (Corollary 1).
* $\beta$ — the brown boundary, default $0.40$ — is purely a colour cut. It partitions the *non*-center cells into two shades and changes no reported number, so it re-colours instantly and is never sent to the server.

The partition is exhaustive and non-overlapping by construction, and $\beta < \tau$ is enforced by the fact that the green test is evaluated first. Sharp transitions, not a gradient, as in version 1.0 — for instant categorical distinctness (Healey & Enns, 2012).

**Consequences of the change, stated explicitly.**

* **What is green is what is counted.** The single most important property of the scale: coverage in the panel is computed over exactly the green cells, so a reader cannot be looking at one set and reading a number about another. In v2.1 the colour threshold ($0.85$) and the reported statistic ($U_{\mathrm{adj}}$) had no relationship at all.
* **A singleton cell is green under the default settings, and the display says so.** With $m = 1$ a cell holding one target-value object has $\hat\pi = 1$ and is a center. Its hover text carries the interval $[\alpha/C, 1]$, i.e. the data are equally consistent with a purity of a few percent. The specification's position is that this is the user's parameter to set, not the framework's to fix: Corollary 2 measures what it costs on pure noise ($K = 2$, coverage $0.09\,\%$, $p = 0.23$), and both remedies are one control away.
* **Yellow is gone.** It encoded "high but not top", which is not a category the product acts on; with a movable $\tau$ the user places that boundary themselves.
* Every cell still carries its $1-\alpha$ Clopper–Pearson interval, and under Rule C that interval, not the point purity, decides. The legend states which rule is in force, both boundaries, $m$, the positive value, the base rate and the $N$ — so a screenshot of the display is self-describing.

### 5.4. Grid axes: from 1D to 3D

Adding a feature turns an axis (1D) into a grid (2D), then into a cube (3D). When $d < 3$, the missing spatial axes collapse (similar to the `dim >= 2`/`dim >= 3` grid-building branches already in the current implementation): 1D view is a line of intervals, 2D is a flat grid, 3D is a full cube of discrete centers.

### 5.5. 4D: film strip metaphor (Film Strip)

When a 4-dimensional branch $S^*_4 = \{j_1,j_2,j_3,j_4\}$ is selected, the first three features are mapped to the spatial axes of the 3D cube, and the fourth is implemented via an **interactive stepper** (slice tabs) below the chart — not a separate spatial axis:

$$\text{Slice}(c) = \{(x_1, x_2, x_3) \mid X_{j_4} = c\}, \quad c \in \text{Dom}(X_{j_4})$$

A discrete stepper (not a continuous auto-animation by default) was chosen for resilience to Change Blindness and to maintain spatial axis constancy between slices — the reasoning is the same as in version 1.0. Optional auto-play (play/pause/speed) is available on top of the discrete slices, not replacing them.

The "All" tab — marginalization over the 4th dimension (aggregates all slices into one 3D view), allows comparing the overall picture with a single slice.

### 5.6. "Collapse/Split" animation between dimensions

**Important clarification regarding version 1.0, necessary due to branch independence (Section 4.4):** the dimensional transition animation in version 2.0 acts **within a single already selected branch**, not between independently found $S^*_d$ of different dimensions. These are two different interactions that must not be confused:

1. **Branch selection:** the user chooses which of the up to 4 independently found branches to view — for example, the top 3D branch $S^*_3 = \{A, B, C\}$. Switching between branches of different $d$ (for example, from $S^*_3$ to an independently found $S^*_2$, which could consist of entirely different features $\{D, E\}$) is an **instant scene rebuild** with new axes, not a collapse animation: the axes physically change, Object Constancy is inapplicable and is not claimed here.
2. **Dimensionality collapse within a branch:** after a branch is selected (e.g., $S^*_3 = \{A,B,C\}$), the user can visually "collapse" it to 2D (marginalization over $C$: aggregate over $\{A,B\}$) or to 1D (marginalization over $B,C$: aggregate over $\{A\}$), and "split" it back. **This** is the animation preserving Object Constancy, exactly as described below — it folds/unfolds the axes of the same branch, never swapping features.

Formally, for case (2), when removing axis $C$ from an already selected 3D branch $\{A,B,C\}$:

* **Collapse (3D → 2D):** discrete centers do not disappear, but fall onto the $(A,B)$ plane. Centers with the same $(A,B)$ but different $C$ smoothly **merge**: masses (radii) are summed, colors are interpolated into a mixed brown ("mud") — visually demonstrating why dropping the axis loses separability.
* **Split (2D → 3D):** the reverse process. A large muddy center in 2D splits along the returning $C$ axis into several purer centers — a visual "aha moment": adding a feature untangles the mixture.
* **Film Strip (3D ↔ 4D):** transition to a slice — scenes slide in/out using a rewinding metaphor, as in Section 5.5.

All transitions preserve the relative sizes of the circles (Object Constancy, Heer & Robertson, 2007) — an invariant kept from version 1.0, crucial for the interpretability of the animation.

### 5.7. Canonical axis sorting and Object Constancy

Categorical axes do not have a natural metric order. The divisions of each axis are ordered by the conditional mean of the target variable:

$$\text{Score}(c) = \mathbb{E}[Z_{\text{canonical}} \mid X_j = c], \quad c \in \text{Dom}(X_j)$$

The geometric structure (order of axis divisions) is calculated once for the selected target with deterministic collision resolution (tie-break by cluster volume, then lexically) and remains fixed when switching the criterion **within the same target column** (e.g., `class = "p"` ↔ `class = "e"`) — centers do not make parasitic movements, only color and size update. When changing the target column itself (switching the branch to a different target), the basis is recalculated.

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

* **mRMR (Peng et al., 2005):** balance of relevance $I(X_j;Z)$ and redundancy. VSF version 2.0, unlike mRMR, does not solve the problem of selecting a single $K$ — instead, it explicitly materializes all 4 dimensions in parallel, removing the need for the user to choose $K$ at the cost of abandoning a statistical stopping criterion (see Section 4.5).
* **Scagnostics (Wilkinson et al., 2005):** 9 graph metrics for ready-made 2D projections (passive diagnostics). VSF is an active synthesizer, deriving axes prior to visualization.
* **Maximally Informative Dimensions (Sharpee et al., 2004):** continuous gradient optimization of projection by MI. VSF replaces it with discrete exact search, which for $d \le 4$ guarantees a global optimum (not local, unlike gradient descent), at the cost of exponential $M$ complexity (Section 4.6).

### 7.3. Summary comparison matrix

| System | Selection Method | Target-Conditioned? | Branches by Dim? | PMD | Statistical Criterion | Unit |
|---|---|:---:|:---:|:---:|:---:|---|
| **Voyager 2** (2017) | Perceptual rules | ❌ Partially | ❌ | ❌ | ❌ | Charts |
| **Scagnostics** (2005) | Graph metrics | ❌ | ❌ | ❌ | ❌ | Points |
| **mRMR** (2005) | MI, manual $K$ | ✔️ | ❌ | ❌ | ❌ | Features |
| **Jeon et al.** (2025) | Structural complexity | ❌ | ❌ | ❌ | ❌ | Points |
| **VSF v1.0** | Greedy + permutation test + FDR | ✔️ | ❌ (single $d^*$) | ✔️ | ✔️ | Discrete centers |
| **VSF v2.0 (current)** | Honest exhaustive search by MI, 4 independent branches | ✔️ | ✔️ (1D–4D parallel) | ✔️ | ❌ (deliberately removed, Sec. 4.5) | Discrete centers |

The loss explicitly reflected in the table compared to v1.0 — the lack of a statistical criterion — is not an accidental omission, it is Section 4.5.

---

## 8. Summary of Contributions (version 2.2)

| # | Contribution | Type |
|---|-------|-----|
| C1 | Independent Branch Discovery — exact (not approximate) search for up to 4 independent MI-optimal feature sets, one per dimensionality, without assuming nestedness | Algorithmic |
| C2 | Explicit formalization of MI non-monotonicity/supermodularity as the reason why a single greedy path fundamentally cannot find what independent searches find (Section 4.4) | Theoretical |
| C3 | Perceptually-Matched Discretization (PMD), carried over from v1.0 without essential changes | Theoretical |
| C4 | 4-zone certificate semantics of a discrete center: every colour boundary is either the simultaneous confidence bound or the dataset base rate, so the scale has no free parameters and a singleton cell is provably never green (Section 5.3) | Metric / UX |
| C5 | Collapse/split animation as explicitly demarcated from switching between independent branches (Section 5.6) — removes the ambiguity inherited from v1.0 where branches were nested by design | System / UX |
| C6 | Bias-corrected ranking: candidate subsets of unequal cardinality are compared by $\hat I - \mathbb{E}_0[\hat I]$ with an exact closed-form null expectation, removing a selection artifact that made a raw-MI search return "moderate associations" on data with no signal at all (Section 0-bis) | Algorithmic / Methodological |
| C7 | $U_{\mathrm{adj}}$ as the reported association statistic, with a documented rejection of both $NMI_{\min}$ and $NMI_{\mathrm{geo}}$ on a reproducible adversarial case (Section 3.2) | Metric |
| C8 | Two-level multiplicity control matched to where multiplicity actually arises: a look-elsewhere-corrected max-statistic null over the $C(M,1{..}4)$ subset search, and Benjamini–Hochberg FDR across the targets swept by a Global Pattern Scan (Sections 4.5, 4.7) | Methodological |
| C9 | Exact per-class decomposition of the reported MI, so a headline number can be attributed to the class that carries it rather than silently reflecting the majority class (Section 3.2) | Metric / UX |
| C10 | **Discrete centers as a stated definition rather than a colour band**: one threshold triple $(\tau, m, \text{rule})$, owned by the user, that simultaneously fixes what is drawn green and what enters every reported number — plus a certified variant with a proven simultaneous error rate, and an explicit account of why $\tau = 1$ is observable but not certifiable (Definition 6, Proposition 2, Corollaries 1–2) | Metric / Methodological |
| C11 | **Coverage as the reported deliverable**, replacing an association share with the quantity the product exists to produce; demonstrated on a reproducible case where $U_{\mathrm{adj}} = 41.3\,\%$ coexists with a maximum cell purity of $2.42\,\%$, no cell above the user's threshold and $\lambda = 0$ (Section 0-ter, Definition 7) | Metric |
| C12 | **Out-of-sample coverage with Nadeau–Bengio corrected variance**, and the first quantity in this framework that can decrease when a branch is over-resolved — which turns "how many characteristics describe this value" into a decidable question with an explicit *no answer* outcome (Definitions 8, 12) | Methodological |
| C13 | Exact multivariate-hypergeometric null for the coverage statistic, with a shared-permutation familywise variant matched to the $\arg\max$ the search actually reports (Definitions 10–11) | Methodological |

---

## 9. Evaluation Plan

This section is essentially preserved from version 1.0, adjusted for the removal of VIR-specific metrics.

### 9.1. Quantitative evaluation (synthetic data)

1. **Ground Truth Benchmark:** datasets with a known true structure of feature synergy (e.g., XOR-like dependencies, where no single feature is informative on its own, but a pair completely determines $Z$) — specifically to verify that Independent Branch Discovery finds such pairs, while the hypothetical v1.0 greedy algorithm does not.
2. **Null Benchmark (added in v2.1, and the more discriminative of the two).** Datasets in which every feature is generated independently of the target, with feature cardinalities spanning at least $2$ to $200$ so that estimator bias varies across candidates by an order of magnitude. The required outcome is that **no branch is reported as significant at any $d$**. This is the benchmark v2.0 fails (Section 0-bis) and is far more sensitive to the defect than the synergy benchmark, which a biased estimator can still pass. It must be run at several $N$, since the bias scales as $1/N$ while the variance the test must resolve scales as $1/\sqrt{N}$.
3. **Micro-class benchmark.** Target classes of prevalence $10^{-4}$ to $10^{-2}$, under both a null and a deterministic construction, verifying that the reported percentage is near $0$ and near $1$ respectively — the case that discriminates $U_{\mathrm{adj}}$ from both $NMI$ variants (Section 3.2), and where the residual power limitation of Section 3.2 should be characterised as a function of the positive-class count.
4. **Calibration of the familywise test.** Under the null benchmark, the empirical distribution of `p_value_familywise` must be uniform on $[0,1]$; the uncorrected `p_value` is expected to be visibly anti-conservative, and the size of that gap as a function of $M$ is itself a reportable result.
5. **Metrics:** match of the found $S^*_d$ with the true synergistic combination; false-positive rate on the null benchmark at $\alpha = 0.01$; execution time of the exhaustive search *and of the permutation layer* as a function of $M$, $N$ and $B$ (verifying the honesty of Section 4.6 estimates on real hardware).

### 9.2. Controlled User Study (minimum $n = 40$ participants)

Task: participants solve analytical tasks (finding clusters, outliers, separating rules) comparing VSF v2.1 (4 independent branches) against (a) VSF v1.0 (single $d^*$, if still available for comparison) and (b) expert manual axis selection. Hypothesis: parallel presentation of 4 branches does not increase cognitive load compared to a single v1.0 branch, while more frequently leading the user to discover synergistic feature combinations.

---

## 10. Open Problems and Limitations (version 2.2)

1. **Multiple comparisons — partially resolved in v2.1, see Section 4.5 for exactly how far.** The look-elsewhere effect of the subset search is corrected by a max-statistic permutation null, and multiplicity across targets in a Global Pattern Scan by Benjamini–Hochberg. Two exposures remain uncorrected and are not hidden: (a) the familywise test is **off by default** because it costs $B$ times the entire search, so any interactive number is uncorrected and labelled as such; (b) the PMD bin counts are selected from the same data, and that (small) exposure is not corrected at all.
2. **Scalability of exhaustive search** — datasets with $M \gtrsim 50$–100 features require tens of minutes for an honest search of all $\binom{M}{\le 4}$ combinations (Section 4.6); neither pre-filtering nor approximate search is allowed by the current specification.
3. **The fate of Auto-Discovery / cross-column insights catalog — resolved.** The user is still required to specify the target column themselves by default — there is no automatic scan on load. However, upon explicit user request, an optional, button-triggered **Global Pattern Scan** feature was added (only `vsf.serve()`, Section 4.7), which iterates through all (column, value) pairs in the dataset using the same honest Section 4.3 exhaustive search and **filters** (does not replace or substitute) the flat Target Selector by a $U_{\mathrm{adj}}$ threshold under Benjamini–Hochberg FDR control across every pair scanned. This is not a return of the greedy $d^*$ of version 1.0: v1.0's test *drove selection*, whereas here selection remains the deterministic exhaustive search of Section 4.3 and the test only attaches a claim to its output. The cost remains honestly disclosed (not approximated) and the operation stays on a background thread with progress and cancellation.
4. **Ranking across unequal category cardinalities — resolved in v2.1** by ranking on $\hat I - \mathbb{E}_0[\hat I]$ (Section 4.2). The v2.0 formulation of this item proposed ranking by $NMI$ instead; Section 3.2 documents why that would not have worked, since $NMI$ rescales the bias rather than removing it.
5. **Out-of-sample validation — resolved for coverage in v2.2, still open for the association layer.** $\mathrm{Coverage}_{\mathrm{CV}}$ (Definition 8) is estimated on held-out folds with the Nadeau–Bengio corrected variance, and it is the statistic behind the reported dimensionality (Definition 12). $\hat I_{\mathrm{adj}}$ and $U_{\mathrm{adj}}$ remain in-sample and are labelled as such; a cross-validated information gain was considered and rejected for the default path because it multiplies the exhaustive search cost by $K$ without changing any decision the product makes — the deliverable is the discrete centers, and those are cross-validated. This is a scope decision, not an oversight, and it is the first thing to revisit if the association layer is ever promoted back to a headline.
6. **The certificate is per-cell; the coverage is not.** Proposition 2 bounds the probability that any cell with $\pi \le \tau$ is certified. It does not make the in-sample coverage unbiased, because the certified set is chosen by inspecting cell contents. The gap is real and measured ($8.00\,\% \to 5.67\,\%$ on the constructed case in Section 3.3), and it is the reason $\mathrm{Coverage}_{\mathrm{CV}}$ exists rather than being an optional extra. A selective-inference treatment that conditions on the selection event would give a tighter interval than cross-validation; that is open.
7. **$\tau$ and $m$ are user choices with no principled defaults, by design.** $\tau = 0.90$, $m = 1$ are conventions. The center set, the coverage and the reported dimensionality all move with them — on `income = ">50K"` the coverage at $d = 3$ goes from $2.4\,\%$ at $\tau = 0.90$ to $24.6\,\%$ at $\tau = 0.75$. The specification makes the dependence explicit and reproducible ($\tau$, $\alpha$, $m$ and the rule appear in the legend and in the analysis cache key) but deliberately does not remove it. A coverage quoted without its $(\tau, m, \text{rule})$ is meaningless.
7-bis. **Coverage and $K$ are not jointly optimized.** The product rule is lexicographic: coverage dominates, $K$ breaks ties (Section 4.3, `vsf.centers.coverage_score`). That is the stated preference, and it means the search will return 55 centers for $2.4\,\%$ coverage in preference to 2 centers for $1.2\,\%$ — technically correct and, at that center count, not a readable finding. A scalarization (coverage minus a penalty per center) or a Pareto front over $(\mathrm{Coverage}, K)$ would let the user express "not at any price"; neither is implemented, and the interface compensates only by showing both numbers side by side.
8. **Bonferroni is conservative at high cell counts.** At the Grid Capacity Limit, $C \approx N/10$ and $\alpha/C \approx 1.5 \times 10^{-4}$ on the reference dataset, so certification requires cells that are both large and nearly pure. This is the correct direction for a certificate, but it costs power exactly where the display is most detailed. A step-down or a false-discovery-rate formulation over cells would recover some of it at the cost of a weaker per-cell guarantee; this trade-off is not currently offered to the user.
9. **Grid Capacity Limit is a variance bound, not a bias bound.** $\prod_j k_j \le N/10$ (Section 2.4) keeps cell counts estimable but leaves $\mathbb{E}_0[\hat I] \approx 0.08$ bits at the ceiling on the reference dataset. v2.1 subtracts that floor exactly, so the point estimates are unbiased, but the *variance* at such grid densities remains high and is what limits power on small $N$. Tightening the ceiling would trade resolution for power; the current specification does not.

---

## Key References

* Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal.*
* Miller, G. A. (1956). The Magical Number Seven, Plus or Minus Two. *Psychological Review.*
* Rissanen, J. (1978). Modeling by Shortest Data Description. *Automatica.*
* Fayyad, U. & Irani, K. (1993). Multi-Interval Discretization of Continuous-Valued Attributes (MDLP). *IJCAI.*
* Peng, H. et al. (2005). Feature Selection Based on Mutual Information: mRMR. *IEEE TPAMI.*
* Krause, A. & Guestrin, C. (2005). Near-optimal sensor placements in Gaussian processes. *ICML.* — justification for MI non-submodularity (Section 4.4), no longer as a guarantee of the greedy algorithm
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
* Clopper, C. J. & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413. — the exact interval underlying Definition 6
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212. — the score interval offered for display only, rejected for certification
* Brown, L. D., Cai, T. T. & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133. — why the Wilson interval's coverage oscillates below its nominal level, measured for $n=30$ in `tests/test_centers.py`
* Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281. — the corrected variance for overlapping training sets used in Definitions 8 and 12
* Vinh, N. X., Epps, J. & Bailey, J. (2010). Information theoretic measures for clusterings comparison. *JMLR*, 11, 2837–2854. — the exact permutation expectation of Definition 2
* DeWeese, M. R. & Meister, M. (1999). How to measure the information gained from one symbol. *Network: Computation in Neural Systems*, 10(4), 325–340. — the per-class decomposition of Section 3.2
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289–300. — the multiplicity control of Section 4.7
* Miller, G. A. & Madow, W. G. (1955). On the maximum likelihood estimate of the Shannon–Wiener measure of information. *Air Force Cambridge Research Center Technical Report* 54-75. — the closed-form bias of Section 0-bis
* Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P. (2007). *Numerical Recipes*, 3rd ed., Sec. 6.4. — the continued-fraction evaluation of the incomplete beta function used to invert Definition 6 without a `scipy` dependency

**Removed from the version 1.0 references list as no longer cited in the text:** Good (2005), Runge et al. (2018), Krause et al. (2008), Elenberg et al. (2018), Borland & Taylor (2007) — all were specific to permutation testing, FDR, or the weak submodularity guarantee of the greedy algorithm, none of which are part of the core anymore.
