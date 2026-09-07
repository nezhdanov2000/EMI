# Visual Sufficiency Framework (VSF)

**Specification of the algorithm, the reported statistics and the display.**

Version 2.3 · `vsf` 2.3.0 · document current as of 2026-09.

VSF is an exploratory data-analysis tool for **categorical data**. The user names one **target value** — one value of one column, for example `occupation = Armed-Forces`. VSF searches every combination of 1, 2, 3 and 4 of the remaining columns, and for each of those four dimensionalities returns the combination whose grid of cells concentrates that target value most tightly. The result is drawn as a grid of circles ("discrete centres") and reported as three numbers: **how much of the target value the display localises (coverage), in how many cells (K), and how clean those cells are (pooled purity)**.

Every column is a set of categories, and every distinct value in it is one category (Section 2). There is no continuous-feature model: no binning, no bin-count heuristic, and no information-theoretic quantity computed anywhere in the framework.

Everything in this document describes code that runs. Where a limitation is known, it is stated in place rather than in a footnote; Section 10 collects the ones that have no fix in the current design.

---

## 1. Problem Formulation

### 1.1. Concept

> *Given a dataset and one target value, find up to four independent sets of features — one per dimensionality from 1 to 4 — and show where that value is concentrated inside certified discrete centres, where it is absent, and where nothing certifies at all.*

The key word is **independent**: the best set of features for 3D does not have to contain the best set for 2D. Coverage is not submodular in the feature set (Krause & Guestrin, 2005) — because of feature synergy, the pair $\{A, B\}$ can beat the pair $\{C, D\}$ at concentrating the target even when $A$ alone beats $C$ alone. Hence four parallel "branches" rather than one chain of "add the next best feature".

### 1.2. Formal definition

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, z_i)\}_{i=1}^{N}$ over $M$ categorical columns, $\mathbf{x}_i \in \prod_{j=1}^{M} \mathcal{A}_j$ where $\mathcal{A}_j$ is the finite set of values column $j$ takes (Section 2), a target column $z$, and a resolved positive value $z^\ast$ (Section 1.4), find for each $d \in \{1,2,3,4\}$ independently

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; \mathrm{CoverageScore}\big(\mathbb{1}[Z = z^\ast];\, \tilde{X}_S\big)$$

where $\mathrm{CoverageScore}$ is the lexicographic key of Definition 5. The result is up to four **branches** $(d, S^*_d)$, $d = 1,\dots,\min(4, M)$, each carrying the full certified-centres report of Section 3.2.

No single "optimal $d^*$" is picked by the search — the user chooses which branch to look at. `select_branch_dimensionality` (Definition 10) answers "which $d$ is smallest and sufficient" separately, out of sample, when asked.

### 1.3. Visual channels

The display has **four encoding channels**:

$$\mathcal{V} = \{X, Y, Z_{depth}, T\}$$

The first three are the spatial axes of the 3D cube. The fourth is not a spatial axis but a **frame channel** (Section 5.5): the values of the 4th feature become a stack of parallel frames ("film strip") through which the user steps manually or with auto-play.

Circle colour and size are **not** selection channels: they encode aggregate statistics of the already-selected branch (share of the target value, and number of objects in the cell — Sections 5.2, 5.3). They play no part in the search for $S^*_d$.

### 1.4. Target value and criterion

A search is not defined without a positive value. It resolves in exactly two ways:

1. **Criterion value** — the user names a value $v$ of one column, and $Z = \mathbb{1}[X_{target} = v]$ (for example `class = "p"`). $z^\ast = 1$ by construction.
2. **Naturally two-valued column** — the raw column has exactly two distinct values, and the higher-sorted one is taken automatically, which is the same value the One-vs-Rest criterion path would pick, so the two rules never disagree.

A column with more than two values and no criterion has **no resolvable positive value**, and `discover_branches` raises instead of guessing. The live application turns that into a `400` response with an explanation, and its first screen is a short usage guide rather than an auto-analysis of some default column. A purity is a statement about a named value; without one there is nothing for this system to report.

---

## 2. Categorical Encoding and Grid Capacity

### 2.1. Everything is a category

VSF is a categorical framework. A feature column is a finite set of category values, and a discrete centre is a statement about one combination of them: "cap-colour = brown **and** odour = foul". Every column is encoded by its distinct values — string, boolean, integer or float alike — with codes assigned in sorted distinct-value order (`vsf.pmd.discretize_feature`).

**Definition 1 (Category encoding).** For column $X_j$ with distinct values $v_1 < v_2 < \dots < v_{k_j}$,

$$\tilde X_j(i) = \ell \quad\text{iff}\quad X_j(i) = v_\ell, \qquad \ell \in \{1,\dots,k_j\}.$$

Three properties follow, and all three are load-bearing:

* **The encoding is a function of the column alone.** It does not depend on the target, on the other columns, or on row order. Nothing about the axis a user reads was fitted to the thing being searched for, so the search inherits no selection effect from it.
* **It is exact.** No value is merged with another, no boundary is invented, and no information is discarded. The only merging that ever happens is the grid-capacity coarsening of Section 2.3, which is applied per candidate combination and stated as such.
* **A numeric column is not special.** `1`, `2`, `3` are three categories, exactly as `low`, `mid`, `high` would be. The order of the codes follows the values' natural order, which is what makes the capacity coarsening below merge *adjacent* levels, but nothing else in the framework reads the codes as quantities.

### 2.2. What this rules out, deliberately

There is **no binning layer**: no bin-count heuristic, no quantile cut points, no per-channel level limits, and no rate-distortion distortion figure. A framework that quantized continuous features would be a different product, and three arguments keep it out of this one:

1. **An interval is not a category.** A boundary chosen by a density heuristic is an artefact of the estimator, and an axis tick reading `(3.7, 4.1]` is not a value an analyst can act on. A centre certified inside such an interval is a statement about a bin edge, not about the data.
2. **Nothing downstream can use it.** A distortion figure describing how much of a fine quantization a coarse code retains is not an input to the ranking key (Definition 5), to the certificate (Definition 3), or to anything the display shows.
3. **Fitting the encoding to the data is an uncorrected exposure.** Bin edges chosen from the same rows the search is then ranked on are a selection effect, and Section 4.5 controls no such thing. Encoding distinct values removes the exposure by construction instead of correcting it.

The same decision removes the last consumer of mutual information: the distortion figure was the only place it was computed. **No information-theoretic quantity is computed anywhere in this framework** (Section 3.1).

**The consequence, stated plainly.** A genuinely continuous column — thousands of distinct floats — now produces thousands of categories instead of at most a few hundred bins. It is not rejected, not truncated and not silently quantized: it is encoded as it stands, and the grid-capacity rule below is what keeps the joint table estimable when such a column enters a candidate combination. Feeding continuous measurements to a categorical framework yields an axis with thousands of ticks; that is a data-preparation decision belonging to the caller, and the design makes it visible rather than papering over it with a binning heuristic.

### 2.3. Grid capacity limit

For a candidate subset $S$ ($|S| \le 4$), the number of cells $\prod_{j \in S} k_j$ must not exceed $\max(1, \lfloor N/10 \rfloor)$ (`vsf.pmd.check_grid_capacity`). If it does, levels are merged (`adaptively_coarsen_bins`) *before* that combination is scored: each column's per-axis cap is $\lfloor (N/10)^{1/d} \rfloor$, and its sorted distinct levels are split into that many near-equal consecutive groups.

This is not cosmetic. The exhaustive search of Section 4 scores every combination, and without the ceiling a fraction of them would be ranked on tables so sparse that a cell holding one or two objects decides the winner. The limit bounds cell-count variance; it does not by itself bound selection optimism (Section 10, item 3).

Two honest notes about the merging rule. It merges **adjacent codes**, i.e. adjacent entries of the column's sorted distinct values — meaningful for an ordered category set, arbitrary though deterministic for an unordered one. And it applies **only** where the alternative is scoring cells of a handful of rows each, never to columns whose combination already fits.

### 2.4. Human-in-the-loop

The system does not assign semantics on its own. The user names the target column and the positive value (Section 1.4). There is nothing else to configure about the encoding: with no bin counts and no cut points, a column has exactly one representation, and two runs on the same data produce the same axes.

---

## 3. Mathematical Core

### 3.1. Notation, and what is deliberately absent

A branch is a subset $S$ of feature columns; $\tilde X_S$ is the joint category code of those columns, i.e. the partition of the rows into **cells**. $C$ denotes the number of *occupied* cells, $\mathbf{c}$ one of them, $n_\mathbf{c}$ its size and $k_\mathbf{c}$ the number of target-value objects in it. $N$ is the number of rows, $N_+$ the number of target-value objects, $N_+/N$ the base rate.

**No information-theoretic quantity is computed anywhere in this framework.** There is no entropy, no mutual information, no normalized variant, and no bias correction for one — not in the search, not in the report, not in the encoding. Two independent reasons put them out of scope, and both are worth stating because each removed a different defect:

* **They answer the wrong question.** An association statistic answers "is there a relationship between the target and this partition". The deliverable is narrower: "which cells are almost purely this target value, and how much of it do they hold". On a rare target the two have opposite answers — a partition can score a high association share while its purest cell holds a few percent of the target value, which is not something any display can act on.
* **The plug-in estimator cannot rank candidates of unequal cardinality.** Its bias under independence grows with the number of cells, so an $\arg\max$ over it selects on grid size rather than on relationship. Correcting that bias is possible and was once done, but it repairs a statistic that fails the first point anyway.

The last remaining use of mutual information was the rate-distortion distortion figure of the old continuous-feature binning layer, and it went with that layer (Section 2.2).

Everything reported is therefore a **count ratio with an exact binomial confidence bound**: coverage, purity, and the certification of a single cell. The one place a probability model enters is the null distribution of Definitions 8–9, which is the exact multivariate hypergeometric law of the cell counts — combinatorics, not information theory.

### 3.2. Reported metrics: certified discrete centres

Throughout, one target value $z^\ast$ is fixed and $\tilde Z = \mathbb{1}[Z = z^\ast]$.

#### Definitions

**Definition 2 (Cell purity).** For an occupied cell $\mathbf{c}$ holding $n_\mathbf{c}$ objects of which $k_\mathbf{c}$ carry $z^\ast$,

$$\pi(\mathbf{c}) = \Pr\big[Z = z^\ast \mid \tilde X_S \in \mathbf{c}\big], \qquad \hat\pi(\mathbf{c}) = k_\mathbf{c} / n_\mathbf{c}.$$

**Definition 3 (Discrete centre).** Fix a purity floor $\tau$, a minimum occupancy $m \ge 1$ and a rule. Let $C$ be the number of *occupied* cells of the branch.

*Rule P (purity — the default).* Cell $\mathbf{c}$ is a centre iff

$$\hat\pi(\mathbf{c}) = \frac{k_\mathbf{c}}{n_\mathbf{c}} \;\ge\; \tau \qquad\text{and}\qquad n_\mathbf{c} \ge m, \qquad \tau \in (0, 1].$$

This is a statement about the observed table, so $\tau = 1$ is admissible and means "cells that are entirely the target value" — a request users actually make. $\tau$ and $m$ belong to the user, and $\tau$ is simultaneously the green colour boundary (Section 5.3), so the set drawn green and the set counted are identical by construction.

*Rule C (certified).* Cell $\mathbf{c}$ is a centre iff the one-sided exact binomial test of

$$H_0: \pi(\mathbf{c}) \le \tau \qquad\text{against}\qquad H_1: \pi(\mathbf{c}) > \tau$$

rejects at level $\alpha / C$, i.e. iff

$$\Pr\big[\mathrm{Bin}(n_\mathbf{c}, \tau) \ge k_\mathbf{c}\big] \;\le\; \alpha / C
\qquad\Longleftrightarrow\qquad
L_{\mathrm{CP}}\big(k_\mathbf{c}, n_\mathbf{c};\, \alpha/C\big) \;\ge\; \tau,$$

where $L_{\mathrm{CP}}(k,n;\alpha) = \mathrm{Beta}^{-1}(\alpha;\, k,\, n-k+1)$ is the Clopper–Pearson lower bound ($L_{\mathrm{CP}} = 0$ for $k = 0$). The implementation evaluates the tail for the decision and the quantile for the displayed bound, and the test suite pins that the two agree cell by cell.

Both rules feed the identical Definition 4, so coverage, $K$, the cross-validated estimate, the permutation nulls and the reported dimensionality are written once and are comparable *in kind* between rules, though not in value. **Under either rule every cell carries its $[L_{\mathrm{CP}}, U_{\mathrm{CP}}]$ interval.** Under Rule C the interval decides; under Rule P it is information attached to the cell — which is what makes Rule P's weakness visible on the object rather than arguable in the abstract: a cell that is green on the strength of one object displays the interval $[\alpha/C,\, 1]$.

**Proposition 2 (Simultaneous validity of Rule C).** Let $\mathcal{N} = \{\mathbf{c} : \pi(\mathbf{c}) \le \tau\}$ be the cells that do *not* deserve certification. Under Rule C,

$$\Pr\big[\exists\, \mathbf{c} \in \mathcal{N} \text{ selected}\big] \;\le\; \sum_{\mathbf{c}\in\mathcal{N}} \frac{\alpha}{C} \;\le\; \alpha .$$

*Proof.* Each test has exact level $\le \alpha/C$ under its own $H_0$ (Clopper–Pearson is conservative, never anti-conservative); a union bound over $|\mathcal{N}| \le C$ tests gives the claim. $\square$

Bonferroni is used rather than a sharper procedure for two reasons. The cell counts are multinomial and hence negatively associated, so the union bound errs conservatively — the right direction for a certificate. And the family is not the selected cells but *every occupied cell*: the analyst scans the whole lattice and reads the green ones, so the multiplicity is $C$ however many turn green.

**Corollary 1 (100 % purity: observable, not certifiable).** Under Rule P, $\tau = 1$ selects exactly the cells with $k_\mathbf{c} = n_\mathbf{c}$. Under Rule C no cell is ever selected at $\tau = 1$ at any sample size, since $H_0: \pi \le 1$ can never be rejected. `CenterSpec` therefore rejects $\tau = 1$ under Rule C rather than clamping it, so a request to *prove* exact purity fails loudly instead of quietly answering a different question. Rule P describes the table; Rule C infers about the population.

**Corollary 2 (what $m = 1$ costs, and what removes it).** Under Rule P with $m = 1$, a cell holding a single target-value object has $\hat\pi = 1$, is a centre, and contributes $1/N_+$ to coverage. Its interval is $[\alpha/C, 1]$. Measured on pure noise at the density ceiling of Section 2.3 ($C = 3\,000$ cells over $N = 32\,561$ rows, target independent of every cell at prevalence $0.24$, $\tau = 0.90$, seed 1): $K = 1$, coverage $0.05\,\%$, permutation $p = 0.22$. The price is real, it is small, and the permutation test does not mistake it for a finding. Either remedy drives it to exactly zero — $m = 20$, or Rule C. Under Rule C the parameter is redundant anyway: $L_{\mathrm{CP}}(1,1;\alpha) = \alpha$, and the smallest fully pure cell selectable at $\tau = 0.90$, $\alpha = 0.05$, $C = 1$ is $n = 29$, since $\alpha^{1/n} \ge \tau \iff n \ge \ln\alpha / \ln\tau = 28.4$.

**Definition 4 (Deliverable triple).** Let $\mathcal{G}$ be the set of centres, $N_+ = \sum_\mathbf{c} k_\mathbf{c}$ the number of target-value objects and $N$ the number of rows.

$$K = |\mathcal{G}|, \qquad
\mathrm{Coverage} = \frac{\sum_{\mathbf{c}\in\mathcal{G}} k_\mathbf{c}}{N_+}, \qquad
\mathrm{Purity}_{\mathrm{pooled}} = \frac{\sum_{\mathbf{c}\in\mathcal{G}} k_\mathbf{c}}{\sum_{\mathbf{c}\in\mathcal{G}} n_\mathbf{c}} .$$

$\mathrm{Coverage}$ is recall and is the **headline**: the share of the target value the display localizes. $\mathrm{Purity}_{\mathrm{pooled}}$ is precision. $K$ is reported because the goal is a *small* readable set of cells, so at equal coverage fewer centres is strictly better. Two further quantities are carried because no single scalar suffices on a rare target: $\mathrm{Mass} = \sum_{\mathcal{G}} n_\mathbf{c} / N$, the fraction of the population one must inspect, and $\mathrm{Lift} = \mathrm{Purity}_{\mathrm{pooled}} / (N_+/N)$, the enrichment over the base rate.

Coverage and pooled purity are each reported with a Clopper–Pearson lower bound at level $\alpha$, unadjusted — each is one pre-specified ratio, not a maximum over cells.

**Definition 5 (Ranking key, `vsf.centers.coverage_score`).** Candidate subsets are ordered by the lexicographic tuple

$$\big(\mathrm{Coverage}(S),\; -K(S),\; -\mathrm{Mass}(S),\; \max_{\mathbf c} L_{\mathrm{CP}}(\mathbf c)\big).$$

Capture as much of the target value as possible; among equal captures prefer fewer centres (a smaller, more readable finding); among equal centre counts prefer less population mass (the more concentrated finding). The fourth component is load-bearing rather than cosmetic: when **no** candidate certifies anything — the common case on a rare target, and precisely the case this layer exists to report honestly — the first three components are identically $(0,0,0)$ for every candidate, and an $\arg\max$ over them alone returns whichever combination the enumeration happened to reach first. Breaking the tie on the best per-cell lower bound returns the branch that came closest to certifying, which is the only informative thing left to say and is fully determined by the data.

#### Out-of-sample coverage

**Definition 6 (Cross-validated coverage).** Over $R$ repetitions of a stratified $K$-fold split, centres are selected on the training fold by Definition 3 (with $C$ recomputed on that fold, and a cell unseen in training treated as not a centre), and $\mathrm{Coverage}$ is evaluated on the held-out fold. $\mathrm{Coverage}_{\mathrm{CV}}$ is the mean of the $RK$ fold values; its standard error is the Nadeau & Bengio (2003) corrected

$$\mathrm{se}^2 = \Big(\tfrac{1}{RK} + \tfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\Big)\, s^2, \qquad \tfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}} = \tfrac{1}{K-1},$$

which accounts for the overlapping training sets. At the default $R = K = 5$ the naive $s/\sqrt{RK}$ understates the spread by a factor of $\approx 2.3$ and manufactures improvements from $d$ to $d+1$ that do not replicate.

**Why this is required, not optional.** Centres are chosen by inspecting their own contents, so in-sample coverage is optimistic even when every individual cell is honest. Rule P controls nothing here at all; Proposition 2 bounds the rate at which an *individual* cell is falsely certified and says nothing about a coverage computed over cells chosen this way. Measured on a constructed case (40 cells of 300 objects, true purities drawn $U[0.85, 0.95]$ so they straddle $\tau = 0.90$, seed 0): in-sample $66.3\,\%$ over $K = 26$ centres, cross-validated $63.9 \pm 2.0\,\%$. The in-sample figure is not wrong — it describes the observed table exactly — but it is not the figure a reader will reproduce. Cross-validation is also what exposes a coverage built out of one-object cells, since such cells do not recur on a held-out fold.

This is the most expensive part of a search ($R \times K = 25$ refits per branch) and `cv_repeats = 0` disables it for a caller that only needs the ranking.

**Definition 7 (Power floor).** With $N_+$ target-value objects and $K$ folds, a held-out fold carries $N_+/K$ of them, so a fold-level coverage has standard error at least $1/(2\sqrt{N_+/K})$; pooling over $RK$ splits and applying the Nadeau–Bengio factor $\sqrt{1/(RK) + 1/(K-1)} = 0.53$ at $R = K = 5$ leaves a pooled standard error no smaller than $\approx 0.5/\sqrt{N_+}$. Requiring it to be at most $0.10$ gives $N_+ \ge 25$ (`MIN_POSITIVES_FOR_CV`). Below that the report says **"undetermined"** and states why, rather than printing a number: a coverage of $0\,\%$ measured on 9 objects and one measured on 900 are different claims and must not render identically.

#### Null distribution and multiplicity

**Definition 8 (Coverage null).** Conditional on the cell sizes $\{n_\mathbf{c}\}$ and on $N_+$, the permutation null of the per-cell counts is exactly the multivariate hypergeometric law. Coverage is evaluated on draws from it, giving an exact permutation $p$-value at $O(C)$ per replicate instead of $O(N)$. The selection threshold depends only on $n_\mathbf{c}$ — it is $\lceil \tau n_\mathbf{c} \rceil$ under Rule P and the exact binomial threshold under Rule C — and permutation preserves cell sizes, so it is tabulated once and every replicate reduces to one integer comparison per cell. Reported as `coverage_p_value`; on by default at 999 replicates.

**Definition 9 (Familywise coverage null).** A reported branch is an $\arg\max$ over $\sum_{d\le 4} \binom{M}{d}$ candidates, so Definition 8's $p$-value is anti-conservative for it. The look-elsewhere-corrected null is the distribution of $\max_S \mathrm{Coverage}(\tilde Z_\pi; \tilde X_S)$ under a **single shared** permutation $\pi$ applied to every candidate. The permutation must be shared: candidates are partitions of the same rows by overlapping feature sets and are strongly dependent, so maximizing over independently drawn per-candidate nulls would overstate the null maximum. Reported as `coverage_p_value_familywise`. **Off by default** (`n_permutations_familywise_coverage = 0`) because it costs $B$ times the whole candidate family — which means every interactively obtained $p$-value is uncorrected, and is labelled as such. A published number must set it.

#### Sufficient dimensionality

**Definition 10 (Smallest sufficient $d$).** Scanning $d = 1,2,3,4$ in increasing order and holding an incumbent, $d$ replaces the incumbent when the **paired** difference of their per-fold coverages exceeds $t^\ast$ Nadeau–Bengio corrected standard errors; the first incumbent must clear zero by the same margin. The answer is the last accepted $d$, and **None** when no $d$ achieves a coverage distinguishable from zero.

Three properties are deliberate:

* It is the smallest *sufficient* dimensionality, not the first useful one. If $d=3$ genuinely beats $d=2$, then two characteristics do not describe the target and the answer is 3.
* The scan does not stop at the first flat step. Coverage is not submodular (Section 1.1), so a flat step from $d$ to $d+1$ does not license skipping $d+2$.
* **None** is a result, not a failure, and is reported as one.

The pairing is valid because the fold assignment is a function of $(\tilde Z, K, R, \text{seed})$ alone and not of the cell partition, so all four branches are evaluated on byte-identical folds. $t^\ast = 2$ is a two-sigma convention on a statistic whose reference distribution is approximately $t$ with $RK-1$ degrees of freedom (Nadeau & Bengio, 2003, Sec. 4); at $R = K = 5$ the exact $0.975$ quantile is $2.06$, so the default is marginally liberal and is stated as a convention rather than derived.

#### Worked example

All rows of the UCI "Adult" / Census Income dataset ($N = 32\,561$; public, not bundled with this library), Rule P, $m = 1$, $\alpha = 0.05$, features = every column except the target's own, defaults otherwise. Reproduced by running `vsf.discover_branches` directly.

| criterion | $\tau$ | $d$ | branch | $K$ | Coverage | $\mathrm{Coverage}_{\mathrm{CV}}$ | $p$ | $d^\ast$ |
|---|:---:|:---:|---|---:|---:|---:|---:|:---:|
| `occupation = Armed-Forces` ($N_+ = 9$) | 0.90 | 2 | `workclass+education` | 0 | 0.0 % | undetermined | 1.0 | none |
| | 0.90 | 3 | `workclass+relationship+race` | 1 | 11.1 % | undetermined | 0.006 | none |
| `income = >50K` ($N_+ = 7\,841$) | 0.90 | 2 | `workclass+education` | 2 | 1.2 % | $1.2 \pm 0.1\,\%$ | 0.001 | 3 |
| | 0.90 | 3 | `workclass+education+occupation` | 55 | 2.4 % | $1.9 \pm 0.1\,\%$ | 0.001 | 3 |
| | 0.75 | 2 | `education+marital_status` | 6 | 17.5 % | $17.4 \pm 0.6\,\%$ | 0.001 | 3 |
| | 0.75 | 3 | `education+marital_status+occupation` | 46 | 24.6 % | $24.8 \pm 0.7\,\%$ | 0.001 | 3 |
| `relationship = Husband` ($N_+ = 13\,193$) | 0.90 | 2 | `marital_status+sex` | 2 | 100.0 % | $100.0 \pm 0.0\,\%$ | 0.001 | 2 |
| | 1.00 | 4 | `workclass+education+marital_status+sex` | 75 | 19.2 % | $22.6 \pm 1.6\,\%$ | 0.001 | 4 |

Four things this table exists to make unavoidable.

1. **`Armed-Forces` at $\tau = 0.90$ is what $m = 1$ buys.** One centre, holding one of the nine target objects — a coverage of $11.1\,\%$ that is a single row of the dataset. $N_+ = 9$ is below the power floor, so there is no out-of-sample estimate to contradict it, and $d^\ast$ is correctly **none**.
2. **Coverage and $K$ are in tension, and the specification does not resolve it silently.** For `income = >50K` at $\tau = 0.90$, going from $d = 2$ to $d = 3$ raises coverage from $1.2\,\%$ to $2.4\,\%$ and the centre count from 2 to 55. Coverage dominates and $K$ only breaks ties (Definition 5), so $d = 3$ wins the search — but 55 cells is not a readable finding whatever the coverage says. The panel shows both numbers; a user who wants the two-centre answer lowers $d$ or raises $\tau$. The framework will not make that trade for them.
3. **$\tau$ is the dominant parameter, not a cosmetic one.** The same target and the same branch family move from $2.4\,\%$ coverage at $\tau = 0.90$ to $24.6\,\%$ at $\tau = 0.75$. **A coverage quoted without its $\tau$ is meaningless.**
4. **$\tau = 1$ is a real setting with a real price.** `Husband` is captured essentially completely by two cells at $\tau = 0.90$. Demanding *entirely* pure cells costs 81 points of coverage and 73 additional centres, and the cross-validated figure ($22.6\,\%$) exceeds the in-sample one — the signature of a rule so strict that fold-to-fold variation in cell contents dominates the selection.

---

## 4. Independent Branch Discovery Algorithm

### 4.1. Overview

Four independent exact searches, one per dimensionality $d \in \{1,2,3,4\}$, each exhaustive, each ranked by Definition 5. There is no greedy chain, no stopping rule that gates selection, and no single $d^*$ returned by the search.

### 4.2. Formal definition of a branch

$$S^*_d = \underset{S \subseteq \{1,\dots,M\},\ |S| = d}{\arg\max} \; \mathrm{CoverageScore}\big(\mathbb{1}[Z=z^\ast];\, \tilde{X}_S\big)$$

per Definition 5: coverage first, then fewer centres, then less mass, then — only when nothing certifies at all — the best per-cell lower bound. For $d > M$ the branch is undefined (not enough features). A positive value $z^\ast$ is required to evaluate this at all (Section 1.4); `discover_branches` raises rather than substituting a different ranking.

Coverage is a recall fraction bounded in $[0,1]$ regardless of how many cells a candidate's partition has, so the ranking needs no cardinality-bias correction. What a wider candidate family *does* buy is a better chance that some cell clears $\tau$ by chance alone — that is **selection optimism**, not estimator bias, and it is handled by the cross-validation of Definition 6 and the familywise null of Definition 9, not by adjusting the point estimate before ranking.

### 4.3. Algorithm

```
Algorithm: Independent Branch Discovery (IBD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:   Dataset D, target column + resolved positive value z*
Output:  Up to 4 branches {(d, S*_d, CenterReport_d)}, d = 1..min(4, M)

1. Encode every feature column as categories (Section 2).
2. Resolve z* (Section 1.4); raise if it cannot be resolved.
3. FOR d = 1 TO min(4, M):
      a. FOR EACH combination S subset of {1..M}, |S| = d:
           Compute X̃_S (joint discrete code of features S,
                         adaptively coarsened to the capacity limit)
           Compute CoverageScore(1[Z=z*]; X̃_S)          (Definition 5)
      b. S*_d <- argmax over all S, by CoverageScore
      c. Compute the full CenterReport for S*_d           (Section 3.2:
         coverage, K, purity, cross-validation, permutation p-value)
      d. Save (d, S*_d, CenterReport_d) as branch d
4. RETURN all found branches
```

No stopping condition or significance threshold gates *selection*. The four searches are independent and always run in full. Statistical control attaches to the *report* of an already-selected branch, never to the search (Section 4.5).

### 4.4. Why branches do not have to be nested

Coverage $f(S) = \mathrm{Coverage}(\tilde{Z}; \tilde{X}_S)$ is not submodular in general (Krause & Guestrin, 2005): with feature synergy $f$ can be supermodular, so $f(S \cup \{a, b\}) - f(S)$ can exceed the sum of the two individual gains. Practically: a pair of features, neither of which belongs to the best single axis $S^*_1$, can jointly form a far better $S^*_2$ — a certified centre that neither axis alone can certify. A greedy algorithm growing $S^*_1 \to S^*_2 \to S^*_3$ one feature at a time **cannot find** such a pair in principle, because it never re-evaluates an already selected feature. Exhaustive search per $d$ is the only exact way to guarantee the true $S^*_d$ under synergy; `tests/test_avr.py`'s XOR-synergy fixture is a live executable instance of exactly this case.

### 4.5. Statistical control: what is guaranteed and what is not

Selection is a deterministic exhaustive $\arg\max$ over Definition 5. The tests below attach a *claim* to an already-selected branch; they never drive the choice. Nothing about the search depends on a random seed except which rows land in which cross-validation fold, and that seed is fixed by default (`random_state=0`).

**Two sources of optimism, handled separately.**

1. **Sampling variability of a pre-specified subset** — addressed by the coverage permutation $p$-value (Definition 8), drawn from the multivariate hypergeometric law of the cell counts. Cheap, $O(C)$ per replicate, on by default.
2. **The look-elsewhere effect of the search itself** — `coverage_p_value` is *not* valid for a branch chosen as the maximum over $\sum_d \binom{M}{d}$ candidates. The corrected statistic is the shared-permutation max null of Definition 9, `coverage_p_value_familywise`. Off by default because it costs $B$ times the entire search; **any published result must set it.**

**What is not controlled.**

* **No FDR correction within a single search.** This specification reports the *maximum* per $d$ and tests that maximum; it does not enumerate which of the $\binom{M}{d}$ candidates are individually significant. That is a different question. Multiplicity across *targets*, where it genuinely accumulates, is the Global Pattern Scan's problem (Section 4.7).
* **No out-of-sample validation of the centre *set*, only of coverage.** Definition 6 answers "does this much coverage survive on held-out rows"; it does not certify that the same cells would win the search on a fresh sample.
* **Nothing about the encoding is fitted, so nothing about it needs correcting.** Category codes are a function of each column alone (Section 2.1). The earlier binning layer chose bin counts and cut points from the same data the search then ranked on; that exposure no longer exists rather than being corrected.

### 4.6. Computational cost

The exhaustive search has no pre-filter, no feature-count limit, and never switches to an approximate search. Combinations evaluated for all four branches:

$$\text{Total}(M) = \sum_{d=1}^{\min(4,M)} \binom{M}{d}$$

| $M$ (features) | $\binom{M}{1}$ | $\binom{M}{2}$ | $\binom{M}{3}$ | $\binom{M}{4}$ | Total |
|:---:|---:|---:|---:|---:|---:|
| 10 | 10 | 45 | 120 | 210 | 385 |
| 22 (mushrooms) | 22 | 231 | 1,540 | 7,315 | 9,108 |
| 30 | 30 | 435 | 4,060 | 27,405 | 31,930 |
| 50 | 50 | 1,225 | 19,600 | 230,300 | 251,175 |
| 100 | 100 | 4,950 | 161,700 | 3,921,225 | 4,087,975 |

Each candidate's score costs $O(N)$ (vectorized joint code + cell tabulation), so the ranking pass is $O(N \cdot \text{Total}(M))$.

**Where the time actually goes.** The ranking pass only orders candidates. The expensive statistics run on the up-to-four *winners*:

| Component | Cost | Default |
|---|---|---|
| `CoverageScore` per candidate (ranking pass) | $O(N)$ per combination, vectorized | always on |
| Cross-validated coverage (Definition 6) | $R \times K = 25$ stratified refits per branch | on (`cv_repeats=5`, `cv_splits=5`) |
| Coverage permutation $p$-value (Definition 8) | $O(C)$ per replicate | on (999 replicates) |
| Familywise coverage null (Definition 9) | $B \times \text{Total}(M)$ table builds | **off** (`n_permutations_familywise_coverage=0`) |

Because the reporting layer touches only the winners, its cost is independent of $M$ for a fixed number of branches and is dominated by $N$ and the 25-fold cross-validation. `cv_repeats=0` and `n_permutations_centers=0` remove those two rows for a caller that needs only the ranking.

**Measured end to end** (`discover_branches` plus four rendered payloads, all defaults, 2-core laptop, 2026-09):

| Dataset | $N$ | $M$ | search | payloads | total |
|---|---:|---:|---:|---:|---:|
| UCI Mushroom (`class = p`) | 8,124 | 22 | 1.6 s | 1.0 s | **2.6 s** |
| UCI Adult / Census Income (`income = >50K`) | 32,561 | 7 | 1.3 s | 1.9 s | **3.1 s** |
| UCI Bank Marketing (`subscribed_term_deposit`) | 41,188 | 13 | 1.7 s | 1.5 s | **3.1 s** |

All three are public UCI datasets, not bundled with this library.

What remains is dominated by first-time Clopper–Pearson inversions on the winning branches' cells and by the cross-validation — both inherent to what is reported, not to how it is computed. The search itself is an exact re-implementation of the straightforward form, pinned against it candidate-by-candidate in `tests/test_fastpaths.py`: a per-column coarsening cache, sort-free dense cell codes, prefix-shared mixed-radix codes with one fused `bincount` for $(n_\mathbf{c}, k_\mathbf{c})$, memoized Clopper–Pearson bounds and a tabulated `lgamma`. `scipy.special.betaincinv` would be another order of magnitude on the bounds and was deliberately not taken: it would make the reported digits depend on whether an optional package is installed.

**Datasets with $M \gtrsim 50$–100 will take tens of minutes.** That is the stated price of an exhaustive search, not an oversight, and the right response is a time estimate shown before launch — not a silent substitution of an approximate algorithm.

**The interactive click path.** A first click on a target value costs roughly 2–3 s on the server (per-cell bounds, the 999-replicate null, and four payloads), plus the browser's first WebGL render, in which most of the time is Plotly triangulating axis tick labels into text meshes. Three mechanisms address this: the frontend pre-triangulates every catalog label off-screen while the welcome guide is shown; the server keeps an LRU of *serialised* responses (16 entries or 512 MB, whichever binds first) keyed by target, value and certificate parameters, with in-flight de-duplication; and after each criterion-mode analysis a background thread precomputes the column's other values through one shared enumeration, cancelled when the user moves to another column. Every prefetched body is byte-identical to a direct request's (`tests/test_serve.py::TestAnalyzeCacheAndPrefetch`), so the only behavioural change is that later clicks within a column answer in tens of milliseconds. `prefetch=False` turns the last one off.

### 4.7. Global Pattern Scan (server only)

An optional, button-triggered operation that applies Section 4.3 not to one target but to **every** (column, value) pair in the dataset, each taken as a One-vs-Rest criterion — the same form as a manual criterion.

* For each pair, $\tilde Z = \mathbb{1}[c = v]$ is built and **exactly the same** exhaustive search is run over all other columns. Not an approximation, not a cheaper variant. From the up-to-four branches the maximum coverage is taken; ties break toward fewer centres, then toward lower $d$.
* It **filters** the existing flat target selector. It never replaces target selection or pre-selects a target, and its statistics never touch selection.
* **Two gates are available, and the shipped UI uses only one.** The server keeps a pair when its best branch's coverage reaches the user's threshold **and** the pair survives Benjamini–Hochberg FDR control at rate `fdr_q` across every pair scanned in the run. The web application deliberately sends `fdr_q = 1.0`, which makes the BH step a verified no-op, and relies on the coverage threshold plus a scan-specific **minimum cell occupancy** instead. The trade is stated rather than hidden: a sweep of $k$ targets with no significance gate produces spurious "patterns" by construction, and minimum occupancy is the only remaining guard. A caller using the API directly can set `fdr_q` and get the control back. When BH is used it must run over the whole scanned family *before* the effect-size filter — filtering first and correcting afterwards is itself a selection effect and voids the guarantee.
* `n_permutations_familywise > 0` additionally upgrades each pair's $p$-value to the look-elsewhere-corrected one, covering both levels of multiplicity (across targets, and within each target's own subset search) at $B$ times the per-pair cost. The shipped UI sends `0`.
* **Cost scales linearly** in the number of (column, value) pairs: with $M$ columns and $\bar k$ values per column, $\approx M \cdot \bar k$ full runs of Section 4.3. The reporting layer, not the ranking pass, dominates. No pre-filter or shortcut is applied.
* Because the operation exceeds any reasonable synchronous HTTP timeout, it runs in a background thread with a client-polled progress endpoint and cooperative cancellation between (not within) branch searches.
* **Server only.** The static export (`export_full_dashboard`) has no backend to run a search in, so the feature does not exist there.

---

### 4.8. Absence search: certifying where a value is *not*

**Motivation.** A purity floor $\tau$ is only meaningful above the base rate of what it is applied to. If the chosen value occupies $p_0 = 90\,\%$ of the rows, then at $\tau = 0.90$ the trivial one-cell partition is already a certified centre with coverage 1, every refinement inherits that, and the display turns uniformly green while saying nothing: a needle-in-a-haystack search for hay. For such a value the informative question is the mirror one — *in which cells is it almost absent, and how much of the data can be certified free of it* — and that question is measured against the base rate of the complement, $1 - p_0 = 10\,\%$, where $\tau = 0.90$ is again demanding.

**Definition (search direction).** Let $z_i = \mathbb{1}[Z_i = v]$ be the indicator of the chosen value. A search has a direction $\delta \in \{\text{presence}, \text{absence}\}$ and runs on the oriented indicator $z^{(\delta)} = z$ for presence and $z^{(\delta)} = 1 - z$ for absence. Every definition of Section 3.2 — centre, certificate, coverage, $K$, mass, cross-validated coverage, the hypergeometric and familywise nulls — and every step of the search (Section 4.3) is applied to $z^{(\delta)}$ unchanged. In particular an *absence centre* of $v$ is a cell whose share of rows **without** $v$ reaches $\tau$ (equivalently, whose share of $v$ is at most $1 - \tau$), and

$$\text{absence}(v) \equiv \text{presence}(\lnot v)$$

holds by construction, not by approximation: `tests/test_absence.py` pins field-for-field equality of the two branch sets. This is deliberately **one algorithm with two directions, and two questions with two reports** — a second search procedure for absence would either duplicate the first or, if it differed, introduce two incompatible statistics for symmetric questions.

**Base-rate invariant.** A search in direction $\delta$ is well-posed only if $\tau > \bar{z}^{(\delta)}$, the prevalence of the oriented indicator. `vsf.avr.base_rate_reason` states the violation; `discover_branches` raises on it, the live server answers HTTP 400 with the reason and (for a presence search) names the remedy, and the Global Pattern Scan records such $(\text{column}, \text{value})$ pairs under `skipped` with their reasons instead of reporting them as trivially covered. The invariant is the same inequality in both directions and is what makes "this value is too common to localise at this $\tau$" a stated result rather than a green screen.

**What differs between the two reports.** The headline of a presence search is coverage (the share of the value captured by certified centres). The headline of an absence search is the **mass** of certified $v$-free cells — the share of *all* rows an analyst may exclude from a search for $v$ — with the coverage of the complement second; at high $\tau$ the two order candidate schemas identically, so the ranking key is unchanged. `CenterReport.prevalence` reads $1 - p_0$. Class labels keep naming the value ("Column = v" / "not Column = v") so a point's hover stays literal, while the analysis title reads "Column ≠ v". Certified cells are drawn **red** ("certified free of $v$"), never green — a green cell is a presence certificate, and the absence search does not make one; rows rich in $v$ but certified for nothing at this $\tau$ are slate grey, the band between the boundaries brown. For a $K$-class target an absence centre means "$v$ is not here", not "some particular other value is here": the complement is heterogeneous, and the legend says so.

**Brown is a residual, not a target.** Cells certified for neither direction remain brown. Maximising their number would reward the least informative schema (under independence every cell sits at the base rate, i.e. every cell is brown), so the mixed zone is never an objective; it is reported as what the schema could not resolve either way, $1 - \text{mass(green)} - \text{mass(red)}$ when both directions are certified on one display, and it is the natural quantity that distinguishes this lattice from a decision tree of the same depth, which has no abstention zone.

**API.** `discover_branches(..., direction=)`, `iter_branches_by_value(..., direction=, skipped=)`, `discover_branches_by_value` likewise, `BranchEngine(direction=)`, `export_full_dashboard(..., direction=)`; `/api/analyze` and `/api/scan/start` take `direction` (`"presence"` default, `"absence"`), the analyze response and every scan row carry it, and it is part of the response cache key and of the sibling-value prefetch.

### 4.9. The solution landscape: every schema the search scored

**What it is.** The exhaustive search evaluates $\text{Total}(M)$ candidate schemas (Section 4.6) and reports four winners. Everything it learned about the other candidates was discarded, although it is exactly the context a reader needs to judge a winner: how many schemas reach a comparable coverage, at how many centres; whether the optimum is isolated or one of hundreds of near-equivalent schemas (the *Rashomon set* of this search, in the sense of Semenova, Rudin & Parr, 2022); and how many schemas certify nothing at all at this $\tau$. `vsf.avr.Landscape` keeps, for every candidate, its dimensionality, its axes and the three integers its ranking key is made of ($k_{\text{sel}}$, $K$, $n_{\text{sel}}$). Recording costs nothing beyond the search itself, and the landscape is computed on demand from the same ranking pass (`compute_landscape`), cached per (target, value, certificate, direction).

**Display.** The landscape is drawn with the product's own visual language — a lattice of cells with a circle in each — rather than a scatter plot: the two axes are *ordered categories*. On $x$, the coverage of the value (under an absence search, the mass of certified value-free cells) in the left-open, right-closed intervals $(0, 10], (10, 20], \ldots, (90, 100]$ percent. On $y$, the schema's number of certified centres as a share of $K_{\max}$ — the largest centre count among the schemas shown (per dimensionality, or over the whole family) — in the same ten intervals. A cell's colour is the number of schemas that fall in it, under a three-zone scale whose two boundaries belong to the user and whose right end is the fullest cell; the count is written in the cell. Schemas that certify no centre have coverage $0$ by construction and are **excluded from the lattice** and counted above it ("$n_0$ of $n$ schemas certify no centre at $\tau$") — placing them in the first cell would swamp it with schemas that are not weak but empty. The search winner of the selected branch is marked on the lattice from its exact numbers, so the reader always sees where the reported result sits in the family.

**Drill-down.** Clicking a cell lists its schemas — axes, exact coverage, $K$, mass — most concentrated first (lowest mass, then fewest centres, then highest coverage), paged from the server (`/api/landscape/cell`) since a cell can hold tens of thousands of schemas at $M \approx 50$. Any listed schema can be opened as a lattice (`/api/analyze` with `features=[\ldots]`, `vsf.avr.report_schema`); the landscape state survives the round trip so the reader returns to the same cell.

**Statistical status of a schema opened from the landscape.** Such a schema is selected by looking at the data. Its in-sample coverage is what the lattice shows, its cross-validated coverage remains an out-of-sample estimate (the folds are held out whatever led to the choice), and the familywise $p$-value, when computed, remains valid and conservative (the null of the maximum over the family bounds any member). Its *uncorrected* permutation $p$-value would be the $p$-value of a hypothesis chosen because it looked good and is therefore **not reported** for an opened schema; the panel says why.

**What the landscape adds to the claims.** It makes the coverage–$K$ tension of limitation 5 visible without changing the ranking rule: the reader sees the two-centre schema at $1.2\,\%$ beside the fifty-five-centre schema at $2.4\,\%$ and chooses. It replaces the "random subsets" baseline a reviewer would ask for with the exact distribution over the whole family. And its empty-schema count is the most direct statement of what a threshold means on a dataset: at $\tau = 0.9$ on `income = >50K`, the honest summary is not the winner's $2.4\,\%$ but that nearly every schema certifies nothing.

## 5. Visualization: Discrete Centres

### 5.1. The visual unit

VSF does not draw dataset rows as points. The unit is a **discrete centre**: one combination of category values of branch $S^*_d$.

$$\mathbf{c} = (\tilde{x}_{j_1}, \dots, \tilde{x}_{j_d}) \in \prod_{j \in S^*_d} \mathcal{A}_j$$

The number of occupied centres is normally $|\mathcal{C}_{occ}| \ll \prod_{j} k_j$ — most category combinations do not occur.

### 5.2. Mass: circle size

Scaling by area (Tufte, 1983) keeps the "lie factor" at 1:

$$\text{Radius}(\mathbf{c}) = R_{\max} \cdot \sqrt{\frac{N_{cell}(\mathbf{c})}{\max_{c'} N_{1D}(c')}}$$

The normalizing denominator is the maximum density in the 1D projection of the same branch, and $2R_{\max} = 1.0$ (the side of one grid cell), so the largest circle exactly fills its cell and never overlaps its neighbours.

### 5.3. Purity: circle colour

Purity is Definition 2, computed on **all $N$ rows**. The renderer may draw only a subsample of individual objects as spheres (`max_display_samples = 10 000`), but every number and every colour attached to a cell comes from the full table.

| Zone | Condition | Colour | Meaning |
|------|---|---------|-----------|
| **Discrete centre** | Definition 3 (Rule P: $\hat\pi(\mathbf{c}) \ge \tau$ and $n_\mathbf{c} \ge m$; or Rule C) | 🟢 Green | The deliverable. **Exactly the cells Definition 4 counts.** |
| **Mixed** | $\beta \le \hat\pi(\mathbf{c}) < \tau$ | 🟤 Brown | Enriched, but below the user's threshold. |
| **Low** | $\hat\pi(\mathbf{c}) < \beta$ | 🔴 Red | Target value largely absent. |

Three zones, no yellow, sharp transitions rather than a gradient, for instant categorical distinctness (Healey & Enns, 2012). The two boundaries are **different kinds of control** and the interface separates them:

* $\tau$ — the green boundary — *is* Definition 3's purity floor and the search's own objective (Definition 5). Moving it changes the certified set, the coverage, the centre count, the cross-validated figure **and which feature subset wins the search**. It triggers a recomputation and is part of the analysis cache key. $\tau = 1$ is permitted (Corollary 1). Default $0.90$.
* $\beta$ — the brown/red boundary, default $0.40$ — is purely a colour cut. It splits the *non*-centre cells into two shades, changes no reported number and no ranking, and is never sent to the server. This is the point users most often expect the wrong thing about: lowering $\beta$ cannot make the search chase lower-purity cells, because $\beta$ never enters Definition 5.

The partition is exhaustive and non-overlapping, and $\beta < \tau$ holds because the green test is evaluated first.

Consequences worth stating plainly:

* **What is green is what is counted, and what is counted is what was searched for.** The coverage in the panel is computed over exactly the green cells, and those same statistics are what ranked the branch. The reader cannot be looking at one set, reading a number about a second, and have the search have optimised a third.
* **A singleton cell is green under the defaults, and the display says so.** With $m = 1$ a cell holding one target-value object has $\hat\pi = 1$ and is a centre; its hover text carries the interval $[\alpha/C, 1]$, i.e. the data are equally consistent with a purity of a few percent. Corollary 2 measures what that costs on pure noise, and both remedies are one parameter away.
* Every cell carries its $1-\alpha$ Clopper–Pearson interval. The legend states the rule in force, both boundaries, $m$, the positive value, the base rate and $N$ — so a screenshot of the display is self-describing.

**What the shipped web application exposes.** Only $\tau$ (the green handle and its numeric input) and $\beta$ (the red handle, cosmetic). $\alpha = 0.05$, $m = 1$ and Rule P are fixed in the app; the Global Pattern Scan panel has its own separate minimum-occupancy field. The full triple $(\tau, m, \text{rule})$, $\alpha$, the bound method and the multiplicity correction are all available through `CenterSpec` in the library and through `/api/analyze`.

### 5.4. Grid axes: 1D to 3D

Adding a feature turns an axis into a grid, then into a cube. Below $d = 3$ the missing spatial axes collapse: 1D is a line of intervals, 2D a flat grid, 3D a full cube of centres.

### 5.5. 4D: the film strip

For $S^*_4 = \{j_1,j_2,j_3,j_4\}$ the first three features map to the spatial axes and the fourth becomes an **interactive stepper** of slices below the chart:

$$\text{Slice}(c) = \{(x_1, x_2, x_3) \mid X_{j_4} = c\}, \quad c \in \text{Dom}(X_{j_4})$$

A discrete stepper rather than a continuous animation is the default, for resilience to change blindness and to keep the spatial axes constant between slices. Auto-play (play/pause/speed) sits on top of the discrete slices rather than replacing them. The **"All" tab** marginalizes over the 4th dimension, so a single slice can be compared against the aggregate picture.

### 5.6. Collapse/split animation

Two interactions must not be confused:

1. **Switching branches** — the user picks one of the up-to-four independently found branches. Because $S^*_2$ can consist of entirely different features from $S^*_3$, this is an **instant scene rebuild** with new axes. Object constancy is inapplicable and is not claimed.
2. **Collapsing within a branch** — after choosing, say, $S^*_3 = \{A,B,C\}$, the user can fold it to 2D (marginalize over $C$) or 1D (marginalize over $B,C$) and unfold it back. **This** is the animated transition, and it never swaps features.

For case 2, removing axis $C$ from $\{A,B,C\}$:

* **Collapse (3D → 2D):** centres do not vanish, they fall onto the $(A,B)$ plane. Centres sharing $(A,B)$ but differing in $C$ merge smoothly: masses (radii) add, colours interpolate into a muddy brown — a visual demonstration of why dropping the axis loses separability.
* **Split (2D → 3D):** the reverse. A large muddy centre splits along the returning $C$ axis into several purer ones — adding a feature untangles the mixture.
* **Film strip (3D ↔ 4D):** slices slide in and out with a rewinding metaphor (Section 5.5).

All transitions preserve relative circle sizes (object constancy, Heer & Robertson, 2007). The statistics displayed during a collapse are recomputed **on the partition actually on screen** (`view_metrics` in `vsf.vis`, keyed by view dimensionality 1–4), never projected down from the branch's full-$d$ numbers — showing the full-branch coverage while collapsed would overstate what the visible axes alone carry.

### 5.7. Canonical axis sorting

Categorical axes have no natural metric order. Divisions are ordered by the conditional mean of the target,

$$\text{Score}(c) = \mathbb{E}[Z \mid X_j = c], \quad c \in \text{Dom}(X_j),$$

with deterministic tie-breaking (by frequency, then lexically). The order is computed once per target and stays fixed when the criterion changes *within the same target column* (`class = "p"` ↔ `class = "e"`), so centres do not drift sideways — only colour and size update. Changing the target column recomputes the basis.

---

## 6. Categorical Axis Ordering: implemented and not implemented

Section 5.7's target-conditioned sort (`vsf.vis.target_conditioned_sort`) is the **only ordering mode in the code**. It answers "which categories shift the target most".

A second mode has been specified but **is not implemented**: a cluster-preserving order (1D spectral ordering or correspondence analysis on $p(\tilde Z \mid X_j)$) answering "which categories behave alike", together with a Kendall $\tau$ diagnostic of the divergence between the two orders,

$$\tau(X_j) = \frac{C - D}{\frac{1}{2} K_j (K_j - 1)},$$

whose small values would signal non-linearity or multimodality — hidden subgroups inside a category that the averaged $\mathbb{E}[Z \mid X_j]$ conceals. There is no spectral ordering, correspondence analysis or Kendall $\tau$ anywhere in `vsf/`. Any claim about this diagnostic is a claim about future work, not about the running system.

---

## 7. Comparison with Existing Approaches

### 7.1. Closest conceptual neighbour: Jeon et al. (IEEE TVCG 2025)

*"Dataset-Adaptive Dimensionality Reduction"* (Jeon et al., *IEEE TVCG*, 2025, DOI: 10.1109/TVCG.2025.3634784) adapts a **projection algorithm** (t-SNE/UMAP/PCA and their hyperparameters) for a fixed 2D scatterplot, unsupervised, minimizing geometric distance distortion. VSF selects the **composition of axes** $S^*_d \subseteq \{1,\dots,M\}$ conditioned on an explicitly named target value, on a discrete certification basis rather than continuous geometry. The approaches are orthogonal and do not compete.

### 7.2. Classical approaches

* **mRMR (Peng et al., 2005)** balances relevance $I(X_j;Z)$ against redundancy. VSF does not ask the user to choose a single $K$ — it materializes all four dimensionalities in parallel — at the cost of a narrower objective (coverage of one named value) than mRMR's general relevance/redundancy trade-off.
* **Scagnostics (Wilkinson et al., 2005)** computes nine graph metrics on ready-made 2D projections: passive diagnostics. VSF is an active synthesizer, deriving axes before the picture exists.
* **Maximally Informative Dimensions (Sharpee et al., 2004)** optimizes a continuous projection by gradient ascent on mutual information. VSF replaces that with a discrete exact search, which for $d \le 4$ guarantees a global optimum rather than a local one, at the cost of exponential growth in $M$ (Section 4.6).

### 7.3. Summary

| System | Selection method | Target-conditioned? | Branches by dim? | Data model | Statistical criterion | Unit |
|---|---|:---:|:---:|:---:|:---:|---|
| **Voyager 2** (2017) | Perceptual rules | Partially | ❌ | mixed | ❌ | Charts |
| **Scagnostics** (2005) | Graph metrics | ❌ | ❌ | continuous | ❌ | Points |
| **mRMR** (2005) | MI, manual $K$ | ✔️ | ❌ | mixed | ❌ | Features |
| **Jeon et al.** (2025) | Structural complexity | ❌ | ❌ | continuous | ❌ | Points |
| **VSF** | Exhaustive search by certified-centre coverage of a named target value | ✔️ | ✔️ (1D–4D in parallel) | categorical only | ✔️ (per-cell certificate, coverage permutation null, familywise max null, cross-validation, FDR across a scan) | Discrete centres |

---

## 8. Summary of Contributions

| # | Contribution | Type |
|---|-------|-----|
| C1 | **Independent Branch Discovery** — exact, not approximate, search for up to four coverage-optimal feature sets, one per dimensionality, with no nestedness assumption | Algorithmic |
| C2 | Explicit formalization of coverage's non-submodularity as the reason a single greedy path cannot find what independent searches find (Section 4.4) | Theoretical |
| C3 | **A purely categorical encoding with no fitted parameters**: every column is represented by its distinct values, so no axis boundary, bin count or channel limit is estimated from the data the search then ranks on — removing a selection exposure rather than correcting one (Section 2) | Methodological |
| C4 | **Discrete centres as a stated definition rather than a colour band**: one triple $(\tau, m, \text{rule})$, owned by the user, fixes simultaneously what is drawn green, what enters every reported number, and what the search optimises (Definitions 3, 5; Proposition 2; Corollaries 1–2) | Metric / Methodological |
| C5 | **Coverage as both the reported deliverable and the sole ranking objective** — the display's own question ("which cells are almost purely this value, and how much of it do they hold") is the question the search answers | Metric / Algorithmic |
| C6 | **Out-of-sample coverage with Nadeau–Bengio corrected variance** — the first quantity here that can *decrease* when a branch is over-resolved, which turns "how many characteristics describe this value?" into a decidable question with an explicit *no answer* outcome (Definitions 6, 10) | Methodological |
| C7 | Exact multivariate-hypergeometric null for coverage, with a shared-permutation familywise variant matched to the $\arg\max$ actually reported (Definitions 8–9) | Methodological |
| C8 | Two-level multiplicity control matched to where multiplicity arises: a look-elsewhere max-statistic null over the subset search, and Benjamini–Hochberg across the targets of a dataset-wide scan (Sections 4.5, 4.7) | Methodological |
| C9 | Collapse/split animation explicitly demarcated from switching between independent branches, with per-view statistics recomputed on the displayed partition (Section 5.6) | System / UX |

---

## 9. Evaluation Plan

### 9.1. Quantitative evaluation (synthetic data)

1. **Ground-truth synergy benchmark.** Datasets with known synergy (XOR-like dependencies where no single feature is informative but a pair determines $Z$), verifying that Independent Branch Discovery finds the pair where a greedy chain cannot. `tests/test_avr.py`'s XOR fixture is a running instance.
2. **Null benchmark.** Every feature generated independently of the target. Required outcome: **no branch certifies any centre at any $d$** (`n_centers == 0`, `has_certified_centers() is False`). Must be run at several $N$ and several feature cardinalities, since a wider candidate family raises the chance that *some* cell clears $\tau$ by chance even though the point estimate is unbiased.
3. **Micro-class benchmark.** Target prevalence $10^{-4}$ to $10^{-2}$, under both a null and a deterministic construction, verifying that reported coverage is near $0$ (or "undetermined" below the power floor, Definition 7) and near $1$ respectively.
4. **Calibration of the familywise test.** Under the null benchmark, `coverage_p_value_familywise` must be uniform on $[0,1]$; the uncorrected `coverage_p_value` is expected to be visibly anti-conservative, and the size of that gap as a function of $M$ is itself a reportable result.
5. **Metrics.** Match of the discovered $S^*_d$ against the true synergistic combination; false-positive rate on the null benchmark; runtime of the ranking pass *and* of the reporting layer as functions of $M$, $N$, `cv_repeats` and `n_permutations_centers`, verifying Section 4.6 on real hardware.

### 9.2. Controlled user study ($n \ge 40$)

Participants solve analytical tasks (finding clusters, outliers, separating rules) with the four-branch display against expert manual axis selection. Hypothesis: parallel presentation of four branches does not increase cognitive load relative to a single branch, while leading users to synergistic feature combinations more often.

Neither part of this section has been run. It is a plan, not a result.

---

## 10. Open Problems and Limitations

1. **Multiple comparisons — partially controlled.** The look-elsewhere effect of the subset search is corrected by a max-statistic permutation null on coverage, and multiplicity across targets by Benjamini–Hochberg. One exposure remains at the framework level: the familywise test is **off by default** (cost), so every interactive number is uncorrected and labelled so. In addition, the shipped Global Pattern Scan UI disables BH by choice (Section 4.7). The encoding itself contributes nothing here — it is fitted to no data (Section 2.1), unlike the bin-selection layer it replaced.
2. **Scalability.** $M \gtrsim 50$–100 features means tens of minutes for an honest search of all $\binom{M}{\le 4}$ combinations. Neither pre-filtering nor approximate search is permitted by this specification.
3. **The certificate is per-cell; the coverage is not.** Proposition 2 bounds the probability that a cell with $\pi \le \tau$ is certified. It does not make in-sample coverage unbiased, because the certified set is chosen by inspecting cell contents. The gap is real and measured (Definition 6's constructed case: $66.3\,\% \to 63.9\,\%$), which is why $\mathrm{Coverage}_{\mathrm{CV}}$ is mandatory rather than optional. A selective-inference treatment conditioning on the selection event would give a tighter interval than cross-validation; that is open.
4. **$\tau$ and $m$ are user choices with no principled defaults, by design.** $\tau = 0.90$, $m = 1$ are conventions. The centre set, the coverage, the winning feature subset and the reported dimensionality all move with them — on `income = >50K` at $d = 3$, coverage goes from $2.4\,\%$ at $\tau = 0.90$ to $24.6\,\%$ at $\tau = 0.75$. The specification makes the dependence explicit and reproducible ($\tau$, $\alpha$, $m$ and the rule appear in the legend and in the analysis cache key) but does not remove it. **A coverage quoted without its $(\tau, m, \text{rule})$ is meaningless.**
5. **Coverage and $K$ are not jointly optimized.** The rule is lexicographic: coverage dominates, $K$ breaks ties. So the search returns 55 centres for $2.4\,\%$ coverage in preference to 2 centres for $1.2\,\%$ — technically correct and, at that centre count, not a readable finding. A scalarization (coverage minus a per-centre penalty) or a Pareto front over $(\mathrm{Coverage}, K)$ would let the user say "not at any price"; neither is implemented. The interface compensates only by showing both numbers side by side.
6. **Bonferroni is conservative at high cell counts.** At the capacity limit, $C \approx N/10$, so $\alpha/C \approx 1.5 \times 10^{-4}$ on a 32k-row dataset and certification demands cells that are both large and nearly pure. That is the right direction for a certificate, but it costs power exactly where the display is most detailed. A step-down or FDR formulation over cells would recover some of it at the cost of a weaker per-cell guarantee; that choice is not currently offered.
7. **The grid capacity limit is a variance bound, nothing more.** $\prod_j k_j \le N/10$ keeps cell counts estimable. It does not bound selection optimism (item 3 does that separately). Tightening it would trade resolution for power on small $N$.
8. **The cluster-preserving axis order is specified but not implemented** (Section 6).
9. **No automatic target discovery.** The user must name the target column and value; the first screen is a usage guide, not an analysis. The optional Global Pattern Scan iterates all (column, value) pairs on request, but it only *filters* the target selector — it never picks a target.
10. **Reintroducing an association-based ranking would be a new feature, not a restoration.** Ranking branches by an association statistic (adjusted mutual information and the like) was removed on the grounds that a discrete-centre display makes a statement about one named value, and an association number answers a different question in the same panel. If it is ever wanted again it needs its own explicit interface surface, so a user cannot land in it by accident, and its estimator bias must be corrected before candidates of unequal cell count are compared.

---

## Appendix. Implementation Map

`vsf` 2.3.0 · Python ≥ 3.10 · dependencies: `numpy`, `pandas` only (`pytest` for the suite). `pyproject.toml` is the single source of truth for packaging.

| Module | Responsibility |
|---|---|
| `vsf/pmd.py` | Category encoding, grid capacity, adaptive level merging (Section 2) |
| `vsf/metrics.py` | Dense joint-cell codes and Benjamini–Hochberg FDR control |
| `vsf/centers.py` | The reported layer: Clopper–Pearson and Wilson bounds without `scipy`, Rules P/C, coverage/$K$/purity, cross-validation, hypergeometric and familywise nulls, dimensionality selection, and `coverage_score` (Section 3.2) |
| `vsf/avr.py` | Independent Branch Discovery: `discover_branches`, `discover_branches_by_value`, `BranchEngine`, `select_branch_dimensionality` (Section 4) |
| `vsf/vis.py` | Render payloads per branch, per-view statistics, axis sorting (Section 5) |
| `vsf/server.py` | Embedded HTTP server, REST API, response cache, prefetch, background Global Pattern Scan |
| `vsf/dashboard.py` | Single-file offline HTML export |
| `vsf/webapp/`, `vsf/templates/` | Live app assets and export templates, read via `importlib.resources` |
| `tests/` | 212 tests across 8 files, including the exactness pins of `test_fastpaths.py` and the witnesses in `test_pmd.py` / `test_metrics.py` that no binning or information-theoretic code returns |

**Defaults that determine a reported number.** All are explicit parameters; none is hidden.

| Parameter | Default | Where |
|---|---|---|
| `tau` | `0.90` | `CenterSpec` — purity floor and green boundary |
| `alpha` | `0.05` | `CenterSpec` — simultaneous error rate over occupied cells |
| `rule` | `"purity"` | `CenterSpec` — Rule P; `"certified"` is Rule C |
| `min_samples` | `1` | `CenterSpec` — no occupancy restriction |
| `method` / `multiplicity` | `"clopper-pearson"` / `"bonferroni"` | `CenterSpec` |
| `max_d` | `4` (`MAX_BRANCH_D`) | `discover_branches` |
| `cv_splits` / `cv_repeats` | `5` / `5` | `discover_branches` |
| `n_permutations_centers` | `999` (`DEFAULT_N_PERMUTATIONS`) | `discover_branches` |
| `n_permutations_familywise_coverage` | `0` (off) | `discover_branches` |
| `random_state` | `0` | `discover_branches` |
| `MIN_POSITIVES_FOR_CV` | `25` | `vsf.centers` — power floor, Definition 7 |
| `t_threshold` | `2.0` | `select_dimensionality`, Definition 10 |
| `max_display_samples` | `10 000` | `prepare_visualization_payload` — rendering only, never statistics |
| grid capacity | $\lfloor N/10 \rfloor$ | `check_grid_capacity` — the only place levels are ever merged |
| $\beta$ (red boundary) | `0.40` | frontend only, cosmetic |

**Public entry points.**

```python
import pandas as pd, vsf

# UCI "Adult" / Census Income dataset -- a public dataset, not bundled with
# this library.
df = pd.read_csv("adult_census.csv")

# 1. Library: four branches for one named target value.
feats = [c for c in df.columns if c != "occupation"]
branches = vsf.discover_branches(
    df[feats].astype(str).values,
    df["occupation"].astype(str).values,
    feature_names=feats,
    positive_class="Armed-Forces",
    center_spec=vsf.CenterSpec(tau=0.90, min_samples=1),
)
d_star = vsf.select_branch_dimensionality(branches)      # None is a valid answer

# 2. Interactive app (prefetch=False disables background precomputation).
vsf.serve(df, host="127.0.0.1", port=8000)

# 3. Single-file offline export; the certificate is baked in and stated in
#    the exported page's own legend, since a static file cannot re-certify.
html = vsf.export_full_dashboard(df, target="class", criterion="p",
                                 center_spec=vsf.CenterSpec(tau=1.0, min_samples=10))
```

**HTTP API** (`vsf.serve`): `GET /api/columns`, `POST /api/analyze` (target, criterion, `tau`, `alpha`, `rule`, `min_samples`), `POST /api/scan/start`, `GET /api/scan/status`, `POST /api/scan/cancel`. A target with no resolvable positive value returns `400` with an explanation rather than a different analysis.

---

## Key References

* Peng, H. et al. (2005). Feature Selection Based on Mutual Information: mRMR. *IEEE TPAMI.*
* Krause, A. & Guestrin, C. (2005). Near-optimal sensor placements in Gaussian processes. *ICML.* — non-submodularity, Section 4.4
* Wilkinson, L. et al. (2005). Graph-Theoretic Scagnostics. *IEEE InfoVis.*
* Seo, J. & Shneiderman, B. (2005). Rank-by-Feature Framework. *IEEE InfoVis.*
* Sharpee, T., Rust, N. C. & Bialek, W. (2004). Analyzing neural responses to natural signals: maximally informative dimensions. *Neural Computation*, 16(2), 223–250.
* Wongsuphasawat, K. et al. (2017). Voyager 2: Augmenting Visual Analysis with Partial View Specifications. *ACM CHI.*
* Jeon, H. et al. (2025). Dataset-Adaptive Dimensionality Reduction. *IEEE TVCG.* DOI: 10.1109/TVCG.2025.3634784
* Ware, C. (2004). Information Visualization: Perception for Design. *Morgan Kaufmann.*
* Munzner, T. (2014). Visualization Analysis and Design. *CRC Press.*
* Healey, C. G. & Enns, J. T. (2012). Attention and Visual Memory in Visualization. *IEEE TVCG.*
* Tufte, E. R. (1983). The Visual Display of Quantitative Information. *Graphics Press.*
* Cleveland, W. S. (1993). Visualizing Data. *Hobart Press.*
* Heer, J. & Robertson, G. (2007). Animated Transitions in Statistical Data Graphics. *IEEE TVCG.* — object constancy, Section 5.6
* Clopper, C. J. & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413. — the exact interval behind Definition 3
* Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212. — the score interval, offered for display only, rejected for certification
* Brown, L. D., Cai, T. T. & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133. — why Wilson coverage oscillates below nominal, measured for $n=30$ in `tests/test_centers.py`
* Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281. — the corrected variance of Definitions 6 and 10
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289–300. — the multiplicity control of Section 4.7
* Press, W. H. et al. (2007). *Numerical Recipes*, 3rd ed., Sec. 6.4. — the continued-fraction incomplete beta used to invert Definition 3 without `scipy`
