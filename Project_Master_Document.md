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
* **A numeric column is not special.** `1`, `2`, `3` are three categories, exactly as `low`, `mid`, `high` would be. The only place the framework uses the fact that a numeric column's values are ordered is the capacity merging of Section 2.3, which merges *adjacent* values of an ordered column and *rarest* levels of a nominal one; nothing else reads the codes as quantities.

**Missing values are one category, coded last.** None, NaN and pandas NA/NaT in a column form a single category placed after the largest value. Until 2026-09 an object column holding NaN beside numbers was encoded as it stood, and since NaN compares unequal to itself and breaks the sort, every NaN became a category of its own and equal non-missing values were split too (`[1, NaN, 2, NaN, 1, NaN, 2, 1]` gave 8 levels instead of 3). `discretize_dataset` casts every column to object, so every numeric column with a gap was affected; every result computed on such data before the fix is invalid and must be recomputed. No bundled dataset contains a value pandas reads as missing, so no benchmark number in this document changes. That is not the same as having no missing data: `audiology` and `soybean_large` mark it with the UCI string `?`, `mushroom` with `missing`, `titanic` with `unknown_*`, and all of these are ordinary categories to the encoding (the dataset screen flags them as placeholder levels, Section 4.12; provenance in `benchmark_data/MANIFEST.csv`). One limitation remains: in an *ordered* column the missing category sits after the largest value, so the adjacent-value merging of Section 2.3 can merge it into the top range.

### 2.2. What this rules out, deliberately

There is **no binning layer**: no bin-count heuristic, no quantile cut points, no per-channel level limits, and no rate-distortion distortion figure. A framework that quantized continuous features would be a different product, and three arguments keep it out of this one:

1. **An interval is not a category.** A boundary chosen by a density heuristic is an artefact of the estimator, and an axis tick reading `(3.7, 4.1]` is not a value an analyst can act on. A centre certified inside such an interval is a statement about a bin edge, not about the data.
2. **Nothing downstream can use it.** A distortion figure describing how much of a fine quantization a coarse code retains is not an input to the ranking key (Definition 5), to the certificate (Definition 3), or to anything the display shows.
3. **Fitting the encoding to the data is an uncorrected exposure.** Bin edges chosen from the same rows the search is then ranked on are a selection effect, and Section 4.5 controls no such thing. Encoding distinct values removes the exposure by construction instead of correcting it.

**No information-theoretic quantity is computed anywhere in this framework** (Section 3.1).

**The consequence, stated plainly.** A genuinely continuous column — thousands of distinct floats — now produces thousands of categories instead of at most a few hundred bins. It is not rejected, not truncated and not silently quantized: it is encoded as it stands, and the grid-capacity rule below is what keeps the joint table estimable when such a column enters a candidate combination. Feeding continuous measurements to a categorical framework yields an axis with thousands of ticks; that is a data-preparation decision belonging to the caller, and the design makes it visible rather than papering over it with a binning heuristic.

### 2.3. Grid capacity

A candidate subset $S$ partitions the rows into cells; the search ranks subsets on per-cell counts and the certificate bounds per-cell proportions, and both need cells that hold more than a handful of rows. The capacity rule is stated on the cells that **exist**:

$$C_{\text{occ}}(S) \;\le\; \max(1, \lfloor N/10 \rfloor),$$

i.e. at least ten rows per occupied cell on average (`vsf.pmd.grid_capacity`, `occupied_cells`, `check_grid_capacity`). The nominal product $\prod_{j \in S} k_j$ is deliberately *not* the quantity: it counts cells that may not exist. Two seven-level columns can occupy 12 of their 49 nominal cells, and a rule on the product would merge their levels for no statistical reason. This was a measured defect of the earlier product-based rule: on the Titanic data (N = 891, capacity 89), `title + fare_level + port_embarked + passenger_class` has $7 \times 3 \times 4 \times 3 = 252$ nominal cells and **71 occupied**; the product rule coarsened it, merged the child-identifying `Master` level of `title` into its alphabetical neighbours, and the 4-D branch reported 17.4 % coverage where the uncoarsened partition certifies 52.2 %.

**Merging, when the rule is violated** (`adaptively_coarsen_bins`, `coarsen_to_capacity`). The column with the most remaining levels (ties: the lowest column index) loses one level at a time, the occupancy is recomputed, and the process stops at the first state that respects the capacity — occupancy is non-increasing along this sequence (merging levels can only merge cells), so the first admissible state is found by bisection over the step count rather than by one pass per step (`merge_state`). How a column loses a level depends on the one thing the encoding knows about it:

* a **nominal** column (strings, booleans, mixed objects) has no neighbourhood between its levels; its rarest kept level joins an "other" group, so every dominant level survives intact and what is merged is, by construction, the part of the column with the fewest rows (`level_frequency_order`, ties by code);
* an **ordered** column (any numeric dtype, `column_is_ordered`) is merged between *adjacent values* into contiguous groups of near-equal row count, so that a measurement with thousands of distinct values coarsens into ranges an analyst can read rather than into "the 299 smallest values and everything else".

Both rules are functions of the column alone — never of the target — so the partition inherits no selection effect from them. Reducing the widest column first removes the most cells per merged level and leaves narrow, usually more meaningful columns untouched for longest. The search (`vsf.avr._CandidateFactory`) applies exactly this rule per candidate, with per-(column, level count) merges cached, and `tests/test_fastpaths.py` pins its partition to `cell_codes(adaptively_coarsen_bins(X[:, S]))` for nominal and ordered columns alike.

**What the rule is and is not.** It bounds the variance of per-cell counts by a floor on average occupancy; it does not bound selection optimism (Section 4.5 does), and with `min_samples = 1` under Rule P it still permits a one-object centre inside a combination that fits (Corollary 2). When a combination is merged, the displayed lattice keeps the raw columns and says so (`certificate.partition_matches_search`), so the numbers the search ranked on and the numbers drawn can differ, and the difference is reported rather than hidden.

### 2.4. Human-in-the-loop

The system does not assign semantics on its own. The user names the target column and the positive value (Section 1.4). There is nothing else to configure about the encoding: with no bin counts and no cut points, a column has exactly one representation, and two runs on the same data produce the same axes.

---

## 3. Mathematical Core

### 3.1. Notation, and what is deliberately absent

A branch is a subset $S$ of feature columns; $\tilde X_S$ is the joint category code of those columns, i.e. the partition of the rows into **cells**. $C$ denotes the number of *occupied* cells, $\mathbf{c}$ one of them, $n_\mathbf{c}$ its size and $k_\mathbf{c}$ the number of target-value objects in it. $N$ is the number of rows, $N_+$ the number of target-value objects, $N_+/N$ the base rate.

**No information-theoretic quantity is computed anywhere in this framework** — no entropy, no mutual information, no normalized variant — not in the search, not in the report, not in the encoding. An association statistic answers "is there a relationship between the target and this partition"; the deliverable is narrower — "which cells are almost purely this target value, and how much of it do they hold" — and on a rare target the two have opposite answers: a partition can be strongly associated with the target while its purest cell holds a few percent of the value, which is not something any display can act on (worked example in `vsf/centers.py`'s module docstring).

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

**Coverage is not monotone in $d$ either.** Non-submodularity concerns the size of the gains; the gains can also be negative. Adding an axis splits every cell, and a cell that clears $\tau$ can split into one part that still does and one that does not, or into parts below `min_samples`. A cell of 10 objects, 9 positive, is a centre at $\tau = 0.9$; split into $(5, 5)$ and $(5, 4)$ it keeps only the first part, and the covered count falls from 9 to 5. Hence the best $(d+1)$-subset can cover less than the best $d$-subset, the per-$d$ envelopes of Section 4.9 may cross, and "more axes" is never assumed to be better — it is measured, out of sample (Definition 10).

### 4.5. Statistical control: what is guaranteed and what is not

Selection is a deterministic exhaustive $\arg\max$ over Definition 5. The tests below attach a *claim* to an already-selected branch; they never drive the choice. Nothing about the search depends on a random seed except which rows land in which cross-validation fold, and that seed is fixed by default (`random_state=0`).

**Two sources of optimism, handled separately.**

1. **Sampling variability of a pre-specified subset** — addressed by the coverage permutation $p$-value (Definition 8), drawn from the multivariate hypergeometric law of the cell counts. Cheap, $O(C)$ per replicate, on by default.
2. **The look-elsewhere effect of the search itself** — `coverage_p_value` is *not* valid for a branch chosen as the maximum over $\sum_d \binom{M}{d}$ candidates. The corrected statistic is the shared-permutation max null of Definition 9, `coverage_p_value_familywise`. Off by default because it costs $B$ times the entire search; **any published result must set it.**

**The per-schema certificate does not survive the search.** Proposition 2 is a statement about a schema fixed before the data are seen. The branch the search reports is the best of $\binom{M}{d}$ schemas, and the Bonferroni divisor $C$ counts only its own cells. Measured on pure noise ($N = 3000$, twelve 4-level columns, prevalence $0.45$, $\tau = 0.5$, Rule C, $\alpha = 0.05$, 100 runs; `experiments/null_certificate.py`): the reported 3-D winner carries a certified centre in **36 %** of runs, the 4-D winner in **60 %**, and at least one of the four branches in **68 %**, against a nominal 5 %. Every certificate the interface draws is therefore a per-schema statement only. Section 4.14 gives the two certificates that hold for the reported schemas.

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

**The anchored colour scale (interface).** The legend's axis is the share, in a cell, of what is searched for: the chosen value under presence, the rows *without* it under absence. The base rate of that quantity — $p_0$ under presence, $1 - p_0$ under absence — is a fixed tick on the axis. The certified boundary starts one point above the tick and can move only further right: "green from $c$", $c > p_0$, under presence; "red from $c$", $c > 1 - p_0$, under absence (cells at least $c$ pure in rows without the value). It can never cross the tick — a certificate at the base rate would call every cell of the trivial partition a centre — so the base-rate invariant is enforced by the control, not only by the server. Read on the value's own axis, the absence certificate is "value share $\le 1 - c < p_0$": for a value that fills $80\,\%$ of the rows the absence axis has its tick at $20\,\%$ and certifies cells where the rows without the value exceed $20\,\%$, up to $100\,\%$ — exactly the tail on the far side of the base rate, drawn in the units in which it is read. The second boundary is decorative: it splits the *uncertified* cells into those above the base rate but short of the certificate (brown) and those below it (red under presence — the value is rarer there than overall; grey under absence — the value is present at least as often as overall). It defaults to the tick so that brown never covers a cell below the base rate, which is the misreading the previous absolute "red up to" boundary allowed. When the user picks a value whose base rate lies above the current boundary, the boundary is moved to the nearest admissible position before the request and the move is reported; the catalog marks such values in advance.

**What differs between the two reports.** The headline of a presence search is coverage (the share of the value captured by certified centres). The headline of an absence search is the **mass** of certified $v$-free cells — the share of *all* rows an analyst may exclude from a search for $v$ — with the coverage of the complement second; at high $\tau$ the two order candidate schemas identically, so the ranking key is unchanged. `CenterReport.prevalence` reads $1 - p_0$. Class labels keep naming the value ("Column = v" / "not Column = v") so a point's hover stays literal, while the analysis title reads "Column ≠ v". Certified cells are drawn **red** ("certified free of $v$"), never green — a green cell is a presence certificate, and the absence search does not make one; rows rich in $v$ but certified for nothing at this $\tau$ are slate grey, the band between the boundaries brown. For a $K$-class target an absence centre means "$v$ is not here", not "some particular other value is here": the complement is heterogeneous, and the legend says so.

**Brown is a residual, not a target.** Cells certified for neither direction remain brown. Maximising their number would reward the least informative schema (under independence every cell sits at the base rate, i.e. every cell is brown), so the mixed zone is never an objective; it is reported as what the schema could not resolve either way, $1 - \text{mass(green)} - \text{mass(red)}$ when both directions are certified on one display, and it is the natural quantity that distinguishes this lattice from a decision tree of the same depth, which has no abstention zone.

**API.** `discover_branches(..., direction=)`, `iter_branches_by_value(..., direction=, skipped=)`, `discover_branches_by_value` likewise, `BranchEngine(direction=)`, `export_full_dashboard(..., direction=)`; `/api/analyze` and `/api/scan/start` take `direction` (`"presence"` default, `"absence"`), the analyze response and every scan row carry it, and it is part of the response cache key and of the sibling-value prefetch.

### 4.9. The solution landscape: every schema the search scored

**What it is.** The exhaustive search evaluates $\text{Total}(M)$ candidate schemas (Section 4.6) and reports four winners. Everything it learned about the other candidates was discarded, although it is exactly the context a reader needs to judge a winner: how many schemas reach a comparable coverage, at how many centres; whether the optimum is isolated or one of hundreds of near-equivalent schemas (the *Rashomon set* of this search, in the sense of Semenova, Rudin & Parr, 2022); and how many schemas certify nothing at all at this $\tau$. `vsf.avr.Landscape` keeps, for every candidate, its dimensionality, its axes and the three integers its ranking key is made of ($k_{\text{sel}}$, $K$, $n_{\text{sel}}$). Recording costs nothing beyond the search itself, and the landscape is computed on demand from the same ranking pass (`compute_landscape`), cached per (target, value, certificate, direction).

**Display.** The landscape is drawn with the product's own visual language — a lattice of cells with a circle in each — rather than a scatter plot: the two axes are *ordered categories*. On $x$, the coverage of the value (under an absence search, the mass of certified value-free cells) in the left-open, right-closed intervals $(0, 10], (10, 20], \ldots, (90, 100]$ percent. On $y$, the schema's number of certified centres as a share of $K_{\max}$ — the largest centre count among the schemas shown (per dimensionality, or over the whole family) — in the same ten intervals. A cell's colour is the number of schemas that fall in it, under a three-zone scale whose two boundaries belong to the user and whose right end is the fullest cell; the count is written in the cell. Schemas that certify no centre have coverage $0$ by construction and are **excluded from the lattice** and counted above it ("$n_0$ of $n$ schemas certify no centre at $\tau$") — placing them in the first cell would swamp it with schemas that are not weak but empty. The search winner of the selected branch is marked on the lattice from its exact numbers, so the reader always sees where the reported result sits in the family.

**Drill-down.** Clicking a cell lists its schemas — axes, exact coverage, $K$, mass — most concentrated first (lowest mass, then fewest centres, then highest coverage), paged from the server (`/api/landscape/cell`) since a cell can hold tens of thousands of schemas at $M \approx 50$. Any listed schema can be opened as a lattice (`/api/analyze` with `features=[\ldots]`, `vsf.avr.report_schema`); the landscape state survives the round trip so the reader returns to the same cell.

**Statistical status of a schema opened from the landscape.** Such a schema is selected by looking at the data. Its in-sample coverage is what the lattice shows, its cross-validated coverage remains an out-of-sample estimate (the folds are held out whatever led to the choice), and the familywise $p$-value, when computed, remains valid and conservative (the null of the maximum over the family bounds any member). Its *uncorrected* permutation $p$-value would be the $p$-value of a hypothesis chosen because it looked good and is therefore **not reported** for an opened schema; the panel says why.

**What the landscape adds to the claims.** It makes the coverage–$K$ tension of limitation 5 visible without changing the ranking rule: the reader sees the two-centre schema at $1.2\,\%$ beside the fifty-five-centre schema at $2.4\,\%$ and chooses. It replaces the "random subsets" baseline a reviewer would ask for with the exact distribution over the whole family. And its empty-schema count is the most direct statement of what a threshold means on a dataset: at $\tau = 0.9$ on `income = >50K`, the honest summary is not the winner's $2.4\,\%$ but that nearly every schema certifies nothing.

**The $\tau$-curves: the landscape's envelope over the purity floor.** The lattice above is a section of the landscape at one $\tau$. Its complement is the dependence on $\tau$ itself: for every $d \le 4$ and every purity floor on the grid $\tau \in \{\lceil 100 p_0 \rceil + 1, \ldots, 100\}$ percent (whole percents strictly above the base rate $p_0$ of the searched indicator; under the certified rule $100\,\%$ is not certifiable and is omitted), the best coverage — under absence, the best mass certified free — that any $d$-subset reaches, with the subset that reaches it and its $K$ (`vsf.avr.compute_tau_curves`, `tau_grid`). For a fixed partition, coverage is a right-continuous non-increasing step function of $\tau$ with steps at the cells' purities: under Rule P a cell of size $n \ge n_{\min}$ is a centre at $\tau$ iff $k/n \ge \tau$, so one sort of the cells by purity and cumulative sums of $k$ and $n$ give the whole curve, read off the grid by one `searchsorted` per candidate; under Rule C the threshold $k_{\min}(n, \tau)$ depends on $n$ and $\tau$ jointly and each grid point is a separate selection (the memoised `min_successes_to_certify` keeps the pass to a few seconds). The per-$d$ envelope is the pointwise maximum over the family, hence non-increasing as well; its point at a grid $\tau$ is *exactly* the winner `discover_branches` would report for that $d$ at that $\tau$ (the same partition, the same ranking key — the Wilson tie-break is never reached, since two candidates tied on coverage and mass have the same $(k_{\text{sel}}, n_{\text{sel}})$), which the test suite checks at several grid points under both rules and both directions. The curves are computed once per (target, value, rule, $\alpha$, $n_{\min}$, direction) — $\tau$ is not part of the key, it is the abscissa — and cached (`/api/landscape/curves`).

*Reading the curves.* Four lines on one chart, one per $d$, drawn in the **Trade-offs** view mode (Section 4.13), where the vertical cursor is the current certified boundary. Where the line for $d+1$ stops lying strictly above the line for $d$, the extra axis has stopped paying for itself at that purity floor — the dimensionality question of Section 3 read along $\tau$ instead of at one $\tau$. A curve reaching zero is the statement that no $d$-subset certifies any centre beyond that floor. Coverage on these curves is monotone in $\tau$ *for a fixed $d$*; across dimensionalities it need not be (Section 4.4: coverage is not monotone in $d$), which is why the four lines may cross.

*Click-through.* Clicking a point $(\tau, y)$ of the $d$-line lists the $d$-subsets whose coverage at the floor $\tau$ lies in the same ten-percent category as $y$ — the landscape at $\tau$ (computed and cached on demand), restricted to $d$, filtered by the $x$-category of the lattice (`Landscape.by_x_category`, `/api/landscape/at`), highest coverage first, then fewest centres, then lowest mass; the envelope's own schema is the first row. Opening a schema from that list moves the certified boundary to $\tau$ and renders it with the same statistical status as a schema opened from a lattice cell: chosen by looking, hence no uncorrected $p$-value; cross-validated coverage still out-of-sample.

### 4.10. Composite targets: a conjunction of values as the indicator

**What it is.** The search never sees a target column; it sees a 0/1 indicator $Z$ (Section 1). A *composite target* is the indicator of a conjunction of $k \ge 2$ (column, value) pairs, $Z_i = \mathbb{1}[\bigwedge_{j=1}^{k} X_{i, c_j} = v_j]$ — "survived *and* female" — and is, to the whole apparatus (the exhaustive search, the certificate, the base-rate invariant, the absence search, the landscape, the $\tau$-curves, cross-validation and the permutation nulls), one more indicator. Nothing in `vsf.avr` changes; what changes is how $Z$ and the feature space are built (`vsf.server._Target`, `_resolve_target`, `_target_arrays`), and the statement the user is making.

**The one rule that makes it well-posed.** Every column named in the conjunction is **removed from the feature space**: $M \to M - k$. Otherwise the search "finds" the target's own columns as its schema — the cell `sex = female` of any schema containing `sex` has purity $P(\text{survived} \mid \text{female})$ by construction, a tautology, not a structure. Removing the columns changes the family $\mathcal{F}$ and hence the familywise null; both are recomputed for the reduced feature set, which is the honest family for this question. A plain target already followed this rule with $k = 1$; the composite target extends it.

**What it asks.** The composite target asks *where objects of this kind concentrate*, on all rows, with base rate $p_0 = P(A \wedge B)$. It is not the conditional question — restrict the rows to $B$ and search for $A$ — which has a different $p_0 = P(A \mid B)$, a different sample and a feature space in which the column of $B$ is constant. The interface names the target in full, `survived = yes ∧ sex = female`, and reports its row count and share before any search (`/api/target`), so the two readings cannot be confused. The conditional question is a row filter, not a target, and is not offered.

**Limits, and why.** At most three pairs (`_MAX_TARGET_CONJUNCTS`). Base rates fall multiplicatively with every conjunct — $26\,\%$ for survived women on `titanic`, $8\,\%$ with a class added — and a certificate at $\tau = 0.9$ against $p_0 = 0.03$ is a lift of thirty: on the datasets of Section 9 such cells are single objects, which `min_samples` alone guards against, and the target panel says so below thirty matching rows. No automatic enumeration of conjunctions exists, by design. A sweep over all value pairs of a dataset would be subgroup discovery over the *target* side (Section 7), with the multiplicity of $\binom{V}{2}$ hypotheses that the Global Pattern Scan deliberately avoids by scanning single pairs; the user names the composite target, and the search finds where it lives. The sibling prefetch of Section 4.6 is per column value and is not started for a conjunction.

**Statistical status.** Identical to a plain target's: the in-sample coverage, the cross-validated coverage on held-out folds, the uncorrected permutation $p$-value of the winner and the familywise $p$-value all refer to the indicator of the conjunction on the reduced feature space, and every one of them is computed by the same code path (`tests/test_composite.py` pins the response to `discover_branches` on the hand-built indicator). Under absence the indicator is $1 - Z$, "not (survived = yes ∧ sex = female)", with the base rate $1 - p_0$; the anchored scale and the catalog dividers follow.

### 4.11. Redundant centres: the same objects under different characteristics

**What it is.** The landscape (Section 4.9) counts *schemas*. Two schemas built from different characteristics can certify centres that hold almost the same rows — on `titanic` at $\tau = 0.7$, `title = Mr` (517 rows) and `sex = male ∧ title = Mr` hold the identical 517 rows, and `sex = male` holds them plus 60 more — yet they appear as unrelated points of the lattice. A reader who counts how often a characteristic appears among certifying schemas therefore counts redundancy, not influence. `vsf.redundancy` works one level down, on the centres of every scored schema, and reports which of them are the same group of objects described by different characteristics.

**Definitions.** Let $R(c)$ be the set of rows of centre $c$ — all rows of the cell, positives and negatives alike: two centres capturing the same positives but different negatives are not the same group. The similarity of two centres is their *mutual containment*
$$s(a, b) = \frac{|R(a) \cap R(b)|}{\max(|R(a)|, |R(b)|)} = \min\left(\frac{|R(a)\cap R(b)|}{|R(a)|}, \frac{|R(a)\cap R(b)|}{|R(b)|}\right),$$
the smaller of the two one-sided shares; $s(a,b) \ge t$ says that *each* centre lies at least a fraction $t$ inside the other, and $s = 1$ iff the row sets are identical. A one-sided share — equivalently the overlap coefficient $|A\cap B|/\min(|A|,|B|)$ — would call a small centre nested in a large one its duplicate, and is rejected for that reason. Jaccard would also serve; mutual containment is preferred because its threshold reads directly as a sentence ("each holds at least 80 % of the other") and $J \le s$ always, so a Jaccard threshold would have to be translated for the reader. The comparison is at the level of row sets, not of feature correlation: it is conditional on the certificate (only certified cells are compared) and on the target (the centres are those of this value at this $\tau$).

**Grouping.** At a user threshold $t \in [0.5, 1]$, the centres are grouped by *leader clustering* (Hartigan, 1975, Sec. 3.2) in a fixed rank order: fewest characteristics first; then the highest one-sided Clopper–Pearson lower bound of the cell's purity at its own schema's Bonferroni level $\alpha / C_{\text{occ}}$ (the interval the lattice prints for that cell); then the larger cell; then enumeration order. A centre joins the already chosen representative it is most similar to if that similarity is at least $t$, and becomes a representative otherwise. Exactly two properties follow, and both are checked by brute force in `tests/test_redundancy.py` at six thresholds: (i) every member satisfies $s(\text{member}, \text{representative}) \ge t$; (ii) no two representatives satisfy $s \ge t$. Two members of one group need *not* be $t$-similar to each other — $s$ is not transitive. Complete linkage would guarantee it at the price of splitting groups a reader sees as one; the interface therefore reports every member's similarity *to the representative*, the quantity the guarantee is about. Centres with identical row sets always share a group, so the computation runs on distinct row sets. At $t = 1$ the groups are exactly the distinct row sets (tested).

**Per-member quantities** (`CenterCatalog.group_detail`, `/api/centers/group`). Both one-sided shares; the similarity expected for two row sets of the same sizes placed at random, $s_0 = \min(|R(a)|, |R(b)|)/N$ (the hypergeometric mean of the intersection divided by the larger size), and the observed similarity rescaled against it, $(s - s_0)/(1 - s_0)$, in the manner of Cohen's $\kappa$ — large centres overlap a great deal by chance: on `titanic`, `sex = male` (577 rows) against `title = Mr` (517) has $s_0 = 517/891 = 0.58$, so its observed $s = 0.90$ is $0.75$ of the way from chance to identity; and the rows each side holds that the other does not, with the share of the value among them. The last pair is what decides between two similar descriptions: `sex = male` adds 60 rows to `title = Mr` at 53 % died, *below* the 61.6 % base rate — the extra rows are not part of the group.

**Scope and cost.** The centres are those of the *search* partition of each candidate, under the same `CenterSpec`, i.e. exactly the cells `Landscape` counts (tested under both rules and both directions); where the grid-capacity rule merged levels, a centre's description lists every category its cell holds (labels unique within a column, so a missing value and the string "missing" cannot be confused), and the conjunction of the descriptions selects exactly the cell's rows (tested on a coarsened schema). Pairs are stored only when $s \ge 0.5$; since $s(a,b) \le \min/\max$ of the two sizes, a centre of size $n$ is compared only with centres of size in $[n/2, 2n]$, which is exact (no stored pair is skipped; tested against the full $S \times S$ matrix). Intersections are integer counts from a blocked 0/1 product in `float32`, exact for $N < 2^{24}$. The cost is $O(S^2 N)$ in the worst case for $S$ distinct row sets; measured on two cores: `titanic` ($\tau = 0.7$) 2 225 centres → 892 distinct sets in 0.1 s; `mushroom` ($\tau = 0.9$) 186 731 centres → 19 327 distinct sets, 184 111 pairs, 10.6 s; `chess_krkp` ($\tau = 0.9$) 65 554 → 18 657 sets, 3.8 M pairs, 12.0 s. The catalogue is computed once per (target, value, certificate, direction, `min_rows`) and cached; changing $t$ only regroups (0.2 s on the two largest). Above `MAX_DISTINCT_SETS` = 60 000 distinct sets the server refuses with the reason and asks for a larger `min_rows`.

**`min_rows`.** Centres with fewer rows are dropped before grouping (library default 1, interface default 10). At `min_samples = 1` (the certificate default, Section 3.2) many centres are tiny: on `titanic` at $\tau = 0.7$, 572 of 2 225 centres are single rows and 1 357 have fewer than ten. Single-row centres on the same row are trivially identical, and a single-row cell's purity interval is $[\alpha/C, 1]$; grouping them says nothing. The parameter changes neither the landscape nor any coverage, and it appears in every response.

**What the data say.** On the observational datasets redundancy is the rule, not the exception: on `titanic` at $\tau = 0.7$ with `min_rows = 10`, 868 centres collapse to 583 distinct row sets and 241 groups at $t = 0.8$; 469 of the 868 centres have another centre with *identical* rows, and the nearest-neighbour distribution of $s$ has no gap between 0.5 and 1 — there is no natural threshold, and the interface shows that histogram next to the control rather than implying one. On the fully crossed designed datasets there is almost none: `car_evaluation` (1 367 centres, 44 pairs above 0.5) and `nursery` (1 631 centres, 446 pairs) — every centre is its own group. On a composite target the grouping exposes proxies the feature-space rule of Section 4.10 cannot remove: for *died ∧ male*, `sex` is removed from the features, and the top representative is built on `title = Mr`, which encodes it.

**Statistical status.** Grouping is descriptive. It changes no centre, no coverage and no $p$-value; it reports which certified cells coincide in the sample. It depends on $t$, the rank rule and `min_rows`, all explicit. No inference is made that two groups differ in the population; a centre's lower bound is the one its own lattice reports, not a bound corrected for having been chosen as a representative.

**Branch view.** The reader's first question is about the branch on screen, not about the whole family: *which of these centres could have been found with other characteristics, or with fewer?* `CenterCatalog.branch_view` answers it for one schema (the selected branch, or a schema opened from the landscape): for each of its centres, every centre of every other scored schema whose mutual containment with **that centre** is at least $t$, most similar first. The comparison is direct, not through a group representative, so every listed alternative carries the guarantee $s \ge t$ with the branch centre itself (checked against the brute-force matrix in `tests/test_redundancy.py`); the simplest alternative (fewest characteristics, then rank) is reported separately. On `titanic` at $\tau = 0.7$, 4 of the 15 centres of the 3D branch `sex + age_group + family_type` have a 2D alternative at $t = 0.8$ — e.g. `sex = male ∧ age_group = unknown_age ∧ family_type = alone` (107 rows) and `title = Mr ∧ age_group = unknown_age` (119 rows) share 89 % — and none of the five centres of the 2D branch `title + family_type` has a 1D one. Centres of the schema below `min_rows` are listed as not compared.

**Display.** A third view mode, *Redundancy*, beside Lattice and Landscape. It opens on the selected branch: one card per centre with its description, rows, purity with its lower bound, share of the value, the number of other descriptions (and how many hold exactly the same rows), and the simplest one when it uses fewer characteristics; clicking a card lists the alternatives with the per-member quantities above, under a header carrying the union of the card and all its alternatives. A toggle switches to *All centres*: every centre of every scored schema grouped by the leader rule, one card per representative, sortable by the share of the value or the number of descriptions, optionally restricted to groups with a centre of the branch's $d$. Both scopes share the threshold ("same if each holds $\ge t$ of the other's rows"), `min_rows` and the *Descriptions* filter (all / related only / unrelated only). Each listed alternative carries its relation to the reference as one cell — a word for the characteristics (`extends`, `shortens`, `other`) and a symbol for the rows ($=$, $\subset$, $\supset$, $\times$) — with $\unicode{x26A0}$ on the pairs where the two disagree. Any listed schema opens as a lattice; the Landscape tab rings the lattice cells of the schemas of the card opened in Redundancy.

**Kinship: two facts, one measured.** A duplicate found by adding characteristics is not the same finding as a duplicate built on other characteristics, so each listed alternative carries both facts separately. *Lineage* compares the two schemas' **column sets**: `child` (the reference's columns are a strict subset of the alternative's), `parent`, or `unrelated` (neither contains the other — necessarily the case between two schemas of equal $d$). *Relation* compares the two **row sets**, from the exact intersection already computed: `identical`, `inside`, `contains`, `crossing`. A filter on lineage ("related" / "unrelated" / all) restricts the list, and every derived quantity of a card — `total`, the identical count, the union, the simplest alternative — is recomputed on the filtered set, so a card never mixes two filters; the unfiltered split is reported alongside whichever filter is on. In the branch scope the filter also drops the centres that have no alternative of the chosen kind — the list then answers "which centres of this branch have a description of this kind", and the summary says how many of the branch's centres are hidden; the branch's own centre count and coverage above it stay whole-branch. In the *All centres* scope the grouping is never filtered (it is the clustering, and the server pages it), so each group card shows the split with the chosen side highlighted and the filter takes effect in the expanded member list. On `titanic` at $\tau = 0.7$, $t = 0.8$, of the 3 326 listed pairs 1 192 are related and 2 134 unrelated.

The second fact is **measured, never inferred from the first**. A schema's partition refines its sub-schema's only while no shared column was coarsened to fit the grid capacity (Section 4.3), and that condition fails in practice: on `titanic` at $\tau = 0.7$, 60 of the 504 nested schema pairs (11.9 %, all $3\text{D} \to 4\text{D}$) do **not** refine their parent, so a "child" cell can straddle two parent cells. Ten of the 3 326 listed pairs are such cases — a child whose rows are not inside its parent's, or even contain them. They are flagged as inconsistent and counted, rather than presented as sub-cells of their reference. Inferring the row relation from the column relation would have mislabelled every one of them; `tests/test_redundancy.py` pins both the classification and the existence of the coarsening case.

**Where the threshold comes from.** Since $t$ has no principled default, every scope shows the distribution the cut is made on, binned in steps of $0.05$ from the pair-store floor $0.5$ to $1$ with exact identity ($s = 1$) on its own bar and everything the store does not hold counted as "below the floor"; the current $t$ is a dashed line. Three scopes, one axis: **all centres** — each centre's similarity to its *nearest* other centre, over the whole family, above the cards in *All centres*; **branch** — the same quantity for the compared centres of the selected schema only, above the cards in *Selected branch*; **centre** — one centre against *every* other centre of the run, drawn inside that centre's own card. The third is the operative one: the bars at or above the dashed line are exactly that card's "other descriptions", so the strip shows what lowering or raising $t$ would add or drop for the centre it sits under, and the reader chooses $t$ per case instead of from a family-wide summary. The first two are nearest-neighbour distributions and answer a different question — whether *any* natural cut exists (on `titanic`, none does). A distribution of maxima is pressed against the right edge by construction, so the branch scope draws a **fourth** picture beside it, on the same axis: every (branch centre, other centre) pair, which is exactly the elementwise sum of the per-centre strips in the cards below (pinned as such in the tests). Its bars at or above the threshold are the branch's whole list of alternatives added up, and unlike the maxima it keeps the entire left tail, which is where a gap would appear if the data had one. It is the branch scope only: pooling every pair of the whole family is quadratic in the centre count, and the per-centre strips it sums are computed there anyway. Under a lineage filter the cards of centres with no alternative of the chosen kind are dropped from the list, so the pooled bars are drawn in two parts: solid for the centres the cards show — that part is exactly the sum of the strips on screen — and faded above it for the centres the filter hides, whose pairs all lie below the threshold and are the reader's warning that lowering it would bring those centres back. Solid plus faded is every pair of the branch, so no observation is lost to the filter. All three are drawn on the population the lineage filter leaves: under "related only" a centre's nearest neighbour is its nearest *related* centre, so the picture never counts pairs the list beside it excludes. The filtered nearest-neighbour pass walks each set's stored neighbours in descending similarity and stops at the first admissible one, which is $O(1)$ steps in practice (on `titanic` at $\tau = 0.7$, 690 centres: 4 ms against 125 ms for the dense route, equality pinned in the tests).

### 4.12. The dataset screen: what each column is, and which columns are renamings

**What it is.** Two columns of a dataset can say the same thing. On `titanic`, `title` fixes `sex` on 888 of the 891 rows — `Officer` and `Royalty` are the three exceptions — so a schema holding both is, on almost every row, a schema holding one; the "4D" branch of Section 4.11's example is really three characteristics. Nothing in the search notices: the encoding is per column (Section 2.1), and the family is every $\binom{M}{\le 4}$ subset whether or not two of its members are the same column twice. `vsf.screen` reports this before the search, together with what the framework will do with each column.

**The measure.** For two columns $X$ ($a$ categories) and $Y$ ($b$), the *determination* of $Y$ by $X$ is
$$\delta(X \to Y) = \frac{1}{N}\sum_{x}\max_{y} n(x, y),$$
the share of rows on which the best deterministic map $X \to Y$ is right. $1 - \delta$ is the $g_3$ error of the functional dependency $X \to Y$ (Kivinen & Mannila, 1995) — the smallest share of rows that must be deleted to make it hold — and the screen reports $N(1-\delta)$ as an integer count of *exception rows*, which is what a reader can check. $\delta = 1$ iff $X$ determines $Y$; $\delta = 1$ in both directions iff the two columns induce the **same partition** of the rows, i.e. one is a renaming of the other.

$\delta$ alone is unreadable, because a constant predictor already reaches the largest class share of $Y$: a column whose largest category covers 99 % of the rows is "determined" to 0.99 by anything at all. The screen therefore ranks by
$$\text{strength} = \frac{\delta - \text{baseline}}{1 - \text{baseline}}, \qquad \text{baseline} = \max_y n(y)/N,$$
Cohen's $\kappa$ applied to the majority rule: the share of the possible improvement over that constant predictor that is realised. An inexact pair is listed when its strength reaches the reporting floor (0.90 by default, the reader moves it); an exact dependency is listed whatever its strength.

**Why the screen must be approximate.** An exact-duplicate check finds nothing on most real data. Across the nine bundled datasets, exact dependencies exist only on `audiology` (2 equivalent pairs, 7 one-way) and `soybean_large` (3); `titanic`, `car_evaluation`, `nursery`, `mushroom`, `chess_krkp` and `chess_krk` have **none** — and `titanic` is exactly the dataset on which the redundancy is doing damage. Three rows out of 891 are the difference between "no duplicates found" and the finding that matters.

**What each finding licenses, and what it does not.**
* *Equivalent* ($\delta = 1$ both ways). The partitions are identical up to renaming. Excluding either column is lossless for every candidate and every certificate; only the size of the family changes.
* *Exact one way* ($X \to Y$). Refining by $Y$ splits no cell of any partition containing $X$, so a schema holding both has the same partition as the schema without $Y$ — **before** the capacity rule. Excluding $Y$ as a column is **not** lossless: $Y$ alone is a coarsening that $X$ alone cannot reproduce, and coarser cells are larger, hence easier to certify. The right response is to prune the redundant *candidates*, not the column (below).
* *Approximate*. Nothing is proved. The screen gives the exception count and stops; the decision is the reader's.

**Pruning renaming candidates** (`prune_dependent=`, off by default). When $X \to Y$ exactly and both are in a candidate $S$, the partition of $S$ equals that of $S \setminus \{Y\}$ *provided neither is coarsened* — and since the two have the same occupancy, that is decided by one comparison: the search prunes $S$ only when the occupancy of $S \setminus \{Y\}$, recorded before coarsening when that subset was enumerated, fits $\lfloor N/10 \rfloor$. Otherwise the two are coarsened from different nominal level counts, can differ, and the candidate is kept and scored. The error is therefore one-sided: a pruned candidate is always an exact duplicate of a scored one (pinned by comparing partitions in `tests/test_screen.py`), never a schema the reader would have wanted. What it buys is not speed: on `audiology` ($226 \times 69$, capacity 22) it removes 8 888 of 919 310 candidates — 0.97 % — and costs about 5 % of the running time, because at $N = 226$ most three-column subsets already exceed the capacity and the rule declines. What it buys is that no reported branch is a renaming: on a constructed dataset with one duplicated column, the unpruned search reports a 3D and a 4D branch with *exactly* the 2D branch's coverage and centre count, and with pruning the 4D branch does not exist at all — four columns of which two are one column cannot span four.

**Per-column profile.** Categories (missing values are one category, coded last, Section 2.1), the largest category with its share, how many categories hold a single row, the number of missing rows, and four advisory flags: `constant` (one category — every schema holding it has the partition of the schema without it), `exceeds_capacity` (more categories than $\lfloor N/10 \rfloor$, so even its own 1D grid is coarsened), `dominant_level` (one category covers $\ge 90\,\%$ of the rows), and `placeholder_level` — a category whose **name** reads as "no value recorded" (`?`, `unknown`, `unknown_deck`, `N/A`, ...) covering at least 5 % of the rows. The last one is the honest form of a problem the framework cannot solve: `deck = unknown_deck` is 77 % of `titanic` and is the strongest 1D branch at $\tau = 0.7$, but nothing in the data says whether that category means "no deck recorded" or "deck U". The screen checks the label and says so; the reading stays with the reader.

**Target screen (leakage).** Once a target is named, the same measure is computed from every remaining column to the target indicator. A column at strength 1 reproduces the target exactly — a tautology or a leak — and the search will return it as the whole answer. This is the one part that depends on the target, so it runs after the target is chosen and is reported beside the pairs.

**Where the decision sits, and why there.** The pair screen and the column profile are functions of the feature columns alone: no target, no certificate, no result. The reader therefore excludes columns at the only point in the workflow where that decision cannot be a reaction to an outcome. The exclusions are then part of the analysis identity — `drop=[...]` travels with every request, enters the cache key, changes the candidate family and hence the familywise null and every landscape count, and is echoed in `/api/analyze` (`dropped_columns`, `prune_dependent`) for the legend and the export.

**What this does not fix.** Redundancy is one of three things the Section 4.11 example showed. Excluding `sex` would not have changed the other two: the 4D branch that loses coverage to its own third condition (a refinement that does not pay), and `deck = unknown_deck` (a missingness proxy the screen can only name). Nor do duplicated columns make any reported number *wrong*: the certificate is per cell and stays valid. They waste the search, inflate the candidate count that multiplicity is corrected over — making the correction more conservative, i.e. costing power, not validity — and overstate the dimensionality of a result.

**Cost.** One contingency table per unordered pair, both directions read off it: $O(M^2 N)$, with a sort-based fallback when $ab$ exceeds $2^{24}$ cells and a budget above which the pair screen is skipped with its reason in the payload (the per-column profile is always computed). Measured: `titanic` ($891 \times 9$) 7 ms, `audiology` ($226 \times 70$) 28 ms, `mushroom` ($8\,124 \times 22$) 153 ms, `chess_krkp` ($3\,196 \times 37$) 97 ms. Cached per (screened columns, floor).

**API.** `vsf.screen_dataset(X, names, min_strength=)`, `vsf.column_profiles`, `vsf.dependency_pairs`, `vsf.exact_dependencies`, `vsf.target_report`; `/api/screen` (with an optional target for the leakage part); `drop=[column, ...]` and `prune=true` on `/api/analyze`, `/api/target` and the whole landscape and redundant-centre family; `prune_dependent=` on `discover_branches`, `compute_landscape`, `compute_tau_curves`, `discover_branches_by_value` and `collect_centers`. Display: a panel on the guide screen, before any target is chosen, and the same panel as the *Data* view mode afterwards; changing an exclusion after an analysis offers to re-run it.

### 4.13. What coverage costs: the trade-offs view

**The question.** The search ranks candidates by coverage alone. Two things it therefore never reports: how the winner's *description length* moves as the purity floor moves, and how much coverage is available for less. On `titanic` at $\tau = 0.9$ for *survived*, the reported 2D winner is `passenger_class + title` — 7 certified centres, 49.7 % coverage. One schema of the same dimensionality, `passenger_class + sex`, certifies **47.1 % with 2 centres**: 95 % of the coverage for under a third of the description, and it appears nowhere in the branch list, because it does not win on the only axis the ranking uses. At 4D the extreme is 52.3 % for **40** centres. A reader who has to write the result down needs that comparison, and until now the interface did not carry it.

**Four pictures, two shared axes.** The *Trade-offs* view mode holds them; the first three share the purity-floor abscissa and are stacked, so a value is read off all of them at one $\tau$.

1. *Coverage against the purity floor* — the $\tau$-curves of Section 4.9, unchanged, moved out of the 300-pixel sidebar into the viewport.
2. *Certified centres against the purity floor*, stacked directly under (1) on the same abscissa, so a value is read off both at one $\tau$ without a second scale. **The schema changes along the envelope**: on `titanic` the winner changes 15 times for $d = 2$, 23 for $d = 3$ and 22 for $d = 4$ over the 62-point grid, and consecutive winners' centre counts jump (28, 13, 6, 12, 22, 11, 4). A cost series of a moving schema read as a trend would be a straightforward misreading, so every change is ticked under **this** panel and no other — on the coverage panel a change of winner leaves no artefact, the envelope being continuous by construction, while here it is a step that is not a property of the data. A *Track* control switches both panels to the selected branch — one fixed schema, drawn at the floors where it is its dimensionality's best.
3. *Which dimensionality leads*, as one band on the same abscissa under (2): the pointwise argmax of (1), each stretch in that $d$'s colour, with ties going to the **smaller** $d$ — at equal coverage the shorter description is the better one, the rule the toolbar states. It turns the crossings of the four lines into something read rather than eyeballed, and it is diagnostic in its own right: on `titanic` the band breaks into 16 stretches alternating $4\text{D} \leftrightarrow 3\text{D}$, which says the two envelopes run together over most of the range, not that the answer changes sixteen times.
4. *Coverage against cost at the current floor* — the **Pareto staircase** (`Landscape.frontier`, `/api/landscape/frontier`). For each $d$, the schemas that no schema of that $d$ beats on both axes at once; a step reads "with at most this many rules, this is the most of the value any $d$-schema certifies". The staircase is computed from the cached landscape by a sort and a running maximum — no new pass over the data — and the selected branch is ringed on it, usually far to the right of the knee.

**Cost is offered in two units, and the choice governs both cost panels at once** — (2) and (4) never measure the description in different units. *Centres* counts rules; *conditions* counts the $(\text{column} = \text{value})$ pairs the description spells out, exactly $d \times K$ because every centre of a $d$-schema fixes $d$ columns. Counting centres is not scale-free across dimensionalities — a 4D rule is four times the text of a 1D rule — so only the second unit puts the four lines of (2) on one axis: on `titanic` the median envelope costs 1, 7, 29.5 and 45 centres for $d = 1 \ldots 4$, which in conditions is 1, 14, 88.5 and 180, turning the 4D-to-2D gap from $\times 5$ into $\times 10$. On the staircase the same schema that costs 4 conditions (2 centres, 2D) is compared with a 4D winner at 160. One caveat: a cell the capacity rule coarsened lists several categories for a column, so that condition reads $\text{column} \in \{a, b\}$ — wider, but still one condition, and the count is unaffected.

**What is not claimed.** All three are maxima over a family that was scored and then chosen by looking. The values on them are optimistic and carry no uncorrected $p$-value, exactly as for a schema opened from a lattice cell (Section 4.9); opening one still reports out-of-sample cross-validated coverage. The frontier is also *not* a ranking: the ratio coverage / centres is deliberately not computed anywhere, because a ratio of a bounded quantity to a count rewards degenerate solutions — a schema certifying 7.9 % with one centre would outscore one certifying 47.1 % with two. The picture presents the trade-off; the choice is the reader's.

### 4.14. Post-selection certificates and nested cross-validation

`vsf.selective` (2026-09). Nothing in Sections 3–4.13 changes meaning; these are separate entry points whose results carry their own guarantee statement.

**The claim and the model.** Rows are i.i.d. and partitions depend on the feature values only (the capacity rule included). Conditional on the features, a cell's count is Poisson-binomial; its mean purity $\pi_c$, the average of $\Pr[Z = z^\ast \mid x_i]$ over the cell's rows, is what a certificate is about: the claim $\pi_c > \tau$, tested by the one-sided exact binomial test (`exact_upper_tail`). For a threshold $k \ge n\tau + 1$ the Poisson-binomial upper tail is at most the binomial one at the same mean (Hoeffding, 1956, Theorem 4), so the test is valid and conservative for $\pi_c < \tau$. Every certifying threshold satisfies $k \ge n\tau + 1$ at per-cell levels up to $0.05$ (checked for $n \le 2000$ on a $0.01$ grid of $\tau$; at $0.2$ it fails for small cells), so `certify_discovery` refuses $\alpha > 0.05$ (`MAX_ALPHA`).

**Family-wide Bonferroni** (`CenterSpec(rule="certified", multiplicity="family")`; `certify_discovery(method="family_bonferroni")`). The family is every occupied cell of every partition the search can report or display: each scored schema's search partition, and, where the capacity rule coarsened it, also its uncoarsened partition, which is what `vsf.vis` draws; over all $d \le 4$ (`family_cell_count`). Every prefix view of a branch is the partition of a smaller schema, and a schema skipped as a renaming has the row sets of a counted one, so every cell the product can colour is one of the $T$ hypotheses. Cells of any size are counted: `min_samples` belongs to the user and can move after the data are seen, so the family cannot depend on it. A cell is certified iff $p_c \le \alpha / T$. The union bound over the whole family makes the search irrelevant: $\Pr[\text{some cell with } \pi_c \le \tau \text{ is certified, anywhere}] \le \alpha$ for any dependence between cells and any rule for picking the reported schema. Because a false certificate at any $\tau \ge \pi_c$ implies that the cell's Clopper–Pearson lower bound at level $\alpha/T$ exceeds $\pi_c$ — one event per cell — the guarantee also holds simultaneously over every $\tau$ the user tries. The search ranks schemas by the coverage of cells certified this way, so the reported branch maximises the certified quantity itself. $T = 8\,983$ on `titanic` (7 252 search-partition cells plus the uncoarsened partitions of the coarsened schemas); the family size is cached per feature set, so every endpoint of an analysis shares one count. Every search entry point resolves $T$ itself (`resolve_center_spec`); a partition-level function given an unresolved family spec raises instead of guessing. `alpha` is limited to $0.05$ (`MAX_CERTIFICATE_ALPHA`), for the reason given in the previous paragraph.

**Sample splitting** (`method="split"`). The rows are split uniformly at random, *independently of the target*, into halves $A$ and $B$ (`random_halves`). The search runs on $A$ and ranks by observed purity at the caller's $\tau$ and $m$, whatever rule the caller names ($A$ only proposes, $B$ certifies); the cells of each reported schema that pass a screen on $A$ (default: $A$-purity $\ge \tau$, $A$-size $\ge m$) are the only hypotheses, tested once on $B$'s counts at $\alpha / T_B$, $T_B$ the number of screened cells of all reported branches together (`branch_multiplicity="all_branches"`, default). Conditional on the features and on $A$, $B$'s targets are independent of everything that chose the hypotheses, so the bound is exact. A split stratified on the target would fix $B$'s positive count from the target values and break that argument; it is deliberately not used. Partitions are built from all rows' feature values, as everywhere else — the capacity rule reads no target.

**Considered and rejected: a Westfall–Young min-$p$ permutation null.** Permuting the target tests independence of target and features, not $\pi_c \le \tau$. With $\tau$ above the base rate, a cell whose true purity is $0.85$ is extreme against independence and would be "certified" as above $0.90$ with high probability. Such a procedure controls false certificates only under the global null (weak control) and cannot back the claim the display makes.

**Measured.** Global null of Section 4.5, 100 runs: both methods certify nothing in any branch (0/100 for every $d$). A global null cannot show *strong* control, since every cell there is far below $\tau$; `experiments/certificate_power.py` adds a cell whose purity is exactly $\tau = 0.80$ beside a true centre at $0.95$ and a background at $0.10$ (eight 3-level columns). A false certificate is a certified cell whose population purity — the mean of $\Pr[Z=1\mid x_i]$ over its rows — is at most $\tau$.

| $N$ | runs | per-schema (current) FWER / power | family Bonferroni FWER / power | split FWER / power |
|---:|---:|:---:|:---:|:---:|
| 500 | 200 | 0.000 / 0.745 | 0.000 / 0.010 | 0.000 / 0.180 |
| 1 000 | 200 | 0.015 / 0.975 | 0.000 / 0.565 | 0.000 / 0.645 |
| 3 000 | 100 | 0.000 / 1.000 | 0.000 / 1.000 | 0.000 / 1.000 |

Power is the share of runs certifying at least one cell of population purity $\ge 0.90$. The price of a valid certificate is concentrated at small $N$: at 500 rows the family-wide correction certifies the true centre in 1 run of 100, splitting in about one run of five. On the real data, `titanic` at $\tau = 0.7$: the family-wide method certifies `passenger_class + sex` (first- and second-class women, 47.1 % of survivors), splitting certifies nothing; at $\tau = 0.9$ neither certifies anything.

**Neither makes the reported coverage out-of-sample.** The evaluation counts both certify the cells and measure their coverage. That number is the next paragraph's.

**Nested cross-validation** (`nested_crossvalidation`). Definition 6 re-selects centres on each training fold but keeps the schema chosen on all rows. The nested estimate repeats the whole search on each training fold (`_exhaustive_search(rows=train)`), applies the fold's own winner and its training-fold centres to the held-out fold, and is summarised exactly as Definition 6 (`summarize_cv`). The folds are a function of the target and the seed only, so the nested and the fixed-schema estimates are paired. `NestedCVResult.select_dimensionality` is Definition 10 on the nested estimates. Every fold's winner is recorded, and `SchemaStability` reports the share of folds whose winner equals the full-data winner, the number of distinct winners and their mean pairwise Jaccard similarity. Cost: $R \times K$ full searches.

**Measured** (`experiments/nested_cv.py`, $R = K = 5$, Rule P, $m = 1$ unless stated; held-out pooled purity in brackets):

| data, target, $\tau$ | $d$ | winner (all rows) | in-sample | fixed-schema CV | nested CV | same winner |
|---|:---:|---|---:|---:|---:|---:|
| `titanic`, survived, 0.9 | 2 | passenger_class + title | 49.7 | 44.9 ± 3.7 (94.8) | 48.3 ± 2.4 (95.1) | 56 % |
| | 3 | passenger_class + sex + age_group | 51.2 | 44.3 ± 4.6 (95.7) | 47.4 ± 3.1 (94.4) | 40 % |
| | 4 | passenger_class + title + deck + fare_level | 52.3 | 38.7 ± 3.0 (89.8) | 45.6 ± 3.1 (91.6) | **8 %** (13 distinct) |
| `titanic`, survived, 0.9, $m = 20$ | 2 | passenger_class + sex | 47.1 | 47.1 ± 2.3 | 47.1 ± 2.3 | 100 % |
| | 4 | passenger_class + sex + fare_level + port_embarked | 39.8 | 32.0 ± 5.1 | 37.0 ± 3.6 | 24 % |
| `car_evaluation`, acc, 0.9 | 4 | buying + maint + persons + safety | 59.9 | 44.0 ± 3.8 | 44.0 ± 3.8 | 100 % |
| `nursery`, priority, 0.7 | 4 | parents + has_nurs + children + health | 75.9 | 72.0 ± 1.4 | 73.9 ± 1.2 | 20 % |
| `chess_krkp`, won, 0.9 ($d \le 3$) | 3 | bxqsq + wknck + wkpos | 56.9 | 56.9 ± 1.3 | 58.6 ± 2.8 | 88 % |
| `mushroom`, poisonous, 0.9 | 4 | bruises + stalk_root + ring_number + spore_print_color | 100.0 | 100.0 | 100.0 | 100 % |

Three things the table makes unavoidable.

1. **The in-sample figure the interface prints overstates what replicates** by up to 14 points (`titanic` 4-D: 52.3 → 38.7 / 45.6), and by 16 points on a planned design with a *stable* schema (`car_evaluation`), where the loss comes from cells of about twenty rows whose purity sits near $\tau$.
2. **The reported 4-D schema of `titanic` is not a finding about the population.** It is the fold winner in 2 of 25 folds, among 13 distinct winners. Its coverage is reproducible (45.6 %) only as the coverage of *the procedure*, not of that schema. $m = 20$ halves the instability at $d = 4$ and removes it at $d = 2$.
3. **Nested is not a lower bound of fixed-schema CV, and exceeded it wherever the schema was unstable.** The two estimate different procedures, and the gap has a measured cause (`experiments/cv_gap_decomposition.py`, same folds). Applying the full-data winner's *full-data* centres to the held-out folds (which leaks the test rows) and subtracting the fixed-schema estimate leaves the held-out positives of cells that are centres on all rows but not on the training fold, to within 0.1 point: 13.7 points of `titanic` 4-D at $\tau = 0.9$, 16.1 of `titanic` 2-D at $\tau = 0.7$, 15.9 of `car_evaluation` 4-D. Cells that become centres only on a fold contribute at most that 0.1 point. The full-data winner is chosen for cells that clear $\tau$ on all rows, and some of them fall below it on 80 % of the rows (regression to the mean under selection). A fold's own winner is chosen on the rows its centres are selected on and loses nothing this way. The simpler story — that the full-data winner holds more cells within $0.05$ above $\tau$ than the fold winners — holds in 9 of the 12 non-trivial (data, $\tau$, $m$, $d$) cases checked, not in all. What follows: a held-out coverage is interpretable only beside its held-out purity — a procedure that selects more cells raises the first and lowers the second — and both columns are reported.

**In the interface.** *The lattice is coloured by the family-wide certificate by default.* Display Settings → *Colouring* offers three choices: **Certified (valid after search)** — `rule="certified"`, `multiplicity="family"`, the default; **Legacy: certified per schema** — the earlier strict mode, Bonferroni over the displayed partition only, optimistic after a search (Section 4.5); **Legacy: observed share** — Rule P, the earlier default, which certifies nothing. In both certified modes the boundary stops at 99 %. The summary under the boundary states which of the three is on screen and, for the default, $T$, the per-cell level and the smallest cell that can be certified at all (a fully pure cell of $n$ rows has $p = \tau^n$; on `titanic` at $\tau = 0.7$, $T = 8\,983$ and $n \ge 34$). The API follows the same rule: `/api/analyze`, the landscape family, `/api/validate` and the scan accept `multiplicity` (`"family"` or `"bonferroni"`); when it is absent a `rule="certified"` request is corrected over the family and a `rule="purity"` request keeps the per-schema intervals, so the per-schema certificate is never obtained by omission. Responses report `family_tests`, `per_cell_level`, `min_certifiable_rows` and `valid_after_search`. `export_full_dashboard` defaults to the same certificate (`DEFAULT_EXPORT_SPEC`) and its legend states which rule it was built under. On `titanic` at $\tau = 0.7$ the default colours 2 cells of the 2-D branch (first- and second-class women, 47.1 % of survivors) and 1 cell each of the 3-D and 4-D branches; the legacy per-schema mode colours 4 and 3 cells at 3-D and 4-D, and the observed share 21 and 24.

Under the branch list, *Check this result* sends the analysis on screen to `/api/validate` with the chosen certificate method and number of repetitions. The server runs `certify_discovery` and `nested_crossvalidation` in a background thread (one at a time; the page polls with the same request body and sees `progress` in exhaustive passes; a failed job stays failed until the button asks again with `retry`). The panel then lists, per dimensionality: the nested held-out coverage and purity; the shown schema re-fitted per fold and its all-rows coverage; how often the same schema wins, with the most frequent winner when it is a different one; and the certified cells with their conditions, counts, certified lower bound and adjusted $p$, flagging in amber a certified schema that differs from the one on screen. With the default colouring, the *all schemas* method certifies exactly the cells the search counted green (the lattice's cells too, unless the capacity rule coarsened the branch); *split halves* is the independent check. Each branch card gains one line: held-out coverage, schema stability, number of valid certificates. The static export does not carry the check.

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
3. **The certificate's price, and the coverage it certifies is in-sample.** The per-schema certificate of Proposition 2 fails for a searched schema (Section 4.5: 68 % of null runs certify something in some branch). The default certificate is now corrected over the whole search family (Section 4.14), which holds after the search at a measured loss of power at small $N$: with 500 rows it finds a true centre in 1 run of 100 where the invalid per-schema certificate finds it in 3 of 4. Sample splitting is less conservative there and is offered as a check. The certified coverage is still measured on the rows that certified it; the nested cross-validated coverage with schema stability is the out-of-sample figure (*Check this result*). The static export carries the certificate but not the check. A selective-inference treatment conditioning on the selection event, or a Tarone-style family that drops cells too small ever to be certified, could recover power; both are open.
4. **$\tau$ and $m$ are user choices with no principled defaults, by design.** $\tau = 0.90$, $m = 1$ are conventions. The centre set, the coverage, the winning feature subset and the reported dimensionality all move with them — on `income = >50K` at $d = 3$, coverage goes from $2.4\,\%$ at $\tau = 0.90$ to $24.6\,\%$ at $\tau = 0.75$. The specification makes the dependence explicit and reproducible ($\tau$, $\alpha$, $m$ and the rule appear in the legend and in the analysis cache key) but does not remove it. **A coverage quoted without its $(\tau, m, \text{rule})$ is meaningless.**
5. **Coverage and $K$ are not jointly optimized.** The rule is lexicographic: coverage dominates, $K$ breaks ties. So the search returns 55 centres for $2.4\,\%$ coverage in preference to 2 centres for $1.2\,\%$ — technically correct and, at that centre count, not a readable finding. A scalarization (coverage minus a per-centre penalty) or a Pareto front over $(\mathrm{Coverage}, K)$ would let the user say "not at any price"; neither is implemented. The interface compensates only by showing both numbers side by side.
6. **Bonferroni is conservative at high cell counts.** At the capacity limit, $C \approx N/10$, so $\alpha/C \approx 1.5 \times 10^{-4}$ on a 32k-row dataset and certification demands cells that are both large and nearly pure. That is the right direction for a certificate, but it costs power exactly where the display is most detailed. A step-down or FDR formulation over cells would recover some of it at the cost of a weaker per-cell guarantee; that choice is not currently offered.
7. **The grid capacity rule is a variance bound, nothing more.** $C_{\text{occ}} \le N/10$ keeps cell counts estimable. It does not bound selection optimism (item 3 does that separately). Tightening it would trade resolution for power on small $N$; and the merge rule for a nominal column, "rarest into other", is a heuristic that keeps dominant levels rather than an optimum of any criterion.
8. **The cluster-preserving axis order is specified but not implemented** (Section 6).
9. **No automatic target discovery.** The user must name the target column and value; the first screen is a usage guide, not an analysis. The optional Global Pattern Scan iterates all (column, value) pairs on request, but it only *filters* the target selector — it never picks a target.
10. **Redundancy is measured, not removed.** The redundant-centre grouping (Section 4.11) reports that different characteristics describe the same rows; it does not change the search, which still ranks schemas independently and may return four branches that are one group of objects under four descriptions. Its threshold has no principled default ($0.8$ is a convention; on `titanic` the nearest-neighbour similarities show no gap), and leader clustering depends on the rank order by design. The lineage filter separates alternatives that extend or shorten the reference's characteristics from those built on other ones, but lineage is a statement about columns only: because the capacity rule may discretise a shared column differently in two schemas (11.9 % of the nested schema pairs on `titanic` at $\tau = 0.7$), a "child" cell is not guaranteed to be a sub-cell of its parent, and the interface marks those pairs instead of claiming containment it has not measured. Attributing the target to characteristics — how much each contributes, net of redundancy — is a separate question that this grouping does not answer.
11. **The dataset screen is advisory, and its thresholds are conventions.** The reporting floor (0.90 of the possible gain over the majority rule), the dominant-level share (0.90), the placeholder share (0.05) and the placeholder vocabulary are defaults, not findings; the screen reports the exception counts so that the reader judges rather than trusts them. Placeholder detection reads the label, not the meaning: a category genuinely named "Other" is flagged, and a placeholder spelled in a way the vocabulary does not know is not. Excluding a column remains a human decision made before the search, and the framework records it rather than validating it.

---

## Appendix. Implementation Map

`vsf` 2.3.0 · Python ≥ 3.10 · dependencies: `numpy`, `pandas` only (`pytest` for the suite). `pyproject.toml` is the single source of truth for packaging.

| Module | Responsibility |
|---|---|
| `vsf/pmd.py` | Category encoding, grid capacity, adaptive level merging (Section 2) |
| `vsf/metrics.py` | Dense joint-cell codes and Benjamini–Hochberg FDR control |
| `vsf/centers.py` | The reported layer: Clopper–Pearson and Wilson bounds without `scipy`, Rules P/C, coverage/$K$/purity, cross-validation, hypergeometric and familywise nulls, dimensionality selection, and `coverage_score` (Section 3.2) |
| `vsf/avr.py` | Independent Branch Discovery: `discover_branches`, `discover_branches_by_value`, `BranchEngine`, `select_branch_dimensionality` (Section 4); the solution landscape, its τ-curves and its cost-coverage frontier: `Landscape` (incl. `Landscape.frontier`), `compute_landscape`, `compute_tau_curves`, `tau_grid`, `report_schema` (Sections 4.9, 4.13) |
| `vsf/screen.py` | The dataset screen: per-column profile, approximate and exact functional dependencies between columns, target leakage: `screen_dataset`, `column_profiles`, `dependency_pairs`, `exact_dependencies`, `target_report` (Section 4.12) |
| `vsf/redundancy.py` | Redundant centres: the centre catalogue of every scored schema, the mutual-containment graph, leader grouping at a threshold, per-member overlap quantities, the three similarity histograms (`nearest_neighbour_histogram` for the family and for one branch, `center_similarity_histogram` for one centre): `collect_centers`, `CenterCatalog`, `CenterGrouping` (Section 4.11) |
| `vsf/selective.py` | Post-selection certificates (`certify_discovery`: family-wide Bonferroni, sample splitting; `exact_upper_tail`, `family_test_count`, `random_halves`), nested cross-validation and schema stability (`nested_crossvalidation`, `schema_stability`) (Section 4.14) |
| `vsf/vis.py` | Render payloads per branch, per-view statistics, axis sorting (Section 5) |
| `vsf/server.py` | Embedded HTTP server, REST API (`/api/analyze` with `also` for composite targets, `/api/target`, `/api/landscape`, `/api/landscape/cell`, `/api/landscape/curves`, `/api/landscape/at`, `/api/centers/groups`, `/api/centers/group`, `/api/centers/branch`, `/api/screen`, scan endpoints), response, landscape and τ-curve caches, prefetch, background Global Pattern Scan |
| `vsf/dashboard.py` | Single-file offline HTML export |
| `vsf/webapp/`, `vsf/templates/` | Live app assets and export templates, read via `importlib.resources` |
| `tests/` | The pytest suite (`pytest --collect-only -q` gives the current count), including the exactness pins of `test_fastpaths.py` and the witnesses in `test_pmd.py` / `test_metrics.py` that no binning or information-theoretic code returns; `test_benchmark_manifest.py` pins every bundled dataset to its manifest hash |

**Defaults that determine a reported number.** All are explicit parameters; none is hidden.

| Parameter | Default | Where |
|---|---|---|
| `tau` | `0.90` | `CenterSpec` — purity floor and green boundary |
| `alpha` | `0.05` | `CenterSpec` — simultaneous error rate over occupied cells |
| `rule` | `"purity"` (library) / `"certified"` (interface, export) | `CenterSpec` — Rule P; `"certified"` is Rule C |
| `min_samples` | `1` | `CenterSpec` — no occupancy restriction |
| `min_strength` | `0.90` (`DEFAULT_MIN_STRENGTH`) | `vsf.screen` — reporting floor of an inexact dependency |
| `prune_dependent` | `False` | `discover_branches` — skip candidates that are renamings of a smaller schema |
| group threshold $t$ | `0.8` (`DEFAULT_GROUP_THRESHOLD`) | `/api/centers/groups` — redundant-centre grouping, range $[0.5, 1]$ (`PAIR_FLOOR`) |
| `min_rows` | `1` (library) / `10` (interface) | `collect_centers` — centres below it are not grouped |
| `method` / `multiplicity` | `"clopper-pearson"` / `"bonferroni"` (library, a partition fixed in advance) / `"family"` (interface, export, and any API request with `rule="certified"`) | `CenterSpec`; `family_tests` filled by `resolve_center_spec` (Section 4.14) |
| `MAX_CERTIFICATE_ALPHA` | `0.05` | `vsf.centers` — largest `alpha` a family certificate accepts |
| `max_d` | `4` (`MAX_BRANCH_D`) | `discover_branches` |
| `cv_splits` / `cv_repeats` | `5` / `5` | `discover_branches` |
| `n_permutations_centers` | `999` (`DEFAULT_N_PERMUTATIONS`) | `discover_branches` |
| `n_permutations_familywise_coverage` | `0` (off) | `discover_branches` |
| `random_state` | `0` | `discover_branches` |
| `MIN_POSITIVES_FOR_CV` | `25` | `vsf.centers` — power floor, Definition 7 |
| `t_threshold` | `2.0` | `select_dimensionality`, Definition 10 |
| `max_display_samples` | `10 000` | `prepare_visualization_payload` — rendering only, never statistics |
| grid capacity | $C_{\text{occ}} \le \lfloor N/10 \rfloor$ | `check_grid_capacity` / `coarsen_to_capacity` — the only place levels are ever merged |
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

**HTTP API** (`vsf.serve`). `GET`: `/api/columns`, `/api/scan/status`. `POST`: `/api/analyze` (target, criterion, `also` for composite targets, `features` for an opened schema, `tau`, `alpha`, `rule`, `min_samples`, `direction`, `drop`, `prune`), `/api/target`, `/api/screen`, `/api/landscape`, `/api/landscape/cell`, `/api/landscape/curves`, `/api/landscape/at`, `/api/landscape/frontier`, `/api/centers/branch`, `/api/centers/groups`, `/api/centers/group`, `/api/validate` (`method`, `repeats`, `retry`), `/api/scan/start`, `/api/scan/cancel`. A target with no resolvable positive value returns `400` with an explanation rather than a different analysis.

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
* Hoeffding, W. (1956). On the distribution of the number of successes in independent trials. *Annals of Mathematical Statistics*, 27(3), 713–721. — the Poisson-binomial tail bound behind Section 4.14
* Hartigan, J. A. (1975). *Clustering Algorithms.* Wiley, Sec. 3.2. — the leader algorithm of Section 4.11
* van Leeuwen, M. & Knobbe, A. (2012). Diverse subgroup set discovery. *Data Mining and Knowledge Discovery*, 25(2), 208–242. — cover-based redundancy between subgroups, the closest prior art to Section 4.11
* Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. — the chance correction of the per-member similarity
* Kivinen, J. & Mannila, H. (1995). Approximate inference of functional dependencies from relations. *Theoretical Computer Science*, 149(1), 129–149. — the $g_3$ error of Section 4.12
* Huhtala, Y., Kärkkäinen, J., Porkka, P. & Toivonen, H. (1999). TANE: an efficient algorithm for discovering functional and approximate dependencies. *The Computer Journal*, 42(2), 100–111. — exact/approximate FD discovery, the prior art for the pair screen
