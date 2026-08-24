# VSF Code Review — Correctness & Rigor Audit

Scope: `vsf/*.py`, `server.py`, `setup.py`, `tests/*.py`, cross-checked against `Project_Master_Document.md`. All 19 unit tests pass, but the test suite does not exercise any of the defects below — several of them (benchmark accuracy, FDR control, purity metric on multi-class targets, sparse-strata permutation) are structurally invisible to the current tests. Every claim below marked "confirmed" was reproduced with a runnable script against this codebase (including the shipped `data/mushrooms.csv`); claims marked "by inspection" are deterministic logic read directly from the source, not requiring a data-dependent repro.

---

## Critical — undermines the paper's core methodological claims

### 1. The spec's headline FDR control does not exist in code (confirmed)

`Project_Master_Document.md` explicitly specifies Benjamini–Hochberg correction as a mandatory safeguard against multiple testing, and the comparison table (line 450) claims VSF uniquely provides `✔️ Yes (α = 0.01, FDR)` versus baselines:

```
STAGE 2: Full AVR cycle + FDR
- Conditional permutation test (p < 0.01)
- Benjamini-Hochberg correction (FDR q ≤ 0.05)
```

`grep -rn -i "benjamini|fdr|bonferroni|holm|multipletests" vsf/ server.py tests/` returns **zero matches**. `AVREngine.fit` (`vsf/avr.py`) runs `marginal_permutation_test` independently for every feature in Phase 1 (line ~106) at the raw, uncorrected `alpha`. With `M` features tested at `α = 0.01` and no correction, the family-wise Type I error rate is `1 - (1-0.01)^M` — ≈ 18% for `M = 20`, and climbing — while the paper's own table asserts this is controlled. This is not a missing nice-to-have; it's a specific, cited, numbered claim (`q ≤ 0.05`) in the spec that the implementation never attempts.

### 2. The accuracy benchmark scores correct behavior as failure (confirmed)

`vsf/benchmark.py::generate_synthetic_dataset` builds `d_true` "informative" features as the *same three* deterministic transforms of `Z` (cycling on `j % 3`) plus small Gaussian noise (`noise_level=0.2`). They are not independent evidence about `Z` — they're redundant restatements of it:

```
d_true=1: NMI(Z; feature_1 alone) = 0.8126   NMI(Z; all 1 informative) = 0.8126
d_true=3: NMI(Z; feature_1 alone) = 0.8126   NMI(Z; all 3 informative) = 0.9982
d_true=5: NMI(Z; feature_1 alone) = 0.8126   NMI(Z; all 5 informative) = 1.0000
d_true=7: NMI(Z; feature_1 alone) = 0.8126   NMI(Z; all 7 informative) = 1.0000
```

A single feature already carries 81% of the achievable information; near-total NMI saturates well before `d_true` features are used. Running the shipped `evaluate_vsf_accuracy` protocol reproduces the consequence directly:

```
d_true=5: d_star_accuracy = 0.0   mean_recall = 0.40   mean_vir = 0.990
d_true=7: d_star_accuracy = 0.0   mean_recall = 0.34   mean_vir = 0.999
```

`d_star_accuracy` is 0.0 exactly where `mean_vir` is ~0.99 — i.e., the engine is finding a near-perfect minimal-sufficient basis (the stated design goal), and the paper's own metric scores that as total failure because it demands `d* == d_true` on a benchmark where `d_true` was never the true minimal dimensionality. If these numbers, or numbers from a benchmark built the same way, are meant to support a paper's quantitative claims, they are invalid on their face. Notably, `tests/test_benchmark.py` only asserts `overall_feature_recall >= 0.5` — it never asserts on `overall_d_star_accuracy`, which would fail immediately given the above.

### 3. Conditional permutation test breaks exchangeability across distinct strata (confirmed)

`vsf/permutation.py::_get_strata_groups` (lines 64–87) is supposed to implement stratified permutation for Definition 4: permute `Z` only within samples sharing the same `X_S` value, to preserve `P(Z|X_S)`. Strata with fewer than 2 members are pooled into one `fallback` bucket and permuted **together, across strata**:

```python
X_S = [[0,0],[0,1],[1,0],[1,1],[0,0],[0,0]]   # 3 distinct singleton strata: (0,1), (1,0), (1,1)
_get_strata_groups(X_S) -> [array([0,4,5]), array([1,2,3])]
```

Indices 1, 2, 3 belong to three *different* conditioning classes but get shuffled as one group. That treats non-exchangeable observations as exchangeable, biasing the null distribution used for `Δ I(j | S)` — and the bias gets worse exactly where the test matters most: as `S` grows toward 4–7D, `X_S` cardinality grows and more strata become singletons, so more of the sample gets swept into this invalid pooled shuffle. This directly affects whether Phase 2 greedy selection stops at the correct dimensionality.

### 4. Graph-reasoning "cascade" score is not a real information-theoretic quantity, and generates false positives (confirmed)

`vsf/graph_miner.py` and `vsf/graph_inference.py` both compute a "chain score" as the **product** of pairwise asymmetric Uncertainty Coefficients along a path (`nmi_inp_z1 * nmi_z1_tgt[...]`), label it "Multiplicative Cascade... DPI compliant", and flag chains where this product exceeds the direct coefficient as `"is_superior": true`. `Project_Master_Document.md` contains **zero** occurrences of "cascade", "DPI", or "mediator" — this construct has no basis in the project's own formal spec; it was invented directly in implementation code with an inline comment claiming theoretical grounding it doesn't have.

The Data Processing Inequality bounds mutual information for a true Markov chain (`I(X;Y) ≤ I(X;Z)`); it says nothing about the product of two *differently normalized* Uncertainty Coefficients from unrelated variable pairs, and multiplying two numbers in `[0,1]` trivially can't exceed either factor — that triviality is being sold as "DPI compliance." Reproduced concretely: built `X`, `Y` with a weak direct link, and `Z1` correlated with each of them through **unrelated** side channels (i.e., `Z1` is not a genuine mediator of the `X→Y` relationship):

```
U(Y|X) direct = 0.2262   U(Z1|X) = 0.6600   U(Y|Z1) = 0.0371
mine_strong_links(...) -> path=['Z1','X','Y']  chain_score=0.0630  direct_nmi=0.0371  is_superior=True
```

The tool reports a fabricated "superior mediator chain" for a variable that has no genuine mediating role. Neither module runs a permutation test or any multiple-comparison correction over what is an `O(M^3)` search space of candidate chains (`min_nmi` is a bare, uncorrected threshold) — this is the same missing-FDR problem as Critical #1, compounded by a scoring function that isn't a valid statistic in the first place.

---

## High severity

### 5. Unsynchronized global mutable cache under a threading HTTP server (confirmed by inspection)

`server.py` serves with `http.server.ThreadingHTTPServer` (a new thread per request) but reads/writes module-level globals `_last_params`, `_last_res`, `_top_columns_cache` with no lock (lines 15–17, 267–289, 111–184). Two concurrent `/api/analyze` requests (two browser tabs, or a user double-clicking) race on these globals: thread A can be mid-`engine.fit()` computing a result for key K1 while thread B reads a half-updated `_last_res` written for K2, or both write in an interleaved order that leaves `_last_params` and `_last_res` inconsistent with each other. This is a genuine, reachable bug in a server explicitly built with concurrency enabled, not a theoretical one.

### 6. Server exposed beyond localhost with wildcard CORS (confirmed by inspection)

`main()` binds `("", PORT)` — all network interfaces, not `127.0.0.1` — and every response sets `Access-Control-Allow-Origin: *` (line 67). Despite being documented as a local tool (`http://localhost:8050`), this exposes the mushroom-dataset analysis API to the LAN (and to the internet if the port is reachable), and the wildcard CORS lets any origin loaded in the user's browser issue cross-origin requests and read the responses. Combined with `except Exception as e: self._send_json_response(500, {"error": str(e)})` on every handler, internal exception text is also returned verbatim to any caller.

### 7. Scenario routing produces a mislabeled, arithmetically nonsensical message (confirmed)

`vsf/avr.py`, Phase 3 routing (lines 293–322): the branch structure leaves a reachable gap for `4 <= d_star <= 6` with `vir < vir_threshold` (i.e. greedy stopped before 7D, but with meaningfully incomplete information capture). That state falls into the final `else`, which is labeled `SCENARIO_C` and prints a message hard-coded for the ">7D" case:

```python
scenario = Scenario.SCENARIO_C
xai_msg = f"... Current visualization is incomplete. {n_features - 7} significant features omitted."
```

Reproduced directly: `d_star=5, n_features=5, vir=0.55` → `"SCENARIO C (Warning: >7D): ... -2 significant features omitted."` — a 5D result labeled as an over-7D warning, with a negative omitted-feature count. `n_features - 7` is also the wrong quantity even when non-negative: it should be counted against the significant-feature set `F`, not the raw column count of the original dataset.

### 8. Visualization can plot axes the noise filter explicitly rejected (confirmed by inspection)

`vsf/vis.py::prepare_visualization_payload` (lines 199–201): when `d_star < 2` or `< 3`, the unfilled Y/Z axes fall back to raw column indices 1 and 2 (`selected_idx[1] if len(selected_idx) > 1 else (1 if n_features > 1 else 0)`), not to the next-best or even Phase-1-significant feature. For a Scenario A result with `d* = 1`, the 3D scatter can render axes 2 and columns that `marginal_permutation_test` rejected as statistically indistinguishable from noise — directly contradicting the tool's own "perceptually sufficient, statistically justified" framing for exactly the samples where a user is most likely to visually over-interpret structure that isn't there.

### 9. "Purity" metric is meaningless for any target with more than 2 classes (confirmed by inspection)

`vsf/vis.py::build_grid` (line 304): `pur = mean(class_id) / (K - 1)`. Class ids are integer codes assigned by `np.unique`'s sort order — there is no ordinal relationship between them for a nominal target. This happens to compute "fraction of the positive class" correctly for `K=2` (the shipped mushroom dataset), but for any `K > 2` categorical target it averages arbitrary label indices into a number with no statistical meaning, and that number is displayed to the user as a purity/confidence percentage in every cell's hover text and voxel opacity/shading.

---

## Methodological gaps versus the project's own spec

### 10. Non-standard "Miller-Madow" correction, faithfully implemented from a non-standard spec

`Project_Master_Document.md` (lines 646, 695) defines the local-NMI bias correction as `I_corrected = I_plugin − (k_Z−1)(k_φ−1) / (2 N_c ln2)`, and `vsf/mining.py::nmi_miller_madow_corrected` implements this exactly. The code is not at fault relative to its spec, but the formula itself does not match the standard Miller–Madow bias correction for mutual information in the cited literature (Miller 1955; Panzeri & Treves 1996), which corrects the three entropy terms separately and subtracts a joint-cardinality term: `[(K_x−1) + (K_y−1) − (K_xy−1)] / (2N ln2)`. The spec's version is a product of category-count offsets, not the standard signed sum of entropy biases. Citing "Miller-Madow" for a materially different derivation is exactly the kind of thing a Q1 reviewer checks first.

### 11. `/api/top_columns` reproduces the exact spurious-discovery failure mode the spec warns against — with zero mitigation (confirmed against the real dataset)

`Project_Master_Document.md` §4.5.2 explicitly names the failure mode ("spurious NMI = 1.0 on micro-samples") and mandates a support threshold (`θ_supp = 0.03` or `N_pos ≥ 30`) before any predicate is scored. `server.py::_handle_top_columns_api` implements the same greedy stepwise NMI search but applies **no** support filter at all. Run against the shipped `data/mushrooms.csv` (8124 rows):

```
cap-surface = 'g'              n=4/8124 (0.05%)   NMI = 0.839
stalk-color-above-ring = 'y'   n=8/8124 (0.10%)   NMI = 1.000
veil-color = 'y'                n=8/8124 (0.10%)   NMI = 1.000
stalk-surface-above-ring = 'y' n=24/8124 (0.30%)  NMI = 1.000
```

48 (column, value) pairs get surfaced to the user as "top columns" at NMI ≥ 0.75, several from single-digit sample counts achieving perfect NMI by construction (any sufficiently rare, unique combination of a handful of other columns can trivially "explain" 4–8 rows). `vsf/mining.py::mine_dirty_center` at least has *a* support floor (`max(30, 0.1·N_c)`), though it doesn't match the spec's OR-semantics (`0.03 support OR N≥30` — a lenient union; the code's `max()` is the stricter of the two, an inconsistency in its own right). The `/api/top_columns` endpoint has nothing.

---

## Efficiency (Q1-grade computational cost, not just correctness)

- **Triple-computed entropies.** `normalized_mutual_information` (`vsf/math.py`) calls `mutual_information`, which computes `h_z`, `h_xs`, `h_z_xs`; then `normalized_mutual_information` recomputes `h_z` and `h_xs` again independently. Every call does 5 entropy passes for what needs 3.
- **Invariant recomputed inside the hot loop.** `vsf/avr.py`, Phase 2 (~line 199): `i_base = mutual_information(Z_discrete, X_S_curr)` is recomputed for every candidate `j` in the inner loop, even though `X_S_curr` — and therefore `i_base` — is constant across that whole loop. This is `O(|candidates|)` redundant full-entropy computations per selection step, avoidable by hoisting one line.
- **Symmetric quantity computed twice.** `vsf/graph_miner.py::compute_predictiveness_matrix` computes `mutual_information(c1, c2)` for both `(c1,c2)` and `(c2,c1)` even though MI is symmetric — 2× the necessary `O(M²)` MI calls, on top of not caching per-column entropies that are recomputed at every pair.
- **Forced object dtype.** `X_arr = np.asarray(X, dtype=object)` (`vsf/avr.py`, `vsf/pmd.py::discretize_dataset`) routes even purely numeric input through NumPy's slow object-array path (`np.unique` on `dtype=object` is materially slower than on a typed array) for every dataset processed, regardless of whether it actually contains mixed types.
- **Unbounded synchronous request.** `_handle_top_columns_api` runs a brute-force stepwise search (`columns × values × 4 steps × ~20 candidates`, each step calling full-dataset MI) synchronously in the request thread on first call, with a single all-or-nothing cache and no way to invalidate it if the dataset changes.

---

## Bottom line

The information-theoretic core (`shannon_entropy`, `joint_entropy`, `mutual_information`, basic NMI) is implemented correctly and matches its own definitions. The problems are concentrated exactly where a Q1 reviewer would push hardest: the multiple-testing control the paper claims as a differentiator isn't implemented anywhere (#1), the evaluation protocol used to claim accuracy penalizes correct behavior and reports 0% accuracy at the exact settings where the method works as designed (#2), the stratified permutation test silently violates its own exchangeability assumption (#3), and an entire "graph reasoning" feature is built on a scoring function with no theoretical basis that demonstrably manufactures false "superior" findings (#4). None of this is caught by the 19 passing tests, which don't touch multi-class targets, sparse strata, the routing edge case, or the benchmark's own headline metric.
