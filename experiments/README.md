# experiments/

Every number quoted in `Project_Master_Document.md` or in a paper must be
produced by a script in this folder, with a fixed seed, from data loaded
through `datasets.load_dataset` (which checks the manifest hash). Results go
to `experiments/results/` (ignored by git; regenerate them).

Run from the repository root with the versions in `requirements-lock.txt`.

| script | what it measures | status |
|---|---|---|
| `datasets.py` | verifies `benchmark_data/` against `MANIFEST.csv` | done |
| `build_adult.py` | builds `data/adult_census.csv` (PMD Section 3.2) from UCI | done; needs network |
| `build_external_benchmarks.py` | builds seven real categorical tables (adult, hmda, attrition, credit, resume, wa_churn, mlc_churn; N 1 470 - 48 842, M 11 - 24) in `benchmark_data/` from pinned public copies, checking SHA-256 and row counts; quantile bins recorded per column | done; needs raw.githubusercontent.com |
| `build_pmlb_benchmarks.py` | builds `connect_4.csv` (67 557 x 42) and `splice.csv` (3 188 x 60) from the raw PMLB TSVs pinned in `benchmark_data/raw/` (PMLB is behind Git LFS) | done |
| `build_cancer_benchmarks.py` | builds the six cancer benchmark files (3 low-, 3 high-dimensional) in `benchmark_data/` from pinned GitHub sources, with checks against the published facts | done; needs access to raw.githubusercontent.com |
| `null_certificate.py` | global null: how often the reported winner carries a false certificate - per-schema (current product), family-wide Bonferroni, split | done; 100 runs: per-schema any branch 68 %, corrected 0 % (nominal 5 %) |
| `certificate_power.py` | strong control and power with a cell exactly at tau beside a true centre | done; one `--sizes` value per call (500, 1000, 3000); power of the default family certificate at N=500 is 1 % |
| `nested_cv.py` | in-sample vs fixed-schema CV vs nested CV (training-only and all-rows encoding), held-out purity, schema stability | done; one `--only <dataset>` per call |
| `cv_gap_decomposition.py` | why nested CV exceeds fixed-schema CV: held-out positives lost in cells that are centres on all rows but not on the training fold | done; one configuration per call |
| `n_min_table.py` | smallest certifiable cell as a function of the family size (bundled data and synthetic M, N) | done; one `--dataset` or `--synthetic M N` per call, appends to results/n_min_table.csv |
| `baselines.py` | library: budget-matched selection for VSF (best schema, top cells), VSF with partial centres (disjoint packing of sub-grid cells, `vsf_partial`), greedy VSF chain, rules (cells of uncoarsened partitions of d <= 4, budgeted max coverage; `rules_disjoint` = no shared training row), CART leaves, SSD++ rule lists (`rulelist`, ordered rules: disjoint by construction) | done; tested in `tests/test_baselines.py` |
| `compare_baselines.py` | held-out coverage, purity, net coverage, description stability and selection time at condition budgets B = 1..32, 5x5 repeated stratified CV, paired differences to VSF (Nadeau-Bengio SE); selection by observed purity or by the family certificate | done; one `--only <dataset>` or `--synthetic <name> [--n N]` per call, repeat until done (per-split cache in results/cache) |
| `partial_centres_external.py` | independent check of partial-axis centres (PMD 4.16) on ten datasets outside `benchmark_data/`: full grid vs partial centres (disjoint / union / overlap) vs free rules (same three), observed-purity and family-certified selection, 5x3 folds, budgets 2-32; no dependency on `vsf/` | done; needs the files listed in its docstring; ~15 min |
| `summarize_partial_external.py` | paired Nadeau-Bengio marks vs the full grid and vs disjoint rules, share of splits with held-out union purity >= tau, share of the grid-to-rules gap closed | done |
| `patch_rulelist.py` | makes the `rulelist` package (SSD++) run under numpy 2 / pandas 2 (install with `--no-deps` plus gmpy2, numba); idempotent | done |
| `ablations.py` | capacity divisor {5, 10, 20, none} x size floor {1, 5, 20}: held-out coverage and purity of the full grid, partial centres and disjoint rules at B = 8, 32 | done; one `--only <dataset>` per call |
| `summarize_comparison.py` | pivot of all compare_*.csv, +/=/- counts per selection rule (uninformative configurations left out) | done |
| `colouring_strictness.py` | one cell under the default colouring: p-value, largest family that would certify it, lower bounds (single / per schema / family / Tarone), cells certified by Bonferroni, Tarone, BH, BY | done |
| `noise_highdim.py` | the product (`discover_branches`, d <= 2) on pure noise shaped like colon (62 rows, 40 positives): observed-share rule vs default certificate, and all cells that pass the observed rule | done; M = 500 about 1 min |

The PMD tables in Section 4.14 are these scripts' output at the default
arguments (seed 0). Every call stays under three minutes on a laptop.

Out of scope until PLAN.md phase 3 is finished: d = 5/6 "plates".
