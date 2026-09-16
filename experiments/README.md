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
| `null_certificate.py` | global null: how often the reported winner carries a false certificate - per-schema (current product), family-wide Bonferroni, split | done; 100 runs: per-schema any branch 68 %, corrected 0 % (nominal 5 %) |
| `certificate_power.py` | strong control and power with a cell exactly at tau beside a true centre | done; one `--sizes` value per call (500, 1000, 3000); power of the default family certificate at N=500 is 1 % |
| `nested_cv.py` | in-sample vs fixed-schema CV vs nested CV (training-only and all-rows encoding), held-out purity, schema stability | done; one `--only <dataset>` per call |
| `cv_gap_decomposition.py` | why nested CV exceeds fixed-schema CV: held-out positives lost in cells that are centres on all rows but not on the training fold | done; one configuration per call |
| `n_min_table.py` | smallest certifiable cell as a function of the family size (bundled data and synthetic M, N) | done; one `--dataset` or `--synthetic M N` per call, appends to results/n_min_table.csv |
| `baselines.py` | library: budget-matched selection for VSF (best schema, top cells), greedy VSF chain, rules (cells of uncoarsened partitions of d <= 4, budgeted max coverage), CART leaves | done; tested in `tests/test_baselines.py` |
| `compare_baselines.py` | held-out coverage, purity, net coverage, description stability and selection time at condition budgets B = 1..32, 5x5 repeated stratified CV, paired differences to VSF (Nadeau-Bengio SE); selection by observed purity or by the family certificate | done; one `--only <dataset>` or `--synthetic <name> [--n N]` per call, repeat until done (per-split cache in results/cache) |
| `summarize_comparison.py` | pivot of all compare_*.csv, +/=/- counts per selection rule (uninformative configurations left out) | done |
| `colouring_strictness.py` | one cell under the default colouring: p-value, largest family that would certify it, lower bounds (single / per schema / family / Tarone), cells certified by Bonferroni, Tarone, BH, BY | done |

The PMD tables in Section 4.14 are these scripts' output at the default
arguments (seed 0). Every call stays under three minutes on a laptop.

Out of scope until PLAN.md phase 3 is finished: d = 5/6 "plates".
