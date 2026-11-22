# Experiments still to run after the 2026-10-19 review

Run from the repository root in the experiments environment
(`python experiments/setup_experiments.py`). Each command resumes from the
per-split cache, so it can be interrupted and restarted. Afterwards:

    python experiments/pairwise_partial.py
    python experiments/summarize_comparison.py > experiments/results/summary.txt
    python experiments/paper_tables.py
    python experiments/compactness.py
    python experiments/check_numbers.py      # must report 0 missing

and then update the sentences that `check_numbers.py` flags.

## 1. Optimal tree (DL8.5) on adult and connect_4  (hours; one process per split is fine)

The fit already has a 600 s time limit per depth and returns DL8.5's best
tree at the limit (`baselines.TREE_OPT_TIME_LIMIT`); the share of splits
that hit it is the `timeouts` column of Table 10. Nothing in the code needs
changing; the two tables were simply never run.

    python experiments/compare_baselines.py --only adult     --methods tree_opt --max-seconds 36000
    python experiments/compare_baselines.py --only connect_4 --methods tree_opt --max-seconds 36000

Then in main.tex: Section 5 (`tree_opt` "Run on 14 of the 17 tables"),
Section 7.6, Limitations, and the "14 tables" wording in the Introduction
and Conclusion.

## 2. WRAcc lists on connect_4  (hours)

    python experiments/compare_baselines.py --only connect_4 --methods wracc,wracc_raw --max-seconds 36000

Then remove the last sentence of Section 7.7 and the clause in Limitations.

## 3. Holdout split on the four certified configurations without it

    python experiments/compare_baselines.py --only car_evaluation --selection split
    python experiments/compare_baselines.py --only resume         --selection split
    python experiments/compare_baselines.py --only breast_cancer  --selection split
    python experiments/compare_baselines.py --only splice         --selection split --max-seconds 36000

Then Limitations ("15 of the 19").

## 4. More repeats on chess and splice (and connect_4 if compute allows)

In `experiments/compare_baselines.py`, change `n_repeats=1` to `n_repeats=3`
for the chess_krkp (and splice / connect_4) configurations. The cache key
includes the configuration, so these tables are recomputed from scratch for
every method:

    python experiments/compare_baselines.py --only chess_krkp --max-seconds 36000
    python experiments/compare_baselines.py --only splice     --max-seconds 36000
    python experiments/parallel_connect4.py        # only if compute allows

Then Section 5 (Folds), Table 3 (repeats column, regenerated), Limitations.

## 5. Fixed (tau, m) grid as a robustness check  (optional, days of compute)

Add, for every real table, `Config(name, positive, tau, min_samples=m)` for
tau in {0.7, 0.9} and m in {1, 20} (purity selection only), run
`compare_baselines.py --only <table>` for each, and report in an appendix
table whether the counts of Table 4 change sign. The rule used in the paper
is stated in Section 5 ("Choice of tau and m"); this check would replace it.

## Already done in this revision (results committed)

* `null_certificate.py --runs 1000`  -> results/null_certificate.csv (per-schema 67 %, family 0/1000)
* `certificate_power.py --runs 200 --sizes 500 ... 20000 --methods family_bonferroni split`
  -> results/certificate_power_curve.csv
* `packing_exact.py` (new) -> results/packing_exact.csv (Section 7.10)
* `permutation_null.py` (new) -> results/permutation_null.csv (Related work, Section 4)
* `subgroup_demo/cv_pysubgroup.py` restored from git history (results/pysubgroup_titanic.csv was already there)
