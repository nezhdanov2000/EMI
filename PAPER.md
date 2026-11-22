# Reproducing the paper

`paper/main.tex` is built from `paper/tables/*.tex`, `paper/fig_*.{pdf,dat,tex}`
and the numbers quoted in the text. Every one of them comes from a script in
`experiments/`, run in the order below from the repository root. Setup:
`python experiments/setup_experiments.py` (pinned versions in
`requirements-experiments.txt`; the core package alone needs only
`requirements-lock.txt`). Seed 0 everywhere; data are loaded through
`experiments/datasets.py`, which checks the SHA-256 in `benchmark_data/MANIFEST.csv`.

The interface (`vsf.serve`, the landscape, Redundancy and Rules tabs,
composite targets) is not used by the paper; `README.md` describes it.

## 1. Data

| step | command | output |
|---|---|---|
| UCI / PMLB / external tables | `python experiments/build_titanic.py`, `build_mushroom.py`, `build_adult.py`, `build_cancer_benchmarks.py`, `build_pmlb_benchmarks.py`, `build_external_benchmarks.py` | `benchmark_data/*.csv`, hashes checked against `MANIFEST.csv` |

## 2. The comparison (Sections 5-7, Tables 3-10, Figure 4)

| step | command | output |
|---|---|---|
| all methods, all budgets, one table at a time (per-split cache; rerun until complete) | `python experiments/compare_baselines.py --only <dataset>` for each dataset of `experiments/run_all.py`, and `--synthetic grid|two_rules|xor [--n 20000]` | `experiments/results/cache/`, `compare_<dataset>.csv` |
| wide tables, several processes | `python experiments/parallel_adult.py`, `parallel_connect4.py`, `parallel_split_remaining.py` | same cache |
| extra methods (WRAcc lists, optimal tree; need `pydl8.5`) | `python experiments/compare_baselines.py --only <dataset> --methods wracc,wracc_raw,tree_opt` | `experiments/results/cache_extra/` |
| summary of every configuration | `python experiments/summarize_comparison.py > experiments/results/summary.txt` | `comparison_summary.csv` |
| Tables 3, 4, 5, 6, 8, 9, 10 | `python experiments/paper_tables.py` | `paper/tables/{datasets,counts,coverage_purity_B32,coverage_certified_B32,split_B32,net_B32,tree_opt_B32}.tex` |
| Table 7 (compactness) | `python experiments/compactness.py` | `paper/tables/compactness_B32.tex` |
| Figure 4 | `python experiments/fig_coverage_budget.py` | `paper/fig_coverage_budget.pdf` |
| ablations (Section 7.9) | `python experiments/ablations.py --only <dataset>` for `car_evaluation nursery hmda credit mlc_churn adult mushroom` | `experiments/results/ablations.csv` |
| partial vs full grid, paired | `python experiments/pairwise_partial.py` | `experiments/results/pairwise_partial.csv` |
| independent re-implementation (Section 7.3; data in `data/external/`) | `python experiments/partial_centres_external.py --selection certified` | `experiments/results/partial_external/part.csv` |
| pysubgroup on titanic (Section 2, Related work) | `python experiments/subgroup_demo/cv_pysubgroup.py` (needs `pysubgroup`) | `experiments/results/pysubgroup_titanic.csv` |

## 3. The certificate (Section 4, Tables 1-2, Figure 3)

| step | command | output |
|---|---|---|
| smallest certifiable cell per family | `python experiments/n_min_table.py --dataset <name>` for the bundled tables and `--synthetic M N` | `experiments/results/n_min_table.csv` |
| Table 1, Table 2, Figure 3 data | `python experiments/certificate_price.py --check` | `paper/tables/certificate_price.tex`, `certificate_n80.tex`, `paper/fig_certificate_power.dat` |
| power on the planted design (Section 4, Power) | `python experiments/certificate_power.py --sizes 500 1000 3000` | `experiments/results/certificate_power_n*.csv` |
| power curve over N | `python experiments/certificate_power.py --runs 200 --sizes 500 1000 2000 5000 10000 20000 --out experiments/results/certificate_power_curve.csv` | `certificate_power_curve.csv` |
| false certificates on pure noise | `python experiments/null_certificate.py --runs 1000` | `experiments/results/null_certificate.csv` |
| permutation null vs level (Related work, Section 4) | `python experiments/permutation_null.py` | `experiments/results/permutation_null.csv` |
| greedy packing vs exact (Section 3.3) | `python experiments/packing_exact.py` | `experiments/results/packing_exact.csv` |

## 4. Figures 1-2 and the final check

| step | command | output |
|---|---|---|
| Figure 1 | `python experiments/fig_example.py` | `paper/fig_example.pdf` |
| Figure 2 | hand-drawn TikZ, `paper/fig_mechanism.tex` | - |
| every number in the text against the results | `python experiments/check_numbers.py` | exits non-zero on any drift |
| PDF | `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` | `paper/main.pdf` |
