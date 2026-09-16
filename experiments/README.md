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
| `null_certificate.py` | how often the *reported* winner carries a false Rule C certificate under a global null | done; 100 runs: d=3 36 %, d=4 60 % against nominal 5 % |
| `nested_cv.py` | in-sample vs fixed-schema CV vs nested CV; schema stability across folds | to write (PLAN.md, phase 1.2) |
| `compare_rules.py` | VSF against exhaustive rules of length <= 4 at equal tau, m, K | to rewrite (the earlier `compare.py` was lost) |

Out of scope until PLAN.md phase 3 is finished: d = 5/6 "plates".
