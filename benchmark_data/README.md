# Benchmark data

Every CSV here is listed in `MANIFEST.csv` with its source, shape, target
column, missing-value marker, preprocessing and the SHA-256 of the exact
bytes. `python experiments/datasets.py` verifies all of them;
`tests/test_benchmark_manifest.py` does the same in the test suite, and
`experiments/` loads data only through `experiments.datasets.load_dataset`,
which refuses a file that does not match.

Changing a file means changing its manifest row in the same commit
(`python experiments/datasets.py --rehash` prints the new values) and
saying why in the `preprocessing` field.

## Contents (2026-09-17)

| type | file | instances (rows) | features (cols) | classes | source |
|---|---|---|---|---|---|
| low-dimensional | breast_cancer.csv | 286 | 9 | 2 | UCI 14 |
| low-dimensional | thyroid_recurrence.csv | 383 | 16 | 2 | UCI 915 |
| low-dimensional | breast_cancer_wisconsin.csv | 699 | 9 | 2 | UCI 15 |
| high-dimensional | lung_discrete.csv | 73 | 325 | 7 | scikit-feature |
| high-dimensional | colon.csv | 62 | 2000 | 2 | scikit-feature |
| high-dimensional | leukemia.csv | 72 | 7070 | 2 | scikit-feature |
| (demo) | titanic.csv | 891 | 8 | 2 | Kaggle |

The six cancer files are rebuilt from pinned sources by
`python experiments/build_cancer_benchmarks.py` (conversions are listed in
its docstring and in the manifest). The previous UCI files (audiology,
car_evaluation, chess_krk, chess_krkp, mushroom, nursery, soybean_large,
splice_junction) were removed on 2026-09-17; they remain in the git history.

## Known gaps

- **titanic.csv** — engineered from Kaggle `train.csv`, but the bin edges
  (age, fare) and the title mapping were never recorded and the script is
  lost. Rebuild it as a script before any titanic number is published.
- **UCI copies** — archive.ics.uci.edu is not reachable from the build
  machines, so the three UCI tables come from public GitHub copies pinned to
  a commit and checked against UCI's published row, class and missing-value
  counts.
- **Inferred labels** — colon (tumor/normal) and leukemia (ALL/AML) label
  names are inferred from the class sizes of the original studies; the
  lung_discrete classes have no published names (c1 ... c7).
- **Missing values** are present in breast_cancer and breast_cancer_wisconsin
  (`?`) and in titanic (`unknown_*`). They are ordinary categories to the
  framework; the dataset screen flags them as placeholder levels.
- **High-dimensional files** have 325 to 7070 columns and fewer than 75 rows.
  Exhaustive search over schemas of d <= 4 is infeasible there without a
  feature pre-filter.
