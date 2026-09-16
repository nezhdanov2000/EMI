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

## Known gaps

- **titanic.csv** — engineered from Kaggle `train.csv`, but the bin edges
  (age, fare) and the title mapping were never recorded and the script is
  lost. Rebuild it as a script before any titanic number is published.
- **audiology.csv, mushroom.csv** — the conversion scripts are lost; the
  manifest describes what was done as far as it can be read off the files.
- **Missing values are present** in audiology and soybean_large (`?`, the UCI
  marker) and in mushroom (`missing`) and titanic (`unknown_*`). They are
  ordinary categories to the framework; the dataset screen flags them as
  placeholder levels.
- **soybean_large.csv** had a broken header until 2026-09-16 (see manifest).
  Any earlier result that names a soybean column refers to the wrong column.
