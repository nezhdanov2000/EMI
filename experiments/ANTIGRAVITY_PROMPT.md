# Task: run the remaining experiments for the paper revision (2026-10-19 review)

Repository root: C:\xampp\htdocs\7D (branch `final`, uncommitted changes present — do not revert or reformat anything).
Read `experiments/REVIEW_FOLLOWUPS.md` and `PAPER.md` first. All commands run from the repository root
in the Python environment that already has `pydl8.5`, `rulelist` (patched), `pysubgroup`, `scikit-learn`
and `scipy` installed (the one Wendy's check ran in; verify with
`python -c "import pydl85, rulelist, pysubgroup, sklearn, scipy; print('ok')"` before starting;
if it fails, run `python experiments/setup_experiments.py` and use `.venv-exp`).

Machine: 16 cores. Every `compare_baselines.py` run resumes from `experiments/results/cache*/` per split,
so a run can be killed and restarted without loss. Run independent jobs as separate processes
concurrently (one terminal/process each; keep total processes <= 12 so each has memory). Log every
process to `experiments/logs/<name>.log` (create the folder) and do not stop until every job below has
finished, i.e. each `compare_baselines.py` call exits reporting the configuration complete.

## Jobs (run concurrently where marked)

A. Optimal tree on the two wide tables (longest jobs; start first, in parallel):
   python experiments/compare_baselines.py --only adult     --methods tree_opt --max-seconds 360000
   python experiments/compare_baselines.py --only connect_4 --methods tree_opt --max-seconds 360000
   The DL8.5 fit has a 600 s limit per depth (baselines.TREE_OPT_TIME_LIMIT) and returns the best
   tree at the limit; do not raise or lower it. If `--methods` with a single extra method is refused,
   read the argparse help of compare_baselines.py and use the accepted spelling.

B. WRAcc lists on connect_4 (parallel with A):
   python experiments/compare_baselines.py --only connect_4 --methods wracc,wracc_raw --max-seconds 360000

C. Holdout split on the four certified configurations that lack it (parallel, cheap except splice):
   python experiments/compare_baselines.py --only car_evaluation --selection split --max-seconds 360000
   python experiments/compare_baselines.py --only resume         --selection split --max-seconds 360000
   python experiments/compare_baselines.py --only breast_cancer  --selection split --max-seconds 360000
   python experiments/compare_baselines.py --only splice         --selection split --max-seconds 360000

D. More repeats on chess and splice. In `experiments/compare_baselines.py`, in `CONFIGS`, change
   `n_repeats=1` to `n_repeats=3` on EVERY chess_krkp and splice configuration (purity, certified, split).
   Leave connect_4 at n_repeats=1. The cache key includes the configuration, so both tables are recomputed
   for every method; run after C-splice has finished:
   python experiments/compare_baselines.py --only chess_krkp --max-seconds 360000
   python experiments/compare_baselines.py --only splice     --max-seconds 360000
   Then also top up the extra methods for them:
   python experiments/compare_baselines.py --only chess_krkp --methods wracc,wracc_raw,tree_opt --max-seconds 360000
   python experiments/compare_baselines.py --only splice     --methods wracc,wracc_raw,tree_opt --max-seconds 360000

## After ALL jobs have finished (sequentially, in this order)

   python experiments/pairwise_partial.py
   python experiments/summarize_comparison.py > experiments/results/summary.txt
   python experiments/paper_tables.py
   python experiments/compactness.py
   python experiments/check_numbers.py

`check_numbers.py` will now FAIL on every sentence of `paper/main.tex` whose number changed. For each
failing check, open the pattern in `experiments/check_numbers.py`, find the sentence in `paper/main.tex`,
and update the NUMBERS in that sentence to the recomputed values (print them from the csv files; do not
guess). Do not change wording, claims or structure of the paper; if a recomputed number reverses a
qualitative statement (e.g. a loss becomes a tie), leave the sentence, and list it in the report instead.
Then make these specific edits, which the new runs make true:
  - Section 5, `tree_opt`: replace "Run on 14 of the 17 tables: not on adult and connect_4, ... exceeds the
    time budget." with the actual time-limit share from Table 10 (`timeouts` column) for the two tables.
  - Section 7.6 (Optimal trees), Introduction (contribution 2) and Conclusion: change "on the 14 tables
    where the tree was run" / "14 of 17 tables" to the tree now being run on every table, and remove the
    caveat sentence about the two widest tables not having been run.
  - Section 7.7 (WRAcc lists): delete the final sentence "connect_4 is not in these counts ...".
  - Section 5 (Folds) and Limitations: chess and splice now have 5x3 folds; connect_4 keeps 5x1;
    update the "four degrees of freedom" sentences accordingly; Table 3 is regenerated automatically.
  - Limitations: delete the clauses about the optimal tree, the WRAcc lists and "15 of the 19" split.
Re-run `python experiments/check_numbers.py` until it reports 0 missing, then rebuild the PDF:
   cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
Finally run `python -m pytest -q` (must pass) and write a report `experiments/logs/REPORT.md`:
per job, wall-clock time, number of splits computed, any fit that hit the DL8.5 time limit, every
sentence of main.tex you changed (old number -> new number), and every qualitative statement that the
new numbers contradict. Do not commit.
