#!/bin/sh
# Full phase-3 comparison, ablations and the two wide tables.
# Every call resumes from experiments/results/cache; datasets are independent,
# so several copies of this script (one --only each) can run in parallel.
# Usage: sh experiments/run_all.sh            # everything, sequentially
#        sh experiments/run_all.sh nursery    # one dataset
set -e
cd "$(dirname "$0")/.."
DS="${1:-titanic breast_cancer thyroid_recurrence breast_cancer_wisconsin car_evaluation nursery hmda attrition credit mlc_churn wa_churn resume mushroom adult chess_krkp splice connect_4}"
for ds in $DS; do
  python3 experiments/compare_baselines.py --only "$ds" --max-seconds 36000
done
ABL="${1:-car_evaluation nursery hmda credit mlc_churn adult mushroom}"
for ds in $ABL; do
  case " car_evaluation nursery hmda credit mlc_churn adult mushroom titanic " in *" $ds "*) python3 experiments/ablations.py --only "$ds";; esac
done
python3 experiments/pairwise_partial.py
python3 experiments/summarize_comparison.py > experiments/results/summary.txt
python3 experiments/paper_tables.py
