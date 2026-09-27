"""
Full phase-3 comparison, ablations and the wide tables runner (Windows/cross-platform).
"""
import subprocess
import sys
import time

DATASETS = [
    "titanic", "breast_cancer", "thyroid_recurrence", "breast_cancer_wisconsin",
    "car_evaluation", "nursery", "hmda", "attrition", "credit", "mlc_churn",
    "wa_churn", "resume", "mushroom", "adult", "chess_krkp", "splice", "connect_4"
]

ABLATIONS = [
    "car_evaluation", "nursery", "hmda", "credit", "mlc_churn", "adult", "mushroom"
]

def run(cmd):
    print(f"\n=======================================================")
    print(f">>> Running: {' '.join(cmd)}")
    print(f"=======================================================")
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    print(f">>> Finished in {dt:.1f}s (exit code {res.returncode})")
    if res.returncode != 0:
        print(f"WARNING: command exited with code {res.returncode}")

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    
    if only:
        ds_list = [only]
    else:
        ds_list = DATASETS
        
    for ds in ds_list:
        run([sys.executable, "experiments/compare_baselines.py", "--only", ds, "--max-seconds", "36000"])
        
    if not only or only in ABLATIONS:
        abl_list = [only] if only else ABLATIONS
        for ds in abl_list:
            run([sys.executable, "experiments/ablations.py", "--only", ds])
            
    if not only:
        run([sys.executable, "experiments/pairwise_partial.py"])
        with open("experiments/results/summary.txt", "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "experiments/summarize_comparison.py"], stdout=f)
        run([sys.executable, "experiments/paper_tables.py"])

if __name__ == "__main__":
    main()
