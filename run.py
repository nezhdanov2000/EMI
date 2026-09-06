"""
Launcher script for Visual Sufficiency Framework (VSF) interactive server.
"""
import sys
from pathlib import Path
import pandas as pd
import vsf

def main():
    dataset_path = None
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        # Default to available benchmark datasets
        candidates = [
            Path("benchmark_data/car_evaluation.csv"),
            Path("benchmark_data/nursery.csv"),
            Path("benchmark_data/audiology.csv"),
            Path("benchmark_data/soybean_large.csv"),
        ]
        for c in candidates:
            if c.exists():
                dataset_path = c
                break
        if not dataset_path:
            csvs = list(Path("benchmark_data").glob("*.csv"))
            if csvs:
                dataset_path = csvs[0]

    if not dataset_path or not dataset_path.exists():
        print("Error: No dataset found. Please provide a CSV path: python run.py <path_to_csv>")
        sys.exit(1)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape} ({len(df.columns)} columns, {len(df)} rows)")

    port = 8000
    for p in range(port, port + 10):
        try:
            print(f"Starting VSF Interactive Visual Dashboard on http://127.0.0.1:{p}/ ...")
            vsf.serve(df, host="127.0.0.1", port=p, open_browser=True)
            break
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, 'winerror', None) == 10048 or getattr(e, 'errno', None) in (98, 10048):
                print(f"Port {p} in use, trying {p+1}...")
                continue
            raise

if __name__ == "__main__":
    main()
