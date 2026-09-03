"""
Interactive launcher for VSF v2.0.
"""
import sys
import pandas as pd
import vsf


DATASETS = {
    "1": ("data/mushrooms.csv", "Mushroom Dataset (UCI, 8,124 records -> edibility)", None),
    "2": ("data/adult_census.csv", "Adult Census Income (US Census, 32,561 records -> income >50K)", None),
}


def main():
    print("=" * 78)
    print(" VSF v2.0 - Premier Dataset Launcher:")
    print("=" * 78)
    for key, (path, desc, _) in DATASETS.items():
        print(f" [{key}] {desc}")
    print("=" * 78)

    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    if choice not in DATASETS:
        choice = "1"

    filepath, desc, translations = DATASETS[choice]
    print(f"\nLaunching dataset [{choice}]: {desc}...")
    df = pd.read_csv(filepath)

    vsf.serve(df, translations=translations, host="127.0.0.1", port=8000, open_browser=True)


if __name__ == "__main__":
    main()
