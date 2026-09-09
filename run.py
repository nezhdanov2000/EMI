"""
Launcher script for Visual Sufficiency Framework (VSF) interactive server.
Allows selecting any available benchmark dataset or providing a custom CSV file.
"""
import sys
from pathlib import Path
import pandas as pd
import vsf


def format_size(size_bytes: int) -> str:
    """Format byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_available_datasets(benchmark_dir: Path = Path("benchmark_data")) -> list[Path]:
    """Find all CSV datasets in the benchmark directory."""
    if not benchmark_dir.exists():
        return []
    return sorted(benchmark_dir.glob("*.csv"))


def print_help(datasets: list[Path]):
    """Print usage instructions."""
    print("""
Visual Sufficiency Framework (VSF) Launcher

Usage:
  python run.py                         # Interactive dataset selection
  python run.py <dataset_name_or_path>  # Direct launch with specified dataset
  python run.py <number>                # Direct launch by index number
  python run.py --list                  # List available datasets
  python run.py --help                  # Show this help message

Examples:
  python run.py
  python run.py mushroom
  python run.py titanic.csv
  python run.py 2
  python run.py benchmark_data/car_evaluation.csv
  python run.py C:/path/to/my_data.csv
""")


def resolve_dataset_arg(arg: str, datasets: list[Path]) -> Path | None:
    """Resolve a command-line argument to a dataset Path."""
    # 1. Check if it's a direct valid path
    p = Path(arg.strip('"').strip("'"))
    if p.exists() and p.is_file():
        return p

    # 2. Check if it's an index number
    if arg.isdigit():
        idx = int(arg)
        if 1 <= idx <= len(datasets):
            return datasets[idx - 1]

    # 3. Check inside benchmark_data/ directly
    benchmark_candidate = Path("benchmark_data") / arg
    if benchmark_candidate.exists() and benchmark_candidate.is_file():
        return benchmark_candidate
    if not arg.lower().endswith(".csv"):
        benchmark_candidate_csv = Path("benchmark_data") / f"{arg}.csv"
        if benchmark_candidate_csv.exists() and benchmark_candidate_csv.is_file():
            return benchmark_candidate_csv

    # 4. Search matching filenames
    arg_lower = arg.lower()
    exact_matches = [d for d in datasets if d.name.lower() == arg_lower or d.stem.lower() == arg_lower]
    if exact_matches:
        return exact_matches[0]

    partial_matches = [d for d in datasets if arg_lower in d.stem.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def select_dataset_interactively(datasets: list[Path]) -> Path:
    """Show an interactive menu allowing the user to choose a dataset."""
    print("=" * 66)
    print("       Visual Sufficiency Framework (VSF) - Dataset Selector")
    print("=" * 66)

    if datasets:
        print("\nAvailable Benchmark Datasets:")
        for idx, path in enumerate(datasets, start=1):
            try:
                df_peek = pd.read_csv(path)
                info = f"{len(df_peek):,} rows, {len(df_peek.columns)} cols"
            except Exception:
                info = format_size(path.stat().st_size)
            print(f"  [{idx:2d}] {path.name:<24} ({info})")
    else:
        print("\n(No CSV datasets found in benchmark_data/)")

    print("  [ 0] Enter custom CSV file path")
    print("  [ q] Quit")
    print("-" * 66)

    # Pick default (titanic.csv if available, else 1)
    default_idx = None
    if datasets:
        for idx, path in enumerate(datasets, start=1):
            if path.name.lower() == "titanic.csv":
                default_idx = idx
                break
        if default_idx is None:
            default_idx = 1

    prompt = f"Select dataset [1-{len(datasets)}]"
    if default_idx:
        prompt += f" (default: [{default_idx}] {datasets[default_idx - 1].name})"
    prompt += ": "

    while True:
        try:
            choice = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not choice:
            if default_idx and 1 <= default_idx <= len(datasets):
                return datasets[default_idx - 1]
            print("Please enter a selection.")
            continue

        if choice.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)

        if choice == "0":
            try:
                custom = input("Enter path to CSV file: ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)
            custom_path = Path(custom)
            if custom_path.exists() and custom_path.is_file():
                return custom_path
            print(f"Error: File '{custom}' not found. Please try again.\n")
            continue

        resolved = resolve_dataset_arg(choice, datasets)
        if resolved and resolved.exists():
            return resolved

        print(f"Invalid choice '{choice}'. Enter a number (1-{len(datasets)}), dataset name, '0' for custom file, or 'q' to quit.")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    datasets = get_available_datasets()

    dataset_path = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("-h", "--help", "/?"):
            print_help(datasets)
            sys.exit(0)
        if arg in ("-l", "--list"):
            print("\nAvailable Datasets:")
            for idx, path in enumerate(datasets, start=1):
                print(f"  [{idx}] {path.name}")
            sys.exit(0)

        dataset_path = resolve_dataset_arg(arg, datasets)
        if not dataset_path:
            print(f"Could not find dataset matching '{arg}'.")
            print("Falling back to dataset selection menu...\n")
            dataset_path = select_dataset_interactively(datasets)
    else:
        dataset_path = select_dataset_interactively(datasets)

    if not dataset_path or not dataset_path.exists():
        print("Error: No valid dataset selected.")
        sys.exit(1)

    print("\n" + "-" * 66)
    print(f"Loading dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    col_preview = ", ".join(repr(c) for c in list(df.columns)[:8])
    if len(df.columns) > 8:
        col_preview += f", ... (+{len(df.columns) - 8} more)"
    print(f"Columns: [{col_preview}]")
    print("-" * 66)

    port = 8000
    for p in range(port, port + 10):
        try:
            print(f"Starting VSF Interactive Visual Dashboard on http://127.0.0.1:{p}/ ...")
            print("Press Ctrl+C in this terminal to stop the server.\n")
            vsf.serve(df, host="127.0.0.1", port=p, open_browser=True)
            break
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, 'winerror', None) == 10048 or getattr(e, 'errno', None) in (98, 10048):
                print(f"Port {p} in use, trying {p+1}...")
                continue
            raise
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")
            break


if __name__ == "__main__":
    main()

