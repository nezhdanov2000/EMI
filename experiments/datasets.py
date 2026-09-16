"""
Benchmark dataset registry and integrity check.

`benchmark_data/MANIFEST.csv` is the single record of what every bundled
dataset is: where it came from, what was done to it, its shape, its target
column and the SHA-256 of the exact bytes in the repository. Every experiment
loads data through `load_dataset`, which refuses a file whose bytes differ
from the manifest, so a number produced by `experiments/` is always tied to a
known file.

Usage:
    python experiments/datasets.py            # verify every entry, exit 1 on mismatch
    python experiments/datasets.py --rehash   # print current hashes/shapes (does not write)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "benchmark_data"
MANIFEST: Path = DATA_DIR / "MANIFEST.csv"

_FIELDS: Tuple[str, ...] = (
    "file", "rows", "columns", "target", "sha256", "source",
    "source_url", "missing_marker", "preprocessing",
)


@dataclass(frozen=True)
class DatasetEntry:
    file: str
    rows: int
    columns: int
    target: str
    sha256: str
    source: str
    source_url: str
    missing_marker: str
    preprocessing: str

    @property
    def path(self) -> Path:
        return DATA_DIR / self.file

    @property
    def name(self) -> str:
        return Path(self.file).stem


class DatasetIntegrityError(RuntimeError):
    """The bytes or the shape of a dataset differ from the manifest."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_frame(path: Path) -> pd.DataFrame:
    """Every value is read as a string and nothing is converted to NaN: the
    framework treats every distinct value, including UCI's '?', as a category."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def load_manifest(path: Path = MANIFEST) -> Dict[str, DatasetEntry]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != _FIELDS:
            raise DatasetIntegrityError(
                f"{path}: header {reader.fieldnames} != expected {list(_FIELDS)}"
            )
        entries: Dict[str, DatasetEntry] = {}
        for row in reader:
            e = DatasetEntry(
                file=row["file"], rows=int(row["rows"]), columns=int(row["columns"]),
                target=row["target"], sha256=row["sha256"], source=row["source"],
                source_url=row["source_url"], missing_marker=row["missing_marker"],
                preprocessing=row["preprocessing"],
            )
            if e.name in entries:
                raise DatasetIntegrityError(f"{path}: duplicate entry {e.name!r}")
            entries[e.name] = e
    return entries


def verify_entry(entry: DatasetEntry) -> List[str]:
    """Return a list of problems (empty when the file matches the manifest)."""
    if not entry.path.is_file():
        return [f"{entry.file}: file missing"]
    problems: List[str] = []
    digest = sha256_of(entry.path)
    if digest != entry.sha256:
        problems.append(f"{entry.file}: sha256 {digest} != manifest {entry.sha256}")
    df = read_frame(entry.path)
    if df.shape != (entry.rows, entry.columns):
        problems.append(f"{entry.file}: shape {df.shape} != manifest {(entry.rows, entry.columns)}")
    if entry.target not in df.columns:
        problems.append(f"{entry.file}: target column {entry.target!r} absent")
    return problems


def load_dataset(name: str) -> Tuple[pd.DataFrame, DatasetEntry]:
    """Load a manifest dataset by stem, refusing it if it fails verification."""
    entries = load_manifest()
    if name not in entries:
        raise KeyError(f"{name!r} is not in {MANIFEST.name}; known: {sorted(entries)}")
    entry = entries[name]
    problems = verify_entry(entry)
    if problems:
        raise DatasetIntegrityError("; ".join(problems))
    return read_frame(entry.path), entry


def unlisted_files(entries: Dict[str, DatasetEntry]) -> List[str]:
    listed = {e.file for e in entries.values()}
    return sorted(p.name for p in DATA_DIR.glob("*.csv") if p.name not in listed and p != MANIFEST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehash", action="store_true",
                    help="print the current sha256 and shape of every CSV instead of verifying")
    args = ap.parse_args()

    if args.rehash:
        for p in sorted(q for q in DATA_DIR.glob("*.csv") if q != MANIFEST):
            df = read_frame(p)
            print(f"{p.name},{df.shape[0]},{df.shape[1]},{sha256_of(p)}")
        return 0

    entries = load_manifest()
    problems: List[str] = []
    for entry in entries.values():
        problems.extend(verify_entry(entry))
    problems.extend(f"{f}: present in benchmark_data/ but not in the manifest" for f in unlisted_files(entries))
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(entries)} datasets checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
