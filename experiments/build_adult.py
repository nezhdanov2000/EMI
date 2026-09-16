"""
Build `data/adult_census.csv`, the reference dataset of PMD Section 3.2, from
the UCI archive.

The file is the 32 561-row training split (`adult.data`) of UCI Adult /
Census Income (CC BY 4.0), restricted to its eight categorical columns:

    workclass, education, marital_status, occupation, relationship, race,
    sex, income

`native-country` and every numeric column are dropped, which is what gives
M = 7 features for any target in the worked example. Values are stripped of
the leading space UCI puts after each comma; UCI's missing marker `?` is kept
as an ordinary category. Two regression tests in `tests/test_centers.py`
skip until this file exists.

Usage (needs network access to archive.ics.uci.edu):
    python experiments/build_adult.py
    python experiments/build_adult.py --zip path/to/adult.zip   # offline
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_PATH: Path = REPO_ROOT / "data" / "adult_census.csv"
UCI_URL: str = "https://archive.ics.uci.edu/static/public/2/adult.zip"

UCI_COLUMNS: Sequence[str] = (
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
)
KEPT: Sequence[str] = (
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "income",
)
# Published facts about adult.data used as a self-check (PMD Section 3.2).
EXPECTED_ROWS: int = 32561
EXPECTED_COUNTS: Dict[tuple[str, str], int] = {
    ("occupation", "Armed-Forces"): 9,
    ("income", ">50K"): 7841,
    ("relationship", "Husband"): 13193,
}


def parse_adult_data(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != len(UCI_COLUMNS):
            raise ValueError(f"adult.data line {lineno}: {len(fields)} fields, expected {len(UCI_COLUMNS)}")
        rows.append(dict(zip(UCI_COLUMNS, fields)))
    return rows


def check(rows: List[Dict[str, str]]) -> List[str]:
    problems: List[str] = []
    if len(rows) != EXPECTED_ROWS:
        problems.append(f"{len(rows)} rows, expected {EXPECTED_ROWS}")
    for (col, val), n in EXPECTED_COUNTS.items():
        got = sum(1 for r in rows if r[col] == val)
        if got != n:
            problems.append(f"{col} = {val}: {got} rows, expected {n}")
    return problems


def write_csv(rows: List[Dict[str, str]], path: Path) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(KEPT)
    for r in rows:
        w.writerow([r[c] for c in KEPT])
    data = buf.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_zip_bytes(zip_path: Path | None) -> bytes:
    if zip_path is not None:
        return zip_path.read_bytes()
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=None, help="use a local copy of adult.zip")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    with zipfile.ZipFile(io.BytesIO(read_zip_bytes(args.zip))) as zf:
        text = zf.read("adult.data").decode("utf-8")
    rows = parse_adult_data(text)
    problems = check(rows)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return 1
    digest = write_csv(rows, args.out)
    print(f"wrote {args.out} ({len(rows)} rows, {len(KEPT)} columns), sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
