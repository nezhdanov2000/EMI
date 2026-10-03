"""
Build `benchmark_data/mushroom.csv` from UCI ML Repository (Dataset 73).

Source:
    https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
    (8 124 instances, 22 attributes + target class)

Transformations:
    - Single-letter codes expanded to descriptive full English words per the
      official UCI `agaricus-lepiota.names` specification.
    - Constant column `veil_type` (all 'p' / 'partial') dropped (22 of 23 columns kept).
    - Missing value '?' in `stalk_root` recoded to category 'missing'.
    - Target `class` placed first ('poisonous' / 'edible').
    - Output written with LF line terminators.

Usage:
    python experiments/build_mushroom.py            # download, check, write
    python experiments/build_mushroom.py --manifest # print manifest row
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
import urllib.request
from typing import Dict

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_DIR: Path = REPO_ROOT / "benchmark_data"
OUT_FILE: Path = OUT_DIR / "mushroom.csv"

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
SOURCE_SHA256 = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e"
EXPECTED_SHA256 = "10e8c326be8469a2aa1f9afcfcdf00eda78b402a30b5b3d4b186328a8b34131e"

COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "class": {"p": "poisonous", "e": "edible"},
    "cap_shape": {"x": "convex", "b": "bell", "s": "sunken", "f": "flat", "k": "knobbed", "c": "conical"},
    "cap_surface": {"s": "smooth", "y": "scaly", "f": "fibrous", "g": "grooves"},
    "cap_color": {
        "n": "brown", "y": "yellow", "w": "white", "g": "gray", "e": "red",
        "p": "pink", "b": "buff", "u": "purple", "c": "cinnamon", "r": "green"
    },
    "bruises": {"t": "bruises", "f": "no_bruises"},
    "odor": {
        "p": "pungent", "a": "almond", "l": "anise", "n": "none", "f": "foul",
        "c": "creosote", "y": "fishy", "s": "spicy", "m": "musty"
    },
    "gill_attachment": {"f": "free", "a": "attached"},
    "gill_spacing": {"c": "close", "w": "crowded"},
    "gill_size": {"n": "narrow", "b": "broad"},
    "gill_color": {
        "k": "black", "n": "brown", "g": "gray", "p": "pink", "w": "white",
        "h": "chocolate", "u": "purple", "e": "red", "b": "buff", "r": "green",
        "y": "yellow", "o": "orange"
    },
    "stalk_shape": {"e": "enlarging", "t": "tapering"},
    "stalk_root": {"e": "equal", "c": "club", "b": "bulbous", "r": "rooted", "?": "missing"},
    "stalk_surface_above_ring": {"s": "smooth", "f": "fibrous", "k": "silky", "y": "scaly"},
    "stalk_surface_below_ring": {"s": "smooth", "f": "fibrous", "y": "scaly", "k": "silky"},
    "stalk_color_above_ring": {
        "w": "white", "g": "gray", "p": "pink", "n": "brown", "b": "buff",
        "e": "red", "o": "orange", "c": "cinnamon", "y": "yellow"
    },
    "stalk_color_below_ring": {
        "w": "white", "p": "pink", "g": "gray", "b": "buff", "n": "brown",
        "e": "red", "y": "yellow", "o": "orange", "c": "cinnamon"
    },
    "veil_color": {"w": "white", "n": "brown", "o": "orange", "y": "yellow"},
    "ring_number": {"o": "one", "t": "two", "n": "none"},
    "ring_type": {"p": "pendant", "e": "evanescent", "l": "large", "f": "flaring", "n": "none"},
    "spore_print_color": {
        "k": "black", "n": "brown", "u": "purple", "h": "chocolate", "w": "white",
        "r": "green", "o": "orange", "y": "yellow", "b": "buff"
    },
    "population": {
        "s": "scattered", "n": "numerous", "a": "abundant", "v": "several",
        "y": "solitary", "c": "clustered"
    },
    "habitat": {
        "u": "urban", "g": "grasses", "m": "meadows", "d": "woods",
        "p": "paths", "w": "waste", "l": "leaves"
    },
}

# Raw column indices corresponding to the 22 retained target columns (col 16 veil_type dropped)
RAW_INDICES = [0] + list(range(1, 16)) + list(range(17, 23))


def fetch_raw() -> bytes:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Source SHA256 mismatch: {digest} != {SOURCE_SHA256}")
    return content


def build() -> pd.DataFrame:
    raw_bytes = fetch_raw()
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str)
    if raw.shape != (8124, 23):
        raise ValueError(f"Unexpected shape: {raw.shape} != (8124, 23)")

    out = pd.DataFrame()
    for raw_idx, (col_name, mapping) in zip(RAW_INDICES, COLUMN_MAPPINGS.items()):
        out[col_name] = raw[raw_idx].map(mapping)
        if out[col_name].isna().any():
            raise ValueError(f"Unmapped value found in column {col_name}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true", help="print manifest row and exit")
    args = ap.parse_args()

    if args.manifest:
        print(
            'mushroom.csv,8124,22,class,10e8c326be8469a2aa1f9afcfcdf00eda78b402a30b5b3d4b186328a8b34131e,'
            'UCI ML Repository: Mushroom,https://archive.ics.uci.edu/dataset/73/mushroom,missing,'
            '"Built by experiments/build_mushroom.py. Single-letter codes expanded to words; '
            'constant column veil_type dropped (22 of 23 columns kept); stalk_root \'\\?\' recoded as the category \'missing\'."'
        )
        return 0

    print(f"Downloading from {SOURCE_URL}...")
    df = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_bytes = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = hashlib.sha256(csv_bytes).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Output SHA256 mismatch: {digest} != {EXPECTED_SHA256}")

    with OUT_FILE.open("wb") as fh:
        fh.write(csv_bytes)

    print(f"Wrote {OUT_FILE} ({df.shape[0]} rows, {df.shape[1]} columns, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
