"""
Build `benchmark_data/titanic.csv` from Kaggle Titanic (891 passengers).

Source:
    Kaggle Titanic train.csv (891 rows, 12 columns), via pinned public copy:
    https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Engineered features:
    - survived: 0 -> 'died', 1 -> 'survived' (target, placed first)
    - passenger_class: 1 -> '1st_Class', 2 -> '2nd_Class', 3 -> '3rd_Class'
    - sex: 'male', 'female' as given
    - title: extracted from Name (r',\\s*([^\\.]+)\\.'):
        * Common titles: Mr, Mrs, Miss (including Mlle, Ms), Master
        * Officer: Capt, Col, Dr, Major, Rev
        * Royalty: Don, Jonkheer, Lady, Sir, the Countess
        * Other: Mme
    - age_group:
        * child (< 13)
        * teenager (13 <= age < 18)
        * adult (18 <= age < 60)
        * senior (>= 60)
        * unknown_age (missing)
    - family_type: derived from SibSp + Parch:
        * alone (0)
        * small_family (1 or 2)
        * large_family (>= 3)
    - deck: first character of Cabin ('deck_A' .. 'deck_G'; 'T' -> 'other_deck'; missing -> 'unknown_deck')
    - fare_level:
        * low_fare (<= 10.0)
        * medium_fare (10.0 < fare <= 30.0)
        * high_fare (> 30.0)
    - port_embarked: 'S' -> 'Southampton', 'C' -> 'Cherbourg', 'Q' -> 'Queenstown'; missing -> 'unknown_port'
    - Raw columns (PassengerId, Name, Ticket, Cabin, Age, Fare, SibSp, Parch, Pclass, Embarked) dropped.

Usage:
    python experiments/build_titanic.py            # download, check, write
    python experiments/build_titanic.py --manifest # print manifest row
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
import urllib.request

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_DIR: Path = REPO_ROOT / "benchmark_data"
OUT_FILE: Path = OUT_DIR / "titanic.csv"

SOURCE_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
SOURCE_SHA256 = "4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7"
EXPECTED_SHA256 = "86398f9d18b9d81932d90198007ce51e1b066bf1797d4a8ca214afe7cacb2b91"

TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Master": "Master",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Capt": "Officer",
    "Col": "Officer",
    "Dr": "Officer",
    "Major": "Officer",
    "Rev": "Officer",
    "Don": "Royalty",
    "Jonkheer": "Royalty",
    "Lady": "Royalty",
    "Sir": "Royalty",
    "the Countess": "Royalty",
    "Mme": "Other",
}

PORT_MAP = {
    "S": "Southampton",
    "C": "Cherbourg",
    "Q": "Queenstown",
}


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
    raw = pd.read_csv(io.BytesIO(raw_bytes))
    if raw.shape != (891, 12):
        raise ValueError(f"Unexpected shape: {raw.shape} != (891, 12)")

    out = pd.DataFrame()
    out["survived"] = raw["Survived"].map({0: "died", 1: "survived"})
    out["passenger_class"] = raw["Pclass"].map({1: "1st_Class", 2: "2nd_Class", 3: "3rd_Class"})
    out["sex"] = raw["Sex"]

    titles = raw["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    out["title"] = titles.map(TITLE_MAP)
    if out["title"].isna().any():
        unmapped = titles[out["title"].isna()].unique().tolist()
        raise ValueError(f"Unmapped titles: {unmapped}")

    def map_age(a: float) -> str:
        if pd.isna(a):
            return "unknown_age"
        if a < 13:
            return "child"
        if a < 18:
            return "teenager"
        if a < 60:
            return "adult"
        return "senior"

    out["age_group"] = raw["Age"].apply(map_age)

    fam_size = raw["SibSp"] + raw["Parch"]

    def map_family(f: int) -> str:
        if f == 0:
            return "alone"
        if f <= 2:
            return "small_family"
        return "large_family"

    out["family_type"] = fam_size.apply(map_family)

    def map_deck(c: object) -> str:
        if pd.isna(c) or not str(c).strip():
            return "unknown_deck"
        d = str(c).strip()[0]
        if d in "ABCDEFG":
            return f"deck_{d}"
        return "other_deck"

    out["deck"] = raw["Cabin"].apply(map_deck)

    def map_fare(f: float) -> str:
        if f <= 10.0:
            return "low_fare"
        if f <= 30.0:
            return "medium_fare"
        return "high_fare"

    out["fare_level"] = raw["Fare"].apply(map_fare)
    out["port_embarked"] = raw["Embarked"].map(PORT_MAP).fillna("unknown_port")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true", help="print manifest row and exit")
    args = ap.parse_args()

    if args.manifest:
        print(
            'titanic.csv,891,9,survived,86398f9d18b9d81932d90198007ce51e1b066bf1797d4a8ca214afe7cacb2b91,'
            'Kaggle Titanic train.csv (891 passengers),https://www.kaggle.com/c/titanic/data,'
            'unknown_age;unknown_deck;unknown_port,'
            '"Built by experiments/build_titanic.py. Engineered features from Kaggle train.csv: '
            'title from Name (Officer/Royalty/Other groupings), age_group, family_type from SibSp+Parch, '
            'deck from Cabin, fare_level from Fare; raw columns dropped."'
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
