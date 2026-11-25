"""
Build the two wide PMLB tables in `benchmark_data/` from the raw TSVs kept in
`benchmark_data/raw/` (branch data-benchmarks; PMLB serves its files through
Git LFS, which the project's machines cannot reach, so the raw copies are
committed and pinned by SHA-256 here).

    file             rows   feat.  target (values)                          PMLB name
    connect_4.csv   67557     42   outcome: win 44473 / loss 16635 / draw 6449   connect_4
    splice.csv       3188     60   site: neither 1655 / site_a 769 / site_b 764  splice

connect_4: the 42 board cells keep PMLB's codes 0/1/2 (as strings); the
outcome names follow the UCI class sizes exactly (win 44 473, loss 16 635,
draw 6 449). splice: 60 nucleotide positions with PMLB's integer codes (0-5;
A, C, G, T and the IUPAC ambiguity codes); PMLB's copy has 3 188 of UCI's
3 190 rows, so which of its classes 0 / 1 is EI and which IE cannot be
recovered from the sizes (UCI: 767 / 768) and they are written site_a
(code 0) and site_b (code 1); code 2 (1 655 rows) is UCI's N, written neither.

Two large recorded tables were added on 2026-11-25 (inclusion rule, fixed
before any protocol run: every PMLB classification table with N >= 50 000
that records observations, is not generated or simulated, and is not already
in the study; `sleep` meets the rule and is left out until the origin of the
PMLB copy and the meaning of its class codes are documented):

    file           rows   feat.  target (values)                                  PMLB name
    fars.csv     100968     16   injury: fatal 42116 / none 20007 / ...            fars
    shuttle.csv   58000      9   class: rad_flow 45586 / high 8903 / bypass 3267   shuttle

fars: KEEL's copy of the 2001 U.S. Fatality Analysis Reporting System person
file (https://sci2s.ugr.es/keel/dataset.php?cod=191). FARS is a census of
crashes with at least one death, so every share is conditional on that.
PMLB codes the eight injury classes 0-7; they are named in the alphabetical
order of KEEL's class names (the order a label encoder gives), checked
against the class sizes. Code 1 is the fatal class on independent evidence:
RACE, which FARS takes from the death certificate, has one code on 58 686
rows (not applicable) and another code on 42 282 rows, and those include
every one of the 42 116 rows of class 1 and only 166 others. Thirteen columns that are filled in
after the outcome are dropped (FARS_POST_OUTCOME): race and Hispanic origin
(death certificate), the alcohol and drug test fields (tests are ordered
mostly for the dead) and whether the person was taken to hospital. With them
a single column covers the fatal class. AGE is cut into 5 quantile groups
on all rows (99 may be FARS's code for an unknown age; it is not separated);
every other column keeps PMLB's integer code as a category.

shuttle: Statlog (Shuttle), 58 000 rows (43 500 + 14 500), nine numeric
sensor attributes (the first is time), seven classes named as in the Statlog
documentation. Each attribute is cut into 5 quantile groups on all rows.

    python experiments/build_pmlb_benchmarks.py --manifest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "benchmark_data" / "raw"
OUT = REPO_ROOT / "benchmark_data"

RAW_SHA256: Dict[str, str] = {
    "connect_4.tsv": "8209a4bca4de6c0073d781d31751dc07b57a38d5c75196ef5b69f2c87374c1a9",
    "splice.tsv": "f57132a4b1437271b173b1073000e660f0a2736106037f5b8e67da85285a4a35",
    "fars.tsv": "6142db15b50dd2af73f125affe378c076b1fa2843e3572afde31decb8e30a20a",
    "shuttle.tsv": "0ee9339cbf69ca929411b0dfd2b3cb53f478035d1eef554347f2c9d448921eb6",
}

#: fars columns recorded after the outcome (see the module docstring).
FARS_POST_OUTCOME: Tuple[str, ...] = (
    "RACE", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL",
    "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE", "ALCOHOL_TEST_RESULT",
    "METHOD_OF_DRUG_DETERMINATION",
    "DRUG_TEST_TYPE", "DRUG_TEST_RESULTS_(1_of_3)",
    "DRUG_TEST_TYPE_(2_of_3)", "DRUG_TEST_RESULTS_(2_of_3)",
    "DRUG_TEST_TYPE_(3_of_3)", "DRUG_TEST_RESULTS_(3_of_3)",
)
#: PMLB code -> injury class, in the alphabetical order of KEEL's class names, with the class sizes.
FARS_CLASSES: Dict[int, Tuple[str, int]] = {
    0: ("died_prior", 9), 1: ("fatal", 42116), 2: ("incapacitating", 15072), 3: ("severity_unknown", 299),
    4: ("none", 20007), 5: ("nonincapacitating", 13890), 6: ("possible", 8674), 7: ("unknown", 901),
}
#: fars columns kept, renamed (PMLB name -> name in fars.csv).
FARS_COLUMNS: Dict[str, str] = {
    "CASE_STATE": "state", "AGE": "age", "SEX": "sex", "PERSON_TYPE": "person_type",
    "SEATING_POSITION": "seating_position", "RESTRAINT_SYSTEM-USE": "restraint",
    "AIR_BAG_AVAILABILITY/DEPLOYMENT": "air_bag", "EJECTION": "ejection", "EJECTION_PATH": "ejection_path",
    "EXTRICATION": "extrication", "NON_MOTORIST_LOCATION": "non_motorist_location",
    "POLICE_REPORTED_ALCOHOL_INVOLVEMENT": "police_alcohol", "POLICE-REPORTED_DRUG_INVOLVEMENT": "police_drug",
    "RELATED_FACTOR_(1)-PERSON_LEVEL": "related_factor_1", "RELATED_FACTOR_(2)-PERSON_LEVEL": "related_factor_2",
    "RELATED_FACTOR_(3)-PERSON_LEVEL": "related_factor_3",
}
#: Statlog (Shuttle) class codes and sizes.
SHUTTLE_CLASSES: Dict[int, Tuple[str, int]] = {
    1: ("rad_flow", 45586), 2: ("fpv_close", 50), 3: ("fpv_open", 171), 4: ("high", 8903),
    5: ("bypass", 3267), 6: ("bpv_close", 10), 7: ("bpv_open", 13),
}
QUANTILE_GROUPS: int = 5


class BuildError(RuntimeError):
    pass


def read_raw(name: str) -> pd.DataFrame:
    path = RAW / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RAW_SHA256[name]:
        raise BuildError(f"{name}: sha256 {digest} != pinned {RAW_SHA256[name]}")
    return pd.read_csv(path, sep="\t")


def build_connect4() -> pd.DataFrame:
    t = read_raw("connect_4.tsv")
    if t.shape != (67557, 43):
        raise BuildError(f"connect_4: shape {t.shape}")
    names = {2: "win", 1: "loss", 0: "draw"}
    out = pd.DataFrame({"outcome": t["target"].map(names)})
    counts = out["outcome"].value_counts().to_dict()
    if counts != {"win": 44473, "loss": 16635, "draw": 6449}:
        raise BuildError(f"connect_4 class sizes {counts}")
    for c in t.columns:
        if c != "target":
            out[c] = t[c].astype(int).astype(str)
    return out


def build_splice() -> pd.DataFrame:
    t = read_raw("splice.tsv")
    if t.shape != (3188, 61):
        raise BuildError(f"splice: shape {t.shape}")
    names = {2: "neither", 0: "site_a", 1: "site_b"}
    out = pd.DataFrame({"site": t["target"].map(names)})
    counts = out["site"].value_counts().to_dict()
    if counts != {"neither": 1655, "site_b": 769, "site_a": 764}:
        raise BuildError(f"splice class sizes {counts}")
    for i, c in enumerate(c for c in t.columns if c != "target"):
        out[f"p{i - 30:+d}".replace("+", "p").replace("-", "m")] = t[c].astype(int).astype(str)
    return out


def quantile_groups(values: pd.Series, name: str) -> pd.Series:
    """QUANTILE_GROUPS quantile groups name_q0 < name_q1 < ... on all rows (duplicate edges collapsed); target-free."""
    codes = pd.qcut(values.astype(float), QUANTILE_GROUPS, duplicates="drop", labels=False)
    if codes.isna().any():
        raise BuildError(f"{name}: {int(codes.isna().sum())} values fell outside the quantile groups")
    return codes.astype(int).map(lambda c: f"{name}_q{c}")


def named_classes(codes: pd.Series, classes: Dict[int, Tuple[str, int]], what: str) -> pd.Series:
    counts = {int(k): int(v) for k, v in codes.value_counts().items()}
    expected = {code: size for code, (_, size) in classes.items()}
    if counts != expected:
        raise BuildError(f"{what}: class sizes {dict(sorted(counts.items()))} != {expected}")
    return codes.astype(int).map(lambda c: classes[c][0])


def build_fars() -> pd.DataFrame:
    t = read_raw("fars.tsv")
    if t.shape != (100968, 30):
        raise BuildError(f"fars: shape {t.shape}")
    missing = [c for c in (*FARS_POST_OUTCOME, *FARS_COLUMNS) if c not in t.columns]
    if missing or len(FARS_POST_OUTCOME) + len(FARS_COLUMNS) + 1 != t.shape[1]:
        raise BuildError(f"fars: columns do not match the kept and dropped lists (missing {missing})")
    # the evidence that code 1 is the fatal class: race comes from the death certificate
    race_recorded = t["RACE"] != t["RACE"].mode().iloc[0]
    if (int(race_recorded.sum()), int((race_recorded & (t["target"] == 1)).sum())) != (42282, 42116):
        raise BuildError("fars: the RACE check of the fatal class does not reproduce")
    out = pd.DataFrame({"injury": named_classes(t["target"], FARS_CLASSES, "fars")})
    for raw, name in FARS_COLUMNS.items():
        out[name] = quantile_groups(t[raw], name) if raw == "AGE" else t[raw].astype(int).astype(str)
    return out


def build_shuttle() -> pd.DataFrame:
    t = read_raw("shuttle.tsv")
    if t.shape != (58000, 10):
        raise BuildError(f"shuttle: shape {t.shape}")
    out = pd.DataFrame({"class": named_classes(t["target"], SHUTTLE_CLASSES, "shuttle")})
    for i in range(1, 10):
        out[f"a{i}"] = quantile_groups(t[f"A{i}"], f"a{i}")
    return out


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="store_true")
    args = ap.parse_args()
    targets = [
        ("connect_4.csv", "outcome", build_connect4, "UCI ML Repository: Connect-4, via PMLB connect_4 (raw TSV pinned in benchmark_data/raw)",
         "https://archive.ics.uci.edu/dataset/26/connect+4",
         "Built by experiments/build_pmlb_benchmarks.py. Board cells keep PMLB codes 0/1/2; outcome named by the UCI class sizes (2 -> win, 1 -> loss, 0 -> draw)."),
        ("splice.csv", "site", build_splice, "UCI ML Repository: Molecular Biology (Splice-junction), via PMLB splice (3 188 of 3 190 rows; raw TSV pinned in benchmark_data/raw)",
         "https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences",
         "Built by experiments/build_pmlb_benchmarks.py. Positions p_m30..p_p29 keep PMLB integer codes; class 2 -> neither (UCI N); classes 0/1 -> site_a/site_b (EI/IE not recoverable from PMLB's row subset)."),
        ("fars.csv", "injury", build_fars, "KEEL fars: U.S. Fatality Analysis Reporting System 2001, person file (100 968 rows), via PMLB fars (raw TSV pinned in benchmark_data/raw)",
         "https://sci2s.ugr.es/keel/dataset.php?cod=191",
         "Built by experiments/build_pmlb_benchmarks.py. A census of crashes with at least one death. 13 columns recorded after the outcome dropped (race, Hispanic origin, alcohol and drug test fields, taken to hospital); age 5 quantile groups on all rows; other columns keep PMLB integer codes; injury classes named in the alphabetical order of KEEL's class names, checked by size (1 -> fatal also by the RACE field: recorded on 42 282 rows, 42 116 of them class 1)."),
        ("shuttle.csv", "class", build_shuttle, "UCI ML Repository: Statlog (Shuttle), 58 000 rows, via PMLB shuttle (raw TSV pinned in benchmark_data/raw)",
         "https://archive.ics.uci.edu/dataset/148/statlog+shuttle",
         "Built by experiments/build_pmlb_benchmarks.py. Nine numeric attributes, each 5 quantile groups on all rows; classes named as in the Statlog documentation (1 rad_flow, 2 fpv_close, 3 fpv_open, 4 high, 5 bypass, 6 bpv_close, 7 bpv_open)."),
    ]
    rows: List[List[str]] = []
    for file, target, build, source, url, prep in targets:
        frame = build()
        path = OUT / file
        write_csv(frame, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{file}: {frame.shape[0]} rows, {frame.shape[1] - 1} features")
        rows.append([file, str(frame.shape[0]), str(frame.shape[1]), target, digest, source, url, "", prep])
    if args.manifest:
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        sys.stdout.write(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
