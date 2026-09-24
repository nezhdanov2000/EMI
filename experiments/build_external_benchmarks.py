"""
Build seven real categorical benchmark tables in `benchmark_data/` from
pinned public copies (PLAN.md phase 3.1: more datasets, N from 1 470 to
48 842, M from 11 to 24, base rates 8-28 %).

    file               rows  feat.  target (positive)      source
    adult.csv         48842    12   income (>50K)          UCI 2, via jbrownlee/Datasets
    hmda.csv           2380    12   deny (yes)             AER::HMDA (Rdatasets)
    attrition.csv      1470    19   attrition (Yes)        modeldata::attrition (Rdatasets)
    credit.csv         4454    13   status (bad)           modeldata::credit_data (Rdatasets)
    resume.csv         4870    24   call (yes)             AER::ResumeNames (Rdatasets)
    wa_churn.csv       7043    19   churn (Yes)            modeldata::wa_churn (Rdatasets)
    mlc_churn.csv      5000    11   churn (yes)            modeldata::mlc_churn (Rdatasets)

Every download is checked against a SHA-256 of its bytes (the Rdatasets
copies are served from the repository's master branch, so the hash is the
pin) and against the published row counts. Numeric columns are binned into
quantile groups computed on ALL rows of the file (recorded per column below);
the bins are part of the dataset, identical for every method that reads it.
Columns that duplicate another (charges = minutes x rate), identify rows
(fnlwgt, ids) or leak the target are dropped. Target first; values as strings;
missing values written as `?`.

Usage:
    python experiments/build_external_benchmarks.py            # download, check, write
    python experiments/build_external_benchmarks.py --manifest # also print manifest rows
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_DIR: Path = REPO_ROOT / "benchmark_data"

RD = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"
BROWNLEE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"

SOURCE_SHA256: Dict[str, str] = {
    BROWNLEE + "adult-all.csv": "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2",
    RD + "AER/HMDA.csv": "d142067a4bad8835339f562f57dd2b402e9a5ba7f3b77238d72f0f4efa6fee54",
    RD + "modeldata/attrition.csv": "22db7ee590589479392aeb9f44b9e11ea8d9c2458cc0fd942e8c233d95393d4b",
    RD + "modeldata/credit_data.csv": "acd2b41f7923a4b7e2264e6b9ae2eff8f3e8f9ba3080fb6967df1cfec9568cac",
    RD + "AER/ResumeNames.csv": "9ece4fea6e77a3fa7ed8737ced8b02250b6b768d9cd15be26360e00a4c29e1aa",
    RD + "modeldata/wa_churn.csv": "720b67923e990d9c6ea5b6fa07c0ff15552d7f5bd79d48843fd5c7505a8e71a6",
    RD + "modeldata/mlc_churn.csv": "58f21b63c526bded47dd6df9d58d03b11611d3ad628c2aa5e353f3abb8d51a9c",
}


class BuildError(RuntimeError):
    pass


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256[url]:
        raise BuildError(f"{url}: sha256 {digest} != pinned {SOURCE_SHA256[url]}")
    return data


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def read(url: str, **kw: object) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(fetch(url)), **kw)  # type: ignore[arg-type]


def qbin(s: pd.Series, q: int, name: str) -> pd.Series:
    """Quantile groups g0 < g1 < ... (duplicate edges collapsed); NaN -> '?'."""
    codes = pd.qcut(s, q, duplicates="drop", labels=False)
    out = codes.map(lambda c: "?" if pd.isna(c) else f"{name}_q{int(c)}")
    return out.astype(str)


def finish(frame: pd.DataFrame, target: str, positive: str, rows: int) -> pd.DataFrame:
    expect(frame.shape[0] == rows, f"{target}: {frame.shape[0]} rows, expected {rows}")
    expect(positive in set(frame[target].astype(str)), f"{target}: positive value {positive!r} absent")
    cols = [target] + [c for c in frame.columns if c != target]
    out = frame[cols].copy()
    for c in out.columns:
        out[c] = out[c].where(out[c].notna(), "?").astype(str)
    return out


def build_adult() -> pd.DataFrame:
    cols = ["age", "workclass", "fnlwgt", "education", "educnum", "marital", "occupation",
            "relationship", "race", "sex", "capgain", "caploss", "hours", "native", "income"]
    t = read(BROWNLEE + "adult-all.csv", header=None, names=cols, skipinitialspace=True)
    f = pd.DataFrame({
        "income": t.income.str.replace(".", "", regex=False),
        "age": qbin(t.age, 5, "age"), "workclass": t.workclass, "education": t.education,
        "marital": t.marital, "occupation": t.occupation, "relationship": t.relationship,
        "race": t.race, "sex": t.sex,
        "capital_gain": (t.capgain > 0).map({True: "yes", False: "no"}),
        "capital_loss": (t.caploss > 0).map({True: "yes", False: "no"}),
        "hours": qbin(t.hours, 4, "hours"),
        "native_us": (t.native == "United-States").map({True: "yes", False: "no"}),
    })
    return finish(f, "income", ">50K", 48842)


def build_hmda() -> pd.DataFrame:
    t = read(RD + "AER/HMDA.csv")
    f = t[["deny", "chist", "mhist", "phist", "selfemp", "insurance", "condomin", "afam", "single", "hschool"]].copy()
    f["pirat"] = qbin(t.pirat, 4, "pirat"); f["lvrat"] = qbin(t.lvrat, 4, "lvrat"); f["unemp"] = qbin(t.unemp, 3, "unemp")
    return finish(f, "deny", "yes", 2380)


def build_attrition() -> pd.DataFrame:
    t = read(RD + "modeldata/attrition.csv")
    keep = ["Attrition", "BusinessTravel", "Department", "Education", "EducationField", "EnvironmentSatisfaction",
            "Gender", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
            "OverTime", "StockOptionLevel", "WorkLifeBalance"]
    f = t[keep].copy()
    for c, q in [("Age", 4), ("MonthlyIncome", 4), ("TotalWorkingYears", 4), ("YearsAtCompany", 4), ("DistanceFromHome", 3)]:
        f[c] = qbin(t[c], q, c)
    f.columns = [c.lower() for c in f.columns]
    return finish(f, "attrition", "Yes", 1470)


def build_credit() -> pd.DataFrame:
    t = read(RD + "modeldata/credit_data.csv")
    f = t[["Status", "Home", "Marital", "Records", "Job"]].copy()
    for c, q in [("Seniority", 4), ("Time", 3), ("Age", 4), ("Expenses", 3), ("Income", 4),
                 ("Assets", 3), ("Amount", 4), ("Price", 4)]:
        f[c] = qbin(t[c], q, c)
    f["Debt"] = (t.Debt.fillna(0) > 0).map({True: "yes", False: "no"})
    f.columns = [c.lower() for c in f.columns]
    return finish(f, "status", "bad", 4454)


def build_resume() -> pd.DataFrame:
    t = read(RD + "AER/ResumeNames.csv")
    keep = ["call", "gender", "ethnicity", "quality", "city", "jobs", "honors", "volunteer", "military",
            "holes", "school", "email", "computer", "special", "college", "equal", "wanted",
            "requirements", "reqexp", "reqcomm", "reqeduc", "reqcomp", "reqorg", "industry"]
    f = t[keep].copy(); f["experience"] = qbin(t.experience, 3, "experience")
    return finish(f, "call", "yes", 4870)


def build_wa_churn() -> pd.DataFrame:
    t = read(RD + "modeldata/wa_churn.csv")
    drop = {"rownames", "tenure", "monthly_charges", "total_charges"}
    f = t[[c for c in t.columns if c not in drop]].copy()
    f["tenure"] = qbin(t.tenure, 4, "tenure"); f["monthly"] = qbin(t.monthly_charges, 4, "monthly")
    f["total"] = qbin(t.total_charges, 4, "total")
    return finish(f, "churn", "Yes", 7043)


def build_mlc_churn() -> pd.DataFrame:
    t = read(RD + "modeldata/mlc_churn.csv")
    f = t[["churn", "area_code", "international_plan", "voice_mail_plan"]].copy()
    f["vmail"] = (t.number_vmail_messages > 0).map({True: "yes", False: "no"})
    for c, q in [("total_day_minutes", 4), ("total_eve_minutes", 4), ("total_night_minutes", 4),
                 ("total_intl_minutes", 4), ("total_intl_calls", 3), ("account_length", 3)]:
        f[c] = qbin(t[c], q, c)
    f["svc_calls"] = t.number_customer_service_calls.clip(upper=4).map(lambda v: f"{int(v)}{'+' if v >= 4 else ''}")
    return finish(f, "churn", "yes", 5000)


@dataclass(frozen=True)
class Target:
    file: str
    target: str
    build: Callable[[], pd.DataFrame]
    source: str
    source_url: str
    preprocessing: str


TARGETS: Sequence[Target] = (
    Target("adult.csv", "income", build_adult, "UCI ML Repository: Adult (48 842 rows), via jbrownlee/Datasets adult-all.csv",
           "https://archive.ics.uci.edu/dataset/2/adult",
           "Built by experiments/build_external_benchmarks.py. age 5 quantile groups, hours 4; capital gain/loss as yes/no (>0); native-country as US yes/no; fnlwgt and education-num dropped; income '.' suffix removed."),
    Target("hmda.csv", "deny", build_hmda, "AER::HMDA (Boston HMDA, Munnell et al. 1996), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/AER/HMDA.html",
           "Built by experiments/build_external_benchmarks.py. pirat and lvrat 4 quantile groups, unemp 3; other columns as given; rownames dropped."),
    Target("attrition.csv", "attrition", build_attrition, "modeldata::attrition (IBM HR analytics), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/attrition.html",
           "Built by experiments/build_external_benchmarks.py. 14 categorical columns kept; Age, MonthlyIncome, TotalWorkingYears, YearsAtCompany 4 quantile groups, DistanceFromHome 3; other numeric columns dropped; names lower-cased."),
    Target("credit.csv", "status", build_credit, "modeldata::credit_data (Spanish credit scoring), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/credit_data.html",
           "Built by experiments/build_external_benchmarks.py. Seniority, Age, Income, Amount, Price 4 quantile groups; Time, Expenses, Assets 3; Debt as yes/no (>0); missing -> ?; names lower-cased."),
    Target("resume.csv", "call", build_resume, "AER::ResumeNames (Bertrand & Mullainathan 2004), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ResumeNames.html",
           "Built by experiments/build_external_benchmarks.py. name and minimum dropped; experience 3 quantile groups; other columns as given."),
    Target("wa_churn.csv", "churn", build_wa_churn, "modeldata::wa_churn (IBM Watson telco churn), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/wa_churn.html",
           "Built by experiments/build_external_benchmarks.py. tenure, monthly_charges, total_charges 4 quantile groups; rownames dropped."),
    Target("mlc_churn.csv", "churn", build_mlc_churn, "modeldata::mlc_churn (MLC++ telecom churn), via Rdatasets",
           "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/mlc_churn.html",
           "Built by experiments/build_external_benchmarks.py. state and the charge columns (= minutes x rate) dropped; minutes columns 4 quantile groups, intl calls and account length 3; customer service calls capped at 4+; vmail messages as yes/no."),
)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="store_true", help="print MANIFEST.csv rows for the written files")
    args = ap.parse_args(argv)
    rows: List[List[str]] = []
    for t in TARGETS:
        frame = t.build()
        path = OUT_DIR / t.file
        write_csv(frame, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        counts = {str(k): int(v) for k, v in frame[t.target].value_counts().items()}
        print(f"{t.file}: {frame.shape[0]} rows, {frame.shape[1] - 1} features, classes {counts}")
        missing = "?" if (frame == "?").any().any() else ""
        rows.append([t.file, str(frame.shape[0]), str(frame.shape[1]), t.target, digest,
                     t.source, t.source_url, missing, t.preprocessing])
    if args.manifest:
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        sys.stdout.write(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
