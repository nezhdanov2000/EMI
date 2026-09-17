"""
Build the six cancer benchmark files requested for the experiments
(low- and high-dimensional) in `benchmark_data/`, from pinned sources.

    file                           rows  features  classes  source
    breast_cancer.csv               286         9        2  UCI 14 (Ljubljana)
    thyroid_recurrence.csv          383        16        2  UCI 915
    breast_cancer_wisconsin.csv     699         9        2  UCI 15 (original)
    lung_discrete.csv                73       325        7  scikit-feature
    colon.csv                        62      2000        2  scikit-feature
    leukemia.csv                     72      7070        2  scikit-feature

archive.ics.uci.edu is not reachable from the machines this project runs on,
so the three UCI tables are taken from public GitHub copies, pinned to a
commit and checked against a SHA-256 of the downloaded bytes and against the
facts UCI publishes (rows, features, class counts, missing values). The
scikit-feature tables are the `.mat` files of jundongl/scikit-feature at a
pinned commit; `lung_discrete` is stored there as `lung_small.mat`.

Every value is written as a string; the target is the first column.
Conversions (also recorded in benchmark_data/MANIFEST.csv):
- breast_cancer: UCI column names; the copy marks missing values with an
  bare `nan`, written back as UCI's `?` (9 cells: node_caps 8, breast_quad 1).
- thyroid_recurrence: snake_case column names (the source's
  "Hx Radiothreapy" typo fixed); `age` binned into 15-29, 30-39, 40-49,
  50-59, 60-69, 70+ (65 distinct ages otherwise); all other values unchanged.
- breast_cancer_wisconsin: sample id dropped; class 2 -> benign, 4 ->
  malignant; UCI's `?` kept in bare_nuclei (16 rows).
- scikit-feature: feature values -2 / 0 / 2 kept as they are; features named
  g0001... (colon, leukemia) and f001... (lung_discrete). Class labels:
  colon -1 -> tumor, 1 -> normal and leukemia -1 -> ALL, 1 -> AML are
  inferred from the class sizes of the original studies (Alon et al. 1999:
  40 tumour / 22 normal; Golub et al. 1999: 47 ALL / 25 AML); lung_discrete
  classes have no published names and are written c1 ... c7.

Usage:
    python experiments/build_cancer_benchmarks.py            # download, check, write
    python experiments/build_cancer_benchmarks.py --manifest # also print manifest rows
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
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
OUT_DIR: Path = REPO_ROOT / "benchmark_data"

SKFEATURE = "https://raw.githubusercontent.com/jundongl/scikit-feature/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/data/"
BROWNLEE = "https://raw.githubusercontent.com/jbrownlee/Datasets/d20fcb6402ae34e653d4513b00f39257bb37ed7f/"
HASHEMI = "https://raw.githubusercontent.com/HashemiScience/data4pnb/f1696a49eb00286d52ae98ba198d8ea718357b41/datasets/"

SOURCE_SHA256: Dict[str, str] = {
    SKFEATURE + "colon.mat": "ffcdeba03eb67cec403fa1dc9f827c22a6e2c57786bf3e01dfe1b4b3e25e0a2f",
    SKFEATURE + "leukemia.mat": "eb92382fb77c968864cb0a92a2c77acde7f88ba7fe5fd5a64b98184e3a05269d",
    SKFEATURE + "lung_small.mat": "93c6a65eb1f6f9f95ac897010c337254a57c57e7f8c0c9a7963a0e03de8233f4",
    BROWNLEE + "breast-cancer.csv": "4523656d14e91168a602301490a8c89674a9b14384c29a5f652ba1a2bec844a9",
    BROWNLEE + "breast-cancer-wisconsin.data": "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64",
    HASHEMI + "Thyroid_Diff.csv": "a2c65f7c8fa0e78a65aa4946961aafd45218c92760d10375fe2c778045c97b44",
}


class BuildError(RuntimeError):
    """A source differs from its pinned bytes or from the published facts."""


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


def class_counts(frame: pd.DataFrame, target: str) -> Dict[str, int]:
    return {str(k): int(v) for k, v in frame[target].value_counts().items()}


# --------------------------------------------------------------------------
# UCI tables
# --------------------------------------------------------------------------
BREAST_COLUMNS: Sequence[str] = (
    "age", "menopause", "tumor_size", "inv_nodes", "node_caps",
    "deg_malig", "breast", "breast_quad", "irradiat", "class",
)


def build_breast_cancer() -> pd.DataFrame:
    raw = fetch(BROWNLEE + "breast-cancer.csv").decode("ascii")
    df = pd.read_csv(io.StringIO(raw), header=None, names=list(BREAST_COLUMNS), dtype=str,
                     quotechar="'", keep_default_na=False, na_filter=False)
    expect(not (df == "").to_numpy().any(), "breast_cancer: empty values")
    df = df.replace("nan", "?")
    df = df[["class"] + list(BREAST_COLUMNS[:-1])]
    expect(df.shape == (286, 10), f"breast_cancer shape {df.shape}")
    expect(class_counts(df, "class") == {"no-recurrence-events": 201, "recurrence-events": 85},
           f"breast_cancer classes {class_counts(df, 'class')}")
    missing = {c: int((df[c] == "?").sum()) for c in df.columns if (df[c] == "?").any()}
    expect(missing == {"node_caps": 8, "breast_quad": 1}, f"breast_cancer missing {missing}")
    return df


WISCONSIN_COLUMNS: Sequence[str] = (
    "sample_id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses", "class",
)


def build_breast_cancer_wisconsin() -> pd.DataFrame:
    raw = fetch(BROWNLEE + "breast-cancer-wisconsin.data").decode("ascii")
    df = pd.read_csv(io.StringIO(raw), header=None, names=list(WISCONSIN_COLUMNS), dtype=str,
                     keep_default_na=False, na_filter=False)
    df["class"] = df["class"].map({"2": "benign", "4": "malignant"})
    expect(not df["class"].isna().any(), "breast_cancer_wisconsin: class outside {2, 4}")
    df = df[["class"] + list(WISCONSIN_COLUMNS[1:-1])]
    expect(df.shape == (699, 10), f"breast_cancer_wisconsin shape {df.shape}")
    expect(class_counts(df, "class") == {"benign": 458, "malignant": 241},
           f"breast_cancer_wisconsin classes {class_counts(df, 'class')}")
    expect(int((df == "?").to_numpy().sum()) == 16 and int((df["bare_nuclei"] == "?").sum()) == 16,
           "breast_cancer_wisconsin: expected 16 '?' in bare_nuclei only")
    return df


THYROID_RENAME: Dict[str, str] = {
    "Age": "age", "Gender": "gender", "Smoking": "smoking", "Hx Smoking": "hx_smoking",
    "Hx Radiothreapy": "hx_radiotherapy", "Thyroid Function": "thyroid_function",
    "Physical Examination": "physical_examination", "Adenopathy": "adenopathy",
    "Pathology": "pathology", "Focality": "focality", "Risk": "risk", "T": "t", "N": "n",
    "M": "m", "Stage": "stage", "Response": "response", "Recurred": "recurred",
}
AGE_BINS: Sequence[Tuple[int, int, str]] = (
    (0, 29, "15-29"), (30, 39, "30-39"), (40, 49, "40-49"),
    (50, 59, "50-59"), (60, 69, "60-69"), (70, 200, "70+"),
)


def age_bin(value: str) -> str:
    age = int(value)
    for lo, hi, label in AGE_BINS:
        if lo <= age <= hi:
            return label
    raise BuildError(f"thyroid_recurrence: age {age} outside the bins")


def build_thyroid_recurrence() -> pd.DataFrame:
    raw = fetch(HASHEMI + "Thyroid_Diff.csv").decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), dtype=str, keep_default_na=False, na_filter=False)
    expect(list(df.columns) == list(THYROID_RENAME), f"thyroid_recurrence columns {list(df.columns)}")
    df = df.rename(columns=THYROID_RENAME)
    ages = df["age"].astype(int)
    expect(int(ages.min()) == 15 and int(ages.max()) == 82, "thyroid_recurrence: age range is not 15-82")
    df["age"] = df["age"].map(age_bin)
    df = df[["recurred"] + [c for c in df.columns if c != "recurred"]]
    expect(df.shape == (383, 17), f"thyroid_recurrence shape {df.shape}")
    expect(class_counts(df, "recurred") == {"No": 275, "Yes": 108},
           f"thyroid_recurrence classes {class_counts(df, 'recurred')}")
    expect(not (df == "").to_numpy().any(), "thyroid_recurrence: empty values")
    return df


# --------------------------------------------------------------------------
# scikit-feature tables
# --------------------------------------------------------------------------
def build_mat(file: str, prefix: str, labels: Callable[[np.ndarray], List[str]],
              shape: Tuple[int, int], counts: Dict[str, int]) -> pd.DataFrame:
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(fetch(SKFEATURE + file)))
    X = np.asarray(mat["X"])
    Y = np.asarray(mat["Y"]).ravel()
    expect(X.shape == shape, f"{file}: X shape {X.shape} != {shape}")
    expect(set(np.unique(X).tolist()) <= {-2, 0, 2}, f"{file}: values outside -2/0/2")
    width = len(str(shape[1]))
    names = [f"{prefix}{j + 1:0{width}d}" for j in range(shape[1])]
    frame = pd.DataFrame(X.astype(np.int64).astype(str), columns=names)
    frame.insert(0, "class", labels(Y))
    expect(class_counts(frame, "class") == counts, f"{file}: classes {class_counts(frame, 'class')}")
    return frame


def _binary(neg: str, pos: str) -> Callable[[np.ndarray], List[str]]:
    def label(y: np.ndarray) -> List[str]:
        mapping = {-1: neg, 1: pos}
        if not set(y.tolist()) <= set(mapping):
            raise BuildError(f"labels {sorted(set(y.tolist()))} are not -1/1")
        return [mapping[int(v)] for v in y]
    return label


def _numbered(y: np.ndarray) -> List[str]:
    return [f"c{int(v)}" for v in y]


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    file: str
    target: str
    build: Callable[[], pd.DataFrame]
    source: str
    source_url: str
    missing: str
    preprocessing: str


TARGETS: Sequence[Target] = (
    Target("breast_cancer.csv", "class", build_breast_cancer,
           "UCI ML Repository: Breast Cancer (Ljubljana), via jbrownlee/Datasets@d20fcb6",
           "https://archive.ics.uci.edu/dataset/14/breast+cancer", "?",
           "Built by experiments/build_cancer_benchmarks.py. UCI column names; class moved first; "
           "the copy's nan written as UCI's ? (node_caps 8, breast_quad 1)."),
    Target("thyroid_recurrence.csv", "recurred", build_thyroid_recurrence,
           "UCI ML Repository: Differentiated Thyroid Cancer Recurrence, via HashemiScience/data4pnb@f1696a4",
           "https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence", "",
           "Built by experiments/build_cancer_benchmarks.py. snake_case names (Hx Radiothreapy -> "
           "hx_radiotherapy); age binned 15-29/30-39/40-49/50-59/60-69/70+; target moved first."),
    Target("breast_cancer_wisconsin.csv", "class", build_breast_cancer_wisconsin,
           "UCI ML Repository: Breast Cancer Wisconsin (Original), via jbrownlee/Datasets@d20fcb6",
           "https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original", "?",
           "Built by experiments/build_cancer_benchmarks.py. Sample id dropped; class 2->benign, "
           "4->malignant, moved first; ? kept in bare_nuclei (16 rows); values 1-10 as categories."),
    Target("lung_discrete.csv", "class",
           lambda: build_mat("lung_small.mat", "f", _numbered, (73, 325),
                             {"c7": 21, "c4": 16, "c6": 13, "c5": 7, "c1": 6, "c2": 5, "c3": 5}),
           "scikit-feature: lung_discrete (lung_small.mat) @48cffad",
           "https://jundongl.github.io/scikit-feature/datasets.html", "",
           "Built by experiments/build_cancer_benchmarks.py. Values -2/0/2 as given; features f001-f325; "
           "classes 1-7 written c1-c7 (no published names)."),
    Target("colon.csv", "class",
           lambda: build_mat("colon.mat", "g", _binary("tumor", "normal"), (62, 2000),
                             {"tumor": 40, "normal": 22}),
           "scikit-feature: colon (colon.mat) @48cffad",
           "https://jundongl.github.io/scikit-feature/datasets.html", "",
           "Built by experiments/build_cancer_benchmarks.py. Values -2/0/2 as given; features g0001-g2000; "
           "labels -1->tumor, 1->normal inferred from class sizes (Alon et al. 1999: 40/22)."),
    Target("leukemia.csv", "class",
           lambda: build_mat("leukemia.mat", "g", _binary("ALL", "AML"), (72, 7070),
                             {"ALL": 47, "AML": 25}),
           "scikit-feature: leukemia (leukemia.mat) @48cffad",
           "https://jundongl.github.io/scikit-feature/datasets.html", "",
           "Built by experiments/build_cancer_benchmarks.py. Values -2/0/2 as given; features g0001-g7070; "
           "labels -1->ALL, 1->AML inferred from class sizes (Golub et al. 1999: 47/25)."),
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
        print(f"{t.file}: {frame.shape[0]} rows, {frame.shape[1] - 1} features, "
              f"classes {class_counts(frame, t.target)}")
        rows.append([t.file, str(frame.shape[0]), str(frame.shape[1]), t.target, digest,
                     t.source, t.source_url, t.missing, t.preprocessing])
    if args.manifest:
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        sys.stdout.write(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
