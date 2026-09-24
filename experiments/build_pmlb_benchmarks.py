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

    python experiments/build_pmlb_benchmarks.py --manifest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "benchmark_data" / "raw"
OUT = REPO_ROOT / "benchmark_data"

RAW_SHA256: Dict[str, str] = {
    "connect_4.tsv": "8209a4bca4de6c0073d781d31751dc07b57a38d5c75196ef5b69f2c87374c1a9",
    "splice.tsv": "f57132a4b1437271b173b1073000e660f0a2736106037f5b8e67da85285a4a35",
}


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
