"""The bundled benchmark files are exactly the bytes recorded in the manifest."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_datasets_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vsf_experiments_datasets", _ROOT / "experiments" / "datasets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


datasets = _load_datasets_module()
ENTRIES = datasets.load_manifest()


def test_manifest_lists_every_csv() -> None:
    assert datasets.unlisted_files(ENTRIES) == []


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_entry_matches_file(name: str) -> None:
    assert datasets.verify_entry(ENTRIES[name]) == []


def test_load_dataset_refuses_modified_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = ENTRIES["breast_cancer"]
    copy = tmp_path / entry.file
    copy.write_bytes(entry.path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    with pytest.raises(datasets.DatasetIntegrityError):
        datasets.load_dataset("breast_cancer")


@pytest.mark.parametrize(
    ("name", "shape", "counts"),
    [
        ("breast_cancer", (286, 10), {"no-recurrence-events": 201, "recurrence-events": 85}),
        ("thyroid_recurrence", (383, 17), {"No": 275, "Yes": 108}),
        ("breast_cancer_wisconsin", (699, 10), {"benign": 458, "malignant": 241}),
        ("lung_discrete", (73, 326), {"c1": 6, "c2": 5, "c3": 5, "c4": 16, "c5": 7, "c6": 13, "c7": 21}),
        ("colon", (62, 2001), {"tumor": 40, "normal": 22}),
        ("leukemia", (72, 7071), {"ALL": 47, "AML": 25}),
    ],
)
def test_cancer_benchmarks_match_the_published_facts(name: str, shape: tuple, counts: dict) -> None:
    frame, entry = datasets.load_dataset(name)
    assert frame.shape == shape
    assert frame.columns[0] == entry.target
    assert frame[entry.target].value_counts().to_dict() == counts


def test_missing_markers_of_the_uci_copies() -> None:
    breast, _ = datasets.load_dataset("breast_cancer")
    assert (breast == "?").sum().to_dict() == {c: {"node_caps": 8, "breast_quad": 1}.get(c, 0) for c in breast.columns}
    wisconsin, _ = datasets.load_dataset("breast_cancer_wisconsin")
    assert int((wisconsin["bare_nuclei"] == "?").sum()) == 16
    assert int((wisconsin.drop(columns="bare_nuclei") == "?").to_numpy().sum()) == 0


def test_high_dimensional_values_are_ternary() -> None:
    for name in ("lung_discrete", "colon", "leukemia"):
        frame, entry = datasets.load_dataset(name)
        values = set(frame.drop(columns=entry.target).to_numpy().ravel().tolist())
        assert values <= {"-2", "0", "2"}


@pytest.mark.parametrize(
    ("name", "shape", "counts"),
    [
        ("fars", (100968, 17), {"fatal": 42116, "none": 20007, "incapacitating": 15072, "nonincapacitating": 13890,
                                "possible": 8674, "unknown": 901, "severity_unknown": 299, "died_prior": 9}),
        ("shuttle", (58000, 10), {"rad_flow": 45586, "high": 8903, "bypass": 3267, "fpv_open": 171,
                                  "fpv_close": 50, "bpv_open": 13, "bpv_close": 10}),
    ],
)
def test_large_recorded_tables_match_their_sources(name: str, shape: tuple, counts: dict) -> None:
    frame, entry = datasets.load_dataset(name)
    assert frame.shape == shape
    assert frame.columns[0] == entry.target
    assert frame[entry.target].value_counts().to_dict() == counts


def test_fars_has_no_column_recorded_after_the_outcome() -> None:
    frame, _ = datasets.load_dataset("fars")
    leaked = [c for c in frame.columns if any(word in c for word in ("race", "hispanic", "hospital", "test"))]
    assert leaked == []
    # no single remaining column separates the fatal class the way the dropped RACE column did
    fatal = frame["injury"] == "fatal"
    for column in frame.columns[1:]:
        share = fatal.groupby(frame[column]).mean()
        size = frame[column].value_counts()
        pure = share[(share >= 0.99) & (size.reindex(share.index) >= 1000)]
        assert pure.empty, (column, pure.to_dict())
