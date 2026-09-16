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
    entry = ENTRIES["car_evaluation"]
    copy = tmp_path / entry.file
    copy.write_bytes(entry.path.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    with pytest.raises(datasets.DatasetIntegrityError):
        datasets.load_dataset("car_evaluation")


def test_soybean_header_is_the_uci_attribute_list() -> None:
    frame, _ = datasets.load_dataset("soybean_large")
    assert list(frame.columns[:3]) == ["disease", "date", "plant_stand"]
    assert list(frame.columns[-2:]) == ["shriveling", "roots"]
    # fruit_spots is the one attribute whose code 3 is unused in soybean-large.names
    assert set(frame["fruit_spots"]) == {"0", "1", "2", "4", "?"}
