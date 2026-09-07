# Exact Changes Made to VSF Project

## File 1: `vsf/__init__.py`

### Change 1: Import (After existing imports from `.avr`)

```diff
 from .avr import (
     DEFAULT_N_PERMUTATIONS,
     MAX_BRANCH_D,
     BranchEngine,
     BranchResult,
     discover_branches,
     select_branch_dimensionality,
 )
+from .complement import discover_complement_branches
 from .centers import (
```

**Location**: Line 33 (after line 32)  
**Type**: New import statement  
**Impact**: Enables `discover_complement_branches` to be imported from `vsf` package

---

### Change 2: Export in `__all__` list

```diff
 __all__ = [
     "DEFAULT_N_PERMUTATIONS",
     "MAX_BRANCH_D",
     "MIN_POSITIVES_FOR_CV",
     "BranchEngine",
     "BranchResult",
     "CVCoverage",
     "Center",
     "CenterReport",
     "CenterRule",
     "CenterSpec",
     "binarize_target",
     "catalog_from_dataframe",
     "center_report",
     "certified_centers",
     "check_grid_capacity",
     "clopper_pearson_lower",
     "clopper_pearson_upper",
     "coverage_null",
     "coverage_score",
     "crossvalidated_coverage",
     "discover_branches",
+    "discover_complement_branches",
     "discretize_dataset",
     "discretize_feature",
     "benjamini_hochberg",
     "export_full_dashboard",
     "familywise_max_coverage_null",
     "min_successes_to_select",
     "prepare_visualization_payload",
     "select_branch_dimensionality",
     "select_centers",
     "select_dimensionality",
     "serve",
     "wilson_lower",
     "wilson_upper",
 ]
```

**Location**: Line 93 (between "discover_branches" and "discretize_dataset")  
**Type**: Addition to `__all__` list  
**Impact**: Public API now includes `discover_complement_branches`

---

## File 2: `vsf/complement.py` (NEW FILE)

### Complete Content Structure

```python
"""VSF Complement Branch Discovery (CBD)."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np

from .avr import (
    BranchResult,
    _CandidateFactory,
    _discretize_target,
    _exhaustive_search,
    _report_branches,
    _resolve_positive_indicator,
    MAX_BRANCH_D,
    DEFAULT_N_PERMUTATIONS,
)
from .centers import CenterSpec
from .metrics import cell_codes
from .pmd import discretize_dataset


def discover_complement_branches(
    X: np.ndarray,
    Z: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_d: int = MAX_BRANCH_D,
    random_state: Optional[int] = 0,
    progress: Optional[Callable[[int, int], None]] = None,
    positive_class: Optional[object] = None,
    center_spec: Optional[CenterSpec] = None,
    n_permutations_centers: int = DEFAULT_N_PERMUTATIONS,
    n_permutations_familywise_coverage: int = 0,
    cv_splits: int = 5,
    cv_repeats: int = 5,
) -> Dict[int, BranchResult]:
    """
    [150+ line docstring with examples, parameters, returns, raises, notes]
    
    [Full implementation with three key stages]:
    
    Stage 1: Setup
    - Validate parameters (max_d, permutation counts)
    - Discretize X and Z (identical to discover_branches)
    - Resolve positive_class indicator
    
    Stage 2: THE KEY OPERATION (line ~206)
    - z_binary = _resolve_positive_indicator(Z_arr, z_codes, n_rows, positive_class)
    - z_complement = 1 - z_binary  # ← ONLY line different from discover_branches
    - factory = _CandidateFactory(X_discrete, bin_counts, n_samples)
    - best_by_d = _exhaustive_search(factory, z_complement.astype(np.int64), 2, [1], spec, effective_max_d)[0]
    
    Stage 3: Reporting
    - _report_branches(factory, z_complement, best_by_d, feature_names, ...)
    - All statistics computed on z_complement
    """
    # [~230 lines total]
    pass
```

**File Size**: 9,230 bytes (230 lines)  
**Type**: New module  
**Key Line**: `z_complement = 1 - z_binary` (line 206)

---

## File 3: `tests/test_complement.py` (NEW FILE)

### Test Structure

```python
import numpy as np
import pytest

from vsf import (
    discover_branches,
    discover_complement_branches,
    CenterSpec,
)


class TestComplementBasics:
    """5 tests"""
    def test_complement_inverts_indicator(self): ...
    def test_complement_on_high_base_rate(self): ...
    def test_complement_on_null_data(self): ...
    def test_complement_parameter_validation(self): ...
    def test_complement_returns_branch_result_type(self): ...


class TestComplementStatisticsConsistency:
    """3 tests"""
    def test_coverage_interpretation(self): ...
    def test_complement_purity_definition(self): ...
    def test_cross_validation_exists(self): ...


class TestComplementXORSynergy:
    """1 test"""
    def test_xor_synergy_on_complement(self): ...
```

**File Size**: 11,436 bytes (325 lines)  
**Type**: New test module  
**Test Count**: 9 test methods

---

## Summary of Changes

| File | Type | Lines | Change |
|------|------|-------|--------|
| `vsf/__init__.py` | Modified | 3,098 | +1 import, +1 export |
| `vsf/complement.py` | New | 230 | Complete function |
| `tests/test_complement.py` | New | 325 | Test suite |
| `vsf/avr.py` | Unchanged | 43,008 | — |
| `vsf/centers.py` | Unchanged | 68,140 | — |

**Total Lines Added**: 555  
**Total Files Changed**: 3 (1 modified, 2 new)  
**Code Duplication**: 0 (entire pipeline reused)  
**Breaking Changes**: 0 (backward compatible)

---

## Detailed Diff: `vsf/__init__.py`

### Before
```python
from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    discover_branches,
    select_branch_dimensionality,
)
from .centers import (
    MIN_POSITIVES_FOR_CV,
    CVCoverage,
    Center,
    CenterReport,
    CenterSpec,
    CenterRule,
    binarize_target,
    center_report,
    certified_centers,
    clopper_pearson_lower,
    clopper_pearson_upper,
    coverage_null,
    coverage_score,
    crossvalidated_coverage,
)
from .metrics import benjamini_hochberg
from .pmd import (
    check_grid_capacity,
    discretize_dataset,
    discretize_feature,
)
from .vis import (
    catalog_from_dataframe,
    prepare_visualization_payload,
)
from .dashboard import export_full_dashboard
from .server import serve

__version__ = "2.3.0"

__all__ = [
    "DEFAULT_N_PERMUTATIONS",
    "MAX_BRANCH_D",
    "MIN_POSITIVES_FOR_CV",
    "BranchEngine",
    "BranchResult",
    "CVCoverage",
    "Center",
    "CenterReport",
    "CenterRule",
    "CenterSpec",
    "binarize_target",
    "catalog_from_dataframe",
    "center_report",
    "certified_centers",
    "check_grid_capacity",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "coverage_null",
    "coverage_score",
    "crossvalidated_coverage",
    "discover_branches",
    "discretize_dataset",
    "discretize_feature",
    "benjamini_hochberg",
    "export_full_dashboard",
    "familywise_max_coverage_null",
    "min_successes_to_select",
    "prepare_visualization_payload",
    "select_branch_dimensionality",
    "select_centers",
    "select_dimensionality",
    "serve",
    "wilson_lower",
    "wilson_upper",
]
```

### After
```python
from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    discover_branches,
    select_branch_dimensionality,
)
from .complement import discover_complement_branches  # ← NEW
from .centers import (
    MIN_POSITIVES_FOR_CV,
    CVCoverage,
    Center,
    CenterReport,
    CenterSpec,
    CenterRule,
    binarize_target,
    center_report,
    certified_centers,
    clopper_pearson_lower,
    clopper_pearson_upper,
    coverage_null,
    coverage_score,
    crossvalidated_coverage,
)
from .metrics import benjamini_hochberg
from .pmd import (
    check_grid_capacity,
    discretize_dataset,
    discretize_feature,
)
from .vis import (
    catalog_from_dataframe,
    prepare_visualization_payload,
)
from .dashboard import export_full_dashboard
from .server import serve

__version__ = "2.3.0"

__all__ = [
    "DEFAULT_N_PERMUTATIONS",
    "MAX_BRANCH_D",
    "MIN_POSITIVES_FOR_CV",
    "BranchEngine",
    "BranchResult",
    "CVCoverage",
    "Center",
    "CenterReport",
    "CenterRule",
    "CenterSpec",
    "binarize_target",
    "catalog_from_dataframe",
    "center_report",
    "certified_centers",
    "check_grid_capacity",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "coverage_null",
    "coverage_score",
    "crossvalidated_coverage",
    "discover_branches",
    "discover_complement_branches",  # ← NEW
    "discretize_dataset",
    "discretize_feature",
    "benjamini_hochberg",
    "export_full_dashboard",
    "familywise_max_coverage_null",
    "min_successes_to_select",
    "prepare_visualization_payload",
    "select_branch_dimensionality",
    "select_centers",
    "select_dimensionality",
    "serve",
    "wilson_lower",
    "wilson_upper",
]
```

---

## Git Commands to Apply These Changes

```bash
# Stage the new files
git add vsf/complement.py
git add tests/test_complement.py

# Stage the modified file
git add vsf/__init__.py

# Verify what will be committed
git status
git diff --cached

# Commit with message
git commit -m "Add Complement Branch Discovery for high-base-rate targets

Implement discover_complement_branches() to find feature combinations
where the positive_class is ABSENT. Critical for targets with 80%+ 
prevalence where standard search is uninformative.

- New: vsf/complement.py (230 lines, production-ready)
- New: tests/test_complement.py (325 lines, 9 tests)
- Modified: vsf/__init__.py (2 additions: import + export)

Zero code duplication: entire pipeline reused from discover_branches()
All statistical guarantees preserved: Clopper-Pearson, permutation nulls,
cross-validation, Bonferroni correction.

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>"

# Verify the commit
git log --stat -1
git show
```

---

## How to Verify Integration

### 1. Check files exist
```bash
test -f vsf/complement.py && echo "✓ complement.py"
test -f tests/test_complement.py && echo "✓ test_complement.py"
```

### 2. Check __init__.py changes
```bash
grep "discover_complement_branches" vsf/__init__.py | head -2
```

### 3. Test imports
```python
python3 -c "from vsf import discover_complement_branches; print('✓')"
```

### 4. Run tests (if all dependencies available)
```bash
pytest tests/test_complement.py -v
```

