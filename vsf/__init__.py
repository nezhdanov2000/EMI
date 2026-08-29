"""
Visual Sufficiency Framework (VSF) — Python Library
Independent Branch Discovery (Mutual-Information) for Multimodal Data Visualization

v2.0 "Clean Core" — see Project_Master_Document.md Section 0 for the full
revision history. Removed relative to v1.0 (deliberately, not by omission):
`AVREngine`/`AVRResult`/`Scenario` (replaced by `BranchEngine`/
`BranchResult`/`discover_branches`), `generate_interactive_html` (unused,
superseded by `export_full_dashboard`), `compute_top_insights` and
`mine_dirty_center` (Auto-Discovery and dirty-center mining, both removed —
their sole remaining consumers, `marginal_permutation_test`/
`conditional_permutation_test`/`benjamini_hochberg`, were removed alongside
them since nothing else called them).
"""

from .avr import (
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    discover_branches,
)
from .benchmark import (
    evaluate_vsf_accuracy,
    generate_synthetic_dataset,
)
from .math import (
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
    shannon_entropy,
)
from .pmd import (
    CHANNEL_LIMITS,
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

__version__ = "2.0.0"

__all__ = [
    "CHANNEL_LIMITS",
    "MAX_BRANCH_D",
    "BranchEngine",
    "BranchResult",
    "catalog_from_dataframe",
    "check_grid_capacity",
    "discover_branches",
    "discretize_dataset",
    "discretize_feature",
    "evaluate_vsf_accuracy",
    "export_full_dashboard",
    "generate_synthetic_dataset",
    "joint_entropy",
    "mutual_information",
    "normalized_mutual_information",
    "prepare_visualization_payload",
    "serve",
    "shannon_entropy",
]
