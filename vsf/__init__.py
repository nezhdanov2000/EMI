"""
Visual Sufficiency Framework (VSF) — Python Library
Independent Branch Discovery (Mutual-Information) for Multimodal Data Visualization

v2.2 "Certified Centers" — see Project_Master_Document.md Section 0 for the
full revision history.

v2.2 adds `vsf.centers`: the reporting layer that answers the question the
display is actually for — which cells are certifiably almost pure in the
target value (K), what share of the target they capture (coverage), and how
much of that survives cross-validation and the permutation null. `u_adj`
remains available and remains the correct DETECTION statistic, but it is no
longer the headline: on a rare target it reads 41.3 % where the highest cell
purity in the branch is 2.42 % (see `vsf.centers`' module docstring for the
fully reproducible case).

v2.1 replaces v2.0's raw-plug-in-MI ranking and NMI_min reporting with
bias-corrected quantities from the new `vsf.metrics` module: branches are
ranked by MI_adj = I_hat - E_0[I_hat] (exact permutation expectation, Vinh
et al. 2010) and reported as U_adj, the corrected share of the target's
entropy. `BranchResult.nmi` is GONE, not deprecated: on a control dataset in
which every feature is independent of the target, v2.0 reported branches at
up to 9.8 % NMI_min, and no consumer should be able to read that number.

Removed relative to v1.0 (deliberately, not by omission):
`AVREngine`/`AVRResult`/`Scenario` (replaced by `BranchEngine`/
`BranchResult`/`discover_branches`), `generate_interactive_html` (unused,
superseded by `export_full_dashboard`), `compute_top_insights` and
`mine_dirty_center` (Auto-Discovery and dirty-center mining, both removed —
their sole remaining consumers, `marginal_permutation_test`/
`conditional_permutation_test`/`benjamini_hochberg`, were removed alongside
them since nothing else called them).
"""

from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    Objective,
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
    familywise_max_coverage_null,
    min_successes_to_select,
    select_centers,
    select_dimensionality,
    wilson_lower,
    wilson_upper,
)
from .metrics import (
    ClassInfo,
    InfoReport,
    adjusted_mutual_information_bits,
    adjusted_uncertainty_coefficient,
    benjamini_hochberg,
    expected_mutual_information_bits,
    familywise_max_null,
    information_report,
    mutual_information_bits,
    permutation_pvalue,
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

__version__ = "2.2.0"

__all__ = [
    "CHANNEL_LIMITS",
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
    "ClassInfo",
    "InfoReport",
    "Objective",
    "adjusted_mutual_information_bits",
    "adjusted_uncertainty_coefficient",
    "benjamini_hochberg",
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
    "expected_mutual_information_bits",
    "export_full_dashboard",
    "familywise_max_coverage_null",
    "familywise_max_null",
    "information_report",
    "joint_entropy",
    "mutual_information",
    "min_successes_to_select",
    "mutual_information_bits",
    "normalized_mutual_information",
    "permutation_pvalue",
    "prepare_visualization_payload",
    "select_branch_dimensionality",
    "select_centers",
    "select_dimensionality",
    "serve",
    "shannon_entropy",
    "wilson_lower",
    "wilson_upper",
]
