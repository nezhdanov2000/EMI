"""
Visual Sufficiency Framework (VSF) - Python Library
Independent Branch Discovery for categorical data visualization.

VSF is a CATEGORICAL framework. Every column - feature or target - is a set
of discrete categories, and every distinct value in it is one category,
whatever dtype it arrives in. There is no continuous-feature model: no
binning, no bin-count heuristic, no rate-distortion distortion figure, and
no information-theoretic quantity computed anywhere in this package. See
`vsf.pmd`'s module docstring for what was removed with that machinery and
why.

For a chosen target column and a named positive value, `discover_branches`
finds, independently for each dimensionality d in {1, 2, 3, 4}, the feature
subset of that size whose grid concentrates the most of that value inside
certified discrete centres, by exhaustive enumeration. `vsf.centers` is the
reporting layer that defines a centre and computes what is reported:
coverage, the number of centres K, pooled purity, cross-validated coverage
and the permutation nulls. `vsf.serve` runs the interactive application;
`vsf.export_full_dashboard` writes a single self-contained HTML file.

See Project_Master_Document.md for the full specification.
"""

from .avr import (
    DEFAULT_N_PERMUTATIONS,
    MAX_BRANCH_D,
    BranchEngine,
    BranchResult,
    Direction,
    Landscape,
    base_rate_reason,
    family_cell_count,
    resolve_center_spec,
    compute_landscape,
    compute_tau_curves,
    tau_grid,
    discover_branches,
    discover_branches_by_value,
    iter_branches_by_value,
    report_schema,
    select_branch_dimensionality,
)
from .centers import (
    FAMILY_TAIL_MARGIN,
    min_successes_to_certify_heterogeneous,
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
from .metrics import benjamini_hochberg
from .selective import (
    CertifiedCell,
    NestedCVResult,
    SchemaStability,
    SelectiveCertificate,
    SelectiveDiscovery,
    certify_discovery,
    exact_upper_tail,
    nested_crossvalidation,
)
from .screen import (
    DEFAULT_MIN_STRENGTH,
    ColumnProfile,
    DatasetScreen,
    DependencyPair,
    column_profiles,
    dependency_pairs,
    exact_dependencies,
    screen_dataset,
    target_report,
)
from .rules import enumerate_rules
from .redundancy import (
    DEFAULT_GROUP_THRESHOLD,
    PAIR_FLOOR,
    CenterCatalog,
    CenterGrouping,
    collect_centers,
)
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
    "FAMILY_TAIL_MARGIN",
    "min_successes_to_certify_heterogeneous",
    "family_cell_count",
    "resolve_center_spec",
    "CertifiedCell",
    "NestedCVResult",
    "SchemaStability",
    "SelectiveCertificate",
    "SelectiveDiscovery",
    "certify_discovery",
    "exact_upper_tail",
    "nested_crossvalidation",
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
    "CenterCatalog",
    "CenterGrouping",
    "DEFAULT_GROUP_THRESHOLD",
    "PAIR_FLOOR",
    "collect_centers",
    "enumerate_rules",
    "ColumnProfile",
    "DatasetScreen",
    "DependencyPair",
    "DEFAULT_MIN_STRENGTH",
    "column_profiles",
    "dependency_pairs",
    "exact_dependencies",
    "screen_dataset",
    "target_report",
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
    "discover_branches_by_value",
    "iter_branches_by_value",
    "Direction",
    "Landscape",
    "base_rate_reason",
    "compute_landscape",
    "compute_tau_curves",
    "tau_grid",
    "report_schema",
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
