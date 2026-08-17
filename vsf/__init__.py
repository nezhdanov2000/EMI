"""
Visual Sufficiency Framework (VSF) — Python Library
Information-Theoretic Adaptive Dimensionality Selection for Multimodal Data Visualization
"""

from .math import (
    shannon_entropy,
    joint_entropy,
    mutual_information,
    normalized_mutual_information,
)
from .pmd import (
    CHANNEL_LIMITS,
    discretize_feature,
    discretize_dataset,
    check_grid_capacity,
)
from .permutation import (
    marginal_permutation_test,
    conditional_permutation_test,
)
from .avr import (
    AVREngine,
    AVRResult,
    Scenario,
)
from .benchmark import (
    generate_synthetic_dataset,
    evaluate_vsf_accuracy,
)
from .vis import (
    prepare_visualization_payload,
    generate_interactive_html,
)

__version__ = "1.0.0"

__all__ = [
    "shannon_entropy",
    "joint_entropy",
    "mutual_information",
    "normalized_mutual_information",
    "CHANNEL_LIMITS",
    "discretize_feature",
    "discretize_dataset",
    "check_grid_capacity",
    "marginal_permutation_test",
    "conditional_permutation_test",
    "AVREngine",
    "AVRResult",
    "Scenario",
    "generate_synthetic_dataset",
    "evaluate_vsf_accuracy",
    "prepare_visualization_payload",
    "generate_interactive_html",
]
