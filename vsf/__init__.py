"""
Visual Sufficiency Framework (VSF) — Python Library
Information-Theoretic Adaptive Dimensionality Selection for Multimodal Data Visualization
"""

from .avr import (
    AVREngine,
    AVRResult,
    Scenario,
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
from .permutation import (
    conditional_permutation_test,
    marginal_permutation_test,
)
from .pmd import (
    CHANNEL_LIMITS,
    check_grid_capacity,
    discretize_dataset,
    discretize_feature,
)
from .vis import (
    generate_interactive_html,
    prepare_visualization_payload,
)

__version__ = "1.0.0"

__all__ = [
    "CHANNEL_LIMITS",
    "AVREngine",
    "AVRResult",
    "Scenario",
    "check_grid_capacity",
    "conditional_permutation_test",
    "discretize_dataset",
    "discretize_feature",
    "evaluate_vsf_accuracy",
    "generate_interactive_html",
    "generate_synthetic_dataset",
    "joint_entropy",
    "marginal_permutation_test",
    "mutual_information",
    "normalized_mutual_information",
    "prepare_visualization_payload",
    "shannon_entropy",
]
