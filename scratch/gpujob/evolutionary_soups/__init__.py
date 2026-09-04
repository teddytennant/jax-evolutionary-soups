"""Evolutionary Soups (arXiv:2608.29978) in JAX, on a toy scale."""

from .entmax import DEFAULT_ALPHA, entmax, entmax_threshold, sparsemax
from .evolution import EvolutionResult, GenerationLog, crossover_mutate, evolve, select
from .experiment import (
    Setup,
    build_setup,
    evolve_gates,
    fixed_coefficient_front,
    front_hypervolume,
    per_layer_reachable,
    population_fitness,
    run_model,
)
from .gating import coefficients, coeff_fn, init_gate, init_population
from .hypervolume import (
    REFERENCE_POINT,
    crowding_distance_selection,
    dominates,
    greedy_hvc_selection,
    hv_contribution,
    hypervolume,
    nondominated,
    reference_point,
)
from .model import ModelConfig, forward, init_backbone, init_experts, merged_ffn
from .preference import (
    linear_utility,
    preference_selection,
    sample_preferences,
    selection_utility,
    tchebyshev_utility,
)

__all__ = [
    "DEFAULT_ALPHA",
    "EvolutionResult",
    "GenerationLog",
    "ModelConfig",
    "REFERENCE_POINT",
    "Setup",
    "build_setup",
    "coeff_fn",
    "coefficients",
    "crossover_mutate",
    "crowding_distance_selection",
    "dominates",
    "entmax",
    "entmax_threshold",
    "evolve",
    "evolve_gates",
    "fixed_coefficient_front",
    "forward",
    "front_hypervolume",
    "greedy_hvc_selection",
    "hv_contribution",
    "hypervolume",
    "init_backbone",
    "init_experts",
    "init_gate",
    "init_population",
    "linear_utility",
    "merged_ffn",
    "nondominated",
    "per_layer_reachable",
    "population_fitness",
    "preference_selection",
    "reference_point",
    "run_model",
    "sample_preferences",
    "select",
    "selection_utility",
    "sparsemax",
    "tchebyshev_utility",
]
