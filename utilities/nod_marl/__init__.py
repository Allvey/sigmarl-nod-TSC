"""Auxiliary NOD-MARL opinion learning components.

The package is deliberately isolated from the policy and PPO computation graph.
It only consumes detached rollout diagnostics until opinion-conditioned control is
enabled in a later implementation phase.
"""

from .interaction import NOD_PAIR_FEATURE_DIM, build_directed_interactions
from .opinion import NODOpinionModel, kl_objective, kl_proximal_update
from .trainer import NODOpinionManager

__all__ = [
    "NOD_PAIR_FEATURE_DIM",
    "NODOpinionManager",
    "NODOpinionModel",
    "build_directed_interactions",
    "kl_objective",
    "kl_proximal_update",
]
