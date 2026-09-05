"""NOD-MARL opinion learning and opinion-conditioned control components."""

from .interaction import NOD_PAIR_FEATURE_DIM, build_directed_interactions
from .opinion import NODOpinionModel, kl_objective, kl_proximal_update
from .trainer import NODOpinionManager
from .policy import (
    NODActorInputModule,
    NODMessageAggregator,
    NOD_ACTOR_OBSERVATION_KEY,
)

__all__ = [
    "NOD_PAIR_FEATURE_DIM",
    "NODOpinionManager",
    "NODOpinionModel",
    "NODActorInputModule",
    "NODMessageAggregator",
    "NOD_ACTOR_OBSERVATION_KEY",
    "build_directed_interactions",
    "kl_objective",
    "kl_proximal_update",
]
