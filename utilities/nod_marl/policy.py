"""Opinion-conditioned actor input and permutation-invariant edge aggregation."""

from __future__ import annotations

import weakref
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from tensordict.nn import TensorDictModuleBase


NOD_ACTOR_OBSERVATION_KEY = ("agents", "info", "nod_actor_observation")
NOD_ACTOR_EDGE_CONTEXT_KEY = ("agents", "info", "nod_actor_edge_context")
NOD_ACTOR_EDGE_MASK_KEY = ("agents", "info", "nod_actor_edge_mask")
NOD_OPINION_ACTIVE_KEY = ("agents", "info", "nod_opinion_active")
NOD_ACTOR_CONTEXT_READY_KEY = ("agents", "info", "nod_actor_context_ready")
NOD_ACTOR_MESSAGE_KEY = ("agents", "info", "nod_actor_message")
NOD_ACTOR_ATTENTION_KEY = ("agents", "info", "nod_actor_message_attention")


class NODMessageAggregator(nn.Module):
    """Shared edge encoder followed by masked, permutation-invariant attention."""

    def __init__(
        self,
        context_dim: int,
        message_dim: int,
        hidden_dim: int,
        message_scale: float = 0.1,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.message_dim = int(message_dim)
        self.message_scale = float(message_scale)
        self.context_norm = nn.LayerNorm(
            self.context_dim, elementwise_affine=False
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.context_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), self.message_dim),
            nn.Tanh(),
        )
        self.score = nn.Linear(self.message_dim, 1)

    def forward(self, edge_context: Tensor, edge_mask: Tensor) -> Tuple[Tensor, Tensor]:
        if edge_context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected edge context dim {self.context_dim}, "
                f"got {edge_context.shape[-1]}"
            )
        if edge_context.shape[:-1] != edge_mask.shape:
            raise ValueError("edge_context and edge_mask dimensions do not match")

        edge_mask = edge_mask.bool()
        edge_messages = self.edge_encoder(self.context_norm(edge_context))
        logits = self.score(edge_messages).squeeze(-1)
        valid_any = edge_mask.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~edge_mask, -torch.inf)
        safe_logits = torch.where(valid_any, safe_logits, torch.zeros_like(safe_logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(edge_mask, weights, torch.zeros_like(weights))
        aggregate = (weights.unsqueeze(-1) * edge_messages).sum(dim=-2)
        aggregate = torch.where(
            valid_any, aggregate, torch.zeros_like(aggregate)
        )
        # A fixed bound prevents a moving NOD representation from gradually
        # dominating the direct physical observation received by the Actor.
        aggregate = self.message_scale * torch.tanh(aggregate)
        return aggregate, weights


class NODActorInputModule(TensorDictModuleBase):
    """Build ``[local observation, opinion message, previous action]``.

    During collection the module asks :class:`NODOpinionManager` for one online
    recurrent step. During PPO replay it finds ``context_ready=True`` and only
    re-runs this stateless aggregator, preserving the context used to collect
    the action while still allowing PPO gradients into the message network.
    """

    def __init__(
        self,
        *,
        observation_key,
        base_observation_dim: int,
        nod_manager,
        action_dim: int,
        message_dim: int,
        message_hidden_dim: int,
        message_scale: float = 0.1,
        opinion_mode: str = "online",
    ):
        super().__init__()
        self.observation_key = observation_key
        self.base_observation_dim = int(base_observation_dim)
        self.message_dim = int(message_dim)
        self.action_dim = int(action_dim)
        if opinion_mode not in {"online", "neutral"}:
            raise ValueError("opinion_mode must be online or neutral")
        self.opinion_mode = opinion_mode
        self.context_dim = int(nod_manager.online_context_dim)
        self.in_keys = [observation_key, NOD_ACTOR_EDGE_CONTEXT_KEY]
        self.out_keys = [
            NOD_ACTOR_OBSERVATION_KEY,
            NOD_ACTOR_EDGE_CONTEXT_KEY,
            NOD_ACTOR_EDGE_MASK_KEY,
            NOD_OPINION_ACTIVE_KEY,
            NOD_ACTOR_CONTEXT_READY_KEY,
            NOD_ACTOR_MESSAGE_KEY,
            NOD_ACTOR_ATTENTION_KEY,
        ]
        object.__setattr__(self, "_nod_manager_ref", weakref.ref(nod_manager))
        self.aggregator = NODMessageAggregator(
            context_dim=self.context_dim,
            message_dim=self.message_dim,
            hidden_dim=int(message_hidden_dim),
            message_scale=float(message_scale),
        )

    @property
    def actor_input_dim(self) -> int:
        return self.base_observation_dim + self.message_dim + self.action_dim

    @property
    def nod_manager(self):
        manager = self._nod_manager_ref()
        if manager is None:
            raise RuntimeError("NOD manager no longer exists")
        return manager

    @staticmethod
    def _ready(tensordict) -> bool:
        ready = tensordict.get(NOD_ACTOR_CONTEXT_READY_KEY, default=None)
        return ready is not None and bool(ready.bool().all())

    def _previous_action(self, tensordict, observation: Tensor) -> Tensor:
        velocity = tensordict.get(("agents", "info", "act_vel"), default=None)
        steering = tensordict.get(("agents", "info", "act_steer"), default=None)
        if velocity is None or steering is None:
            return observation.new_zeros(*observation.shape[:-1], self.action_dim)
        if velocity.ndim == observation.ndim and velocity.shape[-1] == 1:
            velocity = velocity.squeeze(-1)
        if steering.ndim == observation.ndim and steering.shape[-1] == 1:
            steering = steering.squeeze(-1)
        previous = torch.stack([velocity, steering], dim=-1).to(observation.dtype)
        if previous.shape[-1] < self.action_dim:
            previous = torch.nn.functional.pad(
                previous, (0, self.action_dim - previous.shape[-1])
            )
        return previous[..., : self.action_dim]

    def _fallback_context(self, observation: Tensor) -> Dict[str, Tensor]:
        k_neighbors = max(1, int(self.nod_manager.n_neighbors))
        leading_shape = observation.shape[:-1]
        return {
            "edge_context": observation.new_zeros(
                *leading_shape, k_neighbors, self.context_dim
            ),
            "edge_mask": torch.zeros(
                *leading_shape,
                k_neighbors,
                device=observation.device,
                dtype=torch.bool,
            ),
            "opinion_active": torch.zeros(
                *leading_shape,
                k_neighbors,
                device=observation.device,
                dtype=torch.bool,
            ),
        }

    def forward(self, tensordict):
        observation = tensordict.get(self.observation_key)
        if not self._ready(tensordict):
            with torch.no_grad():
                online = self.nod_manager.online_step(
                    tensordict
                )
            if online is None:
                online = self._fallback_context(observation)
            tensordict.set(
                NOD_ACTOR_EDGE_CONTEXT_KEY, online["edge_context"].detach()
            )
            tensordict.set(NOD_ACTOR_EDGE_MASK_KEY, online["edge_mask"].detach())
            tensordict.set(
                NOD_OPINION_ACTIVE_KEY,
                online.get("opinion_active", online["edge_mask"]).detach(),
            )
            ready = torch.ones(
                *observation.shape[:-1],
                1,
                device=observation.device,
                dtype=torch.bool,
            )
            tensordict.set(NOD_ACTOR_CONTEXT_READY_KEY, ready)

        edge_context = tensordict.get(NOD_ACTOR_EDGE_CONTEXT_KEY).detach()
        if self.opinion_mode == "neutral":
            # Preserve hidden state, physical risk features and attention while
            # removing only z, the final context coordinate. Keep the cached
            # online context intact for diagnostics and controlled ablations.
            edge_context = edge_context.clone()
            edge_context[..., -1] = 0.0
        edge_mask = tensordict.get(NOD_ACTOR_EDGE_MASK_KEY).detach().bool()
        message, attention = self.aggregator(edge_context, edge_mask)
        previous_action = self._previous_action(tensordict, observation)
        actor_observation = torch.cat(
            [
                observation[..., : self.base_observation_dim],
                message,
                previous_action,
            ],
            dim=-1,
        )
        tensordict.set(NOD_ACTOR_MESSAGE_KEY, message)
        tensordict.set(NOD_ACTOR_ATTENTION_KEY, attention)
        tensordict.set(NOD_ACTOR_OBSERVATION_KEY, actor_observation)
        return tensordict
