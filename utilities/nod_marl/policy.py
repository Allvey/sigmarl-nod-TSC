"""Opinion-conditioned actor input and permutation-invariant edge aggregation."""

from __future__ import annotations

import weakref
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from tensordict.nn import TensorDictModuleBase

from .opinion import kl_proximal_update, control_attentions


NOD_ACTOR_OBSERVATION_KEY = ("agents", "info", "nod_actor_observation")
NOD_ACTOR_EDGE_CONTEXT_KEY = ("agents", "info", "nod_actor_edge_context")
NOD_ACTOR_EDGE_MASK_KEY = ("agents", "info", "nod_actor_edge_mask")
NOD_OPINION_ACTIVE_KEY = ("agents", "info", "nod_opinion_active")
NOD_FUTURE_LOWER_KEY = ("agents", "info", "nod_future_lower_bound")
NOD_FUTURE_LOWER_VALID_KEY = ("agents", "info", "nod_future_lower_valid")
NOD_ACTOR_CONTEXT_READY_KEY = ("agents", "info", "nod_actor_context_ready")
NOD_ACTOR_MESSAGE_KEY = ("agents", "info", "nod_actor_message")
NOD_ACTOR_ATTENTION_KEY = ("agents", "info", "nod_actor_message_attention")
NOD_EVIDENCE_GATE_CONTEXT_KEY = ("agents", "info", "nod_evidence_gate_context")
NOD_EVIDENCE_WEIGHT_KEY = ("agents", "info", "nod_evidence_weight")


def replay_evidence_sequence(gate_context, collected_z, continuation, gate, solver_kwargs):
    """Differentiate through consecutive opinions; freeze forecast/history.

    Inputs are [window,time,agent,neighbor,...]. Continuation is false at
    identity changes, terminal boundaries, inactive gaps and window starts.
    Only the scalar opinion state and gate remain on the gradient graph.
    """
    if gate_context.ndim != 5 or collected_z.shape != gate_context.shape[:-1]:
        raise ValueError("Gate replay requires [B,T,N,K,D] context and aligned opinions")
    if continuation.shape != collected_z.shape:
        raise ValueError("Gate continuation mask must match opinions")
    gate_context, collected_z = gate_context.detach(), collected_z.detach()
    opinions, weights = [], []
    previous = gate_context[:, 0, ..., -3]
    for t in range(gate_context.shape[1]):
        cached = gate_context[:, t]
        if t:
            previous = torch.where(continuation[:, t], previous, cached[..., -3])
        current = torch.cat([
            cached[..., :-3], previous.unsqueeze(-1), cached[..., -2:]
        ], dim=-1)
        weight = gate(current)
        attention, evidence_attention = control_attentions(
            cached[..., -5], weight, cached[..., -1],
            getattr(gate, "target", "evidence"),
        )
        solved = kl_proximal_update(
            previous, attention, cached[..., -4],
            torch.zeros_like(weight), torch.ones_like(weight),
            cached[..., -2].clamp_min(1e-8), evidence_attention=evidence_attention,
            **solver_kwargs,
        )
        previous = torch.where(cached[..., -1].bool(), solved, collected_z[:, t])
        opinions.append(previous)
        weights.append(weight)
    return torch.stack(opinions, dim=1), torch.stack(weights, dim=1)


class NODEvidenceGate(nn.Module):
    """PPO-owned weight; the supervised forecast remains frozen.

    The default split beta modulates only evidence. Target=attention instead
    supplies the original shared A for both self-amplification and evidence.
    Both start at the frozen risk prior via a zero bounded logit residual.
    """

    def __init__(self, context_dim: int, hidden_dim: int, max_logit_shift: float,
                 target: str = "evidence"):
        super().__init__()
        if target not in {"evidence", "attention"}:
            raise ValueError("Control weight target must be evidence or attention")
        self.target = target
        if target == "attention":
            # Different semantics despite identical MLP shapes: strict policy
            # loading must reject exchanging beta and unified-A checkpoints.
            self.register_buffer("unified_attention_contract", torch.ones((), dtype=torch.uint8))
        self.max_logit_shift = float(max_logit_shift)
        self.net = nn.Sequential(
            nn.Linear(context_dim - 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: Tensor) -> Tensor:
        attention = context[..., -5].clamp(1e-6, 1.0 - 1e-6)
        shift = self.max_logit_shift * torch.tanh(self.net(context[..., :-2]).squeeze(-1))
        weight = torch.sigmoid(torch.logit(attention) + shift)
        return torch.where(context[..., -1].bool(), weight, torch.zeros_like(weight))


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
    re-runs the aggregator. The optional PPO evidence gate also reconstructs
    opinions from detached causal inputs. By default previous z is held fixed;
    ordered-window replay instead differentiates through z within the window,
    cutting gradients at its start and at collected interaction/reset boundaries.
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
        evidence_gate_mode: str = "fixed",
        evidence_gate_hidden_dim: int = 32,
        evidence_gate_max_logit_shift: float = 2.0,
        evidence_gate_sequence_length: int = 1,
        evidence_gate_only_iters: int = 0,
        evidence_gate_target: str = "evidence",
    ):
        super().__init__()
        self.observation_key = observation_key
        self.base_observation_dim = int(base_observation_dim)
        self.message_dim = int(message_dim)
        self.action_dim = int(action_dim)
        if opinion_mode not in {"online", "neutral"}:
            raise ValueError("opinion_mode must be online or neutral")
        self.opinion_mode = opinion_mode
        if evidence_gate_mode not in {"fixed", "ppo"}:
            raise ValueError("evidence_gate_mode must be fixed or ppo")
        self.evidence_gate_mode = evidence_gate_mode
        if evidence_gate_target not in {"evidence", "attention"}:
            raise ValueError("Control weight target must be evidence or attention")
        if evidence_gate_target == "attention" and evidence_gate_mode != "ppo":
            raise ValueError("Unified learned attention requires PPO mode")
        self.evidence_gate_target = evidence_gate_target
        self.evidence_gate_sequence_length = int(evidence_gate_sequence_length)
        self.context_dim = int(nod_manager.online_context_dim)
        self.in_keys = [observation_key, NOD_ACTOR_EDGE_CONTEXT_KEY]
        self.out_keys = [
            NOD_ACTOR_OBSERVATION_KEY,
            NOD_ACTOR_EDGE_CONTEXT_KEY,
            NOD_ACTOR_EDGE_MASK_KEY,
            NOD_OPINION_ACTIVE_KEY,
            NOD_FUTURE_LOWER_KEY,
            NOD_FUTURE_LOWER_VALID_KEY,
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
        if evidence_gate_mode == "ppo":
            if opinion_mode != "online" or nod_manager.evidence_target != "future":
                raise ValueError("PPO evidence gate requires online future evidence")
            if not getattr(nod_manager.parameters, "nod_freeze_training", False):
                raise ValueError("PPO evidence gate requires a frozen NOD predictor")
            # A separate actor submodule avoids registering the frozen NOD
            # encoder/head in the PPO optimizer or its functional parameters.
            # This optional CPU-initialized module must not advance the shared
            # policy/action RNG relative to the existing fixed-weight branch.
            with torch.random.fork_rng(devices=[]):
                self.evidence_gate = NODEvidenceGate(
                    self.context_dim + 3, int(evidence_gate_hidden_dim),
                    float(evidence_gate_max_logit_shift),
                    target=evidence_gate_target,
                )
            if evidence_gate_sequence_length > 1 or evidence_gate_only_iters > 0:
                # Persist phase progress with the corresponding policy weights,
                # including when a best intermediate checkpoint is selected.
                self.evidence_gate.register_buffer("training_batches", torch.zeros((), dtype=torch.long))
            model = nod_manager.model
            self._gate_solver_kwargs = dict(
                bifurcation_gain=model.bifurcation_gain,
                observation_weight=model.observation_weight,
                kl_weight=model.kl_weight, eps=model.z_epsilon,
                max_iterations=model.root_iterations, tolerance=model.root_tolerance,
            )
            self.out_keys = list(self.out_keys) + [
                NOD_EVIDENCE_GATE_CONTEXT_KEY, NOD_EVIDENCE_WEIGHT_KEY
            ]
            self.in_keys = list(self.in_keys) + [NOD_EVIDENCE_GATE_CONTEXT_KEY]
            if evidence_gate_sequence_length > 1:
                self.in_keys += [
                    ("agents", "info", key) for key in
                    ("nod_neighbor_indices", "nod_neighbor_generation", "nod_ego_generation")
                ]

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
            "future_lower_bound": observation.new_zeros(
                *leading_shape, k_neighbors
            ),
            "future_lower_valid": torch.zeros(
                *leading_shape, k_neighbors, device=observation.device,
                dtype=torch.bool,
            ),
        }

    @staticmethod
    def _sequence_continuation(tensordict, gate_context, collected_z):
        """Respect collected identities, episode boundaries and memory resets."""
        info = tensordict.get(("agents", "info"))
        ids = info.get("nod_neighbor_indices").long()
        neighbor = info.get("nod_neighbor_generation").long()
        ego = info.get("nod_ego_generation").long()
        if ego.ndim == 4 and ego.shape[-1] == 1:
            ego = ego.squeeze(-1)
        if ids.shape != collected_z.shape or neighbor.shape != ids.shape or ego.shape != ids.shape[:-1]:
            raise ValueError("Gate sequence identities must match cached opinion dimensions")
        valid = gate_context[..., -1].bool()
        same = ((ids[:, 1:] == ids[:, :-1])
                & (neighbor[:, 1:] == neighbor[:, :-1])
                & (ego[:, 1:] == ego[:, :-1]).unsqueeze(-1))
        # Also honor any collected state reset not expressed by generation.
        connected = torch.isclose(
            gate_context[:, 1:, ..., -3], collected_z[:, :-1], atol=5e-6, rtol=1e-5
        )
        continuation = torch.zeros_like(valid)
        continuation[:, 1:] = same & connected & valid[:, 1:] & valid[:, :-1]
        done = tensordict.get(("next", "done"), default=None)
        if done is not None:
            terminal = done.reshape(*collected_z.shape[:2], -1).bool().any(-1)
            continuation[:, 1:] &= ~terminal[:, :-1, None, None]
        return continuation

    def forward(self, tensordict):
        observation = tensordict.get(self.observation_key)
        context_ready = self._ready(tensordict)
        if not context_ready:
            with torch.no_grad():
                if self.evidence_gate_mode == "ppo":
                    online = self.nod_manager.online_step(
                        tensordict, evidence_gate=self.evidence_gate
                    )
                else:
                    online = self.nod_manager.online_step(tensordict)
            if online is None:
                online = self._fallback_context(observation)
                if self.evidence_gate_mode == "ppo":
                    online["gate_context"] = observation.new_zeros(
                        *online["edge_mask"].shape, self.context_dim + 3
                    )
            if self.evidence_gate_mode == "ppo":
                tensordict.set(NOD_EVIDENCE_GATE_CONTEXT_KEY, online["gate_context"].detach())
            tensordict.set(
                NOD_ACTOR_EDGE_CONTEXT_KEY, online["edge_context"].detach()
            )
            tensordict.set(NOD_ACTOR_EDGE_MASK_KEY, online["edge_mask"].detach())
            tensordict.set(
                NOD_OPINION_ACTIVE_KEY,
                online.get("opinion_active", online["edge_mask"]).detach(),
            )
            tensordict.set(
                NOD_FUTURE_LOWER_KEY,
                online.get("future_lower_bound", torch.zeros_like(
                    online["edge_mask"], dtype=observation.dtype
                )).detach(),
            )
            tensordict.set(
                NOD_FUTURE_LOWER_VALID_KEY,
                online.get("future_lower_valid", torch.zeros_like(
                    online["edge_mask"], dtype=torch.bool
                )).detach(),
            )
            ready = torch.ones(
                *observation.shape[:-1],
                1,
                device=observation.device,
                dtype=torch.bool,
            )
            tensordict.set(NOD_ACTOR_CONTEXT_READY_KEY, ready)

        edge_context = tensordict.get(NOD_ACTOR_EDGE_CONTEXT_KEY).detach()
        if self.evidence_gate_mode == "ppo":
            gate_context = tensordict.get(NOD_EVIDENCE_GATE_CONTEXT_KEY, default=None)
            if gate_context is None:
                raise ValueError("PPO evidence gate replay requires cached causal gate inputs")
            gate_context = gate_context.detach()
            sequence_replay = (context_ready and self.evidence_gate_sequence_length > 1
                               and gate_context.ndim == 5)
            if sequence_replay:
                if gate_context.shape[1] > self.evidence_gate_sequence_length:
                    raise ValueError("Gate replay window exceeds configured sequence length")
                continuation = self._sequence_continuation(tensordict, gate_context, edge_context[..., -1])
                replay_z, weight = replay_evidence_sequence(
                    gate_context, edge_context[..., -1], continuation,
                    self.evidence_gate, self._gate_solver_kwargs,
                )
                edge_context = torch.cat([edge_context[..., :-1], replay_z.unsqueeze(-1)], -1)
            else:
                weight = self.evidence_gate(gate_context)
            if context_ready and not sequence_replay:
                attention, evidence_attention = control_attentions(
                    gate_context[..., -5], weight, gate_context[..., -1],
                    self.evidence_gate_target,
                )
                updated_z = kl_proximal_update(
                    gate_context[..., -3], attention, gate_context[..., -4],
                    torch.zeros_like(weight), torch.ones_like(weight),
                    gate_context[..., -2].clamp_min(1e-8),
                    evidence_attention=evidence_attention, **self._gate_solver_kwargs,
                )
                replay_z = torch.where(gate_context[..., -1].bool(), updated_z, edge_context[..., -1])
                # Keep collected context intact for detached safety advantages
                # and diagnostics. Only the current actor input is reconstructed.
                edge_context = torch.cat([edge_context[..., :-1], replay_z.unsqueeze(-1)], -1)
            if self.evidence_gate_target == "attention":
                attention, _ = control_attentions(
                    gate_context[..., -5], weight, gate_context[..., -1], "attention"
                )
                # Actor sees the same current A used in both opinion terms.
                # Preserve the rollout cache for fixed Safety advantages.
                edge_context = torch.cat([
                    edge_context[..., :-2], attention.unsqueeze(-1), edge_context[..., -1:]
                ], -1)
            tensordict.set(NOD_EVIDENCE_WEIGHT_KEY, weight)
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
