"""History encoder and KL-geometry-preserving directed opinion dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .interaction import (
    APPROACH_CONFIDENCE,
    CLOSING_SPEED,
    CONFLICT_VALID,
    DISTANCE,
    ETA_GAP,
    OVERLAP_RISK,
    TTC,
)


def _atanh_safe(value: Tensor, eps: float) -> Tensor:
    return torch.atanh(value.clamp(-1.0 + eps, 1.0 - eps))


def _stationarity(
    z: Tensor,
    z_previous: Tensor,
    attention: Tensor,
    evidence: Tensor,
    mean_intercept: Tensor,
    slope: Tensor,
    variance: Tensor,
    *,
    bifurcation_gain: float,
    observation_weight: float,
    kl_weight: float,
    eps: float,
) -> Tensor:
    mean = mean_intercept + slope * z
    likelihood_gradient = (
        float(observation_weight)
        * attention
        * (mean - evidence)
        * slope
        / variance
    )
    return (
        float(kl_weight)
        * (_atanh_safe(z, eps) - _atanh_safe(z_previous, eps))
        + z
        - attention * torch.tanh(float(bifurcation_gain) * z)
        + likelihood_gradient
    )


class _ImplicitKLProximalSolve(torch.autograd.Function):
    """Safeguarded scalar solve with implicit, rather than unrolled, gradients."""

    @staticmethod
    def forward(
        ctx,
        z_previous: Tensor,
        attention: Tensor,
        evidence: Tensor,
        mean_intercept: Tensor,
        slope: Tensor,
        variance: Tensor,
        bifurcation_gain: float,
        observation_weight: float,
        kl_weight: float,
        eps: float,
        max_iterations: int,
        tolerance: float,
    ) -> Tensor:
        variance = variance.clamp_min(1e-8)
        lower = torch.full_like(z_previous, -1.0 + float(eps))
        upper = torch.full_like(z_previous, 1.0 - float(eps))

        def stationarity(value: Tensor) -> Tensor:
            return _stationarity(
                value,
                z_previous,
                attention,
                evidence,
                mean_intercept,
                slope,
                variance,
                bifurcation_gain=bifurcation_gain,
                observation_weight=observation_weight,
                kl_weight=kl_weight,
                eps=eps,
            )

        with torch.no_grad():
            gradient_lower = stationarity(lower)
            gradient_upper = stationarity(upper)
            boundary_lower = gradient_lower >= 0.0
            boundary_upper = gradient_upper <= 0.0
            boundary = boundary_lower | boundary_upper
            lo = lower.clone()
            hi = upper.clone()
            for _ in range(int(max_iterations)):
                middle = 0.5 * (lo + hi)
                gradient_middle = stationarity(middle)
                go_left = gradient_middle > 0.0
                hi = torch.where(go_left, middle, hi)
                lo = torch.where(go_left, lo, middle)
                if bool((gradient_middle.abs() <= float(tolerance)).all()):
                    break
            root = 0.5 * (lo + hi)
            root = torch.where(boundary_lower, lower, root)
            root = torch.where(boundary_upper, upper, root)

        ctx.save_for_backward(
            root,
            z_previous,
            attention,
            evidence,
            mean_intercept,
            slope,
            variance,
            boundary,
        )
        ctx.bifurcation_gain = float(bifurcation_gain)
        ctx.observation_weight = float(observation_weight)
        ctx.kl_weight = float(kl_weight)
        ctx.eps = float(eps)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            z,
            z_previous,
            attention,
            evidence,
            mean_intercept,
            slope,
            variance,
            boundary,
        ) = ctx.saved_tensors
        a = ctx.bifurcation_gain
        obs_weight = ctx.observation_weight
        kl_weight = ctx.kl_weight
        eps = ctx.eps

        mean = mean_intercept + slope * z
        sech_squared = 1.0 - torch.tanh(a * z).square()
        curvature = (
            kl_weight / (1.0 - z.square()).clamp_min(eps)
            + 1.0
            - attention * a * sech_squared
            + obs_weight * attention * slope.square() / variance
        ).clamp_min(1e-8)
        scale = -grad_output / curvature
        scale = torch.where(boundary, torch.zeros_like(scale), scale)

        derivative_z_previous = -kl_weight / (
            1.0 - z_previous.square()
        ).clamp_min(eps)
        derivative_attention = -torch.tanh(a * z) + (
            obs_weight * (mean - evidence) * slope / variance
        )
        derivative_evidence = -obs_weight * attention * slope / variance
        derivative_mean_intercept = obs_weight * attention * slope / variance
        derivative_slope = (
            obs_weight
            * attention
            * ((mean - evidence) + slope * z)
            / variance
        )
        derivative_variance = (
            -obs_weight
            * attention
            * (mean - evidence)
            * slope
            / variance.square()
        )

        return (
            scale * derivative_z_previous,
            scale * derivative_attention,
            scale * derivative_evidence,
            scale * derivative_mean_intercept,
            scale * derivative_slope,
            scale * derivative_variance,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def kl_proximal_update(
    z_previous: Tensor,
    attention: Tensor,
    evidence: Tensor,
    mean_intercept: Tensor,
    slope: Tensor,
    variance: Tensor,
    *,
    bifurcation_gain: float,
    observation_weight: float,
    kl_weight: float,
    eps: float = 1e-4,
    max_iterations: int = 48,
    tolerance: float = 1e-7,
) -> Tensor:
    """Solve one bounded Bernoulli-KL proximal opinion update."""

    return _ImplicitKLProximalSolve.apply(
        z_previous,
        attention,
        evidence,
        mean_intercept,
        slope,
        variance,
        float(bifurcation_gain),
        float(observation_weight),
        float(kl_weight),
        float(eps),
        int(max_iterations),
        float(tolerance),
    )


def kl_objective(
    z: Tensor,
    z_previous: Tensor,
    attention: Tensor,
    evidence: Tensor,
    mean_intercept: Tensor,
    slope: Tensor,
    variance: Tensor,
    *,
    bifurcation_gain: float,
    observation_weight: float,
    kl_weight: float,
    eps: float = 1e-6,
) -> Tensor:
    """The scalar objective minimized by :func:`kl_proximal_update`."""

    a = float(bifurcation_gain)
    prior = 0.5 * z.square() - attention / a * torch.log(torch.cosh(a * z))
    mean = mean_intercept + slope * z
    likelihood = 0.5 * float(observation_weight) * attention * (
        (evidence - mean).square() / variance + torch.log(variance)
    )
    probability = ((1.0 + z) * 0.5).clamp(eps, 1.0 - eps)
    probability_previous = ((1.0 + z_previous) * 0.5).clamp(eps, 1.0 - eps)
    kl = probability * torch.log(probability / probability_previous) + (
        1.0 - probability
    ) * torch.log((1.0 - probability) / (1.0 - probability_previous))
    return prior + likelihood + float(kl_weight) * kl


class NODOpinionModel(nn.Module):
    """Topology-conditioned edge history and bounded directed opinions."""

    evidence_dim = 5
    risk_dim = 6

    def __init__(
        self,
        pair_feature_dim: int,
        hidden_dim: int = 64,
        *,
        relation_feature_dim: Optional[int] = None,
        action_dim: int = 0,
        history_mode: str = "gru",
        bifurcation_gain: float = 2.0,
        observation_weight: float = 1.0,
        kl_weight: float = 5.0,
        z_epsilon: float = 1e-4,
        root_iterations: int = 48,
        root_tolerance: float = 1e-7,
        retention_steps: int = 2,
        risk_temperature: float = 2.0,
        risk_threshold: float = 1.25,
        min_log_sigma: float = -2.5,
        max_log_sigma: float = 0.0,
    ):
        super().__init__()
        self.pair_feature_dim = int(pair_feature_dim)
        self.relation_feature_dim = int(
            pair_feature_dim
            if relation_feature_dim is None
            else relation_feature_dim
        )
        self.action_dim = int(action_dim)
        self.history_mode = str(history_mode).lower()
        if self.history_mode not in {"gru", "none"}:
            raise ValueError("history_mode must be one of {'gru', 'none'}")
        self.hidden_dim = int(hidden_dim)
        self.bifurcation_gain = float(bifurcation_gain)
        self.observation_weight = float(observation_weight)
        self.kl_weight = float(kl_weight)
        self.z_epsilon = float(z_epsilon)
        self.root_iterations = int(root_iterations)
        self.root_tolerance = float(root_tolerance)
        self.retention_steps = int(retention_steps)
        self.risk_temperature = float(risk_temperature)
        self.risk_threshold = float(risk_threshold)
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

        # The current evidence is deliberately excluded from the recurrent
        # input.  Otherwise the likelihood head could learn to copy the target
        # it is meant to explain.  In the integrated path this input is the
        # shared topology relation latent plus the neighbor-action prediction.
        self.history = nn.GRUCell(
            self.relation_feature_dim + self.action_dim, self.hidden_dim
        )
        self.likelihood_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 3),
        )

        initial_risk_weights = torch.tensor(
            [0.75, 0.75, 0.50, 0.50, 0.50, 0.75], dtype=torch.float32
        )
        self.raw_risk_weights = nn.Parameter(
            torch.log(torch.expm1(initial_risk_weights))
        )

    @property
    def risk_weights(self) -> Tensor:
        """Nonnegative weights make attention monotone in every risk input."""

        return F.softplus(self.raw_risk_weights)

    @staticmethod
    def risk_components(pair_features: Tensor) -> Tensor:
        conflict = pair_features[..., CONFLICT_VALID].clamp(0.0, 1.0)
        inverse_ttc = 1.0 - pair_features[..., TTC].clamp(0.0, 1.0)
        proximity = 1.0 - pair_features[..., DISTANCE].clamp(0.0, 1.0)
        overlap = pair_features[..., OVERLAP_RISK].clamp(0.0, 1.0)
        approaching = pair_features[..., APPROACH_CONFIDENCE].clamp(0.0, 1.0)
        eta_urgency = conflict * (
            1.0 - pair_features[..., ETA_GAP].abs().clamp(0.0, 1.0)
        )
        return torch.stack(
            [
                proximity,
                approaching,
                inverse_ttc,
                conflict,
                eta_urgency,
                overlap,
            ],
            dim=-1,
        )

    def risk_score(self, pair_features: Tensor) -> Tensor:
        return (self.risk_components(pair_features) * self.risk_weights).sum(dim=-1)

    def risk_attention(
        self, pair_features: Tensor, evidence_components: Optional[Tensor] = None
    ) -> Tensor:
        """Monotone learnable attention from current physical risk only.

        ``evidence_components`` is accepted for source compatibility but is
        intentionally ignored, preventing the current likelihood target from
        leaking into the attention or recurrent encoder.
        """

        risk_score = self.risk_score(pair_features)
        return torch.sigmoid(
            self.risk_temperature * (risk_score - self.risk_threshold)
        )

    @staticmethod
    def evidence_components(current: Tensor, previous: Tensor) -> Tensor:
        """Positive components consistently mean observed risk mitigation."""

        ttc_relief = current[..., TTC] - previous[..., TTC]
        eta_separation = current[..., ETA_GAP].abs() - previous[..., ETA_GAP].abs()
        overlap_relief = previous[..., OVERLAP_RISK] - current[..., OVERLAP_RISK]
        closing_relief = previous[..., CLOSING_SPEED].clamp_min(0.0) - current[
            ..., CLOSING_SPEED
        ].clamp_min(0.0)
        separation = current[..., DISTANCE] - previous[..., DISTANCE]
        return torch.stack(
            [ttc_relief, eta_separation, overlap_relief, closing_relief, separation],
            dim=-1,
        ).clamp(-1.0, 1.0)

    def likelihood_parameters(self, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raw = self.likelihood_head(hidden)
        mean_intercept = torch.tanh(raw[..., 0])
        slope = 0.05 + 0.95 * torch.sigmoid(raw[..., 1])
        log_sigma = self.min_log_sigma + (
            self.max_log_sigma - self.min_log_sigma
        ) * torch.sigmoid(raw[..., 2])
        variance = torch.exp(2.0 * log_sigma)
        return mean_intercept, slope, variance

    def _empty_state(
        self, pair_features: Tensor
    ) -> Dict[str, Tensor]:
        batch, _, n_agents, k_neighbors, _ = pair_features.shape
        shape = (batch, n_agents, k_neighbors)
        return {
            "hidden": pair_features.new_zeros(*shape, self.hidden_dim),
            "z": pair_features.new_zeros(*shape),
            "pair": pair_features.new_zeros(*shape, self.pair_feature_dim),
            "ego_generation": torch.zeros(
                shape, device=pair_features.device, dtype=torch.long
            ),
            "neighbor_generation": torch.zeros(
                shape, device=pair_features.device, dtype=torch.long
            ),
            "age": torch.full(
                shape,
                self.retention_steps + 1,
                device=pair_features.device,
                dtype=torch.long,
            ),
            "has_state": torch.zeros(
                shape, device=pair_features.device, dtype=torch.bool
            ),
        }

    def forward_sequence(
        self,
        pair_features: Tensor,
        edge_mask: Tensor,
        ego_generations: Tensor,
        neighbor_generations: Tensor,
        initial_state: Optional[Dict[str, Tensor]] = None,
        *,
        relation_features: Optional[Tensor] = None,
        predicted_actions: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Unroll edge state with identity-aware reset and short-gap retention."""

        if pair_features.ndim != 5:
            raise ValueError("pair_features must have shape [B,T,N,K,D]")
        if pair_features.shape[-1] != self.pair_feature_dim:
            raise ValueError(
                f"Expected pair feature dim {self.pair_feature_dim}, "
                f"got {pair_features.shape[-1]}"
            )
        if relation_features is None:
            if self.relation_feature_dim != self.pair_feature_dim:
                raise ValueError(
                    "relation_features are required when relation_feature_dim "
                    "differs from pair_feature_dim"
                )
            relation_features = pair_features
        if relation_features.shape[:-1] != pair_features.shape[:-1]:
            raise ValueError("relation_features must share [B,T,N,K] dimensions")
        if relation_features.shape[-1] != self.relation_feature_dim:
            raise ValueError(
                f"Expected relation feature dim {self.relation_feature_dim}, "
                f"got {relation_features.shape[-1]}"
            )
        if predicted_actions is None:
            predicted_actions = pair_features.new_zeros(
                *pair_features.shape[:-1], self.action_dim
            )
        if predicted_actions.shape[:-1] != pair_features.shape[:-1]:
            raise ValueError("predicted_actions must share [B,T,N,K] dimensions")
        if predicted_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected predicted action dim {self.action_dim}, "
                f"got {predicted_actions.shape[-1]}"
            )
        edge_mask = edge_mask.bool()
        state = (
            self._empty_state(pair_features)
            if initial_state is None
            else initial_state
        )
        hidden = state["hidden"]
        z_state = state["z"]
        previous_pair = state["pair"]
        previous_ego_generation = state["ego_generation"]
        previous_neighbor_generation = state["neighbor_generation"]
        age = state["age"]
        has_state = state["has_state"]

        output_lists = {
            key: []
            for key in (
                "z",
                "rho",
                "attention",
                "risk_score",
                "evidence",
                "mean",
                "variance",
                "learning_valid",
                "root_residual",
                "curvature",
                "boundary",
            )
        }
        event_counts = {
            "new_edges": pair_features.new_zeros(()),
            "resumed_edges": pair_features.new_zeros(()),
            "expired_edges": pair_features.new_zeros(()),
        }

        for time_index in range(pair_features.shape[1]):
            current = pair_features[:, time_index]
            current_relation = relation_features[:, time_index]
            current_action = predicted_actions[:, time_index]
            active = edge_mask[:, time_index]
            ego_generation = (
                ego_generations[:, time_index].unsqueeze(-1).expand_as(active)
            )
            neighbor_generation = neighbor_generations[:, time_index]
            identity_same = (
                has_state
                & (previous_ego_generation == ego_generation)
                & (previous_neighbor_generation == neighbor_generation)
            )
            retained = identity_same & (age <= self.retention_steps)
            temporal = active & retained & (age == 0)
            resumed = active & retained & (age > 0)
            new_edge = active & ~retained
            expired = has_state & ~active & ~retained

            evidence_vector = self.evidence_components(current, previous_pair)
            evidence_vector = torch.where(
                temporal.unsqueeze(-1),
                evidence_vector,
                torch.zeros_like(evidence_vector),
            )
            evidence = evidence_vector.mean(dim=-1)
            history_input = torch.cat(
                [current_relation, current_action], dim=-1
            )
            history_previous = (
                torch.where(
                    retained.unsqueeze(-1), hidden, torch.zeros_like(hidden)
                )
                if self.history_mode == "gru"
                else torch.zeros_like(hidden)
            )
            hidden_candidate = self.history(
                history_input.reshape(-1, history_input.shape[-1]),
                history_previous.reshape(-1, self.hidden_dim),
            ).reshape_as(hidden)

            keep_inactive = ~active & retained
            hidden = torch.where(
                active.unsqueeze(-1),
                hidden_candidate,
                torch.where(
                    keep_inactive.unsqueeze(-1), hidden, torch.zeros_like(hidden)
                ),
            )
            mean_intercept, slope, variance = self.likelihood_parameters(hidden)
            risk_score = self.risk_score(current)
            attention = torch.sigmoid(
                self.risk_temperature * (risk_score - self.risk_threshold)
            )
            z_before_update = z_state
            solved_z = kl_proximal_update(
                z_before_update,
                attention,
                evidence,
                mean_intercept,
                slope,
                variance,
                bifurcation_gain=self.bifurcation_gain,
                observation_weight=self.observation_weight,
                kl_weight=self.kl_weight,
                eps=self.z_epsilon,
                max_iterations=self.root_iterations,
                tolerance=self.root_tolerance,
            )
            z_state = torch.where(
                active & retained,
                solved_z,
                torch.where(new_edge, torch.zeros_like(z_state), z_state),
            )
            z_state = torch.where(
                keep_inactive,
                z_state,
                torch.where(active, z_state, torch.zeros_like(z_state)),
            )
            output_z = torch.where(active, z_state, torch.zeros_like(z_state))

            residual = _stationarity(
                output_z,
                torch.where(retained, z_before_update, torch.zeros_like(output_z)),
                attention,
                evidence,
                mean_intercept,
                slope,
                variance,
                bifurcation_gain=self.bifurcation_gain,
                observation_weight=self.observation_weight,
                kl_weight=self.kl_weight,
                eps=self.z_epsilon,
            ).abs()
            sech_squared = 1.0 - torch.tanh(self.bifurcation_gain * output_z).square()
            curvature = (
                self.kl_weight / (1.0 - output_z.square()).clamp_min(self.z_epsilon)
                + 1.0
                - attention * self.bifurcation_gain * sech_squared
                + self.observation_weight * attention * slope.square() / variance
            )
            boundary = output_z.abs() >= (1.0 - 2.0 * self.z_epsilon)
            residual = torch.where(
                active & retained, residual, torch.zeros_like(residual)
            )

            output_lists["z"].append(output_z)
            output_lists["rho"].append(0.5 * (1.0 + output_z))
            output_lists["attention"].append(attention)
            output_lists["risk_score"].append(risk_score)
            output_lists["evidence"].append(evidence)
            output_lists["mean"].append(mean_intercept + slope * output_z)
            output_lists["variance"].append(variance)
            output_lists["learning_valid"].append(active & retained)
            output_lists["root_residual"].append(residual)
            output_lists["curvature"].append(curvature)
            output_lists["boundary"].append(boundary & active & retained)

            event_counts["new_edges"] += new_edge.sum().to(pair_features.dtype)
            event_counts["resumed_edges"] += resumed.sum().to(pair_features.dtype)
            event_counts["expired_edges"] += expired.sum().to(pair_features.dtype)

            has_state = active | keep_inactive
            age = torch.where(
                active,
                torch.zeros_like(age),
                torch.where(
                    has_state,
                    age + 1,
                    torch.full_like(age, self.retention_steps + 1),
                ),
            )
            previous_pair = torch.where(
                active.unsqueeze(-1),
                current,
                torch.where(
                    keep_inactive.unsqueeze(-1),
                    previous_pair,
                    torch.zeros_like(previous_pair),
                ),
            )
            previous_ego_generation = torch.where(
                active, ego_generation, previous_ego_generation
            )
            previous_neighbor_generation = torch.where(
                active, neighbor_generation, previous_neighbor_generation
            )

        outputs = {
            key: torch.stack(value, dim=1) for key, value in output_lists.items()
        }
        final_state = {
            "hidden": hidden,
            "z": z_state,
            "pair": previous_pair,
            "ego_generation": previous_ego_generation,
            "neighbor_generation": previous_neighbor_generation,
            "age": age,
            "has_state": has_state,
        }
        outputs.update(event_counts)
        return outputs, final_state
