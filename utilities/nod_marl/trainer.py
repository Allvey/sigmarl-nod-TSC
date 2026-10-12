"""Independent sequence training for observation-derived NOD opinions."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .counterfactual import (
    build_counterfactual_labels,
    build_responsibility_evidence,
)
from .interaction import NOD_PAIR_FEATURE_DIM, nod_observation_inputs
from .opinion import NODOpinionModel


def _detach_state(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {key: value.detach() for key, value in state.items()}


def _correlation(first: Tensor, second: Tensor) -> float:
    if first.numel() < 2:
        return 0.0
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator) <= 1e-8:
        return 0.0
    return float((first * second).sum() / denominator)


def _expected_calibration_error(
    probability: Tensor, soft_target: Tensor, n_bins: int = 10
) -> float:
    if probability.numel() == 0:
        return 0.0
    probability = probability.float().clamp(0.0, 1.0)
    soft_target = soft_target.float().clamp(0.0, 1.0)
    boundaries = torch.linspace(
        0.0, 1.0, n_bins + 1, device=probability.device
    )
    error = probability.new_zeros(())
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (probability >= boundaries[index]) & (
                probability <= boundaries[index + 1]
            )
        else:
            selected = (probability >= boundaries[index]) & (
                probability < boundaries[index + 1]
            )
        if bool(selected.any()):
            error = error + selected.float().mean() * (
                probability[selected].mean() - soft_target[selected].mean()
            ).abs()
    return float(error)


class NODOpinionManager:
    """Owns NOD state/parameters without sharing gradients or RNG with PPO."""

    checkpoint_version = 3

    def __init__(
        self,
        parameters,
        pair_feature_dim: int = NOD_PAIR_FEATURE_DIM,
    ):
        self.parameters = parameters
        self.observation_mode = getattr(parameters, "nod_observation_mode", "legacy_paths")
        self.label_mode = getattr(parameters, "nod_label_mode", "instantaneous")
        self.predict_future_mitigation = self.label_mode == "future_mitigation"
        self.predict_future_quantile = bool(
            getattr(parameters, "nod_predict_future_quantile", False)
        )
        if self.predict_future_quantile and not self.predict_future_mitigation:
            raise ValueError("Future quantile requires future_mitigation mode")
        self.variance_mode = str(getattr(parameters, "nod_variance_mode", "learned"))
        self.evidence_mode = str(getattr(parameters, "nod_evidence_mode", "geometric"))
        self.variance_supervision = str(
            getattr(parameters, "nod_variance_supervision", "posterior")
        )
        if self.variance_supervision == "evidence_error" and self.label_mode != "responsibility":
            raise ValueError("Evidence-error supervision requires responsibility labels")
        if self.variance_mode == "fixed" and self.label_mode != "responsibility":
            raise ValueError("Fixed variance is only supported in responsibility mode")
        self.checkpoint_version = (
            7 if self.variance_supervision == "evidence_error" else
            6 if self.variance_mode == "fixed" else
            5 if self.predict_future_quantile else
            4 if self.predict_future_mitigation else 3
        )
        self._variance_needs_calibration = (
            self.variance_mode == "fixed"
            and getattr(parameters, "nod_fixed_variance_value", None) is None
        )
        self.enabled = bool(getattr(parameters, "is_using_nod_opinion", True))
        self.n_neighbors = max(1, int(getattr(parameters, "n_agents", 2)) - 1)
        dt = float(parameters.dt)
        tau = float(getattr(parameters, "nod_tau", 0.25))
        bifurcation_gain = float(
            getattr(parameters, "nod_bifurcation_gain", 2.0)
        )
        kl_weight = tau / max(dt, 1e-8)
        if self.enabled and kl_weight + 1.0 <= bifurcation_gain:
            raise ValueError(
                "NOD KL subproblem is not guaranteed unique: require "
                "nod_tau / dt + 1 > nod_bifurcation_gain"
            )

        # Module initialization and sequence shuffling must not advance PPO's
        # global random stream. This preserves phase-1--3 behavior isolation.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.model = NODOpinionModel(
                pair_feature_dim=pair_feature_dim,
                history_mode=str(getattr(parameters, "nod_history_mode", "gru")),
                hidden_dim=int(getattr(parameters, "nod_hidden_dim", 64)),
                bifurcation_gain=bifurcation_gain,
                observation_weight=float(
                    getattr(parameters, "nod_observation_weight", 1.0)
                ),
                kl_weight=kl_weight,
                z_epsilon=float(getattr(parameters, "nod_z_epsilon", 1e-4)),
                root_iterations=int(getattr(parameters, "nod_root_iterations", 48)),
                root_tolerance=float(
                    getattr(parameters, "nod_root_tolerance", 1e-7)
                ),
                retention_steps=int(
                    getattr(parameters, "nod_edge_retention_steps", 2)
                ),
                risk_temperature=float(
                    getattr(parameters, "nod_risk_temperature", 2.0)
                ),
                risk_threshold=float(
                    getattr(parameters, "nod_risk_threshold", 1.25)
                ),
                min_log_sigma=float(
                    getattr(parameters, "nod_min_log_sigma", -2.5)
                ),
                max_log_sigma=float(getattr(parameters, "nod_max_log_sigma", 0.0)),
                max_risk_weight=float(
                    getattr(parameters, "nod_max_risk_weight", 1.0)
                ),
                fixed_evidence_mapping=(
                    bool(getattr(parameters, "nod_fixed_evidence_mapping", False))
                    or self.label_mode in {"responsibility", "future_mitigation"}
                ),
                predict_future_mitigation=self.predict_future_mitigation,
                predict_future_quantile=self.predict_future_quantile,
                variance_mode=self.variance_mode,
                fixed_variance_value=getattr(parameters, "nod_fixed_variance_value", None),
                evidence_mode=self.evidence_mode,
                variance_supervision=self.variance_supervision,
            ).to(parameters.device)
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(getattr(parameters, "nod_lr", 1e-3)),
        )
        self._sequence_generator = torch.Generator(device="cpu")
        seed = int(getattr(parameters, "seed", 0) or 0)
        self._sequence_generator.manual_seed(seed ^ 0x4E4F44)

        # Collection advances this causal state one frame at a time. Offline
        # sequence training still rebuilds its own state from rollout start.
        self.online_state: Optional[Dict[str, Tensor]] = None
        self.responsibility_state: Optional[Dict[str, Tensor]] = None
        self.last_metrics: Dict[str, float] = {}
        self.last_load_info = ""

    @staticmethod
    def _get_rollout_tensor(tensordict, name: str) -> Optional[Tensor]:
        return tensordict.get(("agents", "info", name), default=None)

    def reset_online_state(self) -> None:
        self.online_state = None
        self.responsibility_state = None

    @property
    def online_context_dim(self) -> int:
        """Detached per-edge features consumed by the PPO message encoder."""

        # NOD hidden + physical risk vector + risk attention + bounded opinion.
        return self.model.hidden_dim + self.model.risk_dim + 2

    @staticmethod
    def _online_state_matches(
        state: Optional[Dict[str, Tensor]], pair_features: Tensor
    ) -> bool:
        if state is None:
            return False
        expected = pair_features.shape[:-1]
        return (
            tuple(state["z"].shape) == tuple(expected)
            and state["z"].device == pair_features.device
        )

    @torch.no_grad()
    def online_step(self, tensordict):
        """Advance NOD exactly once from the current, causally available frame.

        The returned tensors are detached rollout context. PPO can shuffle them
        freely and train its stateless message aggregator without replaying the
        recurrent NOD dynamics in a different temporal order.
        """

        if not self.enabled:
            return None
        pair_features = self._get_rollout_tensor(tensordict, "nod_pair_features")
        edge_mask = self._get_rollout_tensor(tensordict, "nod_edge_mask")
        neighbor_indices = self._get_rollout_tensor(
            tensordict, "nod_neighbor_indices"
        )
        ego_generations = self._get_rollout_tensor(
            tensordict, "nod_ego_generation"
        )
        neighbor_generations = self._get_rollout_tensor(
            tensordict, "nod_neighbor_generation"
        )
        positions = self._get_rollout_tensor(tensordict, "nod_world_pos")
        velocities = self._get_rollout_tensor(tensordict, "nod_world_vel")
        required = [
            pair_features,
            edge_mask,
            neighbor_indices,
            ego_generations,
            neighbor_generations,
        ]
        if self.label_mode in {"responsibility", "future_mitigation"}:
            required.extend((positions, velocities))
        if any(value is None for value in required):
            return None
        if pair_features.ndim != 4:
            # Online collection is [B,N,K,D]. Sequence/flat PPO data must use
            # the cached context instead of stepping a recurrent state.
            return None

        pair_features, edge_mask = nod_observation_inputs(
            pair_features.detach(), edge_mask.detach(), self.observation_mode)
        neighbor_indices = neighbor_indices.detach().long()
        ego_generations = ego_generations.detach().long()
        if ego_generations.ndim == 3 and ego_generations.shape[-1] == 1:
            ego_generations = ego_generations.squeeze(-1)
        neighbor_generations = neighbor_generations.detach().long()
        if not self._online_state_matches(self.online_state, pair_features):
            self.reset_online_state()

        opinion_mask = evidence_override = None
        if self.label_mode in {"responsibility", "future_mitigation"}:
            responsibility_shape = pair_features.shape[:-1]
            if (
                self.responsibility_state is not None
                and (
                    tuple(self.responsibility_state["held"].shape)
                    != tuple(responsibility_shape)
                    or self.responsibility_state["held"].device
                    != pair_features.device
                )
            ):
                self.responsibility_state = None
            responsibility, responsibility_state = build_responsibility_evidence(
                positions.detach().unsqueeze(1),
                velocities.detach().unsqueeze(1),
                ego_generations.unsqueeze(1),
                neighbor_indices.unsqueeze(1),
                edge_mask.unsqueeze(1),
                dt=float(self.parameters.dt),
                lookahead=(
                    int(getattr(self.parameters, "nod_counterfactual_horizon", 8))
                    * float(self.parameters.dt)
                ),
                safe_distance=float(
                    getattr(self.parameters, "nod_safe_distance", 0.25)
                ),
                max_age=float(
                    getattr(self.parameters, "nod_reference_seconds", 2.0)
                ),
                neighbor_generations=neighbor_generations.unsqueeze(1),
                initial_state=self.responsibility_state,
            )
            self.responsibility_state = _detach_state(responsibility_state)
            opinion_mask = responsibility["active"]
            evidence_override = responsibility["evidence"]

        self.model.eval()
        outputs, state = self.model.forward_sequence(
            pair_features.unsqueeze(1),
            edge_mask.unsqueeze(1),
            ego_generations.unsqueeze(1),
            neighbor_generations.unsqueeze(1),
            initial_state=self.online_state,
            opinion_mask=opinion_mask,
            evidence_override=evidence_override,
        )
        self.online_state = _detach_state(state)
        if self.predict_future_quantile:
            correction = float(getattr(self.parameters, "nod_future_lower_correction", 0.0))
            future_lower_bound = (
                (outputs["future_quantile_raw"][:, 0] - correction)
                .clamp(0.0, 1.0)
                .minimum(outputs["rho"][:, 0])
            )
            future_lower_valid = outputs["learning_valid"][:, 0]
        else:
            future_lower_bound = torch.zeros_like(outputs["rho"][:, 0])
            future_lower_valid = torch.zeros_like(edge_mask)
        edge_context = torch.cat(
            [
                state["hidden"],
                self.model.risk_components(pair_features),
                outputs["attention"][:, 0].unsqueeze(-1),
                outputs["z"][:, 0].unsqueeze(-1),
            ],
            dim=-1,
        )
        return {
            "edge_context": edge_context.detach(),
            # Keep the Actor's physical neighbor graph unchanged. Outside an
            # active conflict z is neutral, but the Actor still sees the
            # neighbor's physical context.
            "edge_mask": edge_mask.detach(),
            "opinion_active": (
                opinion_mask[:, 0].detach()
                if opinion_mask is not None
                else edge_mask.detach()
            ),
            "opinion": outputs["z"][:, 0].detach(),
            "risk_attention": outputs["attention"][:, 0].detach(),
            "future_lower_bound": future_lower_bound.detach(),
            "future_lower_valid": future_lower_valid.detach(),
        }

    def _losses(
        self,
        outputs: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        edge_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dynamics_valid = outputs["learning_valid"] & edge_mask
        calibration_valid = dynamics_valid & labels["valid"]
        if self.variance_supervision == "evidence_error":
            # Neither target depends on the learned filter or its variance.
            # Incomplete future windows and resets are excluded by valid.
            error = (
                outputs["evidence"].detach()
                - (2.0 * labels["label"].detach() - 1.0)
            )
            nll_mask = calibration_valid
        else:
            error = outputs["evidence"] - outputs["mean"]
            nll_mask = dynamics_valid
        nll_element = 0.5 * (
            error.square() / outputs["variance"]
            + torch.log(outputs["variance"])
            + math.log(2.0 * math.pi)
        )
        if bool(nll_mask.any()):
            nll = nll_element[nll_mask].mean()
        else:
            nll = outputs["variance"].sum() * 0.0 + outputs["mean"].sum() * 0.0
        if bool(calibration_valid.any()):
            calibration = F.binary_cross_entropy(
                outputs["rho"][calibration_valid].clamp(1e-6, 1.0 - 1e-6),
                labels["label"][calibration_valid],
            )
            brier = (
                outputs["rho"][calibration_valid]
                - labels["label"][calibration_valid]
            ).square().mean()
        else:
            calibration = outputs["rho"].sum() * 0.0
            brier = outputs["rho"].new_zeros(())
        loss = (
            float(getattr(self.parameters, "nod_nll_weight", 1.0)) * nll
            + float(getattr(self.parameters, "nod_calibration_weight", 1.0))
            * calibration
        )
        if self.evidence_mode == "learned":
            if bool(dynamics_valid.any()):
                evidence_loss = F.smooth_l1_loss(
                    outputs["evidence_used"][dynamics_valid],
                    outputs["evidence"][dynamics_valid].detach(),
                )
            else:
                evidence_loss = outputs["evidence_used"].sum() * 0.0
            loss = loss + float(
                getattr(self.parameters, "nod_evidence_loss_weight", 1.0)
            ) * evidence_loss
        if self.predict_future_quantile:
            if bool(calibration_valid.any()):
                residual = (
                    labels["label"][calibration_valid]
                    - outputs["future_quantile_raw"][calibration_valid]
                )
                level = float(getattr(self.parameters, "nod_quantile_level", 0.1))
                quantile_loss = torch.maximum(
                    level * residual, (level - 1.0) * residual
                ).mean()
            else:
                quantile_loss = outputs["future_quantile_raw"].sum() * 0.0
            loss = loss + float(getattr(self.parameters, "nod_quantile_weight", 1.0)) * quantile_loss
        return loss, nll, calibration, brier, dynamics_valid, calibration_valid

    def train_on_rollout(self, tensordict) -> Dict[str, float]:
        """Train NOD directly on ordered physical interaction sequences."""

        if not self.enabled:
            self.last_metrics = {"enabled": 0.0}
            return self.last_metrics

        pair_features = self._get_rollout_tensor(tensordict, "nod_pair_features")
        edge_mask = self._get_rollout_tensor(tensordict, "nod_edge_mask")
        neighbor_indices = self._get_rollout_tensor(
            tensordict, "nod_neighbor_indices"
        )
        ego_generations = self._get_rollout_tensor(
            tensordict, "nod_ego_generation"
        )
        neighbor_generations = self._get_rollout_tensor(
            tensordict, "nod_neighbor_generation"
        )
        positions = self._get_rollout_tensor(tensordict, "nod_world_pos")
        velocities = self._get_rollout_tensor(tensordict, "nod_world_vel")
        required = (
            pair_features,
            edge_mask,
            neighbor_indices,
            ego_generations,
            neighbor_generations,
            positions,
            velocities,
        )
        if any(value is None for value in required):
            self.last_metrics = {"enabled": 1.0, "missing_rollout_fields": 1.0}
            return self.last_metrics

        pair_features, edge_mask = nod_observation_inputs(
            pair_features.detach(), edge_mask.detach(), self.observation_mode)
        neighbor_indices = neighbor_indices.detach().long()
        ego_generations = ego_generations.detach().long()
        if ego_generations.ndim == 4 and ego_generations.shape[-1] == 1:
            ego_generations = ego_generations.squeeze(-1)
        neighbor_generations = neighbor_generations.detach().long()
        positions = positions.detach()
        velocities = velocities.detach()

        labels = build_counterfactual_labels(
            positions,
            velocities,
            ego_generations,
            neighbor_indices,
            edge_mask,
            horizon=int(getattr(self.parameters, "nod_counterfactual_horizon", 8)),
            dt=float(self.parameters.dt),
            safe_distance=float(getattr(self.parameters, "nod_safe_distance", 0.25)),
            label_slope=float(getattr(self.parameters, "nod_label_slope", 12.0)),
            label_margin=float(getattr(self.parameters, "nod_label_margin", 0.02)),
            mode=self.label_mode,
            reference_seconds=float(getattr(self.parameters, "nod_reference_seconds", 2.0)),
        )
        opinion_mask = (
            labels.get("reference_active")
            if self.label_mode in {"responsibility", "future_mitigation"}
            else None
        )
        evidence_override = (
            labels.get("online_evidence")
            if self.label_mode in {"responsibility", "future_mitigation"}
            else None
        )
        dynamics_mask = opinion_mask if opinion_mask is not None else edge_mask

        if self._variance_needs_calibration:
            # Use only the first frozen-behavior rollout, before any learner
            # update. Both ablation branches therefore share the same scale.
            self.model.eval()
            with torch.no_grad():
                reference, _ = self.model.forward_sequence(
                    pair_features, edge_mask, ego_generations,
                    neighbor_generations, opinion_mask=opinion_mask,
                    evidence_override=evidence_override,
                    reference_variance=True,
                )
                valid = reference["learning_valid"] & dynamics_mask
                if not bool(valid.any()):
                    self.last_metrics = {"enabled": 1.0, "fixed_variance_pending": 1.0}
                    return self.last_metrics
                self.model.fixed_variance.copy_(reference["variance"][valid].median())
            self._variance_needs_calibration = False

        batch_size, time_steps = pair_features.shape[:2]
        sequence_length = max(
            1, int(getattr(self.parameters, "nod_sequence_length", 32))
        )
        sequence_batch_size = max(
            1,
            min(
                batch_size,
                int(
                    getattr(
                        self.parameters,
                        "nod_sequence_minibatch_size",
                        min(8, batch_size),
                    )
                ),
            ),
        )
        num_epochs = max(1, int(getattr(self.parameters, "nod_num_epochs", 4)))
        update_losses = []
        optimizer_updates = 0
        self.model.train()
        for _ in range(num_epochs):
            order_cpu = torch.randperm(
                batch_size, generator=self._sequence_generator
            )
            for batch_start in range(0, batch_size, sequence_batch_size):
                env_ids = order_cpu[
                    batch_start : batch_start + sequence_batch_size
                ].to(pair_features.device)
                state = None
                for time_start in range(0, time_steps, sequence_length):
                    time_slice = slice(
                        time_start, min(time_start + sequence_length, time_steps)
                    )
                    pair_chunk = pair_features.index_select(0, env_ids)[:, time_slice]
                    mask_chunk = edge_mask.index_select(0, env_ids)[:, time_slice]
                    ego_generation_chunk = ego_generations.index_select(
                        0, env_ids
                    )[:, time_slice]
                    neighbor_generation_chunk = neighbor_generations.index_select(
                        0, env_ids
                    )[:, time_slice]
                    label_chunk = {
                        key: value.index_select(0, env_ids)[:, time_slice]
                        for key, value in labels.items()
                    }
                    opinion_chunk = (
                        opinion_mask.index_select(0, env_ids)[:, time_slice]
                        if opinion_mask is not None
                        else None
                    )
                    evidence_chunk = (
                        evidence_override.index_select(0, env_ids)[:, time_slice]
                        if evidence_override is not None
                        else None
                    )
                    outputs, state = self.model.forward_sequence(
                        pair_chunk,
                        mask_chunk,
                        ego_generation_chunk,
                        neighbor_generation_chunk,
                        initial_state=state,
                        opinion_mask=opinion_chunk,
                        evidence_override=evidence_chunk,
                    )
                    loss, _, _, _, dynamics_valid, calibration_valid = self._losses(
                        outputs,
                        label_chunk,
                        opinion_chunk if opinion_chunk is not None else mask_chunk,
                    )
                    state = _detach_state(state)
                    if (self.predict_future_quantile or self.variance_supervision == "evidence_error") and not bool(calibration_valid.any()):
                        continue
                    if not bool(dynamics_valid.any() | calibration_valid.any()):
                        continue
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(getattr(self.parameters, "nod_max_grad_norm", 1.0)),
                    )
                    self.optimizer.step()
                    update_losses.append(float(loss.detach()))
                    optimizer_updates += 1

        # Reconstruct from the beginning using the updated snapshot. This state
        # is diagnostic only and is never reused as online state.
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model.forward_sequence(
                pair_features,
                edge_mask,
                ego_generations,
                neighbor_generations,
                initial_state=None,
                opinion_mask=opinion_mask,
                evidence_override=evidence_override,
            )
            loss, nll, calibration, brier, dynamics_valid, calibration_valid = (
                self._losses(outputs, labels, dynamics_mask)
            )

        edge_count = dynamics_mask.sum().clamp_min(1)
        active_z = outputs["z"][dynamics_mask]
        active_attention = outputs["attention"][dynamics_mask]
        valid_residual = outputs["root_residual"][dynamics_valid]
        valid_curvature = outputs["curvature"][dynamics_valid]
        risk_score = outputs["risk_score"]
        attention_critical = 1.0 / max(self.model.bifurcation_gain, 1e-8)
        high_risk = dynamics_valid & (
            risk_score
            >= float(getattr(self.parameters, "nod_high_risk_score", 1.25))
        )
        low_risk = dynamics_valid & (
            risk_score <= float(getattr(self.parameters, "nod_low_risk_score", 0.50))
        )
        high_attention_ratio = (
            float(
                (outputs["attention"][high_risk] > attention_critical)
                .float()
                .mean()
            )
            if bool(high_risk.any())
            else 0.0
        )
        low_attention_ratio = (
            float(
                (outputs["attention"][low_risk] < attention_critical)
                .float()
                .mean()
            )
            if bool(low_risk.any())
            else 0.0
        )
        root_failure_threshold = max(
            1e-4, 10.0 * float(self.model.root_tolerance)
        )
        non_boundary_valid = dynamics_valid & ~outputs["boundary"]
        root_failure_ratio = (
            float(
                (
                    outputs["root_residual"][non_boundary_valid]
                    > root_failure_threshold
                )
                .float()
                .mean()
            )
            if bool(non_boundary_valid.any())
            else 0.0
        )
        valid_rho = outputs["rho"][calibration_valid]
        valid_label = labels["label"][calibration_valid]
        valid_gap = labels["gap"][calibration_valid]
        new_edge_mask = dynamics_mask & ~outputs["learning_valid"]
        metrics = {
            "enabled": 1.0,
            "loss": float(loss),
            "training_loss_mean": (
                sum(update_losses) / len(update_losses) if update_losses else 0.0
            ),
            "optimizer_updates": float(optimizer_updates),
            "nll": float(nll),
            "variance_mode_fixed": float(self.variance_mode == "fixed"),
            "candidate_init_scratch": float(
                getattr(self.parameters, "nod_candidate_init_mode", "behavior") == "scratch"
            ),
            "fixed_variance": (
                float(self.model.fixed_variance)
                if self.variance_mode == "fixed" else 0.0
            ),
            "calibration_bce": float(calibration),
            "brier": float(brier),
            "ece": _expected_calibration_error(valid_rho, valid_label),
            "z_counterfactual_gap_correlation": _correlation(
                outputs["z"][calibration_valid], valid_gap
            ),
            "edge_count": float(dynamics_mask.sum()),
            "edge_density": float(dynamics_mask.float().mean()),
            "counterfactual_valid_ratio": float(
                calibration_valid.sum().float() / edge_count
            ),
            "z_mean": float(active_z.mean()) if active_z.numel() else 0.0,
            "z_std": float(active_z.std(unbiased=False)) if active_z.numel() else 0.0,
            "positive_opinion_ratio": float((active_z > 0.0).float().mean())
            if active_z.numel()
            else 0.0,
            "attention_mean": float(active_attention.mean())
            if active_attention.numel()
            else 0.0,
            "high_risk_attention_above_critical_ratio": high_attention_ratio,
            "low_risk_attention_below_critical_ratio": low_attention_ratio,
            "attention_weight_min": float(self.model.risk_weights.min()),
            "attention_monotonic_violation_count": float(
                (self.model.risk_weights < 0.0).sum()
            ),
            "root_residual_max": float(valid_residual.max())
            if valid_residual.numel()
            else 0.0,
            "root_failure_ratio": root_failure_ratio,
            "curvature_min": float(valid_curvature.min())
            if valid_curvature.numel()
            else 0.0,
            "boundary_ratio": float(
                outputs["boundary"][dynamics_valid].float().mean()
            )
            if bool(dynamics_valid.any())
            else 0.0,
            "reset_neutral_error_max": float(
                outputs["z"][new_edge_mask].abs().max()
            )
            if bool(new_edge_mask.any())
            else 0.0,
            "new_edges": float(outputs["new_edges"]),
            "resumed_edges": float(outputs["resumed_edges"]),
            "expired_edges": float(outputs["expired_edges"]),
        }
        self.last_metrics = metrics
        if self.variance_supervision == "evidence_error":
            metrics["variance_supervision_evidence_error"] = 1.0
            metrics["variance_error_nll"] = float(nll)
            if bool(calibration_valid.any()):
                variance = outputs["variance"][calibration_valid]
                error_squared = (
                    outputs["evidence"][calibration_valid]
                    - (2.0 * valid_label - 1.0)
                ).square()
                metrics.update(
                    variance_mean=float(variance.mean()),
                    variance_p10=float(torch.quantile(variance, 0.1)),
                    variance_p90=float(torch.quantile(variance, 0.9)),
                    variance_lower_bound_ratio=float(
                        (variance <= math.exp(2 * self.model.min_log_sigma) * 1.01).float().mean()
                    ),
                    variance_upper_bound_ratio=float(
                        (variance >= math.exp(2 * self.model.max_log_sigma) * 0.99).float().mean()
                    ),
                    evidence_future_error_mse=float(error_squared.mean()),
                    variance_error_correlation=_correlation(variance, error_squared),
                    variance_normalized_error_mean=float((error_squared / variance).mean()),
                )
                # Bins are ordered by predicted scale; tied predictions must
                # not be split arbitrarily into a false ranking diagnostic.
                low_cut = torch.quantile(variance, 0.25)
                high_cut = torch.quantile(variance, 0.75)
                if bool(low_cut < high_cut):
                    for name, selected in (
                        ("low", variance <= low_cut), ("high", variance >= high_cut)
                    ):
                        metrics[f"variance_{name}_bin_predicted"] = float(variance[selected].mean())
                        metrics[f"variance_{name}_bin_error_mse"] = float(error_squared[selected].mean())
        if bool(calibration_valid.any()):
            metrics['label_positive_ratio'] = float((valid_label > .5).float().mean())
            metrics['label_neutral_ratio'] = float((valid_label == .5).float().mean())
        if self.evidence_mode == "learned" and bool(dynamics_valid.any()):
            teacher = outputs["evidence"][dynamics_valid]
            predicted = outputs["evidence_used"][dynamics_valid]
            metrics["evidence_mae"] = float((predicted - teacher).abs().mean())
            signed = teacher.abs() > 0.1
            metrics["evidence_sign_accuracy"] = (
                float(((predicted[signed] > 0) == (teacher[signed] > 0)).float().mean())
                if bool(signed.any()) else 0.0
            )
        if self.predict_future_mitigation:
            prediction_error = valid_rho - valid_label
            metrics['future_mitigation_mae'] = (
                float(prediction_error.abs().mean())
                if prediction_error.numel() else 0.0
            )
            metrics['future_mitigation_bias'] = (
                float(prediction_error.mean())
                if prediction_error.numel() else 0.0
            )
            filtered = outputs['filter_z'][dynamics_mask]
            metrics['filter_z_mean'] = float(filtered.mean()) if filtered.numel() else 0.0
        if self.predict_future_quantile:
            raw = outputs["future_quantile_raw"][calibration_valid]
            if raw.numel():
                residual = valid_label - raw
                level = float(getattr(self.parameters, "nod_quantile_level", 0.1))
                metrics["future_quantile_pinball"] = float(
                    torch.maximum(level * residual, (level - 1.0) * residual).mean()
                )
                metrics["future_quantile_raw_coverage"] = float((valid_label >= raw).float().mean())
                metrics["future_quantile_raw_mean"] = float(raw.mean())
        if 'reference_active' in labels:
            metrics['label_reference_active_ratio'] = float(
                labels['reference_active'][edge_mask].float().mean()) if bool(edge_mask.any()) else 0.
        if 'online_responsibility' in labels:
            active_responsibility = labels['online_responsibility'][dynamics_mask]
            metrics['online_responsibility_mean'] = (
                float(active_responsibility.mean())
                if active_responsibility.numel()
                else 0.0
            )
            metrics['online_evidence_positive_ratio'] = (
                float((labels['online_evidence'][dynamics_mask] > 0.0).float().mean())
                if bool(dynamics_mask.any())
                else 0.0
            )
            metrics['fixed_evidence_mapping'] = float(
                self.model.fixed_evidence_mapping
            )
        return metrics

    def checkpoint_state(self) -> Dict:
        return {
            "version": self.checkpoint_version,
            "observation_mode": self.observation_mode,
            "pair_feature_dim": self.model.pair_feature_dim,
            "history_mode": self.model.history_mode,
            "label_mode": self.label_mode,
            "fixed_evidence_mapping": self.model.fixed_evidence_mapping,
            "variance_mode": self.variance_mode,
            "variance_supervision": self.variance_supervision,
            "variance_log_bounds": [self.model.min_log_sigma, self.model.max_log_sigma],
            "evidence_mode": self.evidence_mode,
            "candidate_init_mode": getattr(self.parameters, "nod_candidate_init_mode", "behavior"),
            "fixed_variance_calibrated": not self._variance_needs_calibration,
            "predict_future_mitigation": self.predict_future_mitigation,
            "predict_future_quantile": self.predict_future_quantile,
            "future_quantile_level": (
                float(getattr(self.parameters, "nod_quantile_level", 0.1))
                if self.predict_future_quantile else None
            ),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sequence_generator_state": self._sequence_generator.get_state(),
        }

    def load_checkpoint(self, checkpoint: Dict, load_optimizer: bool = False) -> bool:
        mode = checkpoint.get("observation_mode", "legacy_paths") if isinstance(checkpoint, dict) else "legacy_paths"
        if mode != self.observation_mode:
            self.last_load_info = "incompatible NOD observation mode"
            self.reset_online_state()
            return False
        version = checkpoint.get("version") if isinstance(checkpoint, dict) else None
        if isinstance(checkpoint, dict) and checkpoint.get("label_mode", self.label_mode) != self.label_mode:
            self.last_load_info = "incompatible NOD label mode"
            self.reset_online_state()
            return False
        if isinstance(checkpoint, dict) and (
            checkpoint.get("variance_mode", "learned") != self.variance_mode
            or checkpoint.get("variance_supervision", "posterior") != self.variance_supervision
            or checkpoint.get("evidence_mode", "geometric") != self.evidence_mode
            or (self.variance_supervision == "evidence_error"
                and not checkpoint.get("fixed_evidence_mapping", False))
            or (self.variance_mode == "fixed"
                and (not checkpoint.get("fixed_variance_calibrated", False)
                     or not checkpoint.get("fixed_evidence_mapping", False)))
        ):
            self.last_load_info = "incompatible NOD evidence/variance contract"
            self.reset_online_state()
            return False
        if self.variance_supervision == "evidence_error" and checkpoint.get("variance_log_bounds") != [
            self.model.min_log_sigma, self.model.max_log_sigma
        ]:
            self.last_load_info = "incompatible evidence-error variance bounds"
            self.reset_online_state()
            return False
        state_dict = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if self.variance_mode == "fixed" and "fixed_variance" not in state_dict:
            self.last_load_info = "fixed NOD checkpoint is missing its calibrated variance"
            self.reset_online_state()
            return False
        configured_variance = getattr(self.parameters, "nod_fixed_variance_value", None)
        if (self.variance_mode == "fixed" and configured_variance is not None
                and not math.isclose(
                    float(state_dict["fixed_variance"]), float(configured_variance),
                    rel_tol=1e-6, abs_tol=1e-8,
                )):
            self.last_load_info = "incompatible fixed NOD variance value"
            self.reset_online_state()
            return False
        if version is not None and int(version) < self.checkpoint_version:
            self.last_load_info = (
                "legacy NOD checkpoint ignored because direct physical-pair "
                "history has a different input contract"
            )
            self.reset_online_state()
            return False
        if version is not None and int(version) > self.checkpoint_version:
            self.last_load_info = "incompatible NOD checkpoint version"
            self.reset_online_state()
            return False
        if self.predict_future_quantile and checkpoint.get("future_quantile_level") != float(
            getattr(self.parameters, "nod_quantile_level", 0.1)
        ):
            self.last_load_info = "incompatible future quantile level"
            self.reset_online_state()
            return False
        self.model.load_state_dict(state_dict)
        self._variance_needs_calibration = False
        if (
            load_optimizer
            and isinstance(checkpoint, dict)
            and "optimizer" in checkpoint
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if (
            isinstance(checkpoint, dict)
            and "sequence_generator_state" in checkpoint
        ):
            self._sequence_generator.set_state(checkpoint["sequence_generator_state"])
        self.reset_online_state()
        self.last_load_info = "loaded"
        return True

    def initialize_error_variance_from_behavior(self, checkpoint: Dict) -> None:
        """Keep the old history/attention, restart the independent scale output."""
        if self.variance_supervision != "evidence_error":
            raise ValueError("Evidence-error variance supervision is disabled")
        if (checkpoint.get("version") != 3
                or checkpoint.get("label_mode") != "responsibility"
                or checkpoint.get("variance_mode", "learned") != "learned"
                or checkpoint.get("variance_supervision", "posterior") != "posterior"
                or checkpoint.get("evidence_mode", "geometric") != "geometric"
                or checkpoint.get("observation_mode") != self.observation_mode
                or checkpoint.get("pair_feature_dim") != self.model.pair_feature_dim
                or checkpoint.get("history_mode") != self.model.history_mode
                or not checkpoint.get("fixed_evidence_mapping")):
            raise ValueError("Expected a compatible version-3 responsibility behavior NOD")
        self.model.load_state_dict(checkpoint["model"])
        # Old scale outputs often saturate at their minimum. Initialize the
        # new objective at variance 1, retaining the shared hidden features.
        fraction = -self.model.min_log_sigma / (
            self.model.max_log_sigma - self.model.min_log_sigma
        )
        raw_initial = math.log(fraction / (1.0 - fraction))
        with torch.no_grad():
            self.model.likelihood_head[-1].weight[2].zero_()
            self.model.likelihood_head[-1].bias[2].fill_(raw_initial)
        self.reset_online_state()
        self.last_load_info = "initialized evidence-error variance output at variance 1"

    def initialize_ablation_from_behavior(self, checkpoint: Dict) -> None:
        """Warm-start a fixed-variance learner from a frozen responsibility NOD."""
        if self.variance_mode != "fixed" or self.label_mode != "responsibility":
            raise ValueError("Expected a fixed-variance responsibility learner")
        if (checkpoint.get("version") != 3
                or checkpoint.get("label_mode") != "responsibility"
                or checkpoint.get("variance_mode", "learned") != "learned"
                or checkpoint.get("evidence_mode", "geometric") != "geometric"
                or checkpoint.get("observation_mode") != self.observation_mode
                or checkpoint.get("pair_feature_dim") != self.model.pair_feature_dim
                or checkpoint.get("history_mode") != self.model.history_mode
                or not checkpoint.get("fixed_evidence_mapping")):
            raise ValueError("Expected a compatible version-3 responsibility behavior NOD")
        missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
        expected_missing = {"fixed_variance"}
        if self.evidence_mode == "learned":
            expected_missing.update(
                f"evidence_head.{key}" for key in self.model.evidence_head.state_dict()
            )
        if set(missing) != expected_missing or unexpected:
            raise ValueError("Behavior NOD weights do not match the ablation learner")
        self.reset_online_state()
        self.last_load_info = "initialized ablation learner from behavior NOD"

    def initialize_future_from_behavior(self, checkpoint: Dict) -> None:
        """Reuse a frozen responsibility filter while training a new future head."""
        if not self.predict_future_mitigation or self.predict_future_quantile:
            raise ValueError("Future initialization requires future_mitigation mode")
        if checkpoint.get("version") != 3 or checkpoint.get("label_mode") != "responsibility":
            raise ValueError("Expected a version-3 responsibility behavior NOD")
        if (checkpoint.get("observation_mode") != self.observation_mode
                or checkpoint.get("pair_feature_dim") != self.model.pair_feature_dim
                or checkpoint.get("history_mode") != self.model.history_mode
                or not checkpoint.get("fixed_evidence_mapping")):
            raise ValueError("Behavior NOD uses an incompatible evidence contract")
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model"], strict=False
        )
        expected_missing = set(self.model.future_head.state_dict())
        expected_missing = {f"future_head.{key}" for key in expected_missing}
        if set(missing) != expected_missing or unexpected:
            raise ValueError("Behavior NOD weights do not match the future filter")
        self.reset_online_state()
        self.last_load_info = "initialized future filter from behavior NOD"

    def initialize_quantile_from_future(self, checkpoint: Dict) -> None:
        """Start stage 2 from a stage-1 predictor; train only the new head."""
        if not self.predict_future_quantile:
            raise ValueError("Future quantile head is disabled")
        if (checkpoint.get("version") != 4
                or checkpoint.get("label_mode") != "future_mitigation"
                or checkpoint.get("observation_mode") != self.observation_mode
                or checkpoint.get("pair_feature_dim") != self.model.pair_feature_dim
                or checkpoint.get("history_mode") != self.model.history_mode
                or not checkpoint.get("fixed_evidence_mapping")):
            raise ValueError("Expected a compatible version-4 future predictor")
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model"], strict=False
        )
        expected_missing = {
            f"future_quantile_head.{key}"
            for key in self.model.future_quantile_head.state_dict()
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError("Stage-1 predictor weights do not match stage 2")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith("future_quantile_head."))
        self.reset_online_state()
        self.last_load_info = "initialized frozen future predictor with trainable quantile head"
