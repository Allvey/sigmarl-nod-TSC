"""Independent sequence training for topology-conditioned NOD opinions."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .counterfactual import build_counterfactual_labels
from .interaction import NOD_PAIR_FEATURE_DIM
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

    checkpoint_version = 2

    def __init__(
        self,
        parameters,
        pair_feature_dim: int = NOD_PAIR_FEATURE_DIM,
        *,
        relation_feature_dim: Optional[int] = None,
        action_dim: int = 0,
    ):
        self.parameters = parameters
        self.enabled = bool(getattr(parameters, "is_using_nod_opinion", True))
        self.relation_feature_dim = int(
            pair_feature_dim
            if relation_feature_dim is None
            else relation_feature_dim
        )
        self.action_dim = int(action_dim)
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
                relation_feature_dim=self.relation_feature_dim,
                action_dim=self.action_dim,
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

        # Reserved for future policy-time inference. Training always rebuilds
        # recurrent state from raw rollout sequences and never writes here.
        self.online_state: Optional[Dict[str, Tensor]] = None
        self.last_metrics: Dict[str, float] = {}
        self.last_load_info = ""

    @staticmethod
    def _get_rollout_tensor(tensordict, name: str) -> Optional[Tensor]:
        return tensordict.get(("agents", "info", name), default=None)

    def reset_online_state(self) -> None:
        self.online_state = None

    def _losses(
        self,
        outputs: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        edge_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dynamics_valid = outputs["learning_valid"] & edge_mask
        calibration_valid = dynamics_valid & labels["valid"]
        nll_element = 0.5 * (
            (outputs["evidence"] - outputs["mean"]).square()
            / outputs["variance"]
            + torch.log(outputs["variance"])
            + math.log(2.0 * math.pi)
        )
        if bool(dynamics_valid.any()):
            nll = nll_element[dynamics_valid].mean()
        else:
            nll = outputs["mean"].sum() * 0.0
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
        return loss, nll, calibration, brier, dynamics_valid, calibration_valid

    def train_on_rollout(self, tensordict, topology_manager=None) -> Dict[str, float]:
        """Train NOD on ordered rollout sequences with detached shared latents."""

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

        pair_features = pair_features.detach()
        edge_mask = edge_mask.detach().bool()
        neighbor_indices = neighbor_indices.detach().long()
        ego_generations = ego_generations.detach().long()
        if ego_generations.ndim == 4 and ego_generations.shape[-1] == 1:
            ego_generations = ego_generations.squeeze(-1)
        neighbor_generations = neighbor_generations.detach().long()
        positions = positions.detach()
        velocities = velocities.detach()

        encoded = (
            topology_manager.encode_nod_inputs(tensordict, neighbor_indices)
            if topology_manager is not None
            else None
        )
        if encoded is None:
            # Standalone/unit-test fallback. The integrated configuration uses
            # the topology manager and therefore never duplicates this encoder.
            if (
                self.relation_feature_dim != pair_features.shape[-1]
                or self.action_dim != 0
            ):
                self.last_metrics = {
                    "enabled": 1.0,
                    "missing_topology_features": 1.0,
                }
                return self.last_metrics
            relation_features = pair_features
            predicted_actions = pair_features.new_zeros(
                *pair_features.shape[:-1], 0
            )
            representation_available = torch.ones_like(edge_mask)
        else:
            relation_features = encoded["relation_features"].detach()
            predicted_actions = encoded["predicted_actions"].detach()
            representation_available = encoded["available"].detach().bool()
        edge_mask = edge_mask & representation_available

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
        )

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
                    relation_chunk = relation_features.index_select(
                        0, env_ids
                    )[:, time_slice]
                    action_chunk = predicted_actions.index_select(
                        0, env_ids
                    )[:, time_slice]
                    label_chunk = {
                        key: value.index_select(0, env_ids)[:, time_slice]
                        for key, value in labels.items()
                    }
                    outputs, state = self.model.forward_sequence(
                        pair_chunk,
                        mask_chunk,
                        ego_generation_chunk,
                        neighbor_generation_chunk,
                        initial_state=state,
                        relation_features=relation_chunk,
                        predicted_actions=action_chunk,
                    )
                    loss, _, _, _, dynamics_valid, calibration_valid = self._losses(
                        outputs, label_chunk, mask_chunk
                    )
                    state = _detach_state(state)
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
                relation_features=relation_features,
                predicted_actions=predicted_actions,
            )
            loss, nll, calibration, brier, dynamics_valid, calibration_valid = (
                self._losses(outputs, labels, edge_mask)
            )

        edge_count = edge_mask.sum().clamp_min(1)
        active_z = outputs["z"][edge_mask]
        active_attention = outputs["attention"][edge_mask]
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
        new_edge_mask = edge_mask & ~outputs["learning_valid"]
        metrics = {
            "enabled": 1.0,
            "loss": float(loss),
            "training_loss_mean": (
                sum(update_losses) / len(update_losses) if update_losses else 0.0
            ),
            "optimizer_updates": float(optimizer_updates),
            "nll": float(nll),
            "calibration_bce": float(calibration),
            "brier": float(brier),
            "ece": _expected_calibration_error(valid_rho, valid_label),
            "z_counterfactual_gap_correlation": _correlation(
                outputs["z"][calibration_valid], valid_gap
            ),
            "edge_count": float(edge_mask.sum()),
            "edge_density": float(edge_mask.float().mean()),
            "topology_relation_coverage": float(
                representation_available.float().mean()
            ),
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
        return metrics

    def checkpoint_state(self) -> Dict:
        return {
            "version": self.checkpoint_version,
            "pair_feature_dim": self.model.pair_feature_dim,
            "relation_feature_dim": self.model.relation_feature_dim,
            "action_dim": self.model.action_dim,
            "history_mode": self.model.history_mode,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sequence_generator_state": self._sequence_generator.get_state(),
        }

    def load_checkpoint(self, checkpoint: Dict, load_optimizer: bool = False) -> bool:
        version = checkpoint.get("version") if isinstance(checkpoint, dict) else None
        state_dict = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if version is not None and int(version) < self.checkpoint_version:
            self.last_load_info = (
                "legacy NOD checkpoint ignored because topology-conditioned "
                "history has a different input contract"
            )
            self.reset_online_state()
            return False
        self.model.load_state_dict(state_dict)
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
