"""Independent optimizer and rollout adapter for NOD opinion learning."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from .counterfactual import build_counterfactual_labels
from .interaction import NOD_PAIR_FEATURE_DIM
from .opinion import NODOpinionModel


def _detach_state(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {key: value.detach() for key, value in state.items()}


class NODOpinionManager:
    """Owns all NOD parameters; it never shares an optimizer with PPO."""

    checkpoint_version = 1

    def __init__(self, parameters, pair_feature_dim: int = NOD_PAIR_FEATURE_DIM):
        self.parameters = parameters
        self.enabled = bool(getattr(parameters, "is_using_nod_opinion", True))
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
        # Module initialization must not advance PPO's global random stream.
        # This is essential for exact same-seed behavior isolation.
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.model = NODOpinionModel(
                pair_feature_dim=pair_feature_dim,
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
                risk_threshold=float(getattr(parameters, "nod_risk_threshold", 1.25)),
                min_log_sigma=float(getattr(parameters, "nod_min_log_sigma", -2.5)),
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
        self.runtime_state: Optional[Dict[str, Tensor]] = None
        self.last_metrics: Dict[str, float] = {}

    @staticmethod
    def _get_rollout_tensor(tensordict, name: str) -> Optional[Tensor]:
        return tensordict.get(("agents", "info", name), default=None)

    def _state_matches(self, pair_features: Tensor) -> bool:
        if self.runtime_state is None:
            return False
        expected = (
            pair_features.shape[0],
            pair_features.shape[2],
            pair_features.shape[3],
        )
        return tuple(self.runtime_state["z"].shape) == expected

    def train_on_rollout(self, tensordict) -> Dict[str, float]:
        """Train only the auxiliary opinion model on a detached rollout."""

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
            # VMAS canonicalizes scalar per-agent info to a trailing singleton.
            ego_generations = ego_generations.squeeze(-1)
        neighbor_generations = neighbor_generations.detach().long()
        positions = positions.detach()
        velocities = velocities.detach()

        initial_state = (
            self.runtime_state if self._state_matches(pair_features) else None
        )
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

        self.model.train()
        outputs, final_state = self.model.forward_sequence(
            pair_features,
            edge_mask,
            ego_generations,
            neighbor_generations,
            initial_state=initial_state,
        )
        self.runtime_state = _detach_state(final_state)

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
        self.optimizer.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(getattr(self.parameters, "nod_max_grad_norm", 1.0)),
            )
            self.optimizer.step()

        edge_count = edge_mask.sum().clamp_min(1)
        active_z = outputs["z"][edge_mask]
        active_attention = outputs["attention"][edge_mask]
        valid_residual = outputs["root_residual"][dynamics_valid]
        valid_curvature = outputs["curvature"][dynamics_valid]
        metrics = {
            "enabled": 1.0,
            "loss": float(loss.detach()),
            "nll": float(nll.detach()),
            "calibration_bce": float(calibration.detach()),
            "brier": float(brier.detach()),
            "edge_count": float(edge_mask.sum()),
            "edge_density": float(edge_mask.float().mean()),
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
            "root_residual_max": float(valid_residual.max())
            if valid_residual.numel()
            else 0.0,
            "curvature_min": float(valid_curvature.min())
            if valid_curvature.numel()
            else 0.0,
            "boundary_ratio": float(outputs["boundary"][dynamics_valid].float().mean())
            if bool(dynamics_valid.any())
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
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: Dict, load_optimizer: bool = False) -> None:
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Accept a raw model state dict for early development checkpoints.
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        if (
            load_optimizer
            and isinstance(checkpoint, dict)
            and "optimizer" in checkpoint
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.runtime_state = None
