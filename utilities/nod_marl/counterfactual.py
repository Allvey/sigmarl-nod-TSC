"""Future-only training labels for local, directed cooperation opinions."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _gather_agent(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,N,...]`` using per-ego ``[B,N,K]`` indices."""

    batch, n_agents, k_neighbors = indices.shape
    extra = values.shape[2:]
    expanded_values = values.unsqueeze(1).expand(batch, n_agents, n_agents, *extra)
    gather_index = indices.view(batch, n_agents, k_neighbors, *([1] * len(extra)))
    gather_index = gather_index.expand(batch, n_agents, k_neighbors, *extra)
    return torch.gather(expanded_values, dim=2, index=gather_index)


@torch.no_grad()
def build_counterfactual_labels(
    positions: Tensor,
    velocities: Tensor,
    generations: Tensor,
    neighbor_indices: Tensor,
    edge_mask: Tensor,
    *,
    horizon: int,
    dt: float,
    safe_distance: float,
    label_slope: float,
    label_margin: float,
) -> Dict[str, Tensor]:
    """Compare actual neighbor motion with a constant-velocity counterfactual.

    Shapes are ``[B,T,N,...]``.  The ego's actual future is kept fixed while
    neighbor ``j`` is replaced by constant-velocity extrapolation from time
    ``t``.  A positive gap means the observed neighbor reduced future proximity
    risk more than that counterfactual.  Labels crossing any agent reset are
    invalidated using generation ids.
    """

    batch, time_steps, n_agents, _ = positions.shape
    k_neighbors = neighbor_indices.shape[-1]
    label = positions.new_zeros(batch, time_steps, n_agents, k_neighbors)
    gap = torch.zeros_like(label)
    valid = torch.zeros_like(edge_mask, dtype=torch.bool)
    risk_actual_out = torch.zeros_like(label)
    risk_counterfactual_out = torch.zeros_like(label)
    horizon = max(1, int(horizon))
    safe_distance = max(float(safe_distance), 1e-6)

    for t in range(time_steps):
        if t + horizon >= time_steps:
            continue
        indices_t = neighbor_indices[:, t].long()
        pos_t = positions[:, t]
        vel_t = velocities[:, t]
        neighbor_pos_t = _gather_agent(pos_t, indices_t)
        neighbor_vel_t = _gather_agent(vel_t, indices_t)
        ego_generation_t = generations[:, t].unsqueeze(-1).expand(-1, -1, k_neighbors)
        neighbor_generation_t = _gather_agent(
            generations[:, t].unsqueeze(-1), indices_t
        ).squeeze(-1)

        actual_risks = []
        counterfactual_risks = []
        identity_valid = edge_mask[:, t].bool().clone()
        for step in range(1, horizon + 1):
            future_pos = positions[:, t + step]
            ego_future = future_pos.unsqueeze(2).expand(-1, -1, k_neighbors, -1)
            neighbor_future = _gather_agent(future_pos, indices_t)
            neighbor_counterfactual = neighbor_pos_t + neighbor_vel_t * (
                float(step) * float(dt)
            )
            distance_actual = torch.linalg.vector_norm(
                ego_future - neighbor_future, dim=-1
            )
            distance_counterfactual = torch.linalg.vector_norm(
                ego_future - neighbor_counterfactual, dim=-1
            )
            actual_risks.append(torch.exp(-distance_actual / safe_distance))
            counterfactual_risks.append(
                torch.exp(-distance_counterfactual / safe_distance)
            )

            future_generation = generations[:, t + step]
            ego_same = future_generation.unsqueeze(-1) == ego_generation_t
            neighbor_same = (
                _gather_agent(future_generation.unsqueeze(-1), indices_t).squeeze(-1)
                == neighbor_generation_t
            )
            identity_valid &= ego_same & neighbor_same

        risk_actual = torch.stack(actual_risks, dim=0).amax(dim=0)
        risk_counterfactual = torch.stack(counterfactual_risks, dim=0).amax(dim=0)
        mitigation_gap = risk_counterfactual - risk_actual
        soft_label = torch.sigmoid(
            float(label_slope) * (mitigation_gap - float(label_margin))
        )
        gap[:, t] = mitigation_gap
        label[:, t] = soft_label
        valid[:, t] = identity_valid
        risk_actual_out[:, t] = risk_actual
        risk_counterfactual_out[:, t] = risk_counterfactual

    return {
        "label": label,
        "gap": gap,
        "valid": valid,
        "risk_actual": risk_actual_out,
        "risk_counterfactual": risk_counterfactual_out,
    }
