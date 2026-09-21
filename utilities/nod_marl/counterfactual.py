"""Future-only training labels for local, directed cooperation opinions."""

from __future__ import annotations

from typing import Dict
import math

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
def _interaction_references(positions, velocities, generations, neighbor_indices,
                            edge_mask, *, dt, lookahead, conflict_distance, max_age):
    """Causal, rollout-local velocity anchors; never read future motion.

    Keep the pre-yield velocity, but extrapolate from the CURRENT position.
    This evaluates the remaining opportunity to yield, rather than a ghost car
    which has already driven past the conflict. Clear on separation, turn,
    visibility/identity loss or timeout. A timeout cannot immediately rearm.
    """
    reference = torch.zeros_like(positions[:, :, :, None, :].expand(
        -1, -1, -1, neighbor_indices.shape[-1], -1))
    active_out = torch.zeros_like(edge_mask, dtype=torch.bool)
    ages_out = positions.new_zeros(edge_mask.shape)
    held = torch.zeros_like(edge_mask[:, 0], dtype=torch.bool)
    locked = torch.zeros_like(held)
    anchor = torch.zeros_like(reference[:, 0])
    age = torch.zeros_like(ages_out[:, 0])
    previous_ids = previous_ego = previous_neighbor = None

    def potential(relative_position, relative_velocity):
        speed_sq = relative_velocity.square().sum(-1)
        closing = -(relative_position * relative_velocity).sum(-1)
        closest_time = (closing / speed_sq.clamp_min(1e-8)).clamp(0., lookahead)
        closest = relative_position + closest_time.unsqueeze(-1)*relative_velocity
        return (closing > 1e-6) & (closest.norm(dim=-1) <= conflict_distance)

    for t in range(positions.shape[1]):
        ids = neighbor_indices[:, t].long()
        neighbor_pos = _gather_agent(positions[:, t], ids)
        neighbor_vel = _gather_agent(velocities[:, t], ids)
        ego_gen = generations[:, t].unsqueeze(-1).expand_as(ids)
        neighbor_gen = _gather_agent(generations[:, t].unsqueeze(-1), ids).squeeze(-1)
        same = torch.zeros_like(held) if previous_ids is None else (
            (ids == previous_ids) & (ego_gen == previous_ego) & (neighbor_gen == previous_neighbor))
        visible = edge_mask[:, t].bool()
        held &= same & visible
        locked &= same & visible
        relative_pos = neighbor_pos - positions[:, t].unsqueeze(2)
        ego_vel = velocities[:, t].unsqueeze(2)
        current_conflict = potential(relative_pos, neighbor_vel-ego_vel)
        retained_conflict = potential(relative_pos, anchor-ego_vel)
        speed = neighbor_vel.norm(dim=-1)
        anchor_speed = anchor.norm(dim=-1)
        # A moving neighbor turning >30 degrees invalidates the straight-line
        # reference. Zero speed is waiting, not a change of heading.
        turned = (speed > .05) & ((neighbor_vel*anchor).sum(-1)
                  < math.cos(math.pi/6)*speed*anchor_speed)
        age = torch.where(held, age+dt, torch.zeros_like(age))
        expired = held & ((age >= max_age) | turned)
        locked |= expired
        locked &= current_conflict | (held & retained_conflict)
        held &= retained_conflict & ~expired
        start = visible & ~held & ~locked & current_conflict & (speed > .05)
        anchor = torch.where(start.unsqueeze(-1), neighbor_vel, anchor)
        age = torch.where(start, torch.zeros_like(age), age)
        held |= start
        reference[:, t] = torch.where(held.unsqueeze(-1), anchor, neighbor_vel)
        active_out[:, t] = held
        ages_out[:, t] = torch.where(held, age, 0.)
        previous_ids, previous_ego, previous_neighbor = ids, ego_gen, neighbor_gen
    return reference, active_out, ages_out


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
    mode: str = "instantaneous",
    reference_seconds: float = 2.0,
) -> Dict[str, Tensor]:
    """Compare actual neighbor motion with a constant-velocity counterfactual.

    Shapes are ``[B,T,N,...]``.  The ego's actual future is kept fixed while
    neighbor ``j`` is replaced by constant-velocity extrapolation from time
    ``t``.  A positive gap means the observed neighbor reduced future proximity
    risk more than that counterfactual.  Labels crossing any agent reset are
    invalidated using generation ids.

    ``instantaneous`` preserves legacy labels. ``interaction`` uses causal
    pre-yield velocity references and a symmetric neutral band. Outside an
    anchored interaction the attributed gap is zero (raw risks are still
    returned for diagnostics). Reference state is local to this rollout.
    """

    if mode not in {"instantaneous", "interaction"}:
        raise ValueError("Unknown NOD label mode")
    if (not math.isfinite(reference_seconds) or reference_seconds <= 0
            or not math.isfinite(dt) or dt <= 0):
        raise ValueError("NOD reference duration and dt must be positive and finite")
    if not math.isfinite(label_margin) or label_margin < 0:
        raise ValueError("NOD label margin must be nonnegative and finite")
    batch, time_steps, n_agents, _ = positions.shape
    k_neighbors = neighbor_indices.shape[-1]
    label = positions.new_zeros(batch, time_steps, n_agents, k_neighbors)
    gap = torch.zeros_like(label)
    valid = torch.zeros_like(edge_mask, dtype=torch.bool)
    risk_actual_out = torch.zeros_like(label)
    risk_counterfactual_out = torch.zeros_like(label)
    horizon = max(1, int(horizon))
    safe_distance = max(float(safe_distance), 1e-6)
    references = active = reference_age = None
    if mode == "interaction":
        references, active, reference_age = _interaction_references(
            positions, velocities, generations, neighbor_indices, edge_mask,
            dt=dt, lookahead=horizon*dt, conflict_distance=1.5*safe_distance,
            max_age=reference_seconds)

    for t in range(time_steps):
        if t + horizon >= time_steps:
            continue
        indices_t = neighbor_indices[:, t].long()
        pos_t = positions[:, t]
        vel_t = velocities[:, t]
        neighbor_pos_t = _gather_agent(pos_t, indices_t)
        neighbor_vel_t = _gather_agent(vel_t, indices_t)
        if references is not None:
            neighbor_vel_t = references[:, t]
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
        if mode == "interaction":
            # No active interaction means no attributed cooperation evidence.
            mitigation_gap = torch.where(active[:, t], mitigation_gap, 0.)
            centered_gap = mitigation_gap.sign() * (mitigation_gap.abs()-float(label_margin)).clamp_min(0.)
        else:
            centered_gap = mitigation_gap-float(label_margin)
        soft_label = torch.sigmoid(float(label_slope)*centered_gap)
        gap[:, t] = mitigation_gap
        label[:, t] = soft_label
        valid[:, t] = identity_valid
        risk_actual_out[:, t] = risk_actual
        risk_counterfactual_out[:, t] = risk_counterfactual

    result = {
        "label": label,
        "gap": gap,
        "valid": valid,
        "risk_actual": risk_actual_out,
        "risk_counterfactual": risk_counterfactual_out,
    }
    if active is not None:
        result.update(reference_active=active, reference_age=reference_age)
    return result
