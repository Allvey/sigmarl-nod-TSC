"""Future-only training labels for local, directed cooperation opinions."""

from __future__ import annotations

from typing import Dict, Optional
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


def _closest_approach_risk(
    relative_position: Tensor,
    relative_velocity: Tensor,
    *,
    lookahead: float,
    safe_distance: float,
) -> tuple[Tensor, Tensor]:
    """Return constant-velocity proximity risk and whether agents approach."""

    speed_squared = relative_velocity.square().sum(dim=-1)
    closing = -(relative_position * relative_velocity).sum(dim=-1)
    closest_time = (closing / speed_squared.clamp_min(1e-8)).clamp(
        0.0, float(lookahead)
    )
    closest = relative_position + closest_time.unsqueeze(-1) * relative_velocity
    distance = torch.linalg.vector_norm(closest, dim=-1)
    risk = torch.exp(-distance / max(float(safe_distance), 1e-6))
    return risk, closing > 1e-6


@torch.no_grad()
def build_responsibility_evidence(
    positions: Tensor,
    velocities: Tensor,
    ego_generations: Tensor,
    neighbor_indices: Tensor,
    edge_mask: Tensor,
    *,
    dt: float,
    lookahead: float,
    safe_distance: float,
    max_age: float,
    neighbor_generations: Optional[Tensor] = None,
    initial_state: Optional[Dict[str, Tensor]] = None,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Build causal evidence from the neighbor's share of required avoidance.

    At the beginning of a predicted conflict, the current ego and neighbor
    velocities are frozen as counterfactual anchors. The ego anchor is used in
    both alternatives, so an ego manoeuvre cannot be credited to the neighbor.
    With anchored risk R0, safety-boundary risk Rs and neighbor-observed risk
    Rj, responsibility is q=clip((R0-Rj)/(R0-Rs), 0, 1). Online evidence is
    2*q-1, so zero means half of the required avoidance.
    """

    if positions.ndim != 4 or velocities.shape != positions.shape:
        raise ValueError("positions and velocities must have shape [B,T,N,2]")
    if neighbor_indices.ndim != 4 or edge_mask.shape != neighbor_indices.shape:
        raise ValueError("neighbor tensors must have shape [B,T,N,K]")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(lookahead) or lookahead <= 0:
        raise ValueError("lookahead must be positive and finite")
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("max_age must be positive and finite")

    batch, time_steps, n_agents, _ = positions.shape
    k_neighbors = neighbor_indices.shape[-1]
    pair_shape = (batch, n_agents, k_neighbors)
    vector_shape = (*pair_shape, 2)
    device = positions.device

    if initial_state is None:
        held = torch.zeros(pair_shape, dtype=torch.bool, device=device)
        locked = torch.zeros_like(held)
        anchor_ego = positions.new_zeros(vector_shape)
        anchor_neighbor = positions.new_zeros(vector_shape)
        age = positions.new_zeros(pair_shape)
        previous_ids = torch.zeros(pair_shape, dtype=torch.long, device=device)
        previous_ego_generation = torch.zeros_like(previous_ids)
        previous_neighbor_generation = torch.zeros_like(previous_ids)
        has_identity = torch.zeros_like(held)
    else:
        held = initial_state["held"]
        locked = initial_state["locked"]
        anchor_ego = initial_state["anchor_ego_velocity"]
        anchor_neighbor = initial_state["anchor_neighbor_velocity"]
        age = initial_state["age"]
        previous_ids = initial_state["neighbor_indices"]
        previous_ego_generation = initial_state["ego_generation"]
        previous_neighbor_generation = initial_state["neighbor_generation"]
        has_identity = initial_state["has_identity"]

    evidence_out = positions.new_zeros(batch, time_steps, n_agents, k_neighbors)
    responsibility_out = torch.zeros_like(evidence_out)
    active_out = torch.zeros_like(edge_mask, dtype=torch.bool)
    age_out = torch.zeros_like(evidence_out)
    anchor_ego_out = positions.new_zeros(batch, time_steps, n_agents, k_neighbors, 2)
    anchor_neighbor_out = torch.zeros_like(anchor_ego_out)
    safe_risk = math.exp(-1.0)

    for time_index in range(time_steps):
        ids = neighbor_indices[:, time_index].long()
        visible = edge_mask[:, time_index].bool()
        ego_generation = ego_generations[:, time_index].unsqueeze(-1).expand_as(ids)
        if neighbor_generations is None:
            neighbor_generation = _gather_agent(
                ego_generations[:, time_index].unsqueeze(-1), ids
            ).squeeze(-1)
        else:
            neighbor_generation = neighbor_generations[:, time_index].long()
        same_identity = (
            has_identity
            & (ids == previous_ids)
            & (ego_generation == previous_ego_generation)
            & (neighbor_generation == previous_neighbor_generation)
        )
        held = held & same_identity & visible
        locked = locked & same_identity & visible

        ego_position = positions[:, time_index].unsqueeze(2)
        ego_velocity = velocities[:, time_index].unsqueeze(2)
        neighbor_position = _gather_agent(positions[:, time_index], ids)
        neighbor_velocity = _gather_agent(velocities[:, time_index], ids)
        relative_position = neighbor_position - ego_position

        current_risk, current_approaching = _closest_approach_risk(
            relative_position,
            neighbor_velocity - ego_velocity,
            lookahead=lookahead,
            safe_distance=safe_distance,
        )
        current_conflict = current_approaching & (current_risk > safe_risk)
        anchored_risk, anchored_approaching = _closest_approach_risk(
            relative_position,
            anchor_neighbor - anchor_ego,
            lookahead=lookahead,
            safe_distance=safe_distance,
        )
        anchored_conflict = anchored_approaching & (anchored_risk > safe_risk)

        age = torch.where(held, age + float(dt), torch.zeros_like(age))
        ended = held & (~anchored_conflict | (age >= float(max_age)))
        held = held & anchored_conflict & (age < float(max_age))
        locked = (locked | ended) & current_conflict
        start = visible & ~held & ~locked & current_conflict
        anchor_ego = torch.where(start.unsqueeze(-1), ego_velocity, anchor_ego)
        anchor_neighbor = torch.where(
            start.unsqueeze(-1), neighbor_velocity, anchor_neighbor
        )
        age = torch.where(start, torch.zeros_like(age), age)
        held = held | start

        baseline_risk, _ = _closest_approach_risk(
            relative_position,
            anchor_neighbor - anchor_ego,
            lookahead=lookahead,
            safe_distance=safe_distance,
        )
        neighbor_observed_risk, _ = _closest_approach_risk(
            relative_position,
            neighbor_velocity - anchor_ego,
            lookahead=lookahead,
            safe_distance=safe_distance,
        )
        required_reduction = (baseline_risk - safe_risk).clamp_min(1e-6)
        responsibility = (
            (baseline_risk - neighbor_observed_risk) / required_reduction
        ).clamp(0.0, 1.0)
        responsibility = torch.where(
            held, responsibility, torch.zeros_like(responsibility)
        )

        evidence_out[:, time_index] = torch.where(
            held, 2.0 * responsibility - 1.0, torch.zeros_like(responsibility)
        )
        responsibility_out[:, time_index] = responsibility
        active_out[:, time_index] = held
        age_out[:, time_index] = torch.where(held, age, torch.zeros_like(age))
        anchor_ego_out[:, time_index] = anchor_ego
        anchor_neighbor_out[:, time_index] = anchor_neighbor

        previous_ids = ids
        previous_ego_generation = ego_generation
        previous_neighbor_generation = neighbor_generation
        has_identity = visible

    state = {
        "held": held,
        "locked": locked,
        "anchor_ego_velocity": anchor_ego,
        "anchor_neighbor_velocity": anchor_neighbor,
        "age": age,
        "neighbor_indices": previous_ids,
        "ego_generation": previous_ego_generation,
        "neighbor_generation": previous_neighbor_generation,
        "has_identity": has_identity,
    }
    outputs = {
        "evidence": evidence_out,
        "responsibility": responsibility_out,
        "active": active_out,
        "age": age_out,
        "anchor_ego_velocity": anchor_ego_out,
        "anchor_neighbor_velocity": anchor_neighbor_out,
    }
    return outputs, state


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

    ``responsibility`` supervises the fraction of required avoidance carried
    by the neighbor. It uses the same causal anchors as online NOD evidence,
    holds the ego counterfactual fixed, and excludes non-interactions instead
    of treating them as neutral cooperation samples.
    """

    if mode not in {"instantaneous", "interaction", "responsibility"}:
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
    references = active = reference_age = responsibility_data = None
    if mode == "interaction":
        references, active, reference_age = _interaction_references(
            positions, velocities, generations, neighbor_indices, edge_mask,
            dt=dt, lookahead=horizon*dt, conflict_distance=1.5*safe_distance,
            max_age=reference_seconds)
    elif mode == "responsibility":
        responsibility_data, _ = build_responsibility_evidence(
            positions,
            velocities,
            generations,
            neighbor_indices,
            edge_mask,
            dt=dt,
            lookahead=horizon * dt,
            safe_distance=safe_distance,
            max_age=reference_seconds,
        )
        active = responsibility_data["active"]
        reference_age = responsibility_data["age"]

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
        elif responsibility_data is not None:
            neighbor_vel_t = responsibility_data["anchor_neighbor_velocity"][:, t]
        ego_generation_t = generations[:, t].unsqueeze(-1).expand(-1, -1, k_neighbors)
        neighbor_generation_t = _gather_agent(
            generations[:, t].unsqueeze(-1), indices_t
        ).squeeze(-1)

        actual_risks = []
        counterfactual_risks = []
        identity_valid = edge_mask[:, t].bool().clone()
        for step in range(1, horizon + 1):
            future_pos = positions[:, t + step]
            if responsibility_data is None:
                ego_future = future_pos.unsqueeze(2).expand(
                    -1, -1, k_neighbors, -1
                )
            else:
                ego_future = pos_t.unsqueeze(2) + responsibility_data[
                    "anchor_ego_velocity"
                ][:, t] * (float(step) * float(dt))
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
        if mode == "responsibility":
            required_reduction = (
                risk_counterfactual - math.exp(-1.0)
            ).clamp_min(1e-6)
            responsibility = (mitigation_gap / required_reduction).clamp(0.0, 1.0)
            responsibility = torch.where(
                active[:, t], responsibility, torch.zeros_like(responsibility)
            )
            soft_label = responsibility
            mitigation_gap = responsibility
        elif mode == "interaction":
            # No active interaction means no attributed cooperation evidence.
            mitigation_gap = torch.where(active[:, t], mitigation_gap, 0.)
            centered_gap = mitigation_gap.sign() * (mitigation_gap.abs()-float(label_margin)).clamp_min(0.)
            soft_label = torch.sigmoid(float(label_slope)*centered_gap)
        else:
            centered_gap = mitigation_gap-float(label_margin)
            soft_label = torch.sigmoid(float(label_slope)*centered_gap)
        gap[:, t] = mitigation_gap
        label[:, t] = soft_label
        valid[:, t] = identity_valid & (
            active[:, t] if mode == "responsibility" else True
        )
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
    if responsibility_data is not None:
        result.update(
            online_evidence=responsibility_data["evidence"],
            online_responsibility=responsibility_data["responsibility"],
        )
    return result
