"""Physical, directed pair features for the NOD-MARL auxiliary branch."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


# Feature layout. Keep this stable: checkpoints and the opinion likelihood head
# depend on it.
REL_POS = slice(0, 2)
REL_VEL = slice(2, 4)
SIN_DYAW = 4
COS_DYAW = 5
DISTANCE = 6
CLOSING_SPEED = 7
TTC = 8
CONFLICT_VALID = 9
EGO_CONFLICT_DISTANCE = 10
NEIGHBOR_CONFLICT_DISTANCE = 11
EGO_ETA = 12
NEIGHBOR_ETA = 13
ETA_GAP = 14
OVERLAP_RISK = 15
APPROACH_CONFIDENCE = 16
VISIBLE = 17
EGO_SPEED = 18
NEIGHBOR_SPEED = 19
NOD_PAIR_FEATURE_DIM = 20


def _gather_agents(values: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,N,...]`` values using ``[B,K]`` agent indices."""

    extra = values.shape[2:]
    index = indices.view(*indices.shape, *([1] * len(extra))).expand(
        *indices.shape, *extra
    )
    return torch.gather(values, dim=1, index=index)


def _cumulative_path_distance(path: Tensor) -> Tensor:
    """Distance from path start to each point, for ``[...,P,2]`` paths."""

    segment_lengths = torch.linalg.vector_norm(
        path[..., 1:, :] - path[..., :-1, :], dim=-1
    )
    zeros = torch.zeros_like(segment_lengths[..., :1])
    return torch.cat([zeros, torch.cumsum(segment_lengths, dim=-1)], dim=-1)


def _cross_2d(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _closest_path_approach(
    ego_path: Tensor, neighbor_path: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact polyline intersection plus minimum segment-corridor distance.

    Args:
        ego_path: ``[B,K,P,2]``.
        neighbor_path: ``[B,K,Q,2]``.

    Returns:
        Minimum distance, distance travelled by ego and neighbor to the closest
        approach, and whether any pair of segments intersects.  Endpoint to
        segment projections cover non-intersecting parallel/merging paths;
        explicit line intersection prevents sparse path samples from missing a
        crossing between sample points.
    """

    p0 = ego_path[..., :-1, :].unsqueeze(-2)
    p1 = ego_path[..., 1:, :].unsqueeze(-2)
    q0 = neighbor_path[..., :-1, :].unsqueeze(-3)
    q1 = neighbor_path[..., 1:, :].unsqueeze(-3)
    r = p1 - p0
    s = q1 - q0
    qp = q0 - p0
    denominator = _cross_2d(r, s)
    non_parallel = denominator.abs() > eps
    safe_denominator = torch.where(
        non_parallel, denominator, torch.ones_like(denominator)
    )
    t_intersection = _cross_2d(qp, s) / safe_denominator
    u_intersection = _cross_2d(qp, r) / safe_denominator
    intersects = (
        non_parallel
        & (t_intersection >= 0.0)
        & (t_intersection <= 1.0)
        & (u_intersection >= 0.0)
        & (u_intersection <= 1.0)
    )

    r_length_sq = r.square().sum(dim=-1).clamp_min(eps)
    s_length_sq = s.square().sum(dim=-1).clamp_min(eps)
    r_length = r_length_sq.sqrt()
    s_length = s_length_sq.sqrt()

    def point_to_q(point: Tensor):
        fraction = ((point - q0) * s).sum(dim=-1) / s_length_sq
        fraction = fraction.clamp(0.0, 1.0)
        closest = q0 + fraction.unsqueeze(-1) * s
        distance = torch.linalg.vector_norm(point - closest, dim=-1)
        return distance, fraction

    def point_to_p(point: Tensor):
        fraction = ((point - p0) * r).sum(dim=-1) / r_length_sq
        fraction = fraction.clamp(0.0, 1.0)
        closest = p0 + fraction.unsqueeze(-1) * r
        distance = torch.linalg.vector_norm(point - closest, dim=-1)
        return distance, fraction

    distance_p0_q, fraction_p0_q = point_to_q(p0)
    distance_p1_q, fraction_p1_q = point_to_q(p1)
    distance_q0_p, fraction_q0_p = point_to_p(q0)
    distance_q1_p, fraction_q1_p = point_to_p(q1)
    distance_intersection = torch.where(
        intersects,
        torch.zeros_like(denominator),
        torch.full_like(denominator, float("inf")),
    )

    ego_cumulative = _cumulative_path_distance(ego_path)
    neighbor_cumulative = _cumulative_path_distance(neighbor_path)
    ego_start = ego_cumulative[..., :-1].unsqueeze(-1)
    ego_end = ego_cumulative[..., 1:].unsqueeze(-1)
    neighbor_start = neighbor_cumulative[..., :-1].unsqueeze(-2)
    neighbor_end = neighbor_cumulative[..., 1:].unsqueeze(-2)

    distance_candidates = torch.stack(
        [
            distance_intersection,
            distance_p0_q,
            distance_p1_q,
            distance_q0_p,
            distance_q1_p,
        ],
        dim=-1,
    )
    ego_progress_candidates = torch.stack(
        [
            ego_start + t_intersection.clamp(0.0, 1.0) * r_length,
            ego_start.expand_as(distance_p0_q),
            ego_end.expand_as(distance_p1_q),
            ego_start + fraction_q0_p * r_length,
            ego_start + fraction_q1_p * r_length,
        ],
        dim=-1,
    )
    neighbor_progress_candidates = torch.stack(
        [
            neighbor_start + u_intersection.clamp(0.0, 1.0) * s_length,
            neighbor_start + fraction_p0_q * s_length,
            neighbor_start + fraction_p1_q * s_length,
            neighbor_start.expand_as(distance_q0_p),
            neighbor_end.expand_as(distance_q1_p),
        ],
        dim=-1,
    )

    flat_distance = distance_candidates.flatten(start_dim=-3)
    min_distance, flat_index = flat_distance.min(dim=-1)
    flat_ego_progress = ego_progress_candidates.flatten(start_dim=-3)
    flat_neighbor_progress = neighbor_progress_candidates.flatten(start_dim=-3)
    ego_progress = torch.gather(
        flat_ego_progress, -1, flat_index.unsqueeze(-1)
    ).squeeze(-1)
    neighbor_progress = torch.gather(
        flat_neighbor_progress, -1, flat_index.unsqueeze(-1)
    ).squeeze(-1)
    return (
        min_distance,
        ego_progress,
        neighbor_progress,
        intersects.flatten(start_dim=-2).any(dim=-1),
    )


def build_directed_interactions(
    positions: Tensor,
    velocities: Tensor,
    yaws: Tensor,
    short_term_paths: Tensor,
    ego_index: int,
    *,
    sensing_range: float,
    interaction_distance: float,
    ttc_limit: float,
    conflict_radius: float,
    max_speed: float,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    """Build stable directed candidates and a dynamic interaction mask.

    Args:
        positions: World positions ``[B,N,2]``.
        velocities: World velocities ``[B,N,2]``.
        yaws: World headings ``[B,N]`` (or ``[B,N,1]``).
        short_term_paths: World-frame reference points ``[B,N,P,2]``.
        ego_index: Receiver of directed edges ``ego <- neighbor``.

    Candidate slots use ascending global agent ids, excluding the ego.  This
    gives temporal identity stability even when distance order changes.
    Missing/conflict information is represented by explicit boolean fields and
    never by an ambiguous numeric zero.
    """

    if yaws.ndim == 3 and yaws.shape[-1] == 1:
        yaws = yaws.squeeze(-1)
    batch_size, n_agents, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    candidate_ids = torch.tensor(
        [j for j in range(n_agents) if j != int(ego_index)],
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1)

    ego_pos = positions[:, ego_index].unsqueeze(1)
    ego_vel = velocities[:, ego_index].unsqueeze(1)
    ego_yaw = yaws[:, ego_index].unsqueeze(1)
    neighbor_pos = _gather_agents(positions, candidate_ids)
    neighbor_vel = _gather_agents(velocities, candidate_ids)
    neighbor_yaw = _gather_agents(yaws.unsqueeze(-1), candidate_ids).squeeze(-1)

    rel_pos_world = neighbor_pos - ego_pos
    rel_vel_world = neighbor_vel - ego_vel
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    rel_pos_local = torch.stack(
        [
            cos_yaw * rel_pos_world[..., 0] + sin_yaw * rel_pos_world[..., 1],
            -sin_yaw * rel_pos_world[..., 0] + cos_yaw * rel_pos_world[..., 1],
        ],
        dim=-1,
    )
    rel_vel_local = torch.stack(
        [
            cos_yaw * rel_vel_world[..., 0] + sin_yaw * rel_vel_world[..., 1],
            -sin_yaw * rel_vel_world[..., 0] + cos_yaw * rel_vel_world[..., 1],
        ],
        dim=-1,
    )

    distance = torch.linalg.vector_norm(rel_pos_world, dim=-1)
    line_of_sight = rel_pos_world / distance.clamp_min(eps).unsqueeze(-1)
    # Positive means the pair is closing.
    closing_speed = -(rel_vel_world * line_of_sight).sum(dim=-1)
    ttc = torch.where(
        closing_speed > eps,
        distance / closing_speed.clamp_min(eps),
        torch.full_like(distance, float(ttc_limit)),
    ).clamp(0.0, float(ttc_limit))

    ego_path = torch.cat(
        [positions[:, ego_index].unsqueeze(1), short_term_paths[:, ego_index]], dim=1
    )
    neighbor_paths = _gather_agents(short_term_paths, candidate_ids)
    neighbor_paths = torch.cat([neighbor_pos.unsqueeze(2), neighbor_paths], dim=2)
    ego_path = ego_path.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1, -1)

    (
        min_path_distance,
        ego_conflict_distance,
        neighbor_conflict_distance,
        path_intersects,
    ) = _closest_path_approach(ego_path, neighbor_paths, eps)
    conflict_valid = path_intersects | (
        min_path_distance <= float(conflict_radius)
    )
    ego_speed = torch.linalg.vector_norm(ego_vel, dim=-1).expand_as(distance)
    neighbor_speed = torch.linalg.vector_norm(neighbor_vel, dim=-1)
    ego_eta = ego_conflict_distance / ego_speed.clamp_min(eps)
    neighbor_eta = neighbor_conflict_distance / neighbor_speed.clamp_min(eps)
    eta_gap = neighbor_eta - ego_eta

    conflict_scale = max(float(conflict_radius), eps)
    overlap_risk = torch.exp(-min_path_distance / conflict_scale)
    overlap_risk = torch.where(
        conflict_valid, overlap_risk, torch.zeros_like(overlap_risk)
    )
    visible = distance <= float(sensing_range)
    approaching = (closing_speed > 0.0) & (ttc < float(ttc_limit))
    edge_mask = visible & (
        (distance <= float(interaction_distance)) | conflict_valid | approaching
    )

    distance_scale = max(float(sensing_range), eps)
    speed_scale = max(float(max_speed), eps)
    time_scale = max(float(ttc_limit), eps)
    approach_confidence = (
        torch.clamp(closing_speed / speed_scale, min=0.0, max=1.0)
        * (1.0 - ttc / time_scale)
    )
    delta_yaw = neighbor_yaw - ego_yaw

    features = torch.stack(
        [
            rel_pos_local[..., 0] / distance_scale,
            rel_pos_local[..., 1] / distance_scale,
            rel_vel_local[..., 0] / speed_scale,
            rel_vel_local[..., 1] / speed_scale,
            torch.sin(delta_yaw),
            torch.cos(delta_yaw),
            (distance / distance_scale).clamp(max=2.0),
            (closing_speed / speed_scale).clamp(-2.0, 2.0),
            ttc / time_scale,
            conflict_valid.to(dtype),
            (ego_conflict_distance / distance_scale).clamp(max=2.0),
            (neighbor_conflict_distance / distance_scale).clamp(max=2.0),
            (ego_eta / time_scale).clamp(max=2.0),
            (neighbor_eta / time_scale).clamp(max=2.0),
            (eta_gap / time_scale).clamp(-2.0, 2.0),
            overlap_risk,
            approach_confidence,
            visible.to(dtype),
            (ego_speed / speed_scale).clamp(max=2.0),
            (neighbor_speed / speed_scale).clamp(max=2.0),
        ],
        dim=-1,
    )
    assert features.shape[-1] == NOD_PAIR_FEATURE_DIM

    return {
        "features": features,
        "edge_mask": edge_mask,
        "neighbor_indices": candidate_ids,
        "conflict_valid": conflict_valid,
        "path_intersects": path_intersects,
        "ttc": ttc,
        "eta_gap": eta_gap,
        "overlap_risk": overlap_risk,
    }
