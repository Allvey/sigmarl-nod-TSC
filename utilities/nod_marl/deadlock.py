"""Stage 6: causal stalled-group labels and an independent Deadlock Critic."""

import math

import torch

from .interaction import _closest_path_approach
from .safety import SafetyCriticManager


DEADLOCK_STATE_DIM = 12


def conflict_components(positions, paths, distance_limit, corridor_radius):
    """Connected groups of nearby cars with overlapping reference corridors."""
    batch, count, points, _ = paths.shape
    first = paths[:, :, None].expand(batch, count, count, points, 2)
    second = paths[:, None, :].expand_as(first)
    path_distance, _, _, _ = _closest_path_approach(first, second, 1e-6)
    eye = torch.eye(count, device=positions.device, dtype=torch.bool)
    adjacent = (
        (torch.cdist(positions, positions) <= distance_limit)
        & (path_distance <= 2 * corridor_radius)
        & ~eye
    )
    connected = adjacent | adjacent.transpose(-1, -2) | eye
    for k in range(count):
        connected = connected | (connected[:, :, k, None] & connected[:, None, k, :])
    return connected


def forward_escape_available(
    positions,
    velocities,
    yaws,
    directions,
    clearance,
    *,
    distance,
    speed,
    safe_distance,
    boundary_margin,
    min_progress,
):
    """Sufficient check for one short, zero-steering forward primitive.

    Other cars keep current velocities. Continuous closest approach enforces
    center separation; clearance >= travel + margin bounds the entire vehicle
    translation away from boundaries. Failure means 'not established', not
    proof that every possible escape manoeuvre is impossible.
    """
    heading = torch.stack((yaws.cos(), yaws.sin()), -1)
    probe_velocity = heading * speed
    relative_pos = positions[:, None, :, :] - positions[:, :, None, :]
    relative_vel = velocities[:, None, :, :] - probe_velocity[:, :, None, :]
    closest_time = -(relative_pos * relative_vel).sum(-1) / relative_vel.square().sum(
        -1
    ).clamp_min(1e-8)
    closest_time = closest_time.clamp(0, distance / speed)
    closest = (relative_pos + relative_vel * closest_time.unsqueeze(-1)).norm(dim=-1)
    eye = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool)
    separation_ok = (closest.masked_fill(eye, torch.inf) >= safe_distance).all(-1)
    progress_ok = (heading * directions).sum(-1) * distance >= min_progress
    return separation_ok & (clearance >= distance + boundary_margin) & progress_ok


class DeadlockTracker:
    """Environment-owned temporal state, advanced once per actual frame.

    No rollout-boundary reset: only physical agent resets clear history.
    State order: low time, waiting time, window progress, history ready,
    own permission, own escape, group size, group stalled, group opportunity,
    eligible time, in conflict group, speed (all normalized).
    """

    def __init__(self, parameters):
        self.p = parameters
        self.window_steps = max(
            1, math.ceil(parameters.deadlock_window_seconds / parameters.dt)
        )
        self.positions = None
        self.cache = None

    def reset(self, env_index, agent_index=None):
        if self.positions is None:
            return
        indices = slice(None) if agent_index is None else agent_index
        affected = (
            self.components[env_index, indices].any(0)
            if agent_index is None
            else self.components[env_index, indices]
        )
        self.eligible_time[env_index, affected] = 0
        for value in (self.low_time, self.wait_time, self.age, self.history):
            value[env_index, indices] = 0
        self.cache = None

    @torch.no_grad()
    def update(
        self,
        positions,
        velocities,
        yaws,
        paths,
        clearance,
        collision,
        generations,
        steps,
        forward_allowed,
    ):
        p = self.p
        yaws = yaws.reshape(*positions.shape[:2])
        steps = steps.reshape(-1).expand(positions.shape[0])
        if (
            self.cache is not None
            and torch.equal(steps, self.steps)
            and torch.equal(generations, self.generations)
        ):
            return self.cache
        route = paths[..., -1, :] - paths[..., 0, :]
        route = route / route.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        shape = positions.shape[:2]
        if self.positions is None:
            self.positions = positions.clone()
            self.directions = route.clone()
            self.generations = generations.clone()
            self.steps = steps.clone()
            self.history = positions.new_zeros(*shape, self.window_steps)
            self.low_time = positions.new_zeros(shape)
            self.wait_time = positions.new_zeros(shape)
            self.eligible_time = positions.new_zeros(shape)
            self.age = positions.new_zeros(shape)
            self.components = torch.zeros(
                *shape, shape[-1], device=positions.device, dtype=torch.bool
            )
        replaced = generations != self.generations
        elapsed = ((steps - self.steps).clamp_min(0) * p.dt).unsqueeze(-1).expand(shape)
        elapsed = torch.where(replaced, torch.zeros_like(elapsed), elapsed)
        for value in (self.low_time, self.wait_time, self.eligible_time, self.age):
            value.masked_fill_(replaced, 0)
        self.history.masked_fill_(replaced.unsqueeze(-1), 0)
        displacement = ((positions - self.positions) * self.directions).sum(-1)
        displacement = torch.where(
            replaced, torch.zeros_like(displacement), displacement
        )
        shifted = torch.cat((self.history[..., 1:], displacement.unsqueeze(-1)), -1)
        self.history = torch.where((elapsed > 0).unsqueeze(-1), shifted, self.history)
        self.age += elapsed
        progress = self.history.sum(-1)
        speed = velocities.norm(dim=-1)
        low = speed <= p.deadlock_speed_threshold
        self.low_time = torch.where(
            low, self.low_time + elapsed, torch.zeros_like(elapsed)
        )
        ready = self.age >= self.window_steps * p.dt - 1e-6
        stalled = (
            low
            & ready
            & (self.low_time >= self.window_steps * p.dt - 1e-6)
            & (progress <= p.deadlock_progress_threshold)
        )
        self.wait_time = torch.where(
            stalled, self.wait_time + elapsed, torch.zeros_like(elapsed)
        )
        # Include the current position so a conflict just before the first
        # sampled reference point is not accidentally dropped.
        route_paths = torch.cat((positions.unsqueeze(-2), paths), -2)
        groups = conflict_components(
            positions, route_paths, p.deadlock_conflict_distance, p.nod_conflict_radius
        )
        group_size = groups.sum(-1)
        group_stalled = ~(groups & ~stalled.unsqueeze(1)).any(-1)
        escape = forward_escape_available(
            positions,
            velocities,
            yaws,
            route,
            clearance,
            distance=p.deadlock_probe_distance,
            speed=p.deadlock_probe_speed,
            safe_distance=p.safety_safe_distance,
            boundary_margin=p.safety_boundary_margin,
            min_progress=p.deadlock_progress_threshold,
        )
        opportunity = (groups & (forward_allowed.bool() & escape).unsqueeze(1)).any(-1)
        in_group = group_size >= 2
        candidate = (
            in_group & group_stalled & opportunity & ~collision.any(-1).unsqueeze(-1)
        )
        membership_changed = (groups != self.components).any(-1)
        # Any identity change invalidates timers for its entire old/new group.
        membership_changed |= (groups & replaced.unsqueeze(1)).any(-1)
        old_eligible = self.eligible_time.clone()
        self.eligible_time = torch.where(
            candidate,
            torch.where(
                membership_changed, torch.zeros_like(elapsed), self.eligible_time
            )
            + elapsed,
            torch.zeros_like(elapsed),
        )
        margin = torch.where(
            candidate,
            self.eligible_time / p.deadlock_duration_seconds - 1,
            -torch.ones_like(elapsed),
        ).clamp(-1, 1)
        onset = (margin > 0) & (old_eligible <= p.deadlock_duration_seconds)
        state = torch.stack(
            (
                (self.low_time / p.deadlock_duration_seconds).clamp_max(2),
                (self.wait_time / p.deadlock_duration_seconds).clamp_max(2),
                (progress / p.deadlock_progress_threshold).clamp(-4, 4),
                ready.float(),
                forward_allowed.float(),
                escape.float(),
                group_size.float() / shape[-1],
                group_stalled.float(),
                opportunity.float(),
                (self.eligible_time / p.deadlock_duration_seconds).clamp_max(2),
                in_group.float(),
                (speed / p.deadlock_speed_threshold).clamp_max(4),
            ),
            -1,
        )
        self.cache = {
            "deadlock_state": state,
            "deadlock_margin": margin.unsqueeze(-1),
            "deadlock_onset": onset.unsqueeze(-1),
            "deadlock_eligible_seconds": self.eligible_time.unsqueeze(-1).clone(),
        }
        self.positions = positions.clone()
        self.directions = route.clone()
        self.generations = generations.clone()
        self.steps = steps.clone()
        self.components = groups
        return self.cache


class DeadlockCriticManager(SafetyCriticManager):
    """Separate model/optimizer/checkpoint, sharing only finite-horizon code."""

    def __init__(
        self, parameters, observation_dim, n_agents, action_dim, observation_key
    ):
        super().__init__(
            parameters,
            observation_dim,
            n_agents,
            action_dim,
            observation_key,
            kind="deadlock",
            context_key="deadlock_state",
            context_dim=DEADLOCK_STATE_DIM,
            margin_key="deadlock_margin",
        )
        self.metadata.update(
            {
                "kind": "deadlock",
                "dt": parameters.dt,
                "conflict_radius": parameters.nod_conflict_radius,
            }
        )
        for name in (
            "window_seconds",
            "duration_seconds",
            "speed_threshold",
            "progress_threshold",
            "conflict_distance",
            "probe_distance",
            "probe_speed",
        ):
            self.metadata[name] = getattr(parameters, "deadlock_" + name)

    def train_on_rollout(self, td):
        metrics = super().train_on_rollout(td)
        metrics = {
            key.replace("unsafe", "deadlock"): value for key, value in metrics.items()
        }
        if self.enabled:
            state = td.get(("agents", "info", "deadlock_state"))
            metrics.update(
                {
                    "waiting_agent_ratio": float((state[..., 1] > 0).float().mean()),
                    "conflict_group_agent_ratio": float(state[..., 10].mean()),
                    "escape_available_agent_ratio": float(state[..., 5].mean()),
                    "eligible_agent_ratio": float((state[..., 9] > 0).float().mean()),
                    "max_eligible_seconds": float(
                        td.get(("agents", "info", "deadlock_eligible_seconds")).max()
                    ),
                    "onset_transition_count": int(
                        td.get(("next", "agents", "info", "deadlock_onset"))
                        .any(-2)
                        .sum()
                    ),
                }
            )
        return metrics
