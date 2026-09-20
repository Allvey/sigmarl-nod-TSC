"""Visualization-only mixed control; never use these rollouts for PPO."""
import math
import csv
import random
from numbers import Real

import torch
from torch import nn

from utilities.testing_rule_coordinator import RuleCoordinator


# Physical profiles: same path tracker/cruise speed; only interaction response differs.
PROFILES = {
    'yielding': dict(horizon=1.8, longitudinal=0.25, lateral=0.13),
    'moderate': dict(horizon=1.0, longitudinal=0.20, lateral=0.10),
    'non_yielding': dict(horizon=1.0, longitudinal=0.20, lateral=0.10),
}
REASONS = {0: 'clear', 1: 'slow-conflict', 2: 'stop-conflict',
           3: 'ignore-traffic', 4: 'least-risk', 5: 'rear-aware',
           7: 'reserved-go', 8: 'reservation-wait', 9: 'keep-reservation'}


def assign_rule_vehicles(n_agents, fraction, profile_weights, *, seed, actor_index=0):
    """Make one reproducible test-only slot assignment without touching global RNGs."""
    if not isinstance(n_agents, int) or isinstance(n_agents, bool) or n_agents < 1:
        raise ValueError('n_agents must be a positive integer')
    if isinstance(fraction, bool) or not isinstance(fraction, Real) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('rule fraction must be between 0 and 1')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('rule assignment seed must be an integer')
    if actor_index is not None and (not isinstance(actor_index, int) or isinstance(actor_index, bool)
                                    or not 0 <= actor_index < n_agents):
        raise ValueError('reserved Actor index is outside the scenario')
    if not isinstance(profile_weights, dict) or set(profile_weights) != set(PROFILES):
        raise ValueError(f'rule profile weights must specify exactly {tuple(PROFILES)}')
    weights = [profile_weights[name] for name in PROFILES]
    total_weight = sum(weights) if all(isinstance(w, Real) for w in weights) else math.nan
    if any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 0
           for w in weights) or not math.isclose(total_weight, 1., rel_tol=0, abs_tol=1e-9):
        raise ValueError('rule profile weights must be nonnegative and sum to 1')
    weights = [w / total_weight for w in weights]

    count = math.floor(n_agents * fraction + .5)
    eligible = [i for i in range(n_agents) if i != actor_index]
    if count > len(eligible):
        raise ValueError(f'{count} rule vehicles requested but only {len(eligible)} slots are available; '
                         'set actor_index=None to allow all vehicles to be rule-controlled')

    expected = [count * w for w in weights]
    counts = [math.floor(value) for value in expected]
    for k in sorted(range(len(counts)), key=lambda k: (-(expected[k] - counts[k]), k))[:count - sum(counts)]:
        counts[k] += 1
    rng = random.Random(seed)
    indices = rng.sample(eligible, count)
    profiles = [name for name, amount in zip(PROFILES, counts) for _ in range(amount)]
    rng.shuffle(profiles)
    return dict(sorted(zip(indices, profiles)))


def route_geometry(pos, path, lengths=None, loops=None):
    """Project the vehicle centre onto its own valid route, excluding padding."""
    result = []
    for b in range(pos.shape[0]):
        count = int(lengths[b]) if lengths is not None else path.shape[1]
        if not 2 <= count <= path.shape[1]:
            raise ValueError('Rule vehicle requires at least two valid route points')
        points = path[b, :count]
        # Remove duplicate points; loop closure is then explicit.
        keep = torch.cat([torch.ones(1, dtype=torch.bool, device=path.device),
                          (points[1:] - points[:-1]).norm(dim=-1) > 1e-6])
        points = points[keep]
        if len(points) < 2:
            raise ValueError('Rule vehicle requires at least two distinct route points')
        loop = bool(loops[b]) if loops is not None else False
        if loop and (points[-1] - points[0]).norm() > 1e-6:
            points = torch.cat([points, points[:1]])
        vectors = points[1:] - points[:-1]
        segment_lengths = vectors.norm(dim=-1)
        tangents = vectors / segment_lengths[:, None]
        along = ((pos[b] - points[:-1]) * tangents).sum(-1).clamp_min(0)
        along = torch.minimum(along, segment_lengths)
        projections = points[:-1] + along[:, None] * tangents
        index = (projections - pos[b]).square().sum(-1).argmin()
        arc = torch.cat([segment_lengths.new_zeros(1), segment_lengths.cumsum(0)])
        progress = arc[index] + along[index]
        offset = pos[b] - projections[index]
        result.append((points, arc, tangents, progress, offset, loop))
    return result


def sample_route(routes, distances):
    """Interpolate by arc length, wrapping closed routes and extending exits."""
    positions, headings = [], []
    for b, (points, arc, tangents, progress, offset, loop) in enumerate(routes):
        query = progress + distances[b]
        if loop:
            query = query.remainder(arc[-1])
        index = torch.searchsorted(arc[1:].contiguous(), query.contiguous()).clamp(max=len(tangents)-1)
        sampled = points[index] + (query - arc[index])[..., None] * tangents[index]
        positions.append(sampled)
        headings.append(tangents[index])
    return torch.stack(positions), torch.stack(headings)


def rule_command(pos, yaw, speed, path, neighbor_pos, neighbor_vel, *, profile,
                 cruise_speed, steering_limit, wheelbase, sensing_range, dt,
                 was_yielding, lengths=None, loops=None, rear_length=None,
                 measured_speed=None, lookahead=0.16, acceleration=1.2, braking=2.0,
                 neighbor_yaw=None, body_length=0.16, body_width=0.08, return_details=False):
    """Centre-consistent path tracking and route-based local speed selection.

    No neighbor routes, roles or opinions are read. Candidate speeds are tested
    along the ego route against constant-velocity neighbor predictions. Nearby
    vehicles behind/beside that do not converge on this route do not force a stop.
    """
    rear_length = wheelbase / 2 if rear_length is None else rear_length
    routes = route_geometry(pos, path, lengths, loops)
    target, _ = sample_route(routes, pos.new_full((len(routes), 1), lookahead))
    target = target[:, 0] - pos
    forward = target[:, 0] * yaw.cos() + target[:, 1] * yaw.sin()
    lateral = -target[:, 0] * yaw.sin() + target[:, 1] * yaw.cos()
    # For centre-state bicycle dynamics: kappa=sin(beta)/l_r and
    # kappa=2*sin(bearing-beta)/distance. Solve beta, then steering.
    steering = torch.atan2(2 * wheelbase * lateral,
                          target.square().sum(-1) + 2 * rear_length * forward)
    steering = steering.clamp(-steering_limit, steering_limit)
    preview, _ = sample_route(routes, pos.new_tensor([0., .06, .12, .18, .24]).expand(len(routes), -1))
    a, b = preview[:, 1:-1] - preview[:, :-2], preview[:, 2:] - preview[:, 1:-1]
    cross = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).abs()
    curvature = (2 * cross / (a.norm(dim=-1) * b.norm(dim=-1)
                  * (a+b).norm(dim=-1)).clamp_min(1e-6)).amax(-1)
    # Bound lateral acceleration and slow recovery from a large tracking error.
    offset = torch.stack([r[4] for r in routes])
    error = offset.norm(dim=-1)
    cruise = torch.minimum(pos.new_full((len(routes),), cruise_speed),
                           (0.60 / curvature.clamp_min(.01)).sqrt())
    cruise = cruise / (1 + 8 * error)
    cfg = PROFILES[profile]
    times = torch.linspace(0., cfg['horizon'], 31, device=pos.device, dtype=pos.dtype)
    fractions = pos.new_tensor([1., .65, .3, 0.])
    candidates = cruise[:, None] * fractions
    current_speed = speed if measured_speed is None else measured_speed
    difference = candidates - current_speed[:, None]
    rate = torch.where(difference >= 0, acceleration, braking)
    ramp = torch.minimum(times[None, None, :], difference.abs()[..., None] / rate[..., None])
    travel = (current_speed[:, None, None] * times
              + difference.sign()[..., None] * .5 * rate[..., None] * ramp.square()
              + difference[..., None] * (times - ramp)).clamp_min(0)
    points, tangent = sample_route(routes, travel.flatten(1))
    points = points.reshape(len(routes), 4, -1, 2)
    tangent = tangent.reshape_as(points)
    # Begin at actual position, smoothly joining the intended route.
    points = points + offset[:, None, None, :] * (1 - travel / .2).clamp(0, 1)[..., None]
    future_neighbor = neighbor_pos[:, None, :, :] + times[None, :, None, None] * neighbor_vel[:, None, :, :]
    delta = future_neighbor[:, None] - points[:, :, :, None]
    longitudinal = (delta * tangent[:, :, :, None]).sum(-1)
    lateral_distance = delta[..., 0] * tangent[:, :, :, None, 1] - delta[..., 1] * tangent[:, :, :, None, 0]
    margin = was_yielding.to(pos.dtype) * .01
    if neighbor_yaw is None:
        neighbor_yaw = torch.atan2(neighbor_vel[..., 1], neighbor_vel[..., 0])
    neighbor_direction = torch.stack([neighbor_yaw.cos(), neighbor_yaw.sin()], -1)
    cosine = (neighbor_direction[:, None, None] * tangent[:, :, :, None]).sum(-1).abs()
    sine = (neighbor_direction[:, None, None, :, 0] * tangent[:, :, :, None, 1]
            - neighbor_direction[:, None, None, :, 1] * tangent[:, :, :, None, 0]).abs()
    longitudinal_extent = body_length/2 + body_length/2*cosine + body_width/2*sine
    lateral_extent = body_width/2 + body_length/2*sine + body_width/2*cosine
    longitudinal_extent += cfg['longitudinal'] - body_length + margin[:, None, None, None]
    lateral_extent += cfg['lateral'] - body_width + margin[:, None, None, None]
    # Project the neighbor footprint into the ego route frame. A crossing car
    # occupies more lateral space than a parallel one; don't treat both as points.
    metric = torch.maximum((longitudinal/longitudinal_extent).abs(),
                           (lateral_distance/lateral_extent).abs())
    relative = neighbor_pos - pos[:, None]
    _, current_tangent = sample_route(routes, pos.new_zeros((len(routes), 1)))
    ahead = (relative * current_tangent).sum(-1)
    alongside = (relative[..., 0] * current_tangent[..., 1]
                 - relative[..., 1] * current_tangent[..., 0]).abs()
    parallel_speed = (neighbor_vel * current_tangent).sum(-1)
    # A follower on a bend can be laterally offset from the ego's instantaneous
    # tangent. Treat an approaching car in the swept rear corridor as a rear
    # threat, even if its heading is not yet parallel to the exit tangent.
    rear_approach = ((ahead < -.06) & (alongside < .22)
                     & (parallel_speed > current_speed[:, None] + .08))
    behind_following = ((ahead < -.08) & (alongside < .10) & (parallel_speed > .03)
                        & (parallel_speed > .8 * neighbor_vel.norm(dim=-1)))
    rear_threat = rear_approach | behind_following
    visible = relative.norm(dim=-1) <= sensing_range
    # A rear threat must not force braking, but braking must still be scored as
    # risky when it leaves us in the follower's path.
    approaching = (metric < metric[:, :, :1] - .02)
    risk_all = ((metric < 1) & approaching & visible[:, None, None])
    risk = risk_all & ~rear_threat[:, None, None]
    blocked = risk.any(dim=-1).any(dim=-1)
    safe_candidate = ~blocked
    rear_risk = risk_all & rear_threat[:, None, None]
    # Rank the *whole trajectory*, including the stopped trajectory. Discount
    # distant overlap; prefer progress only when the resulting risk is similar.
    depth = (1 - metric).clamp_min(0) * approaching.to(pos.dtype)
    depth = depth * visible[:, None, None].to(pos.dtype)
    depth = depth * (1 - .4 * times / cfg['horizon'])[None, None, :, None]
    severity = depth.amax(dim=-1).amax(dim=-1) if neighbor_pos.shape[1] else candidates.new_zeros(candidates.shape)
    rear_severity = (depth * rear_threat[:, None, None]).amax(dim=-1).amax(dim=-1) if neighbor_pos.shape[1] else severity
    preference = severity - .04 * fractions[None]
    # When some forward-safe option exists, keep the forward constraint hard,
    # but compare rear risk instead of unconditionally picking the fastest.
    safe_preference = (rear_severity - .04 * fractions[None]).masked_fill(~safe_candidate, torch.inf)
    all_blocked = ~safe_candidate.any(-1)
    choice = torch.where(all_blocked, preference.argmin(-1), safe_preference.argmin(-1))
    if profile == 'non_yielding':
        choice.zero_()
    target_speed = candidates.gather(1, choice[:, None]).squeeze(1)
    yielding = choice > 0
    command_speed = torch.maximum(torch.minimum(target_speed, speed + acceleration * dt),
                                  (speed - braking * dt).clamp_min(0)).clamp_min(0)
    reason = torch.where(choice == 3, 2, torch.where(yielding, 1, 0))
    reason = torch.where(all_blocked, 4, reason)
    reason = torch.where((choice > 0) & (choice < 3) & ~all_blocked & ~blocked.gather(1, choice[:, None]).squeeze(1)
                         & (rear_severity.gather(1, choice[:, None]).squeeze(1) > 0), 5, reason)
    if profile == 'non_yielding':
        reason.fill_(3)
    if neighbor_pos.shape[1]:
        first_risk = torch.where(risk[:, 0], times[None, :, None], torch.inf)
        ttc, neighbor = first_risk.flatten(1).min(-1)
        neighbor = neighbor.remainder(neighbor_pos.shape[1])
        neighbor = torch.where((yielding | all_blocked) & torch.isfinite(ttc), neighbor, -1)
    else:
        ttc = pos.new_full((len(routes),), torch.inf)
        neighbor = torch.full((len(routes),), -1, device=pos.device, dtype=torch.long)
    details = dict(error=error, curvature=curvature, blocker=neighbor, ttc=ttc,
                   reason=reason, all_blocked=all_blocked, rear_risk=rear_risk.any(dim=-1).any(dim=-1).any(dim=-1))
    result = (torch.stack([command_speed, steering], -1), yielding, target_speed)
    return (*result, details) if return_details else result


class TestingRulePolicy(nn.Module):
    """Run Actor/NOD once, then replace designated slots before environment.step.

    Roles are held outside observations. Agent generation changes clear rule
    hysteresis. The action tensor stores the actually executed mixed actions.
    Rule-slot Actor log probabilities are invalidated, not presented as PPO data.
    """
    def __init__(self, policy, scenario, vehicles, *, cruise_speed=0.6):
        super().__init__()
        self.policy = policy
        self.scenario = scenario
        self.vehicles = dict(vehicles)
        if not scenario.parameters.is_testing_mode or scenario.parameters.is_continue_train:
            raise ValueError('Rule vehicles are supported only in testing mode')
        if scenario.parameters.is_using_prioritized_marl:
            raise ValueError('Testing rule vehicles do not support prioritized control')
        if not math.isfinite(cruise_speed) or not 0 < cruise_speed <= scenario.max_speed:
            raise ValueError('Rule cruise speed must be positive and within the environment limit')
        for i, profile in self.vehicles.items():
            if not isinstance(i, int) or not 0 <= i < scenario.n_agents:
                raise ValueError(f'Rule vehicle index {i} is outside the scenario')
            if profile not in PROFILES:
                raise ValueError(f'Unknown rule profile {profile}; choose from {tuple(PROFILES)}')
        self.cruise_speed = cruise_speed
        self.previous = {}
        self.neighbor_history = {}
        self.status = {}
        self.coordinator = RuleCoordinator(scenario, self.vehicles)
        scenario.testing_rule_profiles = self.vehicles

    @torch.no_grad()
    def forward(self, td):
        s = self.scenario
        if not s.parameters.is_testing_mode or s.parameters.is_continue_train:
            raise RuntimeError('Do not use mixed test actions for training')
        td = self.policy(td)  # NOD observes all real vehicles, without role labels.
        actions = td['agents', 'action'].clone()
        pos = torch.stack([a.state.pos for a in s.world.agents], 1)
        vel = torch.stack([a.state.vel for a in s.world.agents], 1)
        mask = torch.zeros_like(actions[..., :1], dtype=torch.bool)
        yielding_mask = torch.zeros_like(mask)
        targets = torch.zeros_like(actions[..., :1])
        diagnostics = {}
        for i, profile in self.vehicles.items():
            agent = s.world.agents[i]
            generation = s.nod_agent_generation[:, i]
            measured_speed = vel[:, i].norm(dim=-1)
            old_generation, yielding, previous_speed = self.previous.get(
                i, (generation, torch.zeros_like(generation, dtype=torch.bool), measured_speed))
            same_generation = old_generation == generation
            yielding = yielding & same_generation
            previous_speed = torch.where(same_generation, previous_speed, measured_speed)
            # Other rule vehicles are handled together by trajectory reservation;
            # their known intended motion must not also trigger local yielding.
            other = [j for j in range(s.n_agents) if j not in self.vehicles]
            other_generation = s.nod_agent_generation[:, other]
            prior_generation, prior_velocity = self.neighbor_history.get(
                i, (other_generation, vel[:, other]))
            smoothed_velocity = torch.where(
                ((other_generation == prior_generation) & same_generation[:, None])[..., None],
                .65 * vel[:, other] + .35 * prior_velocity, vel[:, other])
            self.neighbor_history[i] = (other_generation.clone(), smoothed_velocity.clone())
            command, yielding, target, details = rule_command(
                pos[:, i], agent.state.rot.squeeze(-1), previous_speed,
                s.ref_paths_agent_related.long_term[:, i], pos[:, other], smoothed_velocity,
                lengths=s.ref_paths_agent_related.n_points_long_term[:, i],
                loops=s.ref_paths_agent_related.is_loop[:, i],
                rear_length=agent.dynamics.l_r, measured_speed=measured_speed, return_details=True,
                neighbor_yaw=torch.stack([s.world.agents[j].state.rot.squeeze(-1) for j in other], -1)
                    if other else pos.new_empty((pos.shape[0], 0)),
                profile=profile, cruise_speed=self.cruise_speed,
                steering_limit=float(s.max_steering_angle),
                wheelbase=agent.dynamics.l_f + agent.dynamics.l_r,
                sensing_range=s.parameters.nod_sensing_range, dt=s.parameters.dt,
                was_yielding=yielding)
            command[:, 0].clamp_(0, s.max_speed)
            actions[:, i] = command
            mask[:, i] = True
            yielding_mask[:, i, 0] = yielding
            targets[:, i, 0] = target
            self.previous[i] = (generation.clone(), yielding.clone(), command[:, 0].clone())
            details['blocker'] = torch.where(details['blocker'] >= 0,
                torch.tensor(other, device=pos.device)[details['blocker'].clamp_min(0)]
                if other else details['blocker'], -1)
            self.status[i] = (yielding.clone(), command.clone(), details)
            for key, value in details.items():
                if key not in diagnostics:
                    diagnostics[key] = torch.zeros_like(actions[..., :1], dtype=value.dtype)
                diagnostics[key][:, i, 0] = value
        actions, coordination, blocking, wait_time, infeasible = self.coordinator.coordinate(actions)
        for i in self.vehicles:
            yielding, _, details = self.status[i]
            active = coordination[:, i] != 0
            details['reason'] = torch.where(active, coordination[:, i], details['reason'])
            details['blocker'] = torch.where(active & (blocking[:, i] >= 0), blocking[:, i], details['blocker'])
            details['reservation_wait'] = wait_time[:, i]
            details['reservation_infeasible'] = infeasible[:, i]
            yielding = yielding | (coordination[:, i] == 8)
            yielding_mask[:, i, 0] = yielding
            targets[:, i, 0] = torch.where(active, actions[:, i, 0], targets[:, i, 0])
            generation = s.nod_agent_generation[:, i]
            self.previous[i] = (generation.clone(), yielding.clone(), actions[:, i, 0].clone())
            self.status[i] = (yielding.clone(), actions[:, i].clone(), details)
            for key, value in details.items():
                if key not in diagnostics:
                    diagnostics[key] = torch.zeros_like(actions[..., :1], dtype=value.dtype)
                diagnostics[key][:, i, 0] = value
        td.set(('agents', 'action'), actions)
        td.set(('agents', 'rule_controlled'), mask)
        td.set(('agents', 'rule_yielding'), yielding_mask)
        td.set(('agents', 'rule_target_speed'), targets)
        for key, value in diagnostics.items():
            td.set(('agents', 'rule_' + key), value)
        log_prob = td.get(('agents', 'sample_log_prob'), default=None)
        if log_prob is not None:
            log_prob = log_prob.clone()
            for i in self.vehicles:
                log_prob[:, i] = float('nan')
            td.set(('agents', 'sample_log_prob'), log_prob)
        return td

    def overlay_lines(self, env_index=0):
        lines = ['Rule vehicles (visualization only)']
        for i, profile in self.vehicles.items():
            if i in self.status:
                yielding, action, details = self.status[i]
                state = REASONS[int(details['reason'][env_index])]
                blocker = int(details['blocker'][env_index])
                who = f'A{blocker+1}' if blocker >= 0 else '-'
                lines.append(f'A{i+1}: {profile} | {state} {who} | v={float(action[env_index,0]):.2f}')
                lines.append(f'  route error={float(details["error"][env_index]):.3f}m')
        return lines

    def save_diagnostics(self, rollout, path):
        """Log every transition, including final steps and pre-reset contacts."""
        fields = ['env', 'step', 't_sec', 'agent', 'generation', 'profile', 'reason',
                  'blocker', 'ttc', 'all_blocked', 'rear_risk', 'reservation_wait', 'reservation_infeasible',
                  'rule_contact', 'speed_command', 'target_speed', 'actual_speed_next',
                  'route_error', 'route_error_next', 'road_contact', 'vehicle_contact',
                  'road_events', 'vehicle_contact_events']
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for env_i in range(rollout.shape[0]):
                for i, profile in self.vehicles.items():
                    road_events = vehicle_events = 0
                    old_road = old_vehicle = False
                    old_generation = None
                    for t in range(rollout.shape[1]):
                        td = rollout[env_i, t]
                        generation = int(td['agents', 'info', 'nod_ego_generation'][i])
                        if generation != old_generation:
                            old_road = old_vehicle = False
                        road = bool(td['next', 'agents', 'info', 'testing_road_contact'][i])
                        vehicle = bool(td['next', 'agents', 'info', 'is_collision_with_agents'][i])
                        road_events += int(road and not old_road)
                        vehicle_events += int(vehicle and not old_vehicle)
                        old_road, old_vehicle, old_generation = road, vehicle, generation
                        blocker = int(td['agents', 'rule_blocker'][i])
                        writer.writerow(dict(env=env_i, step=t+1, t_sec=(t+1)*self.scenario.parameters.dt,
                            agent=i+1, generation=generation, profile=profile,
                            reason=REASONS[int(td['agents', 'rule_reason'][i])],
                            blocker=blocker+1 if blocker >= 0 else '',
                            ttc=float(td['agents', 'rule_ttc'][i]),
                            all_blocked=int(td['agents', 'rule_all_blocked'][i]),
                            rear_risk=int(td['agents', 'rule_rear_risk'][i]),
                            reservation_wait=float(td['agents', 'rule_reservation_wait'][i]),
                            reservation_infeasible=int(td['agents', 'rule_reservation_infeasible'][i]),
                            rule_contact=int(td.get(('next', 'agents', 'info', 'testing_rule_contact'),
                                                    default=torch.zeros(self.scenario.n_agents, dtype=torch.bool))[i]),
                            speed_command=float(td['agents', 'action'][i, 0]),
                            target_speed=float(td['agents', 'rule_target_speed'][i]),
                            actual_speed_next=float(td['next', 'agents', 'info', 'nod_world_vel'][i].norm()),
                            route_error=float(td['agents', 'rule_error'][i]),
                            route_error_next=float(td['next', 'agents', 'info', 'testing_route_error'][i]),
                            road_contact=int(road), vehicle_contact=int(vehicle),
                            road_events=road_events, vehicle_contact_events=vehicle_events))
