"""Central trajectory reservations for test-only rule vehicles.

Every accepted replacement is checked against the other reserved trajectories,
including their terminal stop. Plans survive across steps, so a vehicle cannot
withdraw a previously granted crossing just because another car starts moving.
This coordinator never changes Actor slots or exposes roles to Actor/NOD.
"""
import numpy as np
import torch


CONTROLLER_VERSION = 'coordinated_rules_v4'
DEFAULT_RULE_LATERAL_ACCEL = 1.8


def swept_conflict(a, b, size_a, size_b, margin=.012):
    """Conservative OBB sweep over each VMAS integration interval (x,y,yaw).

    Enclose the whole translation/rotation in a rectangle at the interval's
    midpoint, then apply the separating-axis test. Also detects containment.
    """
    def boxes(trajectory, size):
        mid = (trajectory[:-1] + trajectory[1:]) * .5
        delta = trajectory[1:] - trajectory[:-1]
        u = np.stack([np.cos(mid[:, 2]), np.sin(mid[:, 2])], -1)
        v = np.stack([-u[:, 1], u[:, 0]], -1)
        rotation = np.hypot(*size) * .5 * np.abs(delta[:, 2]) * .5
        half = np.stack([size[0]/2 + margin + rotation + np.abs((delta[:, :2]*u).sum(-1))/2,
                         size[1]/2 + margin + rotation + np.abs((delta[:, :2]*v).sum(-1))/2], -1)
        return mid[:, :2], u, v, half
    ca, ua, va, ha = boxes(a, size_a)
    cb, ub, vb, hb = boxes(b, size_b)
    separated = np.zeros(len(ca), dtype=bool)
    for axis in (ua, va, ub, vb):
        radius = (ha[:, 0]*np.abs((ua*axis).sum(-1)) + ha[:, 1]*np.abs((va*axis).sum(-1))
                  + hb[:, 0]*np.abs((ub*axis).sum(-1)) + hb[:, 1]*np.abs((vb*axis).sum(-1)))
        separated |= np.abs(((cb-ca)*axis).sum(-1)) > radius
    return bool((~separated).any())


def bicycle_step(state, velocity, angular_velocity, action, *, dt, lf, lr,
                 drag, max_speed, integration='rk4'):
    """Match KinematicBicycle.process_action + one VMAS world substep."""
    speed, steering = action
    beta = np.arctan(np.tan(steering)*lr/(lf+lr))
    omega = speed*np.cos(beta)*np.tan(steering)/(lf+lr)
    def derivative(theta):
        return np.array([speed*np.cos(theta+beta), speed*np.sin(theta+beta), omega])
    k1 = derivative(state[2])
    if integration == 'euler':
        change = dt*k1
    else:
        k2 = derivative(state[2]+dt*omega/2)
        k4 = derivative(state[2]+dt*omega)
        change = dt*(k1+4*k2+k4)/6
    new_velocity = change[:2]/dt - drag*velocity
    new_velocity *= min(1., max_speed/max(np.linalg.norm(new_velocity), 1e-12))
    new_angular = change[2]/dt - drag*angular_velocity
    return state + dt*np.r_[new_velocity, new_angular], new_velocity, new_angular


class RuleCoordinator:
    def __init__(self, scenario, vehicles, horizon=1.2, *, lateral_accel_limit=DEFAULT_RULE_LATERAL_ACCEL):
        if not np.isfinite(lateral_accel_limit) or lateral_accel_limit <= 0:
            raise ValueError('Rule lateral acceleration limit must be positive and finite')
        self.scenario = scenario
        self.lateral_accel_limit = float(lateral_accel_limit)
        self.indices = sorted(vehicles)
        self.dt = float(scenario.parameters.dt)
        self.steps = max(8, int(np.ceil(horizon/self.dt)))
        self.plans = {}
        self.ages = {}
        self.owners = {}
        self.requests = {}
        self.zones = {}
        parser = getattr(getattr(scenario, 'map', None), 'parser', None)
        if (getattr(scenario.parameters, 'scenario_type', '').startswith('intersection_')
                and parser is not None):
            references = parser.reference_paths
            geometry = []
            for ref in references:
                p = ref['center_line'].detach().cpu().numpy()
                lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
                arc = np.r_[0., lengths.cumsum()]
                query = np.arange(0., arc[-1], .025)
                index = np.minimum(np.searchsorted(arc[1:], query), len(lengths)-1)
                tangent = np.diff(p, axis=0)/np.maximum(lengths[:, None], 1e-9)
                samples = p[index]+(query-arc[index])[:,None]*tangent[index]
                geometry.append((samples, tangent[index], query, arc))
            centers = []
            for i, first in enumerate(references):
                p = first['center_line'].detach().cpu().numpy()
                for second in references[i+1:]:
                    q = second['center_line'].detach().cpu().numpy()
                    for k, x in enumerate(np.diff(p, axis=0)):
                        for l, y in enumerate(np.diff(q, axis=0)):
                            cross = x[0]*y[1]-x[1]*y[0]
                            if abs(cross) <= .3*np.linalg.norm(x)*np.linalg.norm(y)+1e-10:
                                continue
                            delta = q[l]-p[k]
                            t = (delta[0]*y[1]-delta[1]*y[0])/cross
                            u = (delta[0]*x[1]-delta[1]*x[0])/cross
                            if 0 <= t <= 1 and 0 <= u <= 1:
                                center = p[k]+t*x
                                if all(np.linalg.norm(center-old)>.30 for old in centers):
                                    centers.append(center)
            for zone_id, center in enumerate(centers):
                bounds = {}
                for i, (points, _, query, arc) in enumerate(geometry):
                    conflict = np.linalg.norm(points-center, axis=-1) < .17
                    if conflict.any():
                        bounds[i] = (max(0., float(query[conflict].min())-.09),
                                     min(float(arc[-1]), float(query[conflict].max())+.12))
                self.zones[zone_id] = bounds
            for i, ref in enumerate(references):
                ref['testing_rule_conflict_intervals'] = [bounds[i] for bounds in self.zones.values() if i in bounds]
                ref['testing_rule_arc'] = torch.as_tensor(geometry[i][3], device=ref['center_line'].device)
        # The dynamics predictor intentionally supports the world configuration
        # used here; reject incompatible worlds rather than claiming safety.
        world = scenario.world
        if getattr(world, '_substeps', 1) != 1:
            raise ValueError('Rule coordination requires a single VMAS world substep')

    def _vehicle(self, i, env):
        s = self.scenario
        agent = s.world.agents[i]
        count = int(s.ref_paths_agent_related.n_points_long_term[env, i])
        path = s.ref_paths_agent_related.long_term[env, i, :count].detach().cpu().numpy().copy()
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1)>1e-6]]
        loop = bool(s.ref_paths_agent_related.is_loop[env, i])
        if loop and np.linalg.norm(path[-1]-path[0])>1e-6:
            path = np.concatenate([path, path[:1]])
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        tangents = np.diff(path, axis=0)/lengths[:, None]
        drag = getattr(agent, 'drag', None)
        drag = float(getattr(s.world, '_drag', .25) if drag is None else drag)
        shape = getattr(agent, 'shape', None)
        return dict(path=path, lengths=lengths, tangents=tangents,
                    arc=np.r_[0., lengths.cumsum()], loop=loop,
                    size=(float(getattr(shape, 'length', .16)), float(getattr(shape, 'width', .08))),
                    dt=self.dt, lf=float(agent.dynamics.l_f), lr=float(agent.dynamics.l_r),
                    drag=drag, max_speed=float(s.max_speed),
                    integration=getattr(agent.dynamics, 'integration', 'rk4'),
                    steering_limit=float(s.max_steering_angle))

    @staticmethod
    def _project_to_route(position, car):
        """Return route progress, lateral error, and local tangent."""
        position = np.asarray(position)[..., None, :]
        along = np.clip(((position-car['path'][:-1])*car['tangents']).sum(-1), 0, car['lengths'])
        projected = car['path'][:-1] + along[..., None]*car['tangents']
        squared_error = ((projected-position)**2).sum(-1)
        index = np.argmin(squared_error, axis=-1)
        selected = np.expand_dims(index, -1)
        progress = car['arc'][index] + np.take_along_axis(along, selected, axis=-1)[..., 0]
        error = np.sqrt(np.take_along_axis(squared_error, selected, axis=-1)[..., 0])
        return progress, error, car['tangents'][index]

    @classmethod
    def _progress(cls, position, car):
        return cls._project_to_route(position, car)[0]

    def _leader_relations(self, cars, initial):
        """Find same-lane leaders that must be planned before their followers.

        Paths through an intersection can differ before merging but share the
        same exit.  Projecting the follower onto the leader's route recognizes
        that shared section without treating perpendicular crossing traffic as
        a queue.  The distance bound keeps unrelated parallel lanes separate.
        """
        relations = set()
        for leader in self.indices:
            leader_state = initial[leader][0]
            leader_progress, leader_error, leader_tangent = self._project_to_route(
                leader_state[:2], cars[leader])
            leader_heading = np.array([np.cos(leader_state[2]), np.sin(leader_state[2])])
            if leader_error > .08 or leader_heading @ leader_tangent < .5:
                continue
            for follower in self.indices:
                if follower == leader:
                    continue
                follower_state = initial[follower][0]
                follower_progress, follower_error, follower_tangent = self._project_to_route(
                    follower_state[:2], cars[leader])
                follower_heading = np.array([np.cos(follower_state[2]), np.sin(follower_state[2])])
                gap = float(leader_progress-follower_progress)
                same_direction = (leader_heading @ follower_heading > .6
                                  and follower_heading @ follower_tangent > .5)
                lane_tolerance = .5*(cars[leader]['size'][1]+cars[follower]['size'][1])+.025
                if same_direction and follower_error <= lane_tolerance and .04 < gap < .9:
                    relations.add((leader, follower))
        return relations

    def _priority_order(self, cars, initial, ages, plans):
        """Fair reservation order with hard leader-before-follower precedence."""
        base_key = lambda i: (-(ages[i]+(.5 if plans[i][0, 0] > .05 else 0.)), i)
        predecessors = {i: set() for i in self.indices}
        for leader, follower in self._leader_relations(cars, initial):
            predecessors[follower].add(leader)
        remaining = set(self.indices)
        order = []
        while remaining:
            available = [i for i in remaining if not (predecessors[i] & remaining)]
            # Numerical projection ambiguity can create a cycle.  In that rare
            # case fall back to the established fair order for one vehicle.
            chosen = min(available or list(remaining), key=base_key)
            order.append(chosen)
            remaining.remove(chosen)
        return order

    def _assign_crossing(self, env, cars, initial):
        """Independent geometric crossing grants, retained until the rear clears."""
        # Do not hold a downstream grant while still waiting for an upstream
        # crossing. This can block the very vehicle needed to clear our queue.
        next_zone = {}
        for i in self.indices:
            path_id = int(self.scenario.ref_paths_agent_related.path_id[env,i]) if self.zones else -1
            progress = self._progress(initial[i][0][:2], cars[i])
            ahead = [(bounds[path_id][0], zone_id) for zone_id, bounds in self.zones.items()
                     if path_id in bounds and progress <= bounds[path_id][1]]
            next_zone[i] = min(ahead)[1] if ahead else None
        for zone_id, bounds in self.zones.items():
            self._assign_zone(env, zone_id, bounds, cars, initial, next_zone)

    def _assign_zone(self, env, zone_id, zone_bounds, cars, initial, next_zone):
        s = self.scenario
        progress = {i: self._progress(initial[i][0][:2], cars[i]) for i in self.indices}
        bounds = {i: zone_bounds.get(int(s.ref_paths_agent_related.path_id[env,i])) for i in self.indices}
        owners = self.owners.setdefault((env,zone_id), {})
        for i in list(owners):
            if (owners[i] != int(s.nod_agent_generation[env,i]) or bounds[i] is None
                    or progress[i] > bounds[i][1]
                    or (progress[i] < bounds[i][0] and next_zone[i] != zone_id)):
                del owners[i]
        # Support existing cars already in the junction when wrapping a live
        # scenario; admission remains closed until every occupant has cleared.
        for i in self.indices:
            if bounds[i] and bounds[i][0] < progress[i] <= bounds[i][1]:
                owners[i] = int(s.nod_agent_generation[env,i])
        eligible = []
        for i in self.indices:
            key = (env,zone_id,i)
            generation = int(s.nod_agent_generation[env,i])
            if not bounds[i] or progress[i] > bounds[i][1]:
                self.requests.pop(key, None)
                continue
            entry = bounds[i][0]
            if next_zone[i] == zone_id and progress[i] >= entry-.4:
                old = self.requests.get(key)
                if old is None or old[0] != generation:
                    self.requests[key] = (generation, 0.)
                generation, age = self.requests[key]
                self.requests[key] = (generation, age+self.dt)
                # A rear queue member cannot hold the token needed by its leader.
                obstructed = False
                for j in self.indices:
                    if i == j:
                        continue
                    leader = self._progress(initial[j][0][:2], cars[i])
                    queue_ahead = progress[i]+.05 < leader < entry+.05
                    exit_occupied = bounds[i][1] <= leader < bounds[i][1]+.22
                    if queue_ahead or exit_occupied:
                        index = min(np.searchsorted(cars[i]['arc'][1:], leader), len(cars[i]['lengths'])-1)
                        projected = cars[i]['path'][index]+(leader-cars[i]['arc'][index])*cars[i]['tangents'][index]
                        if np.linalg.norm(projected-initial[j][0][:2]) < .12:
                            obstructed = True
                if not obstructed:
                    eligible.append(i)
        if not owners and eligible:
            winner = min(eligible, key=lambda i: (-self.requests[env,zone_id,i][1], bounds[i][0]-progress[i], i))
            owners[winner] = int(s.nod_agent_generation[env,winner])
        for i in self.indices:
            if bounds[i] and progress[i] <= bounds[i][0] and i not in owners:
                stop_at = bounds[i][0]-.035
                if stop_at < cars[i].get('stop_at', np.inf):
                    cars[i]['stop_at'] = stop_at
                    cars[i]['gate_owner'] = next(iter(owners), -1)

    @staticmethod
    def _steering(state, car):
        path, tangents, lengths = car['path'], car['tangents'], car['lengths']
        along = np.clip(((state[:2]-path[:-1])*tangents).sum(-1), 0, lengths)
        nearest = np.argmin(((path[:-1]+along[:, None]*tangents-state[:2])**2).sum(-1))
        query = car['arc'][nearest]+along[nearest]+.16
        if car['loop']:
            query %= car['arc'][-1]
        index = min(np.searchsorted(car['arc'][1:], query), len(lengths)-1)
        delta = path[index]+(query-car['arc'][index])*tangents[index]-state[:2]
        forward = delta[0]*np.cos(state[2])+delta[1]*np.sin(state[2])
        lateral = -delta[0]*np.sin(state[2])+delta[1]*np.cos(state[2])
        angle = np.arctan2(2*(car['lf']+car['lr'])*lateral,
                           delta@delta+2*car['lr']*forward)
        return np.clip(angle, -car['steering_limit'], car['steering_limit'])

    def _predict(self, initial, car, *, commands=None, desired=None, fraction=1., drive_steps=0):
        state, vel, angular = initial
        state, vel = state.copy(), vel.copy()
        trajectory = [state.copy()]
        generated = []
        speed = float(np.linalg.norm(vel))
        physics = {key: car[key] for key in ('dt', 'lf', 'lr', 'drag', 'max_speed', 'integration')}
        for step in range(self.steps):
            if commands is not None:
                action = commands[step]
            elif desired is not None and step < drive_steps:
                steering = desired[1] if step == 0 else self._steering(state, car)
                curvature = abs(np.cos(np.arctan(np.tan(steering)*car['lr']/(car['lf']+car['lr'])))
                                *np.tan(steering)/(car['lf']+car['lr']))
                target = min(desired[0]*fraction, np.sqrt(self.lateral_accel_limit/max(curvature, .01)))
                # First command already includes the local comfort limiter.
                speed = target if step == 0 else min(target, speed+1.2*self.dt)
                if 'stop_at' in car:
                    distance = car['stop_at']-self._progress(state[:2], car)
                    speed = min(speed, max(0., distance/self.dt)*.7)
                action = np.array([speed, steering])
            else:
                # Emergency braking bypasses the comfort limiter. VMAS drag is
                # included, including its small transient reverse displacement.
                action = np.zeros(2)
                speed = 0.
            state, vel, angular = bicycle_step(state, vel, angular, action, **physics)
            trajectory.append(state.copy())
            generated.append(action)
        return np.asarray(trajectory), np.asarray(generated)

    def coordinate(self, actions):
        s = self.scenario
        output = actions.clone()
        shape = actions.shape[:2]
        reason = torch.zeros(shape, dtype=torch.long, device=actions.device)
        blocker = torch.full(shape, -1, dtype=torch.long, device=actions.device)
        waiting = actions.new_zeros(shape)
        infeasible = torch.zeros(shape, dtype=torch.bool, device=actions.device)
        if len(self.indices) < 2:
            return output, reason, blocker, waiting, infeasible
        for env in range(actions.shape[0]):
            cars, initial, trajectories, plans, ages = {}, {}, {}, {}, {}
            for i in self.indices:
                agent = s.world.agents[i]
                cars[i] = self._vehicle(i, env)
                initial[i] = (np.r_[agent.state.pos[env].detach().cpu().numpy(),
                                    float(agent.state.rot[env, 0])],
                              agent.state.vel[env].detach().cpu().numpy(),
                              float(agent.state.ang_vel[env, 0]) if hasattr(agent.state, 'ang_vel') else 0.)
            self._assign_crossing(env, cars, initial)
            stop_trajectories, stop_plans = {}, {}
            for i in self.indices:
                stop_trajectories[i], stop_plans[i] = self._predict(initial[i], cars[i])
                generation = int(s.nod_agent_generation[env, i])
                old = self.plans.get((env, i))
                if old is not None and old[0] == generation:
                    commands = np.concatenate([old[1][1:], np.zeros((1, 2))])
                    trajectories[i], plans[i] = self._predict(initial[i], cars[i], commands=commands)
                    ages[i] = self.ages.get((env, i), 0.)
                else:
                    trajectories[i], plans[i] = stop_trajectories[i], stop_plans[i]
                    ages[i] = 0.

            def conflicts(i, trajectory, pending=()):
                # A vehicle that has not had its turn exposes its physically
                # predicted braking path.  Otherwise an old rear-vehicle plan
                # can reserve the leader's free road and invert queue priority.
                blocked = [j for j in self.indices if j != i and swept_conflict(
                    trajectory, stop_trajectories[j] if j in pending else trajectories[j],
                    cars[i]['size'], cars[j]['size'])]
                if 'stop_at' in cars[i]:
                    # Retained plans also obey today's gate. A plan that was
                    # allowed yesterday cannot bypass a changed crossing grant.
                    limit = max(cars[i]['stop_at']+.008, self._progress(initial[i][0][:2], cars[i]))
                    if (self._progress(trajectory[1:,:2], cars[i]) > limit+1e-6).any():
                        blocked.append(-2)
                return blocked

            # A respawn can invalidate old reservations. Start again with stop
            # trajectories, then admit feasible motion from the actual state.
            if any(conflicts(i, trajectories[i]) for i in self.indices):
                for i in self.indices:
                    trajectories[i], plans[i] = stop_trajectories[i], stop_plans[i]

            # Existing motion gets a short commitment and waiting age resolves
            # crossing ties, but a same-lane leader always precedes its rear car.
            order = self._priority_order(cars, initial, ages, plans)
            pending = set(order)
            for i in order:
                pending.remove(i)
                desired = actions[env, i].detach().cpu().numpy()
                # Reserve movement plus a terminal stop; short reservations
                # permit approaching the stop line without entering a conflict.
                chosen = False
                first_blocker = cars[i].get('gate_owner', -1)
                for fraction, duration in ((1., self.steps-4), (.65, self.steps-4),
                                           (1., max(1, self.steps//3)), (.3, self.steps-4), (0.,0)):
                    trajectory, commands = self._predict(initial[i], cars[i], desired=desired,
                                                          fraction=fraction, drive_steps=duration)
                    blocked_by = conflicts(i, trajectory, pending)
                    if not blocked_by:
                        trajectories[i], plans[i] = trajectory, commands
                        chosen = True
                        break
                    if first_blocker < 0:
                        first_blocker = blocked_by[0]
                # If a new stop would block a granted crossing, continue the
                # previously reserved, still collision-free action sequence.
                invalid = bool(conflicts(i, trajectories[i], pending))
                infeasible[env, i] = invalid
                executed = plans[i][0]
                output[env, i] = torch.as_tensor(executed, device=actions.device, dtype=actions.dtype)
                limited = executed[0] < desired[0]-1e-4
                reason[env, i] = 8 if limited else (7 if executed[0] > .03 else 0)
                if not chosen:
                    reason[env, i] = 9
                blocker[env, i] = first_blocker if limited or invalid else -1
                ages[i] = ages[i]+self.dt if limited else max(0., ages[i]-self.dt)
                waiting[env, i] = ages[i]
                self.ages[env, i] = ages[i]
                self.plans[env, i] = (int(s.nod_agent_generation[env, i]), plans[i].copy())
            # A later vehicle can resolve an earlier temporary stop conflict.
            # Report feasibility of the final joint plan, not intermediate plans.
            for i in self.indices:
                infeasible[env, i] = bool(conflicts(i, trajectories[i]))
        return output, reason, blocker, waiting, infeasible
