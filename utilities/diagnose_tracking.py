"""Frozen-policy collision windows and paired traffic-removal replay.

Diagnostic only: no training, action correction, or checkpoint selection.
Never tune on these test-map results; use independent validation routes for selection.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utilities.helper_training import Parameters
from utilities.helper_scenario import get_distances_between_agents
from utilities.constants import SCENARIOS


def project_path(position, yaw, velocity, path):
    """Signed lateral error, heading error and actual normal velocity in SI units."""
    segments = path[1:] - path[:-1]
    length2 = segments.square().sum(-1)
    if not (length2 > 1e-12).any():
        raise ValueError('Reference path has no nonzero segment')
    u = ((position - path[:-1]) * segments).sum(-1) / length2.clamp_min(1e-12)
    feet = path[:-1] + u.clamp(0, 1)[:, None] * segments
    distance = (position - feet).square().sum(-1).masked_fill(length2 <= 1e-12, torch.inf)
    index = int(distance.argmin())
    tangent = segments[index] / length2[index].sqrt()
    normal = torch.stack((-tangent[1], tangent[0]))
    heading = yaw - torch.atan2(tangent[1], tangent[0])
    heading = torch.atan2(torch.sin(heading), torch.cos(heading))
    return dict(lateral_error_m=float((position - feet[index]) @ normal),
                heading_error_deg=float(heading * 180 / math.pi),
                normal_velocity_mps=float(velocity @ normal),
                route_s_m=float(length2.sqrt()[:index].sum() + u[index].clamp(0, 1) * length2[index].sqrt()))


def frame(scenario, agent_index):
    a = scenario.world.agents[agent_index]
    pos, yaw, vel = a.state.pos[0], a.state.rot[0, 0], a.state.vel[0]
    refs = scenario.ref_paths_agent_related
    count = int(refs.n_points_long_term[0, agent_index])
    result = project_path(pos, yaw, vel, refs.long_term[0, agent_index, :count])
    others = [b.state.pos[0] for i, b in enumerate(scenario.world.agents) if i != agent_index and b.movable]
    nearest = float((torch.stack(others) - pos).norm(dim=-1).min()) if others else None
    result.update(x_m=float(pos[0]), y_m=float(pos[1]), yaw_rad=float(yaw),
                  speed_mps=float(vel.norm()), nearest_neighbor_m=nearest,
                  road_collision=bool(scenario.collisions.with_lanelets[0, agent_index]),
                  agent_collision=bool(scenario.collisions.with_agents[0, agent_index].any()),
                  route_id=int(refs.path_id[0, agent_index]),
                  generation=int(scenario.nod_agent_generation[0, agent_index]))
    return result


def rng_snapshot():
    return torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())


@contextmanager
def replay_rng(state):
    outer = rng_snapshot()
    try:
        torch.set_rng_state(state[0]); random.setstate(state[1]); np.random.set_state(state[2])
        yield
    finally:
        torch.set_rng_state(outer[0]); random.setstate(outer[1]); np.random.set_state(outer[2])


def fork_env(env):
    # Frozen managers contain non-pickleable RNG objects. They are read-only in
    # this diagnostic (NOD and deadlock are rejected), so share just managers.
    s = env.base_env.scenario
    memo = {id(v): v for k, v in vars(s).items() if k.endswith('_manager')}
    return copy.deepcopy(env, memo)


def step_physical(env, policy, decision, only_agent=None):
    """Capture physical state before VMAS done() respawns cars."""
    s = env.base_env.scenario
    policy(decision)
    if only_agent is not None:
        action = decision['agents', 'action']
        selected = action[:, only_agent].clone()
        action.zero_(); action[:, only_agent] = selected
    actions = decision['agents', 'action'].clone()
    original = s.done
    physical = []
    def capture():
        physical.append([frame(s, i) for i in range(s.n_agents)])
        return original()
    s.done = capture
    try:
        transition, following = env.step_and_maybe_reset(decision)
    finally:
        s.done = original
    if len(physical) != 1:
        raise RuntimeError('Unexpected whole-env reset inside diagnostic window')
    return transition, following, physical[0], actions


def remove_traffic(env, decision, ego):
    """Physically park other slots and use the environment's absent-neighbor encoding.

    Actor input width stays fixed. V2 was trained with masking off, so this
    observation intervention can be out of distribution and is not causal proof.
    """
    s = env.base_env.scenario
    p = s.parameters
    if not (p.is_ego_view and p.is_partial_observation and p.is_observe_vertices
            and p.is_observe_distance_to_agents and not p.is_observe_ref_path_other_agents):
        raise ValueError('Traffic removal supports the v2 vertices/velocity/distance neighbor layout only')
    original_pos = s.world.agents[ego].state.pos.clone()
    original_ref = s.ref_paths_agent_related.long_term[:, ego].clone()
    for i, a in enumerate(s.world.agents):
        if i == ego:
            continue
        a.set_pos(torch.tensor([[100. + 10. * i, 100.]], device=s.world.device), batch_index=None)
        a.state.vel.zero_(); a.state.ang_vel.zero_()
        a._movable = False
        a._collide = False
        s._reset_init_distances_and_short_term_ref_path(0, i, s.world.agents)
    s.distances.agents = get_distances_between_agents(s, s.distances.type, is_set_diagonal=True)
    original_update = s._update_state_before_rewarding
    def without_ghost_collisions(agent, index):
        original_update(agent, index)
        # Ghost slots must never collide, exit, or respawn. Ego road flags stay.
        s.collisions.with_agents.zero_()
        for i in range(s.n_agents):
            if i != ego:
                s.collisions.with_lanelets[:, i] = False
                s.collisions.with_entry_segments[:, i] = False
                s.collisions.with_exit_segments[:, i] = False
    s._update_state_before_rewarding = without_ghost_collisions
    original_other = s._observe_other_agents
    def absent_neighbors(index):
        # Per neighbor: 8 vertex coordinates=1, 2 velocity components=0,
        # distance=1. These are exactly the environment's existing mask values.
        observed = original_other(index)
        absent = torch.ones_like(observed).reshape(s.world.batch_dim, -1, 11)
        absent[..., 8:10] = 0
        return absent.reshape_as(observed)
    s._observe_other_agents = absent_neighbors
    mask = torch.ones(s.world.batch_dim, dtype=torch.bool, device=s.world.device)
    observations = [s.observation(a, refresh_mask=mask).clone() for a in s.world.agents]
    decision = decision.clone()
    decision.set(('agents', 'observation'), torch.stack(observations, 1))
    assert torch.equal(original_pos, s.world.agents[ego].state.pos)
    assert torch.equal(original_ref, s.ref_paths_agent_related.long_term[:, ego])
    return decision


def paired_replay(snapshot, policy, ego, steps, reference):
    outcomes = {}
    for mode in ('traffic', 'isolated'):
        env = fork_env(snapshot['env'])
        decision = snapshot['decision'].clone()
        try:
            with replay_rng(snapshot['rng']):
                if mode == 'isolated':
                    decision = remove_traffic(env, decision, ego)
                start = frame(env.base_env.scenario, ego)
                rows = []
                for t in range(steps):
                    before = frame(env.base_env.scenario, ego)
                    _, decision, physical, actions = step_physical(
                        env, policy, decision, ego if mode == 'isolated' else None)
                    after = physical[ego]
                    if mode == 'traffic':
                        expected = reference[t]
                        if (abs(after['x_m'] - expected['x_m']) > 1e-5
                                or abs(after['y_m'] - expected['y_m']) > 1e-5
                                or after['road_collision'] != expected['road_collision']):
                            raise RuntimeError('Traffic replay differs from original window; rejecting pair')
                    after.update(t_seconds=(t + 1) * env.base_env.scenario.world.dt,
                                 command_speed=float(actions[0, ego, 0]),
                                 command_steer_rad=float(actions[0, ego, 1]),
                                 lateral_growth_m=abs(after['lateral_error_m']) - abs(before['lateral_error_m']))
                    rows.append(after)
                    # Do not interpret recovery after respawn as successful tracking.
                    if (after['road_collision'] or after['agent_collision']
                            or int(env.base_env.scenario.nod_agent_generation[0, ego]) != start['generation']):
                        break
                outcomes[mode] = dict(start=start, rows=rows,
                    road_collision=any(r['road_collision'] for r in rows),
                    agent_collision=any(r['agent_collision'] for r in rows),
                    completed_steps=len(rows),
                    peak_abs_lateral_m=max(abs(r['lateral_error_m']) for r in rows),
                    peak_abs_heading_deg=max(abs(r['heading_error_deg']) for r in rows),
                    mean_speed_mps=sum(r['speed_mps'] for r in rows)/len(rows))
        finally:
            env.close()
    for key in ('x_m', 'y_m', 'yaw_rad', 'speed_mps', 'route_id', 'generation'):
        if outcomes['traffic']['start'][key] != outcomes['isolated']['start'][key]:
            raise RuntimeError('Paired starts differ')
    return outcomes


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def diagnose(args):
    from utilities.mappo_cavs import mappo_cavs
    from utilities.helper_training import find_the_highest_reward_among_all_models
    out = Path(args.output) if args.output else Path('outputs/tracking_diagnostics') / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(seed=args.seed, window_seconds=args.seconds, datasets={},
        caveats=['Finite horizon replay cannot establish long-term safety.',
                 'No-neighbor masking was not enabled during v2 training; isolated inputs can be out of distribution.',
                 'Event samples are capped per route and exclude short lifetimes; report is descriptive, not an unbiased collision rate.',
                 'Test-map diagnostics must not be used as a validation checkpoint leaderboard.'])
    for data_path in map(Path, args.data):
        saved = json.loads(data_path.read_text())['parameters']
        prefix = data_path.name.removesuffix('_data.json')
        found = f'reward{find_the_highest_reward_among_all_models(str(data_path.parent)):.2f}'
        if prefix != found:
            raise ValueError(f'{data_path}: loader would select {found}, refusing to substitute weights')
        for scenario_name in args.scenarios:
            p = Parameters.from_dict(saved)
            if p.is_using_nod_actor or p.is_using_nod_opinion or p.is_using_prioritized_marl or p.is_using_deadlock_critic:
                raise ValueError('Only frozen memoryless DGPPO-minimal policies are supported')
            p.where_to_save=str(data_path.parent) + '/'; p.scenario_type=scenario_name
            p.n_agents=SCENARIOS[scenario_name]['n_agents']; p.num_vmas_envs=1; p.device='cpu'
            p.max_steps=args.steps+2; p.is_testing_mode=True; p.is_load_model=True
            p.is_load_final_model=False; p.is_continue_train=False; p.seed=args.seed
            p.is_real_time_rendering=False; p.is_save_eval_results=False; p.is_add_noise=False
            env, policy, _, _ = mappo_cavs(p)
            horizon = max(1, round(args.seconds / p.dt))
            history = deque(); windows=[]; pairs=[]; route_counts=Counter(); observed_events=0; skipped_short=0
            directory=out / data_path.parent.name / scenario_name; directory.mkdir(parents=True)
            try:
                with set_exploration_type(ExplorationType.MODE):
                    decision=env.reset()
                    for step in range(args.steps):
                        snapshot=dict(env=fork_env(env), decision=decision.clone(), rng=rng_snapshot(),
                                      before=[frame(env.base_env.scenario,i) for i in range(p.n_agents)], step=step)
                        history.append(snapshot)
                        if len(history)>horizon:
                            history.popleft()['env'].close()
                        _, decision, physical, actions=step_physical(env,policy,decision)
                        snapshot['after']=physical; snapshot['actions']=actions
                        for ego, endpoint in enumerate(physical):
                            if not endpoint['road_collision']:
                                continue
                            observed_events+=1
                            continuous=[h for h in history if h['before'][ego]['generation']==endpoint['generation']]
                            event_id=observed_events-1
                            for h in continuous:
                                row=dict(event=event_id, step=h['step'], agent=ego,
                                    seconds_to_collision=(step-h['step'])*p.dt, **h['after'][ego],
                                    lateral_growth_m=abs(h['after'][ego]['lateral_error_m'])-abs(h['before'][ego]['lateral_error_m']),
                                    command_speed=float(h['actions'][0,ego,0]), command_steer_rad=float(h['actions'][0,ego,1]))
                                windows.append(row)
                            if len(continuous)!=horizon:
                                skipped_short+=1;continue
                            route=endpoint['route_id']
                            if len(pairs)>=args.max_events or route_counts[route]>=args.max_per_route:
                                continue
                            pair=paired_replay(continuous[0],policy,ego,horizon,[h['after'][ego] for h in continuous])
                            pair.update(event=event_id, agent=ego, collision_step=step, route=route)
                            pairs.append(pair);route_counts[route]+=1
                write_csv(directory/'collision_windows.csv', windows)
                replay_rows=[dict(event=p['event'],mode=mode,**row) for p in pairs for mode in ('traffic','isolated') for row in p[mode]['rows']]
                write_csv(directory/'paired_windows.csv', replay_rows)
                summary=dict(checkpoint=str(data_path), policy_sha256=hashlib.sha256((data_path.parent/(prefix+'_policy.pth')).read_bytes()).hexdigest(),
                    steps=args.steps, dt=p.dt, observation_refresh=p.refresh_respawn_observations,
                    road_events=observed_events, skipped_short_lifetime=skipped_short,
                    replay_pairs=len(pairs), pairs_by_route=dict(route_counts),
                    isolated_still_road_collision=sum(x['isolated']['road_collision'] for x in pairs),
                    pairs=pairs)
                (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
                report['datasets'][f'{data_path.parent.name}/{scenario_name}']={k:v for k,v in summary.items() if k!='pairs'}
                (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print('TRACKING',scenario_name, json.dumps(report['datasets'][f'{data_path.parent.name}/{scenario_name}']),flush=True)
            finally:
                for h in history:h['env'].close()
                env.close()
    print('Saved diagnostics:',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',nargs='+',default=['outputs/dgppo_minimal_v2/reward6.38_data.json',
        'outputs/dgppo_v2_respawn_training/reward5.74_data.json'])
    parser.add_argument('--scenarios',nargs='+',choices=['intersection_2','roundabout_1'],default=['intersection_2','roundabout_1'])
    parser.add_argument('--steps',type=int,default=1199)
    parser.add_argument('--seconds',type=float,default=.5)
    parser.add_argument('--max-events',type=int,default=20)
    parser.add_argument('--max-per-route',type=int,default=5)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.steps<1 or args.seconds<=0 or args.max_events<1 or args.max_per_route<1:parser.error('All counts/durations must be positive')
    torch.set_num_threads(2)
    diagnose(args)
