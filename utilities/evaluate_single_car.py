"""Paired route-following experiment with one active car and masked parked slots.

Does not train, filter ego actions, respawn cars, or change the regular evaluator.
"""
import argparse
import csv
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters
from utilities.helper_scenario import interX
from utilities.map_manager import MapManager
from utilities.mappo_cavs import mappo_cavs


def make_cases(routes, fractions, speeds):
    if not routes or any(r < 0 for r in routes):
        raise ValueError('Specify nonnegative route indices.')
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError('Start fractions must be between 0 and 1.')
    if not speeds or any(not 0 <= s <= 1 for s in speeds):
        raise ValueError('Speed fractions must be between 0 and 1.')
    return [dict(route=r, start_fraction=f, speed_fraction=s)
            for r, f, s in itertools.product(routes, fractions, speeds)]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def no_lanelet_mask(map_manager, agent_idx, nearing_agents_indices):
    """Single-car isolation uses distance masks, not lanelet membership.

    During VMAS's first observation current_lanelet_idx is still an empty
    list. It is irrelevant here: every dummy is outside the sensing radius.
    The scenario still applies its normal distance mask and missing encoding.
    """
    return torch.zeros_like(nearing_agents_indices, dtype=torch.bool)


@torch.no_grad()
def evaluate(data_file, args, cases, output):
    data_file = Path(data_file).resolve()
    p = Parameters(**json.loads(data_file.read_text())['parameters'])
    if p.is_using_nod_actor or p.is_using_nod_opinion or p.is_using_prioritized_marl:
        raise ValueError('This diagnostic supports the non-NOD, non-prioritized DGPPO controls only.')
    if not p.is_partial_observation or p.n_agents <= p.n_nearing_agents_observed:
        raise ValueError('A fixed-neighbor partial-observation checkpoint is required.')
    p.where_to_save = str(data_file.parent) + '/'
    p.scenario_type = args.scenario
    p.num_vmas_envs = len(cases)
    p.seed = args.seed
    p.max_steps = args.steps + 1
    p.frames_per_batch = p.max_steps * len(cases)
    p.is_testing_mode = p.is_load_model = True
    p.is_load_final_model = p.is_continue_train = False
    p.is_save_eval_results = p.is_save_agent_speed = False
    p.is_save_simulation_video = p.is_real_time_rendering = False
    p.is_prb = p.is_challenging_initial_state_buffer = False
    p.is_using_deadlock_critic = False
    # Preserve checkpoint dimensions; use the existing missing-neighbor encoding.
    p.is_apply_mask = True
    initial, results, trajectory = {}, {}, []

    def reset_state(s, env_i, i_agent, is_reset_single_agent, is_use_state_buffer,
                    initial_state, ref_paths_scenario, agents):
        e, a = int(env_i), int(i_agent)
        case = cases[e]
        route = case['route'] if a == 0 else 0
        if route >= len(ref_paths_scenario):
            raise ValueError(f'Route {route} does not exist in {args.scenario}.')
        ref = ref_paths_scenario[route]
        n = len(ref['center_line'])
        if n < 8:
            raise ValueError('Route is too short for interior starting points.')
        point = 3 + round(case['start_fraction'] * (n - 7))
        agent = agents[a]
        pos = ref['center_line'][point].clone()
        yaw = ref['center_line_yaw'][point].clone()
        speed = case['speed_fraction'] * agent.max_speed if a == 0 else 0.
        if a:
            pos = pos.new_tensor([1000. + 100. * a, 1000.])
            yaw = yaw * 0
        vel = torch.stack([torch.cos(yaw), torch.sin(yaw)]) * speed
        agent.set_pos(pos, batch_index=e)
        agent.set_rot(yaw, batch_index=e)
        agent.set_vel(vel, batch_index=e)
        agent.state.ang_vel[e] = 0
        s.ref_paths_agent_related.path_id[e, a] = route
        s.ref_paths_agent_related.point_id[e, a] = point
        if a == 0:
            initial[e] = dict(case=e, **case, point=point, x=float(pos[0]), y=float(pos[1]),
                              yaw=float(yaw), vx=float(vel[0]), vy=float(vel[1]))
        return ref, route

    def observe_done(s):
        # Never call the regular done(): it respawns collided cars.
        for e in range(len(cases)):
            step = int(s.timer.step[e])
            if step == 0 or e in results:
                continue
            car = s.world.agents[0]
            lane = bool(s.collisions.with_lanelets[e, 0])
            other = bool(s.collisions.with_agents[e, 0].any())
            entry = bool(s.collisions.with_entry_segments[e, 0])
            end = bool(s.collisions.with_exit_segments[e, 0])
            # Refuse to report a supposedly isolated run with visible dummy cars.
            distances = s.distances.agents[e, 0, 1:]
            if bool((distances < s.thresholds.distance_mask_agents).any()):
                raise RuntimeError('A parked slot entered the ego sensing range.')
            trajectory.append(dict(case=e, route=cases[e]['route'], step=step, time=step*p.dt,
                                   x=float(car.state.pos[e,0]), y=float(car.state.pos[e,1]),
                                   yaw=float(car.state.rot[e,0]), speed=float(car.state.vel[e].norm()),
                                   deviation=float(s.distances.ref_paths[e,0]),
                                   action_0=float(car.action.u[e,0]), action_1=float(car.action.u[e,1]),
                                   lane_collision=lane))
            status = ('lane_collision' if lane else 'unexpected_agent_collision' if other else
                      'exit' if end else 'entry_exit' if entry else 'timeout' if step >= args.steps else None)
            if status:
                results[e] = dict(case=e, route=cases[e]['route'], status=status, steps=step)
        return torch.zeros(s.world.batch_dim, dtype=torch.bool, device=s.world.device)

    env = None
    with patch.object(ScenarioRoadTraffic, '_reset_init_state', reset_state), \
         patch.object(ScenarioRoadTraffic, 'done', observe_done), \
         patch.object(MapManager, 'determine_masked_agents_by_lanelets', no_lanelet_mask), \
         set_exploration_type(ExplorationType.MODE):
        try:
            env, policy, *_ = mappo_cavs(p)
            # check_env_specs during construction may step the world; discard it.
            results.clear()
            trajectory.clear()
            td = env.reset()
            s = env.scenario
            invalid = interX(s.vertices[:, 0], s.ref_paths_agent_related.left_boundary[:, 0], False) | interX(
                s.vertices[:, 0], s.ref_paths_agent_related.right_boundary[:, 0], False)
            if bool(invalid.any()):
                raise ValueError(f'Initial car footprint touches a route boundary in cases {invalid.nonzero().flatten().tolist()}; adjust --fractions.')

            def isolated_policy(td):
                td = policy(td)
                # Dummy slots stay stationary; the ego action is untouched.
                action = td['agents', 'action'].clone()
                action[:, 1:] = 0
                # Finished cases are no longer measured and remain stopped.
                for e in results:
                    action[e, 0] = 0
                td.set(('agents', 'action'), action)
                return td

            env.rollout(args.steps, isolated_policy, tensordict=td, auto_reset=False,
                        break_when_any_done=False)
            if len(results) != len(cases):
                raise RuntimeError('Not all cases produced a terminal outcome.')
            rows = []
            for e in range(len(cases)):
                samples = [r for r in trajectory if r['case'] == e]
                rows.append(dict(**initial[e], **{k:v for k,v in results[e].items() if k not in ('case','route')},
                                 mean_speed=sum(r['speed'] for r in samples)/len(samples),
                                 mean_deviation=sum(r['deviation'] for r in samples)/len(samples)))
            write_csv(output/'cases.csv', rows)
            write_csv(output/'trajectories.csv', trajectory)
            summary = dict(data=str(data_file), policy=p.model_name, dt=p.dt,
                           neighbor_mode='existing_mask_encoding_with_parked_slots',
                           scenario=args.scenario, seed=p.seed, cases=len(cases),
                           outcomes={status:sum(r['status']==status for r in rows)
                                     for status in ['lane_collision','exit','entry_exit','timeout','unexpected_agent_collision']},
                           by_route={str(route):{status:sum(r['route']==route and r['status']==status for r in rows)
                                      for status in ['lane_collision','exit','entry_exit','timeout']}
                                     for route in sorted({c['route'] for c in cases})})
            (output/'summary.json').write_text(json.dumps(summary, indent=2))
            return initial, summary
        finally:
            if env is not None:
                env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs=2, default=['outputs/archive/dgppo_task_only/reward4.57_data.json',
                                                 'outputs/archive/dgppo_minimal_v2/reward6.38_data.json'])
    parser.add_argument('--scenario', choices=['intersection_2','roundabout_1','on_ramp_1'], default='intersection_2')
    parser.add_argument('--routes', nargs='+', type=int, default=[1,5])
    parser.add_argument('--fractions', nargs='+', type=float, default=[0., .15, .35])
    parser.add_argument('--speeds', nargs='+', type=float, default=[.3,.7])
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    cases = make_cases(args.routes, args.fractions, args.speeds)
    out = Path(args.output) if args.output else Path('outputs/archive/single_car_comparison')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    starts, summaries = [], []
    for index, data in enumerate(args.data):
        folder = out/f'model_{index}'
        folder.mkdir()
        initial, summary = evaluate(data, args, cases, folder)
        starts.append(initial)
        summaries.append(summary)
    if starts[0] != starts[1] or summaries[0]['dt'] != summaries[1]['dt']:
        raise RuntimeError('The two models did not receive identical initial states and dt.')
    (out/'comparison.json').write_text(json.dumps(dict(initial_states_match=True, models=summaries), indent=2))
    print(json.dumps(summaries, indent=2))
    print('Results:', out)
