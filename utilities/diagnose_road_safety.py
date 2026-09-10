"""Read-only Actor/Value rollout diagnostics; no policy filtering or training."""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utilities.constants import SCENARIOS
from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs
from utilities.nod_marl.dgppo import dgppo_advantage


def history_indices(generations, event_step, window):
    """Include the collision transition, stopping at any respawn boundary."""
    start = event_step
    while start > max(0, event_step - window + 1):
        if generations[start - 1] != generations[event_step]:
            break
        start -= 1
    return range(start, event_step + 1)


@torch.no_grad()
def diagnose(args):
    data_path = Path(args.data).resolve()
    data = json.loads(data_path.read_text())
    p = Parameters(**data['parameters'])
    if p.safety_control_mode != 'dgppo':
        raise ValueError('This diagnostic requires a DGPPO checkpoint.')
    p.where_to_save = str(data_path.parent) + '/'
    p.scenario_type = args.scenario
    p.n_agents = SCENARIOS[args.scenario]['n_agents']
    p.num_vmas_envs = args.envs
    p.max_steps = args.steps + 1
    p.frames_per_batch = args.envs * p.max_steps
    if args.seed is not None:
        p.seed = args.seed
    p.is_testing_mode = p.is_load_model = True
    p.is_load_final_model = p.is_continue_train = False
    p.is_save_eval_results = p.is_save_agent_speed = False
    p.is_save_simulation_video = p.is_real_time_rendering = False
    p.is_prb = p.is_challenging_initial_state_buffer = False
    p.is_using_deadlock_critic = False
    out = Path(args.output) if args.output else data_path.parent / 'road_diagnostics' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with set_exploration_type(ExplorationType.MODE):
        env, policy, *_ = mappo_cavs(p)
        scenario = env.scenario
        manager = scenario.safety_value_manager
        original_done = scenario.done
        snapshots = []
        try:
            if not manager.last_load_info.startswith('loaded'):
                raise RuntimeError('A matching trained Safety Value checkpoint is required.')

            def capture_before_reset():
                # VMAS also calls done() during rollout initialization/reset.
                # These calls are not transitions in the returned TensorDict.
                if bool((scenario.timer.step == 0).all()):
                    return original_done()
                snapshots.append(dict(
                    lane=scenario.collisions.with_lanelets.clone(),
                    collision=scenario.collisions.with_agents.any(-1).clone(),
                    path=scenario.ref_paths_agent_related.path_id.clone(),
                    generation=scenario.nod_agent_generation.clone(),
                    clearance=torch.minimum(scenario.distances.left_boundaries.amin(-1), scenario.distances.right_boundaries.amin(-1)).clone(),
                    pos=torch.stack([a.state.pos for a in scenario.world.agents], 1).clone(),
                    speed=torch.stack([a.state.vel.norm(dim=-1) for a in scenario.world.agents], 1).clone(),
                    yaw=torch.stack([a.state.rot.squeeze(-1) for a in scenario.world.agents], 1).clone(),
                ))
                return original_done()

            scenario.done = capture_before_reset
            td = env.rollout(args.steps, policy, break_when_any_done=False)
            if len(snapshots) != args.steps:
                raise RuntimeError(f'Physical snapshots ({len(snapshots)}) and rollout steps ({args.steps}) do not align.')
            snap = {k: torch.stack([s[k] for s in snapshots], 1) for k in snapshots[0]}
            current, following = manager.state(td), manager.state(td['next'])
            value, next_value = manager.model(current), manager.model(following)
            if not torch.equal(snap['generation'].reshape_as(following['ego_gen'][..., -2]), following['ego_gen'][..., -2]):
                raise RuntimeError('Next-state info was captured after respawn; refusing misaligned diagnostics.')
            # Training terminates the entire environment on any collision;
            # evaluation respawns individual cars. Report both residuals.
            terminal = (snap['lane'] | snap['collision']).any(-1) | td['next', 'done'].squeeze(-1)
            training_next = torch.where(terminal[..., None, None], following['g'], next_value)
            task = torch.zeros_like(value[..., :1])
            _, detail = dgppo_advantage(task, current, following, value, training_next,
                                       dt=p.dt, alpha=p.dgppo_alpha, eps=p.dgppo_eps, weight=p.dgppo_weight)
            raw_c = (next_value - value) / p.dt + p.dgppo_alpha * value
            events, rows = [], []
            window = max(1, round(args.seconds / p.dt))
            for e, t, a in snap['lane'].nonzero().tolist():
                event_id = len(events)
                indices = history_indices(snap['generation'][e, :, a].flatten().tolist(), t, window)
                event_rows = []
                for k in indices:
                    valid = bool(detail['valid'][e, k, a, -2])
                    row = dict(event=event_id, env=e, agent=a, path=int(snap['path'][e, k, a]),
                               generation=int(snap['generation'][e, k, a]), step=k,
                               seconds_before_collision=(t-k)*p.dt,
                               x=float(snap['pos'][e,k,a,0]), y=float(snap['pos'][e,k,a,1]),
                               speed=float(snap['speed'][e,k,a]), yaw=float(snap['yaw'][e,k,a]),
                               action_0=float(td['agents','action'][e,k,a,0]), action_1=float(td['agents','action'][e,k,a,1]),
                               clearance_next=float(snap['clearance'][e,k,a]),
                               g=float(current['g'][e,k,a,-2]), g_next=float(following['g'][e,k,a,-2]),
                               value=float(value[e,k,a,-2]), value_next=float(next_value[e,k,a,-2]),
                               c_raw=float(raw_c[e,k,a,-2]), c_training=float(detail['delta'][e,k,a,-2]),
                               valid=valid, training_terminal=bool(terminal[e,k]),
                               road_penalty=p.dgppo_weight*max(0.,float(detail['delta'][e,k,a,-2])+p.dgppo_eps) if valid else 0.,
                               all_head_penalty=p.dgppo_weight*float(detail['penalty'][e,k,a,0]),
                               task_masked=bool(detail['violation'][e,k,a]))
                    event_rows.append(row)
                rows.extend(event_rows)
                warnings = [r['seconds_before_collision'] for r in event_rows if r['valid'] and r['c_training'] > 0 and r['seconds_before_collision'] > 0]
                before = [r for r in event_rows if r['valid'] and r['seconds_before_collision'] > 0]
                events.append(dict(event=event_id, env=e, agent=a, path=int(snap['path'][e,t,a]), step=t,
                                   history_steps=len(event_rows), warned_before_collision=bool(warnings),
                                   valid_pre_collision_steps=len(before), warning_steps=len(warnings),
                                   raw_warning_steps=sum(r['c_raw'] > 0 for r in before),
                                   warning_before_geometric_threshold_steps=sum(r['c_training'] > 0 and r['g'] <= 0 for r in before),
                                   first_warning_seconds=max(warnings) if warnings else None))
            if rows:
                with (out/'collision_windows.csv').open('w', newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            summary = dict(data=str(data_path), scenario=args.scenario, seed=p.seed, steps=args.steps, envs=args.envs,
                           dt=p.dt, window_steps=window, value_load=manager.last_load_info,
                           collision_events=len(events), by_path=dict(Counter(e['path'] for e in events)),
                           events_with_early_warning=sum(e['warned_before_collision'] for e in events),
                           lane_collision_step_percent=float(snap['lane'].any(-1).float().mean()*100),
                           notes='Offline deterministic rollout. c_training substitutes physical next g at collision terminals as training does; c_raw uses network values. Positions/clearance are next-state, actions/current Value are pre-transition. Warnings are not safety guarantees. No Actor updates or action filtering.',
                           events=events)
            (out/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps({k:v for k,v in summary.items() if k != 'events'}, indent=2))
            print('Diagnostics:', out)
        finally:
            scenario.done = original_done
            env.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Exact reward*_data.json associated with the checkpoint')
    parser.add_argument('--scenario', default='intersection_2', choices=list(SCENARIOS))
    parser.add_argument('--steps', type=int, default=1199)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=1.)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', help='New diagnostic directory; never overwrite existing results')
    args=parser.parse_args()
    if args.steps < 1 or args.envs < 1 or args.seconds <= 0:
        parser.error('steps, envs and seconds must be positive')
    diagnose(args)
