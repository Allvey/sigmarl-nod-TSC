"""Pinned v2/final-endpoint comparison with separate navigation violation metrics.

Uses unchanged v2 decisions/respawns. Metrics come from physical next.info before
respawn. Navigation rate is occupancy (violating frames), not incident count.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.constants import SCENARIOS, AGENTS
from utilities.helper_training import Parameters, find_the_highest_reward_among_all_models
from utilities.mappo_cavs import mappo_cavs


def environment_event_rates(flags):
    """Percent of steps with any colliding agent, per environment.

    VMAS scalar info can be [env,time,agent,1]; reduce both agent and
    optional feature axes before averaging over time.
    """
    return flags.bool().reshape(flags.shape[0], flags.shape[1], -1).any(-1).float().mean(-1) * 100


@torch.no_grad()
def evaluate(args):
    output = Path(args.output) if args.output else Path('outputs/dgppo_navigation_comparison') / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(metric_source='physical next.agents.info before respawn', metric_version=2,
                  note='Occupancy rates are per-agent; road_env_event_pct counts steps with any road collision. Speed includes all agents and collision frames.',
                  results=[])
    for folder in map(Path, args.models):
        reward = find_the_highest_reward_among_all_models(str(folder))
        best = f'reward{reward:.2f}'
        data = json.loads((folder / f'{best}_data.json').read_text())
        is_v2 = folder.name == 'dgppo_minimal_v2'
        if is_v2 and best != 'reward6.38':
            raise ValueError('Expected pinned v2 reward6.38')
        checkpoint = best if is_v2 else 'final'
        for suffix in ('policy', 'critic', 'safety_value'):
            if not (folder / f'{checkpoint}_{suffix}.pth').is_file():
                raise FileNotFoundError(folder / f'{checkpoint}_{suffix}.pth')
        for scene in args.scenarios:
            p = Parameters.from_dict(data['parameters'])
            p.where_to_save = str(folder) + '/'
            p.is_testing_mode = True
            p.is_load_model = True
            p.is_load_final_model = not is_v2
            p.is_continue_train = False
            p.scenario_type = scene
            p.n_agents = SCENARIOS[scene]['n_agents']
            p.num_vmas_envs = args.envs
            p.max_steps = args.steps + 2
            p.device = 'cpu'
            p.seed = args.seed
            p.is_add_noise = False
            p.record_navigation_metrics = True
            p.is_real_time_rendering = False
            p.is_save_eval_results = False
            env, policy, _, _ = mappo_cavs(p)
            try:
                with set_exploration_type(ExplorationType.MODE):
                    td = env.rollout(max_steps=args.steps, policy=policy, break_when_any_done=False)
                info = td['next', 'agents', 'info']
                def occupancy(key):
                    return info[key].float().reshape(args.envs, -1).mean(-1) * 100
                nav = occupancy('navigation_violation')
                road = occupancy('physical_road_collision')
                car = occupancy('physical_agent_collision')
                # Environment-event rate is also retained to compare old log semantics.
                event = environment_event_rates(info['physical_road_collision'])
                speed = info['vel'].norm(dim=-1).reshape(args.envs, -1).mean(-1)
                speed *= env.base_env.scenario.normalizers.v / AGENTS['max_speed_achievable'] * 100
                deviation = info['distance_ref'].reshape(args.envs, -1).mean(-1)
                deviation *= env.base_env.scenario.normalizers.distance_ref / AGENTS['width'] * 100
                row = dict(model=str(folder), checkpoint=checkpoint, scene=scene,
                           navigation_boundary=p.use_navigation_boundary, seed=args.seed,
                           observe_navigation_boundary=p.observe_navigation_boundary,
                           navigation_boundary_mode=p.navigation_boundary_mode,
                           navigation_penalty_ratio=p.navigation_penalty_ratio,
                           envs=args.envs, steps=args.steps)
                for name, values in [('navigation_violation_pct', nav), ('road_collision_pct', road),
                                     ('agent_collision_pct', car), ('road_env_event_pct', event),
                                     ('relative_speed_pct', speed), ('relative_deviation_pct', deviation)]:
                    row[name] = dict(mean=float(values.mean()), std=float(values.std(unbiased=False)),
                                     per_env=values.tolist())
                report['results'].append(row)
                (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                line = (f'{scene} | {folder.name} ({checkpoint}, mode={p.navigation_boundary_mode}) | '
                        f'road events={event.mean():.2f}% | navigation occupancy={nav.mean():.2f}% | '
                        f'speed={speed.mean():.2f}% | deviation={deviation.mean():.2f}%')
                print(line, flush=True)
                with (output / 'log.txt').open('a') as file:
                    file.write(line + '\n')
            finally:
                env.close()
    print('Saved:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['outputs/dgppo_minimal_v2',
        'outputs/dgppo_navigation_control', 'outputs/dgppo_navigation'])
    parser.add_argument('--scenarios', nargs='+', choices=list(SCENARIOS),
                        default=['CPM_entire', 'intersection_2', 'on_ramp_1', 'roundabout_1'])
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1199)
    parser.add_argument('--seed', type=int, default=5080868027432654403)
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.envs, args.steps) < 1:
        parser.error('envs and steps must be positive')
    torch.set_num_threads(2)
    evaluate(args)
