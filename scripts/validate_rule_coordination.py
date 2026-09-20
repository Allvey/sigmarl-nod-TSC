"""Run repeatable test-only traffic checks without rendering.

Example: python -m scripts.validate_rule_coordination --fraction 1 --seeds 123 456
"""
import argparse
import json
from pathlib import Path

import torch

from utilities.helper_training import SaveData
from utilities.mappo_cavs import mappo_cavs
from utilities.testing_rule_policy import TestingRulePolicy, assign_rule_vehicles
from utilities.testing_rule_coordinator import CONTROLLER_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', type=float, default=1.)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123])
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--checkpoint', default='outputs/dgppo_nod_opinion_gain2_finetune')
    parser.add_argument('--output', default='outputs/rule_coordination_checks')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    source = next(checkpoint.glob('*.json'))
    for seed in args.seeds:
        p = SaveData.from_dict(json.loads(source.read_text())).parameters
        p.where_to_save = str(checkpoint)+'/'
        p.is_testing_mode = True
        p.is_continue_train = False
        p.is_real_time_rendering = False
        p.is_save_simulation_video = False
        p.is_save_eval_results = False
        p.is_load_model = True
        p.is_load_final_model = False
        p.is_load_out_td = False
        p.is_using_deadlock_critic = False
        p.scenario_type = 'intersection_2'
        p.n_agents = 8
        p.num_vmas_envs = 1
        p.max_steps = args.steps+1
        p.seed = seed
        p.dgppo_alpha_gain = 2.
        p.is_print_agent_speed = False
        env, actor, priority, p = mappo_cavs(parameters=p)
        assignment = assign_rule_vehicles(8, args.fraction,
            {'yielding': .25, 'moderate': .5, 'non_yielding': .25},
            seed=123, actor_index=None if args.fraction == 1 else 0)
        policy = TestingRulePolicy(actor, env.scenario, assignment, cruise_speed=1.)
        def progress(env, td):
            step=int(env.scenario.timer.step[0])
            if step % 200 == 0:
                print(f'fraction={args.fraction} seed={seed} step={step}', flush=True)
        rollout = env.rollout(max_steps=args.steps, policy=policy, priority_module=priority,
                              callback=progress, auto_cast_to_device=True,
                              break_when_any_done=False, is_save_simulation_video=False)
        if isinstance(rollout, tuple):
            rollout = rollout[0]
        folder = Path(args.output)/f'{CONTROLLER_VERSION}_fraction{args.fraction:g}_seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        policy.save_diagnostics(rollout, folder/'rule_diagnostics.csv')
        report = dict(seed=seed, fraction=args.fraction, steps=args.steps, vehicles={})
        for i, profile in assignment.items():
            contacts = rollout['next','agents','info','testing_rule_contact'][0,:,i].bool()
            generation = rollout['agents','info','nod_ego_generation'][0,:,i]
            events = contacts.clone()
            events[1:] &= (~contacts[:-1] | (generation[1:] != generation[:-1]))
            speed = rollout['next','agents','info','nod_world_vel'][0,:,i].norm(dim=-1)
            run = longest = 0
            for step, stopped in enumerate((speed < .03).tolist()):
                if step and generation[step] != generation[step-1]:
                    run = 0
                run = run+1 if stopped else 0
                longest = max(longest, run)
            report['vehicles'][i+1] = dict(profile=profile, rule_contact_events=int(events.sum()),
                max_stop_seconds=longest*p.dt, mean_speed=float(speed.mean()),
                infeasible_steps=int(rollout['agents','rule_reservation_infeasible'][0,:,i].sum()),
                road_contact_steps=int(rollout['next','agents','info','testing_road_contact'][0,:,i].sum()))
        (folder/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        env.close()


if __name__ == '__main__':
    main()
