"""Compare frozen Safety Values on identical trajectories; never train a model.

Observed-safe means only that no violation was seen in the specified finite
window. It is not a certified false-positive label for an infinite-horizon Value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utilities.helper_training import Parameters, find_the_highest_reward_among_all_models
from utilities.nod_marl.safety_value import PairSafetyValue, value_state, same_entities, VALUE_HEADS


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed_window(current, following, done, horizon):
    """Physical suffix maxima, capped at horizon transitions and identity cuts."""
    if horizon < 1:
        raise ValueError('horizon must be positive')
    g, gn = current['g'], following['g']
    valid = current['valid'] & following['valid'] & same_entities(current, following)
    valid &= torch.isfinite(g) & torch.isfinite(gn)
    connected = torch.zeros_like(valid)
    connected[:, :-1] = (valid[:, :-1] & valid[:, 1:]
        & ~done.bool().reshape(*g.shape[:2], 1, 1)[:, :-1]
        & (following['ego_gen'][:, :-1] == current['ego_gen'][:, 1:])
        & (following['other_gen'][:, :-1] == current['other_gen'][:, 1:]))
    observed = torch.maximum(g, gn)
    steps = valid.long()
    for _ in range(horizon - 1):
        tail = observed.clone()
        counts = steps.clone()
        observed[:, :-1] = torch.where(connected[:, :-1],
            torch.maximum(observed[:, :-1], tail[:, 1:]), observed[:, :-1])
        steps[:, :-1] = torch.where(connected[:, :-1], 1 + counts[:, 1:], steps[:, :-1])
    return observed, steps, valid


def score_predictions(current, following, predictions, successors, done, horizon, dt, alpha):
    observed, steps, valid = observed_window(current, following, done, horizon)
    def rate(event, selection):
        return float(event[selection].float().mean()) if selection.any() else None
    result = {}
    for label, prediction in predictions.items():
        delta = (successors[label] - prediction) / dt + alpha * prediction
        row = {}
        for head, selection in VALUE_HEADS:
            mask = valid[..., selection]
            pred, obs = prediction[..., selection], observed[..., selection]
            unsafe = mask & (obs > 0)
            safe = mask & (obs <= 0) & (steps[..., selection] >= horizon)
            early = unsafe & (current['g'][..., selection] <= 0)
            row[head] = dict(samples=int(mask.sum()), observed_unsafe=int(unsafe.sum()),
                complete_window_safe=int(safe.sum()), early_warning_samples=int(early.sum()),
                unsafe_recall=rate(pred > 0, unsafe), early_warning_recall=rate(pred > 0, early),
                observed_safe_alarm_rate=rate(pred > 0, safe),
                positive_delta_rate=rate(delta[..., selection] > 0, mask),
                prediction_mean=float(pred[mask].mean()) if mask.any() else None)
        row['any_gate_rate'] = rate((valid & (delta > 0)).any(-1), valid.any(-1))
        result[label] = row
    result['prediction_change'] = {
        head: float((predictions['final'][..., selection] - predictions['best'][..., selection])
                    [valid[..., selection]].abs().mean())
        if valid[..., selection].any() else None for head, selection in VALUE_HEADS
    }
    return result


@torch.no_grad()
def diagnose(args):
    from utilities.constants import SCENARIOS
    from utilities.mappo_cavs import mappo_cavs
    root = Path(args.model_dir)
    params_path = root / (args.checkpoint + '_data.json')
    saved = json.loads(params_path.read_text())['parameters']
    found = f'reward{find_the_highest_reward_among_all_models(str(root)):.2f}'
    if found != args.checkpoint:
        raise ValueError(f'Expected {args.checkpoint}, found {found}; pin the intended checkpoint explicitly')
    paths = {'best': root / (args.checkpoint + '_safety_value.pth'),
             'final': root / 'final_safety_value.pth'}
    snapshots = {name: torch.load(path, map_location='cpu') for name, path in paths.items()}
    contract = snapshots['best']['contract']
    if contract != snapshots['final']['contract']:
        raise ValueError('Safety Value contracts differ; comparison is not controlled')
    if contract.get('mode') != 'dgppo_minimal_value':
        raise ValueError('This diagnostic requires DGPPO minimal local Value inputs')
    models = {}
    for name, snapshot in snapshots.items():
        model = PairSafetyValue(contract['observation_dim'], contract['width'])
        model.load_state_dict(snapshot['model']); model.eval(); model.requires_grad_(False)
        models[name] = model
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=args.checkpoint, safety_hashes={k: file_hash(v) for k, v in paths.items()},
        horizon_steps=args.horizon, dt=saved['dt'], seed=args.seed,
        interpretation='Finite-window observations, not certified infinite-horizon false positives.', datasets={})
    for scenario in args.scenarios:
        for actor_name in ('best', 'final'):
            policy_path = root / ((args.checkpoint if actor_name == 'best' else 'final') + '_policy.pth')
            metadata = dict(scenario=scenario, actor=actor_name, actor_sha256=file_hash(policy_path),
                parameters_sha256=file_hash(params_path), seed=args.seed, steps=args.steps,
                num_envs=args.num_envs, testing_mode=scenario != 'CPM_mixed', version=1)
            cache = out / f'{scenario}_{actor_name}_trajectories.pth'
            if args.reuse:
                package = torch.load(cache, map_location='cpu')
                if package['metadata'] != metadata:
                    raise ValueError(f'{cache}: metadata mismatch; regenerate without --reuse')
                td = package['rollout']
            else:
                p = Parameters.from_dict(saved)
                p.where_to_save = str(root) + '/'
                p.device='cpu'; p.seed=args.seed; p.num_vmas_envs=args.num_envs
                p.scenario_type=scenario; p.n_agents=SCENARIOS[scenario]['n_agents'] if scenario != 'CPM_mixed' else 4
                p.max_steps=128 if scenario == 'CPM_mixed' else args.steps + 1
                p.is_load_model=True; p.is_load_final_model=actor_name == 'final'; p.is_continue_train=False
                p.is_testing_mode=metadata['testing_mode']; p.is_real_time_rendering=False
                p.is_save_eval_results=False; p.is_add_noise=False
                env, policy, _, _ = mappo_cavs(p)
                try:
                    with set_exploration_type(ExplorationType.MODE):
                        td = env.rollout(args.steps, policy, break_when_any_done=False).cpu()
                finally:
                    env.close()
                torch.save(dict(metadata=metadata, rollout=td), cache)
            current = value_state(td, ('agents', 'observation'), contract['safe_distance'],
                                  local_sensing_range=contract['sensing_range'])
            following = value_state(td['next'], ('agents', 'observation'), contract['safe_distance'],
                                    local_sensing_range=contract['sensing_range'])
            predictions = {name: model(current) for name, model in models.items()}
            successors = {name: model(following) for name, model in models.items()}
            if not all(torch.isfinite(v).all() for v in [*predictions.values(), *successors.values()]):
                raise ValueError('Non-finite Value predictions')
            key = f'{scenario}/{actor_name}_actor'
            report['datasets'][key] = dict(metadata=metadata, metrics=score_predictions(
                current, following, predictions, successors, td['next', 'done'], args.horizon,
                saved['dt'], saved['dgppo_alpha']))
            print('DIAGNOSTIC', key, json.dumps(report['datasets'][key]['metrics']), flush=True)
    (out / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {out / "diagnostics.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='outputs/dgppo_v2_respawn_training')
    parser.add_argument('--checkpoint', default='reward5.74')
    parser.add_argument('--output', default='outputs/dgppo_safety_stability')
    parser.add_argument('--scenarios', nargs='+', default=['CPM_mixed', 'intersection_2', 'roundabout_1'])
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if args.steps < args.horizon or args.horizon < 1 or args.num_envs < 1:
        parser.error('Require steps >= horizon >= 1 and num-envs >= 1')
    torch.set_num_threads(2)
    diagnose(args)
