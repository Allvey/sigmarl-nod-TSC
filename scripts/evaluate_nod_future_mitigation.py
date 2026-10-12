"""Evaluate a candidate future-mitigation NOD on independent behavior rollouts.

The behavior policy and its responsibility NOD stay frozen. The candidate NOD
only reads the completed rollout, so its predictions cannot alter the actions
whose future mitigation is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs
from utilities.nod_marl.counterfactual import build_counterfactual_labels
from utilities.nod_marl.interaction import nod_observation_inputs
from utilities.nod_marl.trainer import NODOpinionManager


def collect_one(
    seed: int, *, config: str, checkpoint: str, steps: int,
    rule_vehicles: dict[int, str] | None = None,
) -> dict:
    candidate_parameters = Parameters.from_json(config)
    if candidate_parameters.nod_label_mode != "future_mitigation":
        raise ValueError("Candidate configuration must use future_mitigation mode")
    behavior_prefix = Path(candidate_parameters.training_init_checkpoint)
    if not behavior_prefix.name.startswith("reward"):
        raise ValueError("Expected a pinned reward* behavior checkpoint prefix")

    behavior_parameters = Parameters.from_json(config)
    behavior_parameters.nod_label_mode = "responsibility"
    # The pinned behavior checkpoint is version 3: it has neither future head.
    # The candidate below retains the quantile head for passive evaluation.
    behavior_parameters.nod_predict_future_quantile = False
    behavior_parameters.nod_training_mode = "joint"
    behavior_parameters.training_init_checkpoint = None
    behavior_parameters.where_to_save = str(behavior_prefix.parent) + "/"
    behavior_parameters.is_load_model = True
    behavior_parameters.is_load_final_model = False
    behavior_parameters.is_continue_train = False
    behavior_parameters.is_save_simulation_video = False
    behavior_parameters.is_save_eval_results = False
    behavior_parameters.is_testing_mode = bool(rule_vehicles)
    behavior_parameters.seed = seed
    behavior_parameters.max_steps = steps

    env, policy, priority_module, loaded = mappo_cavs(behavior_parameters)
    if loaded.model_name != behavior_prefix.name:
        raise ValueError(
            f"Loaded {loaded.model_name}, expected {behavior_prefix.name}; "
            "the behavior directory contains another higher-reward checkpoint"
        )
    if rule_vehicles:
        from utilities.testing_rule_policy import TestingRulePolicy
        policy = TestingRulePolicy(
            policy, env.scenario, rule_vehicles,
            cruise_speed=1.0, lateral_accel_limit=1.8,
        )
    try:
        rollout = env.rollout(
            max_steps=steps - 1,
            policy=policy,
            priority_module=priority_module,
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=False,
        )
        td = rollout[0] if isinstance(rollout, tuple) else rollout
    finally:
        env.close()

    info = td.get(("agents", "info"))
    pair, edge = nod_observation_inputs(
        info["nod_pair_features"],
        info["nod_edge_mask"],
        candidate_parameters.nod_observation_mode,
    )
    ids = info["nod_neighbor_indices"].long()
    ego_generations = info["nod_ego_generation"].long()
    if ego_generations.ndim == 4 and ego_generations.shape[-1] == 1:
        ego_generations = ego_generations.squeeze(-1)
    neighbor_generations = info["nod_neighbor_generation"].long()
    labels = build_counterfactual_labels(
        info["nod_world_pos"],
        info["nod_world_vel"],
        ego_generations,
        ids,
        edge,
        horizon=candidate_parameters.nod_counterfactual_horizon,
        dt=candidate_parameters.dt,
        safe_distance=candidate_parameters.nod_safe_distance,
        label_slope=candidate_parameters.nod_label_slope,
        label_margin=candidate_parameters.nod_label_margin,
        mode="future_mitigation",
        reference_seconds=candidate_parameters.nod_reference_seconds,
    )
    candidate = NODOpinionManager(candidate_parameters)
    saved = torch.load(checkpoint, map_location=candidate_parameters.device)
    if not candidate.load_checkpoint(saved):
        raise ValueError(f"Incompatible candidate NOD: {candidate.last_load_info}")
    candidate.model.eval()
    with torch.no_grad():
        outputs, _ = candidate.model.forward_sequence(
            pair,
            edge,
            ego_generations,
            neighbor_generations,
            opinion_mask=labels["reference_active"],
            evidence_override=labels["online_evidence"],
        )
        valid = edge & labels["valid"] & outputs["learning_valid"]
        if not bool(valid.any()):
            raise ValueError("No valid future-mitigation labels in this rollout")
        prediction = outputs["rho"][valid]
        target = labels["label"][valid]
        error = prediction - target
        centered_prediction = prediction - prediction.mean()
        centered_target = target - target.mean()
        denominator = (
            centered_prediction.square().sum().sqrt()
            * centered_target.square().sum().sqrt()
        )
        correlation = (
            float((centered_prediction * centered_target).sum() / denominator)
            if float(denominator) > 1e-8 else 0.0
        )
        neutral_error = 0.5 - target
        profile_codes = torch.zeros_like(ids)
        for vehicle_index, profile in (rule_vehicles or {}).items():
            code = {"yielding": 1, "moderate": 2, "non_yielding": 3}[profile]
            profile_codes = torch.where(
                ids == vehicle_index, torch.full_like(profile_codes, code), profile_codes
            )
        return {
            "seed": seed,
            "behavior_prefix": str(behavior_prefix),
            "candidate_checkpoint": checkpoint,
            "steps": steps,
            "valid_pairs": int(valid.sum()),
            "future_mae": float(error.abs().mean()),
            "future_brier": float(error.square().mean()),
            "future_bias": float(error.mean()),
            "future_correlation": correlation,
            "neutral_mae": float(neutral_error.abs().mean()),
            "neutral_brier": float(neutral_error.square().mean()),
            "prediction_mean": float(prediction.mean()),
            "target_mean": float(target.mean()),
            "positive_prediction_ratio": float((prediction > 0.5).float().mean()),
            "positive_target_ratio": float((target > 0.5).float().mean()),
            "rule_vehicles": rule_vehicles or {},
            "_prediction": prediction.detach().cpu(),
            "_target": target.detach().cpu(),
            "_quantile_raw": outputs["future_quantile_raw"][valid].detach().cpu(),
            "_neighbor_profile": profile_codes[valid].detach().cpu(),
        }


def evaluate_one(seed: int, *, config: str, checkpoint: str, steps: int) -> dict:
    result = collect_one(seed, config=config, checkpoint=checkpoint, steps=steps)
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/current/config_dgppo_nod_future_candidate_only.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/dependencies/dgppo_nod_future_candidate_only/final_nod.pth",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--output",
        default="outputs/dependencies/dgppo_nod_future_candidate_only/validation.json",
    )
    args = parser.parse_args()
    if args.steps < 2 or any(seed < 0 for seed in args.seeds):
        parser.error("Steps must be at least 2 and seeds must be non-negative")
    results = [
        evaluate_one(seed, config=args.config, checkpoint=args.checkpoint,
                     steps=args.steps)
        for seed in args.seeds
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(json.dumps(result, sort_keys=True))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
