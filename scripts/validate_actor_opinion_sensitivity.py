"""Sweep a cached directed opinion while keeping the physical state fixed.

This diagnostic searches one deterministic rollout for a safe, active
ego-neighbor interaction. It then changes only the cached z coordinate and
re-evaluates the same Actor state. No environment transition is taken between
the probes, so position, velocity, history, previous action and all physical
neighbor features remain identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from utilities.helper_training import SaveData
from utilities.mappo_cavs import mappo_cavs
from utilities.nod_marl.dgppo import dgppo_advantage, opinion_alpha
from utilities.nod_marl.safety_value import same_entities
from utilities.nod_marl.policy import (
    NOD_ACTOR_CONTEXT_READY_KEY,
    NOD_ACTOR_EDGE_CONTEXT_KEY,
    NOD_ACTOR_MESSAGE_KEY,
    NOD_OPINION_ACTIVE_KEY,
)


def _checkpoint_json(model_path: Path) -> Path:
    candidates = list(model_path.glob("reward*_data.json"))
    if not candidates:
        raise FileNotFoundError(f"No reward*_data.json found in {model_path}")

    def reward(path: Path) -> float:
        match = re.match(r"reward(-?[0-9]*\.?[0-9]+)_data\.json", path.name)
        return float(match.group(1)) if match else float("-inf")

    return max(candidates, key=reward)


def _load(model_path: Path, *, seed: int, max_steps: int, final: bool):
    data_file = _checkpoint_json(model_path)
    with data_file.open(encoding="utf-8") as file:
        parameters = SaveData.from_dict(json.load(file)).parameters
    parameters.where_to_save = str(model_path.resolve()) + os.sep
    parameters.seed = seed
    parameters.max_steps = max_steps + 1
    parameters.num_vmas_envs = 1
    parameters.is_testing_mode = True
    parameters.is_load_model = True
    parameters.is_load_final_model = final
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.is_save_eval_results = False
    parameters.is_save_simulation_video = False
    parameters.is_real_time_rendering = False
    parameters.is_using_deadlock_critic = False
    # The sweep must expose the learned response to z even if a neutral-z
    # ablation checkpoint is supplied explicitly.
    parameters.nod_actor_opinion_mode = "online"
    env, policy, _, parameters = mappo_cavs(parameters)
    return env, policy, parameters, data_file


@torch.no_grad()
def _find_fixed_state(env, policy, parameters, *, ego: int, max_steps: int):
    manager = env.scenario.safety_value_manager
    if not manager.enabled or manager.model is None:
        raise ValueError("A loaded DGPPO Safety Value is required")
    td = env.reset()
    best = None
    mapped_extent = min(1.0, float(parameters.dgppo_alpha_gain))
    alpha_low = parameters.dgppo_alpha - parameters.dgppo_alpha_span * mapped_extent
    alpha_high = parameters.dgppo_alpha + parameters.dgppo_alpha_span * mapped_extent
    for step in range(max_steps):
        decision = policy(td)
        active = decision.get(NOD_OPINION_ACTIVE_KEY, default=None)
        ids = decision.get(("agents", "info", "nod_neighbor_indices"), default=None)
        state = manager.state(decision)
        value = manager.model(state)
        transition = env.step(decision)
        following = manager.state(transition.get("next"))
        next_value = manager.model(following)
        if active is not None and ids is not None:
            _, details = opinion_alpha(
                decision, state, value,
                alpha=parameters.dgppo_alpha,
                span=parameters.dgppo_alpha_span,
                gain=parameters.dgppo_alpha_gain,
                deadzone=parameters.dgppo_opinion_deadzone,
            )
            continuous = (
                state["valid"] & following["valid"]
                & same_entities(state, following)
                & torch.isfinite(value) & torch.isfinite(next_value)
            )
            for slot in active[0, ego].bool().nonzero().flatten().tolist():
                neighbor = int(ids[0, ego, slot])
                if (0 <= neighbor < parameters.n_agents
                        and bool(details["available"][0, ego, neighbor])
                        and bool(details["safe"][0, ego, neighbor])
                        and bool(continuous[0, ego, neighbor])):
                    current_v = float(value[0, ego, neighbor])
                    following_v = float(next_value[0, ego, neighbor])
                    change_rate = (following_v - current_v) / parameters.dt
                    delta_low = change_rate + alpha_low * current_v
                    delta_base = change_rate + parameters.dgppo_alpha * current_v
                    delta_high = change_rate + alpha_high * current_v
                    flips = delta_low > 0 >= delta_high
                    distance = 0.0 if flips else min(abs(delta_low), abs(delta_high))
                    sensitivity = abs(delta_low - delta_high)
                    score = (int(flips), -distance, sensitivity)
                    if best is None or score > best[0]:
                        best = (
                            score,
                            decision.detach().clone(),
                            following,
                            next_value,
                            step,
                            slot,
                            neighbor,
                            {
                                "alpha_low": alpha_low,
                                "alpha_high": alpha_high,
                                "delta_low": delta_low,
                                "delta_base": delta_base,
                                "delta_high": delta_high,
                                "alpha_changes_pair_gate": flips,
                            },
                        )
        td = step_mdp(transition)
    if best is not None:
        _, fixed, following, next_value, step, slot, neighbor, selection = best
        return fixed, following, next_value, step, slot, neighbor, selection
    raise RuntimeError(
        f"No continuous safe active interaction for agent {ego + 1} in {max_steps} steps; "
        "try another --seed, --ego or a larger --max-search-steps"
    )


@torch.no_grad()
def _probe(policy, manager, parameters, fixed, following, next_value, *,
           ego, slot, neighbor, z_values):
    rows = []
    for z in z_values:
        probe = fixed.clone()
        context = probe.get(NOD_ACTOR_EDGE_CONTEXT_KEY).clone()
        context[0, ego, slot, -1] = float(z)
        probe.set(NOD_ACTOR_EDGE_CONTEXT_KEY, context)
        ready = probe.get(NOD_ACTOR_CONTEXT_READY_KEY)
        if not bool(ready.all()):
            raise RuntimeError("Fixed state does not contain a complete Actor context")
        probe = policy(probe)
        state = manager.state(probe)
        value = manager.model(state)
        coefficients, details = opinion_alpha(
            probe, state, value,
            alpha=parameters.dgppo_alpha,
            span=parameters.dgppo_alpha_span,
            gain=parameters.dgppo_alpha_gain,
            deadzone=parameters.dgppo_opinion_deadzone,
        )
        task = value.new_zeros(*value.shape[:-1], 1)
        _, barrier = dgppo_advantage(
            task, state, following, value, next_value,
            dt=parameters.dt, alpha=coefficients,
            eps=parameters.dgppo_eps, weight=parameters.dgppo_weight,
            task_mode=parameters.dgppo_task_mode,
        )
        action = probe.get(("agents", "action"))[0, ego]
        loc = probe.get(("agents", "loc"))[0, ego]
        message = probe.get(NOD_ACTOR_MESSAGE_KEY)[0, ego]
        rows.append({
            "z_injected": float(z),
            "z_aligned": float(details["opinions"][0, ego, neighbor]),
            "alpha": float(coefficients[0, ego, neighbor]),
            "alpha_applied": int(details["applied"][0, ego, neighbor]),
            "pair_g": float(state["g"][0, ego, neighbor]),
            "pair_value": float(value[0, ego, neighbor]),
            "fixed_successor_pair_delta": float(
                barrier["delta"][0, ego, neighbor]
            ),
            "fixed_successor_pair_violation": int(
                barrier["delta"][0, ego, neighbor] > 0
            ),
            "fixed_successor_any_violation": int(
                barrier["violation"][0, ego]
            ),
            "fixed_successor_penalty": float(barrier["penalty"][0, ego, 0]),
            "velocity_action": float(action[0]),
            "steering_action": float(action[1]),
            "policy_loc_velocity": float(loc[0]),
            "policy_loc_steering": float(loc[1]),
            "message_norm": float(message.norm()),
        })
    neutral = min(rows, key=lambda row: abs(row["z_injected"]))
    for row in rows:
        row["velocity_delta_from_neutral"] = (
            row["velocity_action"] - neutral["velocity_action"]
        )
        row["steering_delta_from_neutral"] = (
            row["steering_action"] - neutral["steering_action"]
        )
    return rows


def _summary(rows):
    ordered = sorted(rows, key=lambda row: row["z_injected"])
    velocities = [row["velocity_action"] for row in ordered]
    alphas = [row["alpha"] for row in ordered]
    tolerance = 1e-6
    negative = [row["velocity_action"] for row in rows if row["z_injected"] < 0]
    positive = [row["velocity_action"] for row in rows if row["z_injected"] > 0]
    return {
        "alpha_nondecreasing_with_z": all(
            right + tolerance >= left
            for left, right in zip(alphas, alphas[1:])
        ),
        "velocity_nondecreasing_with_z": all(
            right + tolerance >= left
            for left, right in zip(velocities, velocities[1:])
        ),
        "velocity_action_range": max(velocities) - min(velocities),
        "positive_minus_negative_velocity": (
            sum(positive) / len(positive) - sum(negative) / len(negative)
            if positive and negative else None
        ),
        "interpretation": (
            "A positive value means the Actor commands more speed when the "
            "neighbor carries more avoidance responsibility."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="outputs/current/dgppo_nod_gain1_control_finetune/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ego", type=int, default=1,
                        help="One-based agent number whose directed opinion is swept.")
    parser.add_argument("--max-search-steps", type=int, default=1200)
    parser.add_argument("--z-values", type=float, nargs="+",
                        default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.seed < 0 or args.max_search_steps < 1:
        parser.error("seed must be non-negative and max-search-steps must be positive")
    if any(not -1 <= value <= 1 for value in args.z_values):
        parser.error("all z-values must lie in [-1, 1]")
    if len(set(args.z_values)) != len(args.z_values):
        parser.error("z-values must not contain duplicates")
    if 0.0 not in args.z_values:
        parser.error("z-values must contain 0 for the neutral reference")

    model_path = Path(args.model_path).resolve()
    output = (Path(args.output).resolve() if args.output else
              model_path / "actor_opinion_sensitivity" / f"seed{args.seed}_agent{args.ego}")
    output.mkdir(parents=True, exist_ok=True)
    env = None
    with set_exploration_type(ExplorationType.MODE):
        try:
            env, policy, parameters, data_file = _load(
                model_path, seed=args.seed, max_steps=args.max_search_steps,
                final=args.final,
            )
            ego = args.ego - 1
            if not 0 <= ego < parameters.n_agents:
                parser.error(f"--ego must be in 1..{parameters.n_agents}")
            fixed, following, next_value, step, slot, neighbor, selection = _find_fixed_state(
                env, policy, parameters, ego=ego,
                max_steps=args.max_search_steps,
            )
            manager = env.scenario.safety_value_manager
            rows = _probe(
                policy, manager, parameters, fixed, following, next_value,
                ego=ego, slot=slot, neighbor=neighbor,
                z_values=args.z_values,
            )
            result = {
                "model_path": str(model_path),
                "checkpoint_json": str(data_file),
                "seed": args.seed,
                "decision_step": step,
                "decision_time": step * parameters.dt,
                "ego_agent": ego + 1,
                "neighbor_agent": neighbor + 1,
                "edge_slot": slot,
                "state_selection": selection,
                "alpha_rule": {
                    "base": parameters.dgppo_alpha,
                    "span": parameters.dgppo_alpha_span,
                    "gain": parameters.dgppo_alpha_gain,
                    "deadzone": parameters.dgppo_opinion_deadzone,
                },
                "summary": _summary(rows),
                "probes": rows,
            }
            with (output / "sweep.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output / "summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print("z       alpha    velocity   steering   delta-v   gate   penalty")
            for row in rows:
                print(f"{row['z_injected']:+.3f}  {row['alpha']:8.3f}  "
                      f"{row['velocity_action']:+9.4f}  "
                      f"{row['steering_action']:+9.4f}  "
                      f"{row['velocity_delta_from_neutral']:+9.4f}  "
                      f"{row['fixed_successor_any_violation']:4d}  "
                      f"{row['fixed_successor_penalty']:.5f}")
            print(json.dumps(result["summary"], indent=2))
            print(f"Saved: {output}")
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
