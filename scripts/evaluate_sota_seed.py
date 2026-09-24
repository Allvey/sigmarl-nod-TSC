"""Evaluate one learned policy on one environment seed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from vmas.simulator.utils import save_video

from utilities.constants import SCENARIOS
from utilities.mappo_cavs import mappo_cavs
from utilities.sota_evaluation import load_benchmark_parameters, summarize_rollout


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--expected-prioritized-marl", choices=("true", "false"), required=True)
    parser.add_argument("--stochastic-actions", action="store_true")
    parser.add_argument("--checkpoint-observation-noise", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.max_steps < 2:
        parser.error("--max-steps must be at least 2")

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    parameters, raw_parameters, checkpoint_prefix = load_benchmark_parameters(model_path)
    expected_prioritized = args.expected_prioritized_marl == "true"
    actual_prioritized = bool(raw_parameters.get("is_using_prioritized_marl", False))
    if actual_prioritized != expected_prioritized:
        raise ValueError(
            f"{args.method}: manifest expects prioritized_marl={expected_prioritized}, "
            f"checkpoint says {actual_prioritized}"
        )

    parameters.where_to_save = str(model_path.resolve()) + os.sep
    parameters.seed = args.seed
    parameters.scenario_type = args.scenario
    parameters.n_agents = SCENARIOS[args.scenario]["n_agents"]
    parameters.max_steps = args.max_steps
    parameters.num_vmas_envs = 1
    parameters.frames_per_batch = args.max_steps
    parameters.is_testing_mode = True
    parameters.is_load_model = True
    parameters.is_load_final_model = False
    parameters.is_continue_train = False
    parameters.is_load_out_td = False
    parameters.is_save_eval_results = False
    parameters.is_save_simulation_video = args.save_video
    parameters.is_real_time_rendering = False
    parameters.is_visualize_short_term_path = False
    parameters.is_visualize_lane_boundary = False
    parameters.is_visualize_extra_info = False
    parameters.is_add_noise = (
        bool(raw_parameters.get("is_add_noise", False))
        if args.checkpoint_observation_noise else False
    )

    env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)
    deterministic = not args.stochastic_actions
    exploration = ExplorationType.MODE if deterministic else ExplorationType.RANDOM
    callback = (
        (lambda env, _: env.render(mode="rgb_array", visualize_when_rgb=False))
        if args.save_video else None
    )
    with torch.no_grad(), set_exploration_type(exploration):
        rollout = env.rollout(
            max_steps=args.max_steps,
            policy=policy,
            priority_module=priority_module,
            callback=callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=args.save_video,
        )
    if isinstance(rollout, tuple):
        out_td, frames = rollout
    else:
        out_td, frames = rollout, []

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rollout(
        out_td,
        method=args.method,
        label=args.label,
        model_path=model_path,
        checkpoint_prefix=checkpoint_prefix,
        scenario=args.scenario,
        seed=args.seed,
        dt=float(parameters.dt),
        deterministic_actions=deterministic,
        observation_noise=bool(parameters.is_add_noise),
    )
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    if frames:
        save_video(str(output_dir / "video"), frames, fps=1 / float(parameters.dt))
    print("[SOTA run] " + json.dumps(summary, sort_keys=True))
    print(f"[SOTA run] summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
