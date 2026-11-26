"""Four-method, five-metric evaluation on four legacy and two new traffic maps."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation
from baselines.original_sigmarl.runtime import BACKENDS
from baselines.tsc.runtime import DEFAULT_TSC_PROJECT


ROOT = Path(project_root)
MODEL_PATHS = [
    "checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/",
    "checkpoints/sigmarl_original/config_v1_2_0/",
    "checkpoints/icra25/M1 (XP-MARL)/",
]
RELEASED_MODEL_PATHS = [
    MODEL_PATHS[0],
    "checkpoints/itsc24/M0 (our)/",
    "checkpoints/icra25/M1 (XP-MARL)/",
]
LEGENDS = ["NOD-DGPPO", "SigmaRL", "XP-MARL"]
DEFAULT_SCENARIOS = ["CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1",
                     "roundabout_realistic", "bottleneck_1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-paths", nargs="+",
        help="Optional model directories, in the same order as --legends; default compares four methods including TSC.",
    )
    parser.add_argument("--legends", nargs="+")
    parser.add_argument("--without-tsc", action="store_true", help="Keep the fixed three-method comparison")
    parser.add_argument("--tsc-project", default=str(DEFAULT_TSC_PROJECT), help="Original TSC project directory")
    parser.add_argument("--tsc-model-path", default="outputs/TSC/seed5/", help="TSC model directory relative to --tsc-project, or absolute")
    parser.add_argument(
        "--model-backends", nargs="+", choices=BACKENDS,
        help="One runtime per model; default auto uses upstream for original metadata and current for NOD.",
    )
    parser.add_argument(
        "--baseline-checkpoints", choices=("mixed", "released", "retrained"), default="mixed",
        help="Default mixed: retrained SigmaRL and released ICRA25 XP-MARL; alternatively use both released or both retrained.",
    )
    parser.add_argument(
        "--output-root",
        help="Separate result root for a new comparison, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=sorted(SCENARIOS),
        default=DEFAULT_SCENARIOS,
    )
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--stochastic-actions", action="store_true",
        help="Match the historical scripts' sampled policy actions.",
    )
    parser.add_argument(
        "--checkpoint-observation-noise", action="store_true",
        help="Keep each checkpoint's saved noise setting instead of disabling noise for all.",
    )
    parser.add_argument("--measure-inference-time", action="store_true")
    args = parser.parse_args()
    use_default_models = args.model_paths is None
    include_tsc = use_default_models and not args.without_tsc
    args.output_root = args.output_root or (
        ("outputs/benchmarks/paper_metrics_with_tsc/"
         + Path(args.tsc_model_path).name + "/" + args.baseline_checkpoints + "/four_methods/six_scenarios")
        if include_tsc else
        ("outputs/benchmarks/paper_metrics_original_baselines/" + args.baseline_checkpoints
        + ("/config_v1_2_0" if args.baseline_checkpoints == "retrained" else "")
        + ("/three_methods" if use_default_models else "/custom_methods"))
    )
    if use_default_models:
        if args.baseline_checkpoints == "released":
            args.model_paths = list(RELEASED_MODEL_PATHS)
        elif args.baseline_checkpoints == "retrained":
            args.model_paths = list(MODEL_PATHS)
            for index, method in ((1, "sigmarl"), (2, "xp_marl")):
                profile = ROOT / "configs/baselines" / (method + "_original.json")
                args.model_paths[index] = json.loads(profile.read_text())["where_to_save"]
        else:
            args.model_paths = list(MODEL_PATHS)
        if include_tsc:
            args.model_paths.append(str((Path(args.tsc_project) / args.tsc_model_path).resolve()))
    if args.legends is None:
        args.legends = list(LEGENDS) + (["TSC"] if include_tsc else [])
    if args.num_simulations < 1:
        parser.error("--num-simulations must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if len(args.model_paths) != len(args.legends):
        parser.error("--model-paths and --legends must have the same number of entries")
    if len(set(args.legends)) != len(args.legends):
        parser.error("--legends must be unique")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios must be unique")
    if args.model_backends is not None and len(args.model_backends) != len(args.model_paths):
        parser.error("--model-backends must have one entry per --model-paths")

    for relative_path in args.model_paths:
        if not (ROOT / relative_path).is_dir():
            raise FileNotFoundError(
                f"Missing model directory: {ROOT / relative_path}. "
                "Train the corresponding branch or prepare the official comparison checkpoints first."
            )

    fig_sizes = {
        "episode_reward": (3.8, 4.2),
        "collision_rate": (3.5, 2.0),
        "throughput": (3.5, 2.0),
        "average_speed": (3.5, 2.0),
        "smoothness": (3.5, 2.0),
    }
    if len(args.model_paths) > 3:
        for metric in ("collision_rate", "throughput", "average_speed", "smoothness"):
            fig_sizes[metric] = (4.8, 2.4)
    y_limits = {
        "episode_reward": [-1, 8],
        "collision_rate": [0, None],
        "throughput": [0, None],
        "average_speed": [0, 100],
        "smoothness": [0, None],
    }
    model_paths = [str((ROOT / path).resolve()) + os.sep for path in args.model_paths]
    output_root = (ROOT / args.output_root).resolve()

    for scenario in args.scenarios:
        print("*" * 72)
        print(f"[INFO] Matched paper evaluation: {scenario}")
        print("*" * 72)
        output_dir = (
            output_root / scenario
            / f"steps_{args.steps}_envs_{args.num_simulations}_seed_{args.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = Evaluation(
            scenario_type=scenario,
            model_paths=model_paths,
            model_backends=args.model_backends or ["auto"] * len(model_paths),
            tsc_project=args.tsc_project,
            fitst_model_index=0,
            idx_our=0,
            num_agents=SCENARIOS[scenario]["n_agents"],
            fig_sizes=fig_sizes,
            y_limits=y_limits,
            simulation_steps=args.steps,
            is_show_different_collisions=True,
            x_ticks=args.legends,
            x_tick_label_rotation=15,
            where_to_save_eva_results=str(output_dir),
            where_to_save_logging=str(output_dir / "log.txt"),
            legends=args.legends,
            render_titles=args.legends,
            num_simulations_per_model=args.num_simulations,
            is_render=False,
            is_save_simulation_video=False,
            is_measure_policy_inference_time=args.measure_inference_time,
            video_names=[f"method_{index}" for index in range(len(model_paths))],
            evaluation_seed=args.seed,
            deterministic_actions=not args.stochastic_actions,
            observation_noise=(None if args.checkpoint_observation_noise else False),
            reuse_cached_rollouts=False,
            save_rollout_cache=False,
            export_machine_readable=True,
            measure_centerline_deviation=False,
            ttc_threshold_seconds=None,
            measure_throughput=True,
            measure_trajectory_jerk=True,
        )
        evaluator.run_evaluation()


if __name__ == "__main__":
    main()
