"""Matched paper-metric evaluation of NOD-DGPPO, SigmaRL, and XP-MARL."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation


ROOT = Path(project_root)
MODEL_PATHS = [
    "outputs/current/dgppo_nod_gain1_control_finetune/",
    "checkpoints/itsc24/M0 (our)/",
    "checkpoints/icra25/M1 (XP-MARL)/",
]
LEGENDS = ["NOD-DGPPO", "SigmaRL", "XP-MARL"]
DEFAULT_SCENARIOS = ["CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if args.num_simulations < 1:
        parser.error("--num-simulations must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    for relative_path in MODEL_PATHS:
        if not (ROOT / relative_path).is_dir():
            raise FileNotFoundError(
                f"Missing model directory: {ROOT / relative_path}. "
                "Run `python -m scripts.prepare_sota_checkpoints` first."
            )

    fig_sizes = {
        "episode_reward": (3.8, 4.2),
        "collision_rate": (3.5, 2.0),
        "centerline_deviation": (3.5, 2.0),
        "average_speed": (3.5, 2.0),
        "smoothness": (3.5, 2.0),
    }
    y_limits = {
        "episode_reward": [-1, 8],
        "collision_rate": [0, 3],
        "centerline_deviation": [0, 100],
        "average_speed": [0, 100],
        "smoothness": [0, 100],
    }
    model_paths = [str((ROOT / path).resolve()) + os.sep for path in MODEL_PATHS]

    for scenario in args.scenarios:
        print("*" * 72)
        print(f"[INFO] Matched paper evaluation: {scenario}")
        print("*" * 72)
        output_dir = (
            ROOT / "outputs" / "benchmarks" / "paper_metrics" / scenario
            / f"steps_{args.steps}_envs_{args.num_simulations}_seed_{args.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = Evaluation(
            scenario_type=scenario,
            model_paths=model_paths,
            fitst_model_index=0,
            idx_our=0,
            num_agents=SCENARIOS[scenario]["n_agents"],
            fig_sizes=fig_sizes,
            y_limits=y_limits,
            simulation_steps=args.steps,
            is_show_different_collisions=True,
            x_ticks=LEGENDS,
            x_tick_label_rotation=15,
            where_to_save_eva_results=str(output_dir),
            where_to_save_logging=str(output_dir / "log.txt"),
            legends=LEGENDS,
            render_titles=LEGENDS,
            num_simulations_per_model=args.num_simulations,
            is_render=False,
            is_save_simulation_video=False,
            is_measure_policy_inference_time=args.measure_inference_time,
            video_names=["nod_dgppo", "sigmarl", "xp_marl"],
            evaluation_seed=args.seed,
            deterministic_actions=not args.stochastic_actions,
            observation_noise=(None if args.checkpoint_observation_noise else False),
            reuse_cached_rollouts=False,
            save_rollout_cache=False,
            export_machine_readable=True,
        )
        evaluator.run_evaluation()


if __name__ == "__main__":
    main()
