"""Run reproducible mixed-controller evaluations and aggregate their summaries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from utilities.testing_evaluation import aggregate_rule_runs, rule_run_name
from utilities.testing_rule_coordinator import CONTROLLER_VERSION
from utilities.constants import SCENARIOS


DEFAULT_WEIGHTS = {"yielding": 0.25, "moderate": 0.5, "non_yielding": 0.25}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--model-path", default="outputs/current/dgppo_nod_gain1_control_finetune/")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="intersection_2")
    parser.add_argument("--rule-assignment-seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--no-video", action="store_true",
                        help="Keep CSV/JSON statistics but skip MP4 encoding.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse a seed only when its summary.json already exists.")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    if args.rule_assignment_seed < 0:
        parser.error("--rule-assignment-seed must be non-negative")
    if args.max_steps < 2:
        parser.error("--max-steps must be at least 2")

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path = str(model_path.resolve()) + os.sep
    summary_paths = []
    for seed in args.seeds:
        name = rule_run_name(
            args.scenario,
            .5,
            DEFAULT_WEIGHTS,
            assignment_seed=args.rule_assignment_seed,
            environment_seed=seed,
            actor_index=0,
            controller_version=CONTROLLER_VERSION,
        )
        output_dir = os.path.join(model_path, "rule_vehicle_visualization", name)
        summary_path = os.path.join(output_dir, "summary.json")
        if not (args.skip_existing and os.path.isfile(summary_path)):
            command = [
                sys.executable,
                str(ROOT / "main_testing.py"),
                "--model-path", model_path,
                "--env-seed", str(seed),
                "--rule-assignment-seed", str(args.rule_assignment_seed),
                "--scenario", args.scenario,
                "--max-steps", str(args.max_steps),
                "--no-speed-print",
                "--no-realtime",
            ]
            if args.no_video:
                command.append("--no-video")
            print(f"\n[Multi-seed] environment seed {seed}")
            subprocess.run(command, check=True, cwd=ROOT)
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Evaluation did not produce {summary_path}")
        summary_paths.append(summary_path)

    aggregate_dir = os.path.join(
        model_path,
        "rule_vehicle_multiseed",
        f"{args.scenario}_assignseed{args.rule_assignment_seed}_{CONTROLLER_VERSION}",
        "envseeds_" + "_".join(str(seed) for seed in args.seeds),
    )
    result, json_path, csv_path = aggregate_rule_runs(summary_paths, aggregate_dir)
    print("\n[Multi-seed aggregate]")
    print(f"runs={result['n_runs']} seeds={result['seeds']}")
    for name in ("total_collisions", "vehicle_collisions", "road_collisions",
                 "all_mean_speed", "all_stop_fraction", "emergency_seconds"):
        metric = result["aggregate"][name]
        print(f"{name}: mean={metric['mean']:.6f}, std={metric['std']:.6f}, "
              f"min={metric['min']:.6f}, max={metric['max']:.6f}")
    print(f"collision-free runs: {result['collision_free_runs']}/{result['n_runs']} "
          f"({result['collision_free_rate']:.1%})")
    print(f"rule-rule contact runs: {result['rule_rule_contact_runs']}/{result['n_runs']}")
    print(f"JSON: {os.path.abspath(json_path)}")
    print(f"CSV:  {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()
