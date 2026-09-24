"""Run paired all-Actor evaluations for NOD-DGPPO, SigmaRL, and XP-MARL."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utilities.sota_evaluation import aggregate_sota_runs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/sota_all_actor.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Optional method names from the manifest.")
    parser.add_argument("--output-root", default="outputs/benchmarks/sota_all_actor")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    seeds = args.seeds if args.seeds is not None else config["seeds"]
    scenario = args.scenario or config["scenario"]
    max_steps = args.max_steps or config["max_steps"]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("Evaluation seeds must be unique non-negative integers")
    if max_steps < 2:
        parser.error("--max-steps must be at least 2")

    methods = config["methods"]
    if args.methods:
        selected = set(args.methods)
        methods = [method for method in methods if method["name"] in selected]
        missing = selected - {method["name"] for method in methods}
        if missing:
            parser.error(f"Unknown methods: {sorted(missing)}")
    if not methods:
        parser.error("At least one method is required")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    protocol_dir = output_root / scenario / f"steps_{max_steps}" / (
        "seeds_" + "_".join(str(seed) for seed in seeds)
    )
    summary_paths = []
    for method in methods:
        model_path = Path(method["model_path"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        for seed in seeds:
            run_dir = protocol_dir / method["name"] / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            if not (args.skip_existing and summary_path.is_file()):
                command = [
                    sys.executable,
                    "-m", "scripts.evaluate_sota_seed",
                    "--method", method["name"],
                    "--label", method["label"],
                    "--model-path", str(model_path),
                    "--output-dir", str(run_dir),
                    "--seed", str(seed),
                    "--scenario", scenario,
                    "--max-steps", str(max_steps),
                    "--expected-prioritized-marl",
                    str(bool(method["expected_prioritized_marl"])).lower(),
                ]
                if not config.get("deterministic_actions", True):
                    command.append("--stochastic-actions")
                if config.get("observation_noise", False):
                    command.append("--checkpoint-observation-noise")
                if args.save_video:
                    command.append("--save-video")
                print(f"\n[benchmark] {method['label']} | seed={seed}")
                subprocess.run(command, check=True, cwd=ROOT)
            if not summary_path.is_file():
                raise FileNotFoundError(f"Evaluation did not produce {summary_path}")
            summary_paths.append(summary_path)

    result = aggregate_sota_runs(
        summary_paths,
        protocol_dir,
        reference_method=config.get("reference_method", "nod_dgppo"),
    )
    print("\n[SOTA benchmark aggregate]")
    print(json.dumps(result["protocol"], indent=2))
    for method, values in result["aggregate"].items():
        print(f"\n{values['label']} ({method})")
        print(f"collision_free_rate: {values['collision_free_rate']:.1%}")
        for metric in ("vehicle_collisions", "mean_speed_mps", "stop_fraction",
                       "longest_stop_seconds", "mean_agent_return"):
            if metric in values["metrics"]:
                stats = values["metrics"][metric]
                print(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                      f"sum={stats['sum']:.6f}")
    print(f"\nJSON: {(protocol_dir / 'summary.json').resolve()}")
    print(f"Runs CSV: {(protocol_dir / 'runs.csv').resolve()}")
    print(f"Aggregate CSV: {(protocol_dir / 'aggregate.csv').resolve()}")
    print(f"Pairwise CSV: {(protocol_dir / 'pairwise.csv').resolve()}")


if __name__ == "__main__":
    main()
