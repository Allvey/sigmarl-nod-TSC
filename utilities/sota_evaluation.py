"""Shared loading, metrics, and aggregation for matched all-Actor benchmarks."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import torch

from utilities.helper_training import Parameters, SaveData


REWARD_JSON = re.compile(r"^reward(-?[0-9]+(?:\.[0-9]+)?)_data\.json$")


def select_best_checkpoint(model_dir: Path) -> tuple[float, Path, str]:
    """Return reward, metadata JSON, and exact checkpoint prefix."""
    candidates = []
    for path in model_dir.glob("reward*_data.json"):
        match = REWARD_JSON.match(path.name)
        if not match:
            continue
        reward = float(match.group(1))
        prefix = path.name[: -len("_data.json")]
        if (model_dir / f"{prefix}_policy.pth").is_file():
            candidates.append((reward, path, prefix))
    if not candidates:
        raise FileNotFoundError(
            f"No reward*_data.json with a matching policy was found in {model_dir}"
        )
    return max(candidates, key=lambda item: item[0])


def load_benchmark_parameters(model_dir: Path) -> tuple[Parameters, dict[str, Any], str]:
    """Load a checkpoint while keeping pre-NOD models architecture-compatible."""
    _, metadata_path, prefix = select_best_checkpoint(model_dir)
    with metadata_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    raw_parameters = raw.get("parameters", raw)
    if "parameters" in raw:
        parameters = SaveData.from_dict(raw).parameters
    else:
        parameters = Parameters.from_dict(raw_parameters)

    # Old official checkpoints predate these modules. Parameters.from_dict fills
    # new defaults, so leaving them untouched would silently change the Actor
    # architecture and make the historical weights impossible to load.
    if "is_using_nod_opinion" not in raw_parameters:
        parameters.is_using_nod_opinion = False
        parameters.is_using_nod_actor = False
        parameters.is_using_safety_critic = False
        parameters.is_using_safety_value_shadow = False
        parameters.is_using_safety_constraint = False
        parameters.is_using_deadlock_critic = False
        parameters.nod_freeze_training = False
        parameters.safety_control_mode = "legacy_q"
    return parameters, raw_parameters, prefix


def _tensor(out_td, *keys):
    for key in keys:
        value = out_td.get(key, default=None)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    return None


def _longest_below(speed: torch.Tensor, threshold: float) -> float:
    # speed shape: [environment, time, agent]
    longest = 0
    for series in speed.permute(0, 2, 1).reshape(-1, speed.shape[1]):
        current = 0
        for stopped in (series < threshold).tolist():
            current = current + 1 if stopped else 0
            longest = max(longest, current)
    return float(longest)


def summarize_rollout(
    out_td,
    *,
    method: str,
    label: str,
    model_path: Path,
    checkpoint_prefix: str,
    scenario: str,
    seed: int,
    dt: float,
    deterministic_actions: bool,
    observation_noise: bool,
    stop_threshold: float = 0.05,
) -> dict[str, Any]:
    velocity = _tensor(
        out_td,
        ("next", "agents", "info", "vel"),
        ("agents", "info", "vel"),
    )
    if velocity is None:
        raise KeyError("Rollout has no agent velocity tensor")
    speed = velocity.norm(dim=-1)
    route_error = _tensor(
        out_td,
        ("next", "agents", "info", "testing_route_error"),
        ("agents", "info", "testing_route_error"),
    )
    reward = _tensor(
        out_td,
        ("next", "agents", "reward"),
        ("agents", "reward"),
    )
    actions = _tensor(out_td, ("agents", "action"))
    vehicle_events = _tensor(
        out_td,
        ("next", "agents", "info", "testing_vehicle_collision_events"),
        ("agents", "info", "testing_vehicle_collision_events"),
    )
    road_events = _tensor(
        out_td,
        ("next", "agents", "info", "testing_road_collision_events"),
        ("agents", "info", "testing_road_collision_events"),
    )
    if vehicle_events is None or road_events is None:
        raise KeyError("Rollout has no cumulative testing collision counters")

    steps = int(speed.shape[1])
    summary = {
        "method": method,
        "label": label,
        "model_path": str(model_path.resolve()),
        "checkpoint": checkpoint_prefix,
        "scenario": scenario,
        "environment_seed": int(seed),
        "steps": steps,
        "duration_seconds": steps * float(dt),
        "n_agents": int(speed.shape[-1]),
        "deterministic_actions": bool(deterministic_actions),
        "observation_noise": bool(observation_noise),
        "vehicle_collisions": int(vehicle_events.max().item()),
        "road_collisions": int(road_events.max().item()),
        "mean_speed_mps": float(speed.mean().item()),
        "median_speed_mps": float(speed.median().item()),
        "stop_fraction": float((speed < stop_threshold).float().mean().item()),
        "slow_fraction": float((speed < 0.2).float().mean().item()),
        "longest_stop_seconds": _longest_below(speed, stop_threshold) * float(dt),
    }
    if route_error is not None:
        summary["mean_route_error_m"] = float(route_error.abs().mean().item())
        summary["p95_route_error_m"] = float(
            torch.quantile(route_error.abs().reshape(-1).float(), 0.95).item()
        )
    if reward is not None:
        if reward.shape[-1] == 1:
            reward = reward.squeeze(-1)
        per_agent_return = reward.sum(dim=1)
        summary["mean_agent_return"] = float(per_agent_return.mean().item())
        summary["team_return"] = float(per_agent_return.sum(dim=-1).mean().item())
    if actions is not None and actions.shape[1] > 1:
        delta = actions[:, 1:] - actions[:, :-1]
        summary["mean_abs_accel_change"] = float(delta[..., 0].abs().mean().item())
        summary["mean_abs_steer_change"] = float(delta[..., 1].abs().mean().item())
    return summary


PRIMARY_METRICS = (
    "vehicle_collisions",
    "road_collisions",
    "mean_speed_mps",
    "stop_fraction",
    "longest_stop_seconds",
    "mean_agent_return",
    "mean_route_error_m",
    "mean_abs_accel_change",
    "mean_abs_steer_change",
)

LOWER_IS_BETTER = {
    "vehicle_collisions",
    "road_collisions",
    "stop_fraction",
    "longest_stop_seconds",
    "mean_route_error_m",
    "mean_abs_accel_change",
    "mean_abs_steer_change",
}


def _stats(values: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def aggregate_sota_runs(
    summary_paths: list[Path], output_dir: Path, *, reference_method: str = "nod_dgppo"
) -> dict[str, Any]:
    runs = []
    for path in summary_paths:
        with path.open(encoding="utf-8") as file:
            run = json.load(file)
        run["summary_path"] = str(path.resolve())
        runs.append(run)
    if not runs:
        raise ValueError("No benchmark summaries were provided")

    settings = {
        (run["scenario"], run["steps"], run["deterministic_actions"], run["observation_noise"])
        for run in runs
    }
    if len(settings) != 1:
        raise ValueError(f"Runs do not share one evaluation protocol: {settings}")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_method.setdefault(run["method"], []).append(run)
    seed_sets = {
        method: {int(run["environment_seed"]) for run in method_runs}
        for method, method_runs in by_method.items()
    }
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError(f"Methods do not share paired seeds: {seed_sets}")

    aggregate = {}
    for method, method_runs in by_method.items():
        aggregate[method] = {
            "label": method_runs[0]["label"],
            "n_runs": len(method_runs),
            "collision_free_rate": statistics.mean(
                float(run["vehicle_collisions"] == 0) for run in method_runs
            ),
            "metrics": {
                metric: _stats(run[metric] for run in method_runs)
                for metric in PRIMARY_METRICS
                if all(metric in run and math.isfinite(float(run[metric])) for run in method_runs)
            },
        }

    pairwise = {}
    if reference_method in by_method:
        reference_by_seed = {
            int(run["environment_seed"]): run for run in by_method[reference_method]
        }
        for method, method_runs in by_method.items():
            if method == reference_method:
                continue
            method_by_seed = {int(run["environment_seed"]): run for run in method_runs}
            metric_results = {}
            for metric in PRIMARY_METRICS:
                if not all(
                    metric in reference_by_seed[seed] and metric in method_by_seed[seed]
                    for seed in reference_by_seed
                ):
                    continue
                advantages = []
                for seed in sorted(reference_by_seed):
                    reference_value = float(reference_by_seed[seed][metric])
                    competitor_value = float(method_by_seed[seed][metric])
                    advantage = (
                        competitor_value - reference_value
                        if metric in LOWER_IS_BETTER
                        else reference_value - competitor_value
                    )
                    advantages.append(advantage)
                tolerance = 1e-12
                metric_results[metric] = {
                    "reference_advantage": _stats(advantages),
                    "wins": sum(value > tolerance for value in advantages),
                    "ties": sum(abs(value) <= tolerance for value in advantages),
                    "losses": sum(value < -tolerance for value in advantages),
                }
            pairwise[method] = {
                "reference_method": reference_method,
                "competitor_method": method,
                "metrics": metric_results,
            }

    result = {
        "benchmark_name": "sota_all_actor",
        "protocol": {
            "scenario": runs[0]["scenario"],
            "steps": runs[0]["steps"],
            "duration_seconds": runs[0]["duration_seconds"],
            "deterministic_actions": runs[0]["deterministic_actions"],
            "observation_noise": runs[0]["observation_noise"],
            "seeds": sorted(next(iter(seed_sets.values()))),
        },
        "aggregate": aggregate,
        "pairwise": pairwise,
        "runs": sorted(runs, key=lambda run: (run["method"], run["environment_seed"])),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    csv_fields = [
        "method", "label", "environment_seed", "checkpoint", "scenario", "steps",
        *PRIMARY_METRICS, "summary_path",
    ]
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["runs"])

    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["method", "label", "metric", "mean", "std", "min", "max", "sum"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method, method_result in aggregate.items():
            for metric, stats in method_result["metrics"].items():
                writer.writerow({
                    "method": method,
                    "label": method_result["label"],
                    "metric": metric,
                    **stats,
                })
    with (output_dir / "pairwise.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "reference_method", "competitor_method", "metric", "advantage_mean",
            "advantage_std", "wins", "ties", "losses",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method_result in pairwise.values():
            for metric, comparison in method_result["metrics"].items():
                writer.writerow({
                    "reference_method": method_result["reference_method"],
                    "competitor_method": method_result["competitor_method"],
                    "metric": metric,
                    "advantage_mean": comparison["reference_advantage"]["mean"],
                    "advantage_std": comparison["reference_advantage"]["std"],
                    "wins": comparison["wins"],
                    "ties": comparison["ties"],
                    "losses": comparison["losses"],
                })
    return result
