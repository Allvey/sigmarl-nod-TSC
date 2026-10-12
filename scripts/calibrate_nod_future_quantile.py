"""Calibrate a future-mitigation lower bound on disjoint environment seeds.

The reported coverage is empirical. Pair-time windows within one rollout are
correlated, so pooled calibration does not by itself prove conditional or
distribution-free collision-risk coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.evaluate_nod_future_mitigation import collect_one
from utilities.helper_training import Parameters


def lower_bound(raw: torch.Tensor, mean: torch.Tensor, correction: float) -> torch.Tensor:
    return (raw - correction).clamp(0.0, 1.0).minimum(mean)


def calibration_correction(raw: torch.Tensor, target: torch.Tensor, eta: float) -> float:
    """A conservative empirical residual quantile, without upward correction."""
    if raw.numel() == 0 or raw.shape != target.shape:
        raise ValueError("Calibration requires matching, nonempty raw predictions and labels")
    errors = (raw - target).sort().values
    rank = min(errors.numel(), math.ceil((errors.numel() + 1) * (1.0 - eta)))
    return max(0.0, float(errors[rank - 1]))


def relaxation_diagnostics(
    bound: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> dict:
    """Measure mistakes precisely where a future alpha map could relax."""
    if bound.shape != target.shape:
        raise ValueError("Bound and target shapes must match")
    selected = bound > threshold
    count = int(selected.sum())
    total = target.numel()
    if not count:
        return {
            "threshold": threshold,
            "count": 0,
            "fraction_of_valid_pairs": 0.0,
            "target_below_threshold_count": 0,
            "target_below_threshold_rate": None,
            "bound_violation_count": 0,
            "bound_violation_rate": None,
            "target_mean": None,
            "bound_mean": None,
            "bound_excess_mean_on_violations": None,
            "bound_excess_max_on_violations": None,
        }
    selected_target = target[selected]
    selected_bound = bound[selected]
    insufficient = selected_target <= threshold
    violation = selected_target < selected_bound
    excess = (selected_bound - selected_target)[violation]
    return {
        "threshold": threshold,
        "count": count,
        "fraction_of_valid_pairs": count / total,
        "target_below_threshold_count": int(insufficient.sum()),
        "target_below_threshold_rate": float(insufficient.float().mean()),
        "bound_violation_count": int(violation.sum()),
        "bound_violation_rate": float(violation.float().mean()),
        "target_mean": float(selected_target.mean()),
        "bound_mean": float(selected_bound.mean()),
        "bound_excess_mean_on_violations": float(excess.mean()) if excess.numel() else None,
        "bound_excess_max_on_violations": float(excess.max()) if excess.numel() else None,
    }


def summarize(
    result: dict, correction: float, profile_code: int | None = None,
    relax_threshold: float = 0.5,
) -> dict:
    raw, mean, target = result["_quantile_raw"], result["_prediction"], result["_target"]
    if profile_code is not None:
        selected = result["_neighbor_profile"] == profile_code
        raw, mean, target = raw[selected], mean[selected], target[selected]
    if target.numel() == 0:
        return {"seed": result["seed"], "valid_pairs": 0, "calibrated_coverage": None}
    bound = lower_bound(raw, mean, correction)
    metrics = {
        "seed": result["seed"],
        "valid_pairs": int(target.numel()),
        "raw_coverage": float((target >= raw).float().mean()),
        "calibrated_coverage": float((target >= bound).float().mean()),
        "calibrated_bound_mean": float(bound.mean()),
        "bound_zero_ratio": float((bound <= 0.0).float().mean()),
        "mean_minus_bound": float((mean - bound).mean()),
        "future_mae": float((mean - target).abs().mean()),
        "relaxation": relaxation_diagnostics(bound, target, relax_threshold),
    }
    # Outcome strata diagnose asymmetric misses; they are not vehicle-policy labels.
    for name, selected in {
        "low_target": target < 0.25,
        "middle_target": (target >= 0.25) & (target <= 0.75),
        "high_target": target > 0.75,
    }.items():
        metrics[name] = {
            "count": int(selected.sum()),
            "coverage": float((target[selected] >= bound[selected]).float().mean())
            if bool(selected.any()) else None,
            "bound_mean": float(bound[selected].mean()) if bool(selected.any()) else None,
        }
    return metrics


def combine_results(results: list[dict]) -> dict:
    if not results:
        raise ValueError("Cannot combine an empty result list")
    return {
        "seed": "pooled",
        **{
            key: torch.cat([item[key] for item in results])
            for key in ("_quantile_raw", "_prediction", "_target", "_neighbor_profile")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/current/config_dgppo_nod_future_quantile_candidate.json"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/dependencies/dgppo_nod_future_quantile_candidate/final_nod.pth",
    )
    parser.add_argument("--calibration-seeds", type=int, nargs="+", default=[404, 505, 606])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[707, 808, 909])
    parser.add_argument(
        "--rule-test-seeds", type=int, nargs="+", default=[1001, 1002],
        help="Separate OOD diagnostic seeds; each runs one yielding, moderate, and non-yielding neighbor",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--relax-threshold", type=float, default=0.5,
        help="Report diagnostics where the calibrated lower bound exceeds this value",
    )
    parser.add_argument(
        "--output",
        default="outputs/dependencies/dgppo_nod_future_quantile_candidate/calibration.json",
    )
    args = parser.parse_args()
    if (args.steps < 2 or not 0.5 <= args.relax_threshold < 1.0
            or len(set(args.calibration_seeds)) != len(args.calibration_seeds)
            or len(set(args.test_seeds)) != len(args.test_seeds)
            or len(set(args.rule_test_seeds)) != len(args.rule_test_seeds)
            or set(args.calibration_seeds) & (set(args.test_seeds) | set(args.rule_test_seeds))
            or set(args.test_seeds) & set(args.rule_test_seeds)
            or any(seed < 0 for seed in args.calibration_seeds + args.test_seeds + args.rule_test_seeds)):
        parser.error("Use nonnegative, unique, disjoint seeds and at least two steps")
    parameters = Parameters.from_json(args.config)
    if not parameters.nod_predict_future_quantile:
        parser.error("Configuration must enable nod_predict_future_quantile")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    eta = float(parameters.nod_quantile_level)
    if (checkpoint.get("version") != 5
            or checkpoint.get("label_mode") != "future_mitigation"
            or checkpoint.get("future_quantile_level") != eta):
        parser.error("Checkpoint does not match the configured future quantile")

    calibration = [
        collect_one(seed, config=args.config, checkpoint=args.checkpoint, steps=args.steps)
        for seed in args.calibration_seeds
    ]
    correction = calibration_correction(
        torch.cat([item["_quantile_raw"] for item in calibration]),
        torch.cat([item["_target"] for item in calibration]),
        eta,
    )
    # Test seeds are not used to fit the network or select the correction.
    test = [
        collect_one(seed, config=args.config, checkpoint=args.checkpoint, steps=args.steps)
        for seed in args.test_seeds
    ]
    rule_tests = []
    rule_results = []
    if args.rule_test_seeds:
        if parameters.n_agents < 2:
            parser.error("Rule-vehicle diagnostics require at least two agents")
        for seed in args.rule_test_seeds:
            for profile_code, profile in enumerate(
                ("yielding", "moderate", "non_yielding"), start=1
            ):
                result = collect_one(
                    seed, config=args.config, checkpoint=args.checkpoint,
                    steps=args.steps, rule_vehicles={1: profile},
                )
                rule_results.append((profile_code, profile, result))
                rule_tests.append({
                    "profile": profile,
                    "neighbor_agent_index": 1,
                    "all_pairs": summarize(result, correction, relax_threshold=args.relax_threshold),
                    "rule_neighbor_pairs": summarize(
                        result, correction, profile_code, args.relax_threshold
                    ),
                })
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "behavior_prefix": calibration[0]["behavior_prefix"],
        "steps": args.steps,
        "eta": eta,
        "relax_threshold": args.relax_threshold,
        "correction": correction,
        "bound_rule": "min(mean, clamp(raw_quantile - correction, 0, 1))",
        "calibration_seeds": args.calibration_seeds,
        "calibration": [summarize(item, correction, relax_threshold=args.relax_threshold)
                        for item in calibration],
        "test_seeds": args.test_seeds,
        "test": [summarize(item, correction, relax_threshold=args.relax_threshold)
                 for item in test],
        "test_overall": summarize(
            combine_results(test), correction, relax_threshold=args.relax_threshold
        ),
        "rule_test_seeds": args.rule_test_seeds,
        "rule_tests": rule_tests,
        "rule_profile_overall": [
            {
                "profile": profile,
                "rule_neighbor_pairs": summarize(
                    combine_results([result for code, name, result in rule_results
                                     if name == profile]),
                    correction, profile_code, args.relax_threshold,
                ),
            }
            for profile_code, profile in enumerate(
                ("yielding", "moderate", "non_yielding"), start=1
            )
        ] if rule_results else [],
        "interpretation": (
            "Empirical pair-time coverage on independent seeds; overlapping windows "
            "and behavior-policy homogeneity prevent a finite-sample conditional "
            "or collision-safety guarantee. Outcome strata are not rule-vehicle types; "
            "rule_tests additionally report each controlled rule neighbor type."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in summary["test"]:
        print(json.dumps(row, sort_keys=True))
    print("Pooled test relaxation:", json.dumps(summary["test_overall"]["relaxation"], sort_keys=True))
    for row in summary["rule_profile_overall"]:
        print("Rule neighbor relaxation:", row["profile"],
              json.dumps(row["rule_neighbor_pairs"].get("relaxation"), sort_keys=True))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
