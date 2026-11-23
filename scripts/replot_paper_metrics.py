"""Redraw saved evaluation distributions without running simulation or changing metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import torch
from utilities.evaluation_base import Evaluation


def replot(output_dir):
    summary = json.loads((output_dir / "metrics_summary.json").read_text())
    with (output_dir / "metrics_runs.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    models = sorted(summary["models"], key=lambda model: model["model_index"])
    labels = [model["label"] for model in models]
    groups = [sorted(
        (row for row in rows if int(row["model_index"]) == model["model_index"]),
        key=lambda row: int(row["simulation_index"]),
    ) for model in models]
    if not all(groups):
        raise ValueError(f"Missing model samples in {output_dir}")
    for model, group in zip(models, groups):
        if any(row["label"] != model["label"] or row["checkpoint"] != model["checkpoint"] for row in group):
            raise ValueError(f"Inconsistent model identity in {output_dir}")
    scenario = summary["protocol"]["scenario"]
    evaluator = Evaluation(
        model_paths=[model["model_path"].rstrip("/") + "/" for model in models],
        legends=labels, x_ticks=labels, idx_our=0,
        x_tick_label_rotation=15, where_to_save_eva_results=str(output_dir),
    )
    evaluator.parameters = SimpleNamespace(scenario_type=scenario, is_save_eval_results=True)
    # Replace existing metric figures at their original paths; preserve CSV/JSON,
    # logs and the training-reward plots.
    def save(fig, name):
        for ext, dpi in (("pdf", 300), ("png", 600)):
            matches = list(output_dir.glob(f"*_{scenario}_{name}.{ext}"))
            matches += list(output_dir.glob(f"*_{scenario}_{name} copy.{ext}"))
            targets = matches or [output_dir / f"{name}.{ext}"]
            for path in targets:
                fig.savefig(path, format=ext, bbox_inches="tight", pad_inches=0.01, dpi=dpi)
    evaluator._save_metric_figure = save

    def tensor(key):
        return torch.tensor([
            [float(row[key]) if row[key] else math.nan for row in group]
            for group in groups
        ], dtype=torch.float64)

    collision_keys = (
        "total_collision_rate_percent", "agent_agent_collision_rate_percent",
        "agent_lanelet_collision_rate_percent",
    )
    if all(key in rows[0] for key in collision_keys):
        evaluator._plot_collision_breakdown(
            *(tensor(key) for key in collision_keys),
            y_label=r"Collision rate [$\%$] $\downarrow$", fig_name="collision_rate",
            y_limits=[0, None],
        )
    mod = summary.get("metric_source") == "utilities.evaluation_mod"
    specs = [
        ("average_speed_percent", "average_speed", r"Rel. avg. spd. [$\%$] $\uparrow$", r"$\%$", [0, 100]),
        ("action_change_percent", "smoothness", r"Action change [$\%$] $\downarrow$", r"$\%$", [0, None]),
        ("throughput_vehicles_per_second", "throughput_vehicles_per_second" if mod else "vehicle_throughput",
         r"Exit throughput [vehicles/s] $\uparrow$", " vehicles/s", [0, None]),
        ("trajectory_jerk_rms_m_per_s3", "trajectory_jerk_rms_m_per_s3" if mod else "trajectory_jerk_rms",
         r"Trajectory jerk RMS [m/s$^3$] $\downarrow$", r" m/s$^3$", [0, None]),
        ("centerline_deviation_percent", "deviation_average", r"Relative centerline deviation [$\%$]", r"$\%$", [0, 100]),
    ]
    for key, name, label, unit, limits in specs:
        if key in rows[0]:
            evaluator._boxplot(
                figsize=(3.5, 2), data=tensor(key), y_label=label,
                fig_name=name, y_limits=limits, value_unit=unit,
            )
    print(f"[replotted] {output_dir} | " + ", ".join(
        f"{model['label']}={model['checkpoint']}" for model in models
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=[
        "outputs/benchmarks/paper_metrics", "outputs/benchmarks/paper_metrics_mod_collision_rate",
    ])
    args = parser.parse_args()
    for root in args.roots:
        for path in sorted((ROOT / root).glob("*/steps_*/metrics_summary.json")):
            replot(path.parent)


if __name__ == "__main__":
    main()
