import csv
import json
from types import SimpleNamespace

import torch

from utilities.evaluation_base import Evaluation, _compute_exit_throughput


def _exit_info():
    shape = (2, 5, 2, 1)
    return {
        name: torch.zeros(shape, dtype=torch.bool if name != "nod_ego_generation" else torch.long)
        for name in (
            "nod_ego_generation", "testing_route_is_loop", "testing_exit_reached",
            "is_collision_with_agents", "testing_road_contact", "testing_entry_contact",
        )
    }


def test_throughput_counts_exits_once_per_generation_and_excludes_reset_contacts():
    info = _exit_info()
    info["testing_exit_reached"][0, :, 0] = True
    info["nod_ego_generation"][0, 3:, 0] = 1
    info["testing_exit_reached"][0, :3, 1] = True
    for step, key in enumerate((
        "is_collision_with_agents", "testing_road_contact", "testing_entry_contact"
    )):
        info[key][0, step, 1] = True
    info["testing_route_is_loop"][1] = True

    values = _compute_exit_throughput(info, dt=0.2)

    assert values[0] == 2  # Two successes in five frames of 0.2 seconds.
    assert values[1].isnan()
    assert info["testing_exit_reached"][0, :3, 1].all()


def test_throughput_reports_zero_for_open_routes_without_exits():
    torch.testing.assert_close(_compute_exit_throughput(_exit_info(), 0.1), torch.zeros(2))


def test_disabled_metrics_are_absent_from_collection_log_export_and_plots(tmp_path, monkeypatch):
    evaluator = Evaluation(
        model_paths=["/unused/"], legends=["Test"], render_titles=["Test"],
        scenario_type="intersection_2", measure_centerline_deviation=False,
        measure_throughput=True, measure_trajectory_jerk=True,
        where_to_save_eva_results=str(tmp_path),
        where_to_save_logging=str(tmp_path / "log.txt"),
    )
    evaluator.parameters = SimpleNamespace(
        num_vmas_envs=2, device="cpu", dt=0.2,
        scenario_type="intersection_2", is_save_eval_results=False,
    )
    evaluator._init_eva_matrices()
    info = _exit_info()
    info["testing_exit_reached"][0, -1, 0] = True
    info["testing_route_is_loop"][1] = True
    info["nod_world_vel"] = torch.ones(2, 5, 2, 2)
    # No centerline or TTC-specific fields: disabled metrics must not read them.
    before = {
        "pos": torch.zeros(2, 5, 2, 2), "vel": torch.ones(2, 5, 2, 2),
        "is_collision_with_agents": torch.zeros(2, 5, 2, 1),
        "is_collision_with_lanelets": torch.zeros(2, 5, 2, 1),
    }
    rollout = {("next", "agents", "info"): info, ("agents", "action"): torch.zeros(2, 5, 2, 2)}
    rollout.update({("agents", "info", name): value for name, value in before.items()})
    monkeypatch.setattr(evaluator, "_get_simulation_outputs", lambda: rollout)
    evaluator.model_idx = 0
    evaluator._evaluate_model_i()
    evaluator.compute_performance_metrics()
    evaluator.selected_checkpoint_prefixes = ["reward1.0"]
    evaluator._export_metrics()
    plots = []
    monkeypatch.setattr(evaluator, "_boxplot", lambda **kw: plots.append(kw["fig_name"]))
    monkeypatch.setattr(evaluator, "_plot_collision_breakdown", lambda *args, **kw: None)
    evaluator.plot()

    assert "vehicle_throughput" in plots
    assert "trajectory_jerk_rms" in plots
    assert "average_speed" in plots
    assert "smoothness" in plots
    assert "deviation_average" not in plots
    assert "low_ttc_exposure" not in plots
    log = (tmp_path / "log.txt").read_text()
    assert "Vehicle throughput [vehicles/s]" in log
    assert "Relative average speed [%]" in log
    assert "Smoothness [%]" in log
    assert "N/A" in evaluator._format_metric_log("Throughput", torch.tensor([[float("nan")]]))
    assert "centerline" not in log and "TTC" not in log and "Composite" not in log
    result = json.loads((tmp_path / "metrics_summary.json").read_text())
    metrics = result["models"][0]["metrics"]
    assert metrics["throughput_vehicles_per_second"]["mean"] == 1
    assert metrics["throughput_vehicles_per_second"]["num_valid_simulations"] == 1
    assert metrics["average_speed_percent"]["mean"] > 0
    assert metrics["action_change_percent"]["mean"] == 0
    assert "centerline_deviation_percent" not in metrics
    assert "low_ttc_exposure_percent" not in metrics
    with (tmp_path / "metrics_runs.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert float(rows[0]["throughput_vehicles_per_second"]) == 1
    assert float(rows[0]["average_speed_percent"]) > 0
    assert float(rows[0]["action_change_percent"]) == 0
    assert rows[1]["throughput_vehicles_per_second"] == ""


def test_paper_entrypoint_enables_only_requested_metrics(tmp_path, monkeypatch):
    from utilities import evaluation_tase26

    captured = []

    class FakeEvaluation:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def run_evaluation(self):
            pass

    monkeypatch.setattr(evaluation_tase26, "Evaluation", FakeEvaluation)
    monkeypatch.setattr("sys.argv", [
        "evaluation_tase26.py", "--model-paths", str(tmp_path), "--legends", "Test",
        "--scenarios", "intersection_2", "--output-root", str(tmp_path),
    ])
    evaluation_tase26.main()
    assert captured[0]["measure_centerline_deviation"] is False
    assert captured[0]["ttc_threshold_seconds"] is None
    assert captured[0]["measure_throughput"] is True
    assert captured[0]["measure_trajectory_jerk"] is True
