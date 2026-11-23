import json

import pytest
import torch
from tensordict import TensorDict

from utilities.evaluation_mod import EvaluationMod, compute_rollout_metrics, replot_saved_results


def _rollout():
    shape = (2, 5, 2, 1)
    fields = {
        "testing_vehicle_collision_events": torch.zeros(shape, dtype=torch.long),
        "testing_road_collision_events": torch.zeros(shape, dtype=torch.long),
        "testing_exit_reached": torch.zeros(shape, dtype=torch.bool),
        "testing_route_is_loop": torch.zeros(shape, dtype=torch.bool),
        "testing_entry_contact": torch.zeros(shape, dtype=torch.bool),
        "testing_road_contact": torch.zeros(shape, dtype=torch.bool),
        "is_collision_with_agents": torch.zeros(shape, dtype=torch.bool),
        "is_collision_with_lanelets": torch.zeros(shape, dtype=torch.bool),
        "nod_ego_generation": torch.zeros(shape, dtype=torch.long),
        "nod_world_vel": torch.zeros(2, 5, 2, 2),
    }
    fields["nod_world_vel"][..., 0] = 1
    before = TensorDict(fields, batch_size=(2, 5, 2)).clone()
    after = before.clone()
    return TensorDict({
        "agents": TensorDict({"info": before}, batch_size=(2, 5, 2)),
        "next": TensorDict({
            "agents": TensorDict({"info": after}, batch_size=(2, 5, 2)),
        }, batch_size=(2, 5)),
    }, batch_size=(2, 5))


def test_collision_frequency_counts_events_and_all_vehicle_kilometers():
    rollout = _rollout()
    before = rollout["agents", "info"]
    after = rollout["next", "agents", "info"]
    before["testing_vehicle_collision_events"][0, 1:] = 1
    after["testing_vehicle_collision_events"][0] = 1
    before["testing_road_collision_events"][0, 3:] = 1
    after["testing_road_collision_events"][0, 2:] = 1

    metrics = compute_rollout_metrics(rollout, dt=0.2)

    assert metrics["vehicle_pair_contact_events"].tolist() == [1, 0]
    assert metrics["weighted_collision_events"].tolist() == [3, 0]
    torch.testing.assert_close(metrics["distance_km"], torch.tensor([0.002, 0.002], dtype=torch.float64))
    assert metrics["total_collision_events_per_km"].tolist() == [1500, 0]
    torch.testing.assert_close(
        metrics["total_collision_events_per_km"],
        metrics["agent_agent_collision_events_per_km"]
        + metrics["agent_boundary_collision_events_per_km"],
    )


def test_collision_rates_match_legacy_pretransition_time_fraction():
    rollout = _rollout()
    before = rollout["agents", "info"]
    after = rollout["next", "agents", "info"]
    # Multiple vehicles and persistent contacts count once per frame per type.
    before["is_collision_with_agents"][0, 1:4] = True
    before["is_collision_with_lanelets"][0, 3:] = True
    before["is_collision_with_agents"][1] = True
    before["is_collision_with_lanelets"][1] = True
    # The legacy evaluator uses the pre-transition flags, not these next flags.
    after["is_collision_with_agents"].zero_()
    after["nod_world_vel"].zero_()
    metrics = compute_rollout_metrics(rollout, dt=0.2)
    assert metrics["agent_agent_collision_rate_percent"].tolist() == [60, 100]
    assert metrics["agent_lanelet_collision_rate_percent"].tolist() == [40, 100]
    assert metrics["total_collision_rate_percent"].tolist() == [100, 200]
    # Cross-check against the exact reductions used by Evaluation.
    expected = sum(
        before[key].bool().squeeze(-1).any(dim=-1).sum(dim=-1).double() / 5 * 100
        for key in ["is_collision_with_agents", "is_collision_with_lanelets"]
    )
    torch.testing.assert_close(metrics["total_collision_rate_percent"], expected)


def test_exit_throughput_excludes_resets_and_preserves_loop_na():
    rollout = _rollout()
    after = rollout["next", "agents", "info"]
    # Persistent exit contact counts once; a new generation may arrive again.
    after["testing_exit_reached"][0, :, 0] = True
    after["nod_ego_generation"][0, 3:, 0] = 1
    # Colliding or leaving an entry cannot count as an exit success.
    after["testing_exit_reached"][0, :2, 1] = True
    after["testing_road_contact"][0, 0, 1] = True
    after["testing_entry_contact"][0, 1, 1] = True
    after["testing_route_is_loop"][1] = True

    metrics = compute_rollout_metrics(rollout, dt=0.2)

    assert metrics["exit_arrivals"].tolist() == [2, 0]
    assert metrics["throughput_vehicles_per_second"][0] == 2
    assert metrics["throughput_vehicles_per_second"][1].isnan()
    # Measurement leaves the original exit snapshots intact.
    assert after["testing_exit_reached"][0, :2, 1].all()


def test_zero_distance_and_reset_jerk_export_as_json_null(tmp_path):
    rollout = _rollout()
    after = rollout["next", "agents", "info"]
    after["nod_world_vel"].zero_()
    after["nod_ego_generation"][:] = torch.arange(5).view(1, 5, 1, 1)
    after["testing_route_is_loop"][:] = True
    samples = compute_rollout_metrics(rollout, dt=0.2)
    assert samples["total_collision_events_per_km"].isnan().all()
    assert samples["trajectory_jerk_rms_m_per_s3"].isnan().all()

    evaluator = EvaluationMod(
        model_paths=["/unused/"], legends=["Test"],
        num_simulations_per_model=2, where_to_save_eva_results=str(tmp_path),
        where_to_save_logging=str(tmp_path / "log.txt"),
    )
    evaluator.run_metrics = [samples]
    evaluator.model_protocols = [{"dt_seconds": 0.2, "observation_noise": False}]
    evaluator.selected_checkpoint_prefixes = ["reward1.0"]
    evaluator.compute_performance_metrics()
    evaluator.plot()
    evaluator._export_metrics()
    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    exported = summary["models"][0]["metrics"]
    assert exported["total_collision_rate_percent"]["pooled"] == 0
    assert exported["trajectory_jerk_rms_m_per_s3"]["pooled"] is None
    assert exported["throughput_vehicles_per_second"]["pooled"] is None
    assert "N/A" in (tmp_path / "log.txt").read_text()
    assert len(list(tmp_path.glob("*.pdf"))) == 3


def test_collision_rate_export_and_replot_roundtrip(tmp_path):
    rollout = _rollout()
    before = rollout["agents", "info"]
    after = rollout["next", "agents", "info"]
    before["testing_vehicle_collision_events"][0, 1:] = 1
    after["testing_vehicle_collision_events"][0] = 1
    before["is_collision_with_agents"][0, :2] = True
    after["nod_world_vel"][1, :, :, 0] = 2
    evaluator = EvaluationMod(
        model_paths=["/unused/"], legends=["Test"],
        num_simulations_per_model=2, where_to_save_eva_results=str(tmp_path),
        where_to_save_logging=str(tmp_path / "log.txt"),
    )
    evaluator.run_metrics = [compute_rollout_metrics(rollout, dt=0.2)]
    evaluator.model_protocols = [{"dt_seconds": 0.2, "observation_noise": False}]
    evaluator.selected_checkpoint_prefixes = ["reward1.0"]
    evaluator.compute_performance_metrics()
    evaluator.plot()
    evaluator._export_metrics()
    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    metrics = summary["models"][0]["metrics"]
    # Two collision frames / ten environment frames, independent of distance.
    collision = metrics["total_collision_rate_percent"]
    assert collision["pooled"] == 20
    assert collision["mean"] == 20
    assert metrics["throughput_vehicles_per_second"]["pooled"] == 0
    assert metrics["trajectory_jerk_rms_m_per_s3"]["pooled"] == 0
    assert metrics["agent_agent_collision_rate_percent"]["pooled"] == 20
    replot_saved_results(tmp_path)
    regenerated = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert regenerated["models"][0]["metrics"] == metrics

    # Old events/km CSVs cannot be converted without collision-frame counts.
    import csv
    path = tmp_path / "metrics_runs.csv"
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row.pop("agent_agent_collision_steps")
        row.pop("agent_lanelet_collision_steps")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="rerun evaluation"):
        replot_saved_results(tmp_path)
    assert json.loads((tmp_path / "metrics_summary.json").read_text()) == regenerated


def test_plot_matches_reference_distributions(monkeypatch, tmp_path):
    from matplotlib.collections import PathCollection
    import matplotlib.pyplot as plt

    evaluator = EvaluationMod(
        model_paths=["/unused/"], legends=["Test"],
        where_to_save_eva_results=str(tmp_path),
    )
    evaluator.run_metrics = [compute_rollout_metrics(_rollout(), dt=0.2)]
    figures = {}
    monkeypatch.setattr(evaluator, "_save_metric_figure", lambda fig, name: figures.update({name: fig}))
    evaluator.plot()
    collision = figures["collision_rate"]
    assert "Collision rate" in collision.axes[0].get_ylabel()
    assert tuple(collision.get_size_inches()) == (3.5, 2.0)
    ax = collision.axes[0]
    assert [label.get_text() for label in ax.get_legend().get_texts()] == [
        "Total", "Agent-agent", "Agent-boundary",
    ]
    for fig in figures.values():
        ax = fig.axes[0]
        assert [label.get_text() for label in ax.get_xticklabels()] == ["Test"]
        assert any(line.get_linestyle() == "--" for line in ax.lines)
        assert not any(isinstance(collection, PathCollection) for collection in ax.collections)
        plt.close(fig)


def test_exit_snapshot_survives_testing_respawn(monkeypatch):
    from torchrl.envs.libs.vmas import VmasEnv
    from scenarios.road_traffic import ScenarioRoadTraffic
    from utilities.helper_training import Parameters

    scenario = ScenarioRoadTraffic()
    scenario.parameters = Parameters(
        scenario_type="on_ramp_1", n_agents=8, is_testing_mode=True,
        is_using_deadlock_critic=False, is_apply_mask=False,
        is_use_mtv_distance=False, max_steps=16,
    )
    env = VmasEnv(scenario=scenario, num_envs=1, continuous_actions=True,
                  max_steps=16, device="cpu", n_agents=8)
    try:
        td = env.reset()
        exit_midpoint = scenario.ref_paths_agent_related.exit[:, 0].mean(dim=1)
        scenario.world.agents[0].set_pos(exit_midpoint, batch_index=None)
        old_generation = scenario.nod_agent_generation[0, 0].item()
        monkeypatch.setattr(scenario.world, "step", lambda: None)
        td = env.rand_action(td)
        td["agents", "action"].zero_()
        transition = env.step(td)
        info = transition["next", "agents", "info"]
        assert info["testing_exit_reached"][0, 0].item()
        assert not info["testing_route_is_loop"][0, 0].item()
        assert info["nod_ego_generation"][0, 0].item() == old_generation
        assert scenario.nod_agent_generation[0, 0].item() > old_generation
    finally:
        env.close()
