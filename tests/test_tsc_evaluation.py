"""External TSC routing, checkpoint integrity and matched protocol; no simulation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baselines.tsc import runtime, worker
from baselines.original_sigmarl.runtime import resolve_backend


def test_tsc_launcher_isolates_imports_and_runs_in_original_project(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    runtime.launch_worker(["--inspect", "--output", "/tmp/inspection.json"])
    command, options = calls[0]
    assert command[0][1] == str(runtime.PACKAGE / "worker.py")
    assert options["cwd"] == runtime.DEFAULT_TSC_PROJECT
    assert options["env"]["PYTHONPATH"] == str(runtime.DEFAULT_TSC_PROJECT)
    assert options["check"]
    assert resolve_backend("auto", {"use_topology_neighbor_selection": True}) == "tsc_external"


def test_actual_tsc_modules_import_from_sibling_without_environment(tmp_path):
    output = tmp_path / "inspection.json"
    runtime.launch_worker(["--inspect", "--output", str(output)])
    record = json.loads(output.read_text())
    for field, relative in (("trainer", "utilities/mappo_cavs.py"),
                            ("helper", "utilities/helper_training.py"),
                            ("scenario", "scenarios/road_traffic.py")):
        assert Path(record[field]).resolve() == runtime.DEFAULT_TSC_PROJECT / relative
    assert len(record["source_sha256"]["utilities/topology_module.py"]) == 64


def test_default_comparison_adds_seed5_without_replacing_fixed_three(tmp_path, monkeypatch):
    from utilities import evaluation_tase26 as entry
    sibling = tmp_path / "sigmarl-traffic"
    (sibling / "outputs/TSC/seed5").mkdir(parents=True)
    for name in entry.MODEL_PATHS:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    calls = []

    class FakeEvaluation:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run_evaluation(self):
            pass

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "Evaluation", FakeEvaluation)
    monkeypatch.setattr("sys.argv", ["evaluation_tase26.py", "--tsc-project", str(sibling)])
    entry.main()
    assert len(calls) == 6  # Six maps, no rollout is executed.
    assert [call["scenario_type"] for call in calls] == [
        "CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1",
        "roundabout_realistic", "bottleneck_1"]
    for call in calls:
        assert call["legends"] == ["NOD-DGPPO", "SigmaRL", "XP-MARL", "TSC"]
        assert [Path(p) for p in call["model_paths"][:3]] == [tmp_path / p for p in entry.MODEL_PATHS]
        assert Path(call["model_paths"][3]) == sibling / "outputs/TSC/seed5"
        assert call["tsc_project"] == str(sibling)
        assert call["simulation_steps"] == 1200 and call["num_simulations_per_model"] == 32
        assert call["evaluation_seed"] == 101 and call["deterministic_actions"]
        assert call["observation_noise"] is False
        assert not call["reuse_cached_rollouts"] and not call["save_rollout_cache"]
        assert call["measure_throughput"] and call["measure_trajectory_jerk"]
        assert "/paper_metrics_with_tsc/seed5/mixed/four_methods/" in call["where_to_save_eva_results"]
        assert "/six_scenarios/" in call["where_to_save_eva_results"]


def test_parent_uses_external_tsc_worker_and_unified_metric_pipeline(tmp_path, monkeypatch):
    from utilities import evaluation_base as base
    evaluator = base.Evaluation(
        model_paths=[str(tmp_path) + "/"], model_backends=["tsc_external"],
        legends=["TSC"], num_agents=8, scenario_type="intersection_2",
        simulation_steps=1200, evaluation_seed=101, deterministic_actions=True,
        observation_noise=False, is_render=False, num_simulations_per_model=32,
        tsc_project=str(runtime.DEFAULT_TSC_PROJECT),
    )
    evaluator.model_i = 0
    evaluator.selected_checkpoint_metadata = tmp_path / "reward7.02_data.json"
    evaluator.parameters = SimpleNamespace(num_vmas_envs=32, device="cpu")
    expected = torch.zeros(32, 1199)

    def fake_worker(args, project):
        assert "--method" not in args and "rollout" not in args
        assert args[args.index("--metadata") + 1].endswith("reward7.02_data.json")
        assert args[args.index("--steps") + 1] == "1200"
        assert args[args.index("--seed") + 1] == "101"
        assert "--deterministic" in args and project == runtime.DEFAULT_TSC_PROJECT
        torch.save(expected, args[args.index("--output") + 1])
        Path(args[args.index("--provenance-output") + 1]).write_text(json.dumps({"runtime_backend": "tsc_external"}))

    monkeypatch.setattr(base, "launch_tsc_worker", fake_worker)
    monkeypatch.setattr(base, "mappo_cavs", lambda **kw: pytest.fail("TSC used the NOD runtime"))
    monkeypatch.setattr(base, "launch_worker", lambda *a, **k: pytest.fail("TSC used official SigmaRL runtime"))
    torch.testing.assert_close(evaluator._get_simulation_outputs(), expected)
    assert evaluator.runtime_provenance[0]["runtime_backend"] == "tsc_external"


def test_worker_reapplies_seed_checks_auxiliaries_and_preserves_tsc_rollout(tmp_path, monkeypatch):
    metadata = tmp_path / "reward7.02_data.json"
    raw = dict(use_topology_neighbor_selection=True, is_using_opponent_modeling=True,
               device="cpu", dt=.05, is_add_noise=True, seed=5,
               n_nearing_agents_observed=1, n_topology_nearing_agents_observed=3)
    metadata.write_text(json.dumps({"parameters": raw}))
    for suffix in ("policy", "topology", "action_predictor"):
        torch.save({"weight": torch.ones(1)}, tmp_path / ("reward7.02_" + suffix + ".pth"))
    events = []
    policy = object()

    class FakeEnv:
        base_env = SimpleNamespace(scenario=SimpleNamespace())

        def set_seed(self, seed):
            events.append(("seed", seed))

        def rollout(self, **kwargs):
            from torchrl.envs.utils import exploration_type, ExplorationType
            assert exploration_type() == ExplorationType.MODE
            assert kwargs["policy"] is policy and kwargs["priority_module"] is None
            assert not kwargs["break_when_any_done"]
            assert kwargs["max_steps"] == 1199
            events.append(("rollout", 1199))
            return torch.zeros(32, 1199)

        def close(self):
            events.append(("close", None))

    def build(parameters):
        assert parameters.seed == 101 and not parameters.is_add_noise
        assert parameters.n_nearing_agents_observed == 1  # Preserve saved TSC architecture.
        assert parameters.n_topology_nearing_agents_observed == 3
        assert parameters.is_testing_mode and not parameters.is_continue_train
        parameters.model_name = "reward7.02"
        events.append(("construct", parameters.is_load_out_td))
        return FakeEnv(), policy, None, parameters

    trainer = SimpleNamespace(ScenarioRoadTraffic=object, mappo_cavs=build)
    helper = SimpleNamespace(Parameters=SimpleNamespace(from_dict=lambda raw: SimpleNamespace(**raw)))
    monkeypatch.setattr(worker, "load_runtime", lambda project: (trainer, helper, {"runtime_backend": "tsc_external"}))
    monkeypatch.setattr(worker, "verify_loaded_modules", lambda *a: events.append(("verify", None)))
    args = SimpleNamespace(project=str(runtime.DEFAULT_TSC_PROJECT), metadata=str(metadata),
                           scenario="intersection_2", num_agents=8, num_envs=32, steps=1200,
                           seed=101, observation_noise="off", deterministic=True,
                           video_output=None, render=False, measure_inference_time=False,
                           output=str(tmp_path / "out.pth"), provenance_output=str(tmp_path / "runtime.json"))
    worker.rollout(args)
    assert events == [("construct", False), ("verify", None), ("seed", 101), ("rollout", 1199), ("close", None)]
    record = json.loads(Path(args.provenance_output).read_text())
    assert record["policy_neighbors"] == 1 and record["topology_neighbors"] == 3
    assert record["checkpoint"] == "reward7.02" and record["recorded_steps"] == 1199


def test_tsc_validation_fails_if_auxiliary_loading_was_silently_skipped(tmp_path):
    prefix = str(tmp_path / "reward7.02")
    learner, predictor = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    for module, suffix in ((learner, "topology"), (predictor, "action_predictor")):
        torch.save(module.state_dict(), prefix + "_" + suffix + ".pth")
    scenario = SimpleNamespace(topology_learner=learner, topology_action_predictor=predictor)
    p = SimpleNamespace(is_topology_action_predictor_loaded=True)
    worker.verify_loaded_modules(scenario, p, prefix)
    with torch.no_grad():
        predictor.bias.add_(1)
    with pytest.raises(RuntimeError, match="different weights"):
        worker.verify_loaded_modules(scenario, p, prefix)
    p.is_topology_action_predictor_loaded = False
    with pytest.raises(RuntimeError, match="failed to load"):
        worker.verify_loaded_modules(scenario, p, prefix)


def test_tsc_metadata_requires_the_complete_network_bundle(tmp_path):
    metadata = str(tmp_path / "reward7.02_data.json")
    with pytest.raises(ValueError, match="topology-based"):
        worker.validate_checkpoint(metadata, {})
    raw = dict(use_topology_neighbor_selection=True, is_using_opponent_modeling=True)
    with pytest.raises(FileNotFoundError, match="policy"):
        worker.validate_checkpoint(metadata, raw)
