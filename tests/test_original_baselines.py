"""Original source identity, import isolation and passive measurement checks."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest
import torch

from baselines.original_sigmarl import runtime
from baselines.original_sigmarl.instrumentation import instrument_scenario
from baselines.original_sigmarl.worker import configure_training


def test_vendored_trainer_keeps_original_distribution_and_centralized_critic():
    provenance = runtime.verify_upstream()
    assert provenance["commit"] == "5fe715bdfba4ff3e33d901d69dfa220f1222c060"
    trainer = ast.parse((runtime.UPSTREAM / "utilities/mappo_cavs.py").read_text())
    assert any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and node.value.value is True
               and any(isinstance(target, ast.Name) and target.id == "mappo" for target in node.targets)
               for node in ast.walk(trainer))
    calls = {node.func.id for node in ast.walk(trainer)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "NormalParamExtractor" in calls
    assert "BoundedNormalParamExtractor" not in calls


@pytest.mark.parametrize("method", ["sigmarl", "xp_marl"])
def test_original_training_profiles_preserve_repository_config(method):
    config = json.loads((runtime.PROJECT_ROOT / "configs/baselines" / (method + "_original.json")).read_text())
    official = json.loads((runtime.UPSTREAM / "config.json").read_text())
    expected = dict(official, where_to_save=f"checkpoints/{method}_original/config_v1_2_0/")
    if method == "xp_marl":
        expected.update(is_using_prioritized_marl=True, prioritization_method="marl")
    assert config == expected
    configured, output = configure_training(config, method)
    assert configured["n_agents"] == 4
    assert configured["is_add_noise"] is False
    assert configured["num_epochs"] == 60 and configured["lr"] == .0002
    assert output == runtime.PROJECT_ROOT / "checkpoints" / (method + "_original") / "config_v1_2_0"
    assert configured["where_to_save"].endswith("/")
    assert not configured["is_load_model"]
    helper = ast.parse((runtime.UPSTREAM / "utilities/helper_training.py").read_text())
    cls = next(n for n in helper.body if isinstance(n, ast.ClassDef) and n.name == "Parameters")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert set(configured) <= {a.arg for a in init.args.args}


@pytest.mark.parametrize("raw,expected", [
    ({}, "sigmarl_1_2_0"),
    ({"is_using_prioritized_marl": True}, "xpmarl_1_2_0"),
    ({"is_using_nod_opinion": True}, "current"),
    ({"is_using_nod_opinion": False}, "current"),
])
def test_original_runtime_selection_does_not_mistake_modified_disabled_nod_for_original(raw, expected):
    assert runtime.resolve_backend("auto", raw) == expected


def test_original_method_validation_rejects_wrong_or_modified_weights():
    with pytest.raises(ValueError, match="priority setting"):
        runtime.validate_method({"is_using_prioritized_marl": False}, "xp_marl")
    with pytest.raises(ValueError, match="Modified NOD"):
        runtime.validate_method({"is_using_nod_opinion": False}, "sigmarl")
    with pytest.raises(ValueError, match="learned priorities"):
        runtime.validate_method({"is_using_prioritized_marl": True, "prioritization_method": "random"}, "xp_marl")


def test_launcher_isolates_module_search_path_without_starting_simulation(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kw: calls.append((args, kw)))
    runtime.launch_worker(["train", "--method", "sigmarl", "--config", "/test/config.json"])
    args, kwargs = calls[0]
    assert args[0][1] == str(runtime.PACKAGE / "worker.py")
    assert kwargs["cwd"] == runtime.UPSTREAM
    assert kwargs["env"]["PYTHONPATH"] == str(runtime.UPSTREAM)
    assert kwargs["check"] is True


def test_actual_worker_imports_upstream_in_fresh_interpreter_without_rollout(tmp_path):
    output = tmp_path / "imports.json"
    runtime.launch_worker(["inspect", "--output", str(output)])
    record = json.loads(output.read_text())
    for key, filename in (("trainer", "utilities/mappo_cavs.py"),
                          ("helper", "utilities/helper_training.py"),
                          ("scenario", "scenarios/road_traffic.py")):
        assert Path(record[key]).resolve() == runtime.UPSTREAM / filename


@pytest.mark.parametrize("arguments,expected_method", [
    (["--method", "sigmarl"], "sigmarl"),
    (["--method", "xp_marl"], "xp_marl"),
    (["--config", "configs/baselines/sigmarl_original.json"], "sigmarl"),
])
def test_main_training_routes_original_methods_without_entering_nod_trainer(monkeypatch, arguments, expected_method):
    calls = []
    monkeypatch.chdir(runtime.PROJECT_ROOT)
    monkeypatch.setattr("sys.argv", ["main_training.py"] + arguments)
    monkeypatch.setattr(runtime, "launch_worker", lambda args: calls.append(args))
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(runtime.PROJECT_ROOT / "main_training.py"), run_name="__main__")
    assert stopped.value.code == 0
    assert calls[0][calls[0].index("--method") + 1] == expected_method


def _measured_fixture():
    class OriginalScenario:
        def reset_world_at(self, env_index=None, agent_index=None):
            self.reset_calls += 1
            self.collisions.with_exit_segments.zero_()
            self.collisions.with_agents.zero_()
            return "original_reset"

        def done(self):
            self.done_calls += 1
            self.reset_world_at(env_index=0, agent_index=0)
            return torch.tensor([False])

        def info(self, agent):
            return {"original_field": torch.tensor([[7.]])}

        def reward(self, agent):
            return torch.tensor([3.])

    scenario = instrument_scenario(OriginalScenario)()
    agent = SimpleNamespace(state=SimpleNamespace(
        pos=torch.tensor([[1., 2.]]), vel=torch.tensor([[.5, 0.]]), rot=torch.tensor([[.2]])
    ))
    scenario.world = SimpleNamespace(batch_dim=1, agents=[agent], device="cpu")
    scenario.collisions = SimpleNamespace(
        with_exit_segments=torch.zeros(1, 1, dtype=torch.bool),
        with_entry_segments=torch.zeros(1, 1, dtype=torch.bool),
        with_lanelets=torch.zeros(1, 1, dtype=torch.bool),
        with_agents=torch.zeros(1, 1, 1, dtype=torch.bool),
    )
    scenario.ref_paths_agent_related = SimpleNamespace(is_loop=torch.zeros(1, 1, dtype=torch.bool))
    scenario.reset_calls = scenario.done_calls = 0
    assert scenario.reset_world_at() == "original_reset"
    return scenario, agent


def test_measurement_keeps_upstream_reset_and_captures_terminal_exit_before_respawn():
    scenario, agent = _measured_fixture()
    scenario.collisions.with_exit_segments.fill_(True)
    original_position = agent.state.pos.clone()
    result = scenario.done()
    info = scenario.info(agent)
    assert result.tolist() == [False]
    assert scenario.reset_calls == 2 and scenario.done_calls == 1
    assert not scenario.collisions.with_exit_segments.any()
    assert info["testing_exit_reached"].item() is True
    assert info["evaluation_generation"].item() == 1
    assert scenario._evaluation_generation.item() == 2
    torch.testing.assert_close(info["original_field"], torch.tensor([[7.]]))
    torch.testing.assert_close(agent.state.pos, original_position)
    info["evaluation_world_pos"].zero_()
    torch.testing.assert_close(agent.state.pos, original_position)


def test_snapshot_refreshes_before_info_on_every_vmas_reward_cycle():
    scenario, agent = _measured_fixture()
    scenario.collisions.with_exit_segments.fill_(True)
    torch.testing.assert_close(scenario.reward(agent), torch.tensor([3.]))
    assert scenario.info(agent)["testing_exit_reached"].item() is True
    scenario.done()
    # Next VMAS step: rewards precede info, which precedes done().
    agent.state.vel.fill_(.7)
    scenario.reward(agent)
    info = scenario.info(agent)
    assert info["testing_exit_reached"].item() is False
    assert info["evaluation_generation"].item() == 2
    torch.testing.assert_close(info["evaluation_world_vel"], agent.state.vel)


def test_throughput_uses_original_terminal_contact_snapshot_instead_of_cleared_flags():
    from utilities.evaluation_base import _compute_exit_throughput
    info = {
        "evaluation_generation": torch.ones(1, 3, 1, 1, dtype=torch.long),
        "testing_route_is_loop": torch.zeros(1, 3, 1, 1, dtype=torch.bool),
        "testing_exit_reached": torch.tensor([False, True, False]).reshape(1, 3, 1, 1),
        "testing_entry_contact": torch.zeros(1, 3, 1, 1, dtype=torch.bool),
        "testing_road_contact": torch.zeros(1, 3, 1, 1, dtype=torch.bool),
        "is_collision_with_agents": torch.zeros(1, 3, 1, 1, dtype=torch.bool),
        "evaluation_vehicle_contact": torch.ones(1, 3, 1, 1, dtype=torch.bool),
    }
    assert _compute_exit_throughput(info, .1).item() == 0
    info["evaluation_vehicle_contact"].zero_()
    assert _compute_exit_throughput(info, .1).item() == pytest.approx(1 / .3)


def test_formal_evaluator_dispatches_to_worker_and_exports_backend(tmp_path, monkeypatch):
    import utilities.evaluation_base as base

    evaluator = base.Evaluation(
        model_paths=[str(tmp_path) + "/"], model_backends=["xpmarl_1_2_0"],
        legends=["XP-MARL"], scenario_type="intersection_2", num_agents=8,
        simulation_steps=200, deterministic_actions=True,
    )
    evaluator.model_i = 0
    evaluator.selected_checkpoint_metadata = tmp_path / "reward5.71_data.json"
    evaluator.parameters = SimpleNamespace(num_vmas_envs=2, device="cpu")
    expected = torch.tensor([1., 2.])

    def fake_worker(args):
        assert args[args.index("--method") + 1] == "xp_marl"
        assert args[args.index("--num-agents") + 1] == "8"
        assert "--deterministic" in args
        torch.save(expected, args[args.index("--output") + 1])
        Path(args[args.index("--provenance-output") + 1]).write_text(json.dumps({
            "runtime_backend": "xpmarl_1_2_0", "commit": "official",
        }))

    monkeypatch.setattr(base, "launch_worker", fake_worker)
    monkeypatch.setattr(base, "mappo_cavs", lambda **kw: pytest.fail("Used modified NOD runtime"))
    torch.testing.assert_close(evaluator._get_simulation_outputs(), expected)
    assert evaluator.runtime_provenance[0]["runtime_backend"] == "xpmarl_1_2_0"


def test_worker_reapplies_matched_seed_after_construction_and_preserves_priority_rollout(tmp_path, monkeypatch):
    from baselines.original_sigmarl import worker
    raw = json.loads((runtime.PROJECT_ROOT / "configs/baselines/xp_marl_original.json").read_text())
    metadata = tmp_path / "reward5.71_data.json"
    metadata.write_text(json.dumps({"parameters": raw}))
    events = []
    policy, priority = object(), object()

    class FakeEnv:
        def set_seed(self, seed):
            events.append(("seed", seed))

        def rollout(self, **kwargs):
            events.append(("rollout", kwargs["max_steps"]))
            assert kwargs["policy"] is policy
            assert kwargs["priority_module"] is priority
            return torch.zeros(2, 4)

        def close(self):
            events.append(("close", None))

    def build(parameters):
        events.append(("construct", parameters.is_add_noise))
        return FakeEnv(), policy, priority, parameters

    trainer = SimpleNamespace(ScenarioRoadTraffic=object, mappo_cavs=build, __file__="original_trainer.py")
    helper = SimpleNamespace(Parameters=SimpleNamespace(from_dict=lambda data: SimpleNamespace(**data)),
                             __file__="original_helper.py")
    monkeypatch.setattr(worker, "load_runtime", lambda: (trainer, helper, {"commit": "official"}))
    args = SimpleNamespace(metadata=str(metadata), method="xp_marl", scenario="intersection_2",
                           num_agents=8, num_envs=2, steps=5, video_output=None,
                           observation_noise="off", seed=202, deterministic=True,
                           measure_inference_time=False, render=False,
                           output=str(tmp_path / "out.pth"),
                           provenance_output=str(tmp_path / "runtime.json"))
    worker.rollout(args)
    assert events == [("construct", False), ("seed", 202), ("rollout", 4), ("close", None)]
    record = json.loads(Path(args.provenance_output).read_text())
    assert record["runtime_backend"] == "xpmarl_1_2_0"
    assert record["seed"] == 202 and record["recorded_steps"] == 4


@pytest.mark.parametrize("checkpoint_source", [None, "released", "retrained"])
def test_formal_entrypoint_selects_weights_and_separates_result_directories(tmp_path, monkeypatch, checkpoint_source):
    from utilities import evaluation_tase26 as entry
    profiles = {}
    for method in ("sigmarl", "xp_marl"):
        profile = runtime.PROJECT_ROOT / "configs/baselines" / (method + "_original.json")
        profiles[method] = json.loads(profile.read_text())
        target = tmp_path / "configs/baselines" / profile.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profiles[method]))
    for name in entry.MODEL_PATHS + entry.RELEASED_MODEL_PATHS + [p["where_to_save"] for p in profiles.values()]:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    calls = []

    class FakeEvaluation:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run_evaluation(self):
            pass

    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(entry, "Evaluation", FakeEvaluation)
    arguments = ["evaluation_tase26.py", "--scenarios", "intersection_2"]
    if checkpoint_source is not None:
        arguments += ["--baseline-checkpoints", checkpoint_source]
    monkeypatch.setattr("sys.argv", arguments)
    entry.main()
    source = checkpoint_source or "retrained"
    assert calls[0]["model_backends"] == ["auto"] * 3
    assert calls[0]["legends"] == ["NOD-DGPPO (PPO beta)", "SigmaRL", "XP-MARL"]
    assert calls[0]["x_ticks"] == calls[0]["legends"]
    assert len(calls[0]["model_paths"]) == 3
    expected = profiles["sigmarl"]["where_to_save"] if source == "retrained" else "checkpoints/itsc24/M0 (our)"
    assert Path(calls[0]["model_paths"][1]) == tmp_path / expected
    if source == "retrained":
        assert Path(calls[0]["model_paths"][2]) == tmp_path / profiles["xp_marl"]["where_to_save"]
        assert "/retrained/config_v1_2_0/" in calls[0]["where_to_save_eva_results"]
    else:
        assert Path(calls[0]["model_paths"][2]) == tmp_path / "checkpoints/icra25/M1 (XP-MARL)"
    assert "/" + source + "/" in calls[0]["where_to_save_eva_results"]
    assert "/three_methods/" in calls[0]["where_to_save_eva_results"]
