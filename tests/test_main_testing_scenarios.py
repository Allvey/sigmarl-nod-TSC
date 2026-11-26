"""Visualization configuration and sequential dispatch without running a simulator."""
import json
from pathlib import Path
import runpy

import pytest
import torch

from utilities.constants import SCENARIOS


ROOT = Path(__file__).resolve().parents[1]
NEW_SCENARIOS = ["intersection_3", "intersection_4", "on_ramp_2_multilane", "roundabout_2", "roundabout_realistic", "bottleneck_1"]
MODEL = "checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/"


@pytest.mark.parametrize("method,model", [
    ("nod", MODEL), ("sigmarl", "checkpoints/sigmarl_original/config_v1_2_0/"),
])
def test_batch_dispatch_keeps_fixed_model_and_all_actor_default(monkeypatch, method, model):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr("sys.argv", ["main_testing.py", "--scenarios"] + NEW_SCENARIOS
                        + ["--method", method, "--no-realtime", "--no-speed-print"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "main_testing.py"), run_name="__main__")
    assert stopped.value.code == 0
    assert len(calls) == len(NEW_SCENARIOS)
    for scenario, (command, options) in zip(NEW_SCENARIOS, calls):
        assert command[command.index("--scenario") + 1] == scenario
        assert command[command.index("--model-path") + 1] == model
        assert command[command.index("--method") + 1] == method
        assert command[command.index("--rule-fraction") + 1] == "0.0"
        assert command[command.index("--env-seed") + 1] == "101"
        assert "--no-realtime" in command and "--no-speed-print" in command
        assert "--no-video" not in command and "--stochastic-actions" not in command
        assert options["check"] is True


@pytest.mark.parametrize("scenario", NEW_SCENARIOS)
@pytest.mark.parametrize("method", ["nod", "sigmarl"])
def test_single_scenario_selects_matching_metadata_and_separate_video_path(tmp_path, monkeypatch, scenario, method):
    import utilities.mappo_cavs as training
    import utilities.testing_rule_policy as rules
    from torchrl.envs.utils import exploration_type, ExplorationType

    raw = json.loads((ROOT / "configs/current/error_scale_repeat/02_control.json").read_text())
    if method == "sigmarl":
        raw = {"is_using_prioritized_marl": False, "is_add_noise": True}
    # Unrelated and lower-score JSON must not determine the loaded parameters.
    (tmp_path / "unrelated.json").write_text("{}")
    for score in ("6.00", "7.38"):
        saved = dict(raw, model_name="reward" + score, seed=17 if score == "6.00" else 19)
        (tmp_path / ("reward" + score + "_data.json")).write_text(json.dumps({"parameters": saved}))
        (tmp_path / ("reward" + score + "_policy.pth")).write_bytes(b"fixture")
    events = []
    policy = (torch.nn.Sequential(training.BoundedNormalParamExtractor())
              if method == "sigmarl" else object())

    class FakeEnv:
        def set_seed(self, seed):
            events.append(("seed", seed))

        def rollout(self, **kwargs):
            assert exploration_type() == ExplorationType.MODE and not torch.is_grad_enabled()
            assert kwargs["policy"] is policy
            assert kwargs["max_steps"] == 1199 and kwargs["is_save_simulation_video"]
            assert not kwargs["break_when_any_done"]
            events.append(("rollout", None))
            return {}, ["frame"]

        def close(self):
            events.append(("close", None))

    def build(parameters):
        assert parameters.model_name == "reward7.38"
        assert parameters.seed == 101
        assert parameters.scenario_type == scenario
        assert parameters.n_agents == SCENARIOS[scenario]["n_agents"]
        assert parameters.num_vmas_envs == 1 and not parameters.is_add_noise
        assert parameters.is_testing_mode and not parameters.is_continue_train
        assert parameters.is_visualize_nod_alpha == (method == "nod")
        assert parameters.is_visualize_agent_id
        if method == "sigmarl":
            assert not parameters.is_using_nod_opinion and not parameters.is_using_nod_actor
            assert not parameters.is_using_safety_critic and not parameters.is_using_safety_constraint
            assert not parameters.is_using_safety_value_shadow and not parameters.is_using_deadlock_critic
        assert parameters.where_to_save == str(tmp_path) + "/"
        assert str(Path(parameters.agent_speed_log_path).parent) == str(
            tmp_path / "visualization" / scenario / "envseed101_reward7.38")
        events.append(("construct", None))
        return FakeEnv(), policy, None, parameters

    monkeypatch.setattr(training, "mappo_cavs", build)
    monkeypatch.setattr(rules, "TestingRulePolicy", lambda *a, **k: pytest.fail("All-Actor run wrapped rule control"))
    monkeypatch.setattr("vmas.simulator.utils.save_video",
                        lambda filename, frames, **kw: events.append(("video", filename)))
    monkeypatch.setattr("sys.argv", ["main_testing.py", "--model-path", str(tmp_path),
                                    "--method", method, "--scenario", scenario,
                                    "--no-realtime", "--no-speed-print"])
    runpy.run_path(str(ROOT / "main_testing.py"), run_name="__main__")
    output = tmp_path / "visualization" / scenario / "envseed101_reward7.38"
    setup = json.loads((output / "test_setup.json").read_text())
    assert setup["method"] == method
    if method == "sigmarl":
        from tensordict.nn.distributions import NormalParamExtractor
        assert type(policy[0]) is NormalParamExtractor
    assert setup["checkpoint"] == "reward7.38" and setup["rule_vehicles"] == {}
    assert setup["rule_fraction"] == 0 and setup["deterministic_actions"]
    assert setup["environment_seed"] == 101 and setup["n_agents"] == SCENARIOS[scenario]["n_agents"]
    assert (output / "agent_speeds.csv").is_file()
    assert events == [("construct", None), ("seed", 101), ("rollout", None),
                      ("video", str(output / "video")), ("close", None)]


def test_original_actor_restores_upstream_scale_without_changing_weights():
    from tensordict.nn.distributions import NormalParamExtractor
    from utilities.mappo_cavs import BoundedNormalParamExtractor
    from utilities.testing_baseline import restore_original_actor_distribution

    policy = torch.nn.Sequential(torch.nn.Linear(4, 4), BoundedNormalParamExtractor())
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    logits = torch.tensor([[0., 0., 5., 5.]])
    assert (policy[1](logits)[1] <= 1.).all()
    restore_original_actor_distribution(policy)
    assert type(policy[1]) is NormalParamExtractor
    assert (policy[1](logits)[1] > 1.).all()
    assert all(torch.equal(before[name], value) for name, value in policy.state_dict().items())
    with pytest.raises(ValueError, match="Expected one"):
        restore_original_actor_distribution(policy)


def test_sigmarl_visualization_rejects_modified_or_prioritized_metadata():
    from utilities.helper_training import Parameters
    from utilities.testing_baseline import configure_sigmarl_visualization

    for raw in ({"is_using_nod_opinion": False}, {"is_using_prioritized_marl": True}):
        with pytest.raises(ValueError):
            configure_sigmarl_visualization(Parameters(), raw)


@pytest.mark.parametrize("arguments", [
    ["--rule-fraction", "-0.1"], ["--rule-fraction", "1.1"],
    ["--scenarios", "intersection_4", "intersection_4"],
    ["--scenario", "intersection_4", "--scenarios", "roundabout_2"],
])
def test_invalid_visualization_requests_stop_before_simulation(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["main_testing.py"] + arguments)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "main_testing.py"), run_name="__main__")
    assert stopped.value.code == 2
