"""Physical transition labels must survive next-decision respawn refresh."""

import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_scenario import CircularBuffer
from utilities.helper_training import Parameters, TransformedEnvCustom


def test_refresh_replaces_only_latest_selected_environment():
    buffer = CircularBuffer(torch.zeros(3, 2, 4))
    for value in (1., 2., 3., 4.):
        buffer.add(torch.full((2, 4), value))
    previous = buffer.buffer.clone()
    pointer, size = buffer.pointer, buffer.valid_size
    buffer.replace_latest(torch.full((2, 4), 9.), torch.tensor([True, False]))
    assert (buffer.pointer, buffer.valid_size) == (pointer, size)
    torch.testing.assert_close(buffer.get_latest()[0], torch.full((4,), 9.))
    torch.testing.assert_close(buffer.buffer[:, 1], previous[:, 1])
    torch.testing.assert_close(buffer.get_latest(2), torch.full((2, 4), 3.))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rollout_mode", ["step", "stop_early", "nonstop"])
def test_forced_collision_preserves_labels_and_refreshes_neighbors(monkeypatch, enabled, rollout_mode):
    scenario = ScenarioRoadTraffic()
    scenario.parameters = Parameters(
        scenario_type="CPM_mixed", n_agents=4, max_steps=8,
        is_testing_mode=True, is_apply_mask=False, is_add_noise=False,
        is_using_nod_actor=False, is_using_nod_opinion=False,
        is_using_deadlock_critic=False, is_using_safety_value_shadow=True,
        refresh_respawn_observations=enabled,
    )
    env = TransformedEnvCustom(VmasEnv(
        scenario=scenario, num_envs=2, continuous_actions=True,
        max_steps=8, device="cpu", n_agents=4,
    ))
    try:
        env.set_seed(23)
        initial = env.reset()
        old_reward = scenario.reward
        old_done = scenario.done
        snapshots = []
        decisions = []
        first_rewards = []

        def reward(agent):
            result = old_reward(agent)
            if not snapshots:
                first_rewards.append(result.clone())
            if not snapshots and agent is scenario.world.agents[-1]:
                scenario.collisions.with_lanelets[0, 0] = True
            return result

        def done():
            result = old_done()
            if not snapshots:
                snapshots.append({
                    "pos": torch.stack([a.state.pos.clone() for a in scenario.world.agents], dim=1),
                    "generation": scenario.nod_agent_generation.clone(),
                    "buffers": {
                        name: (value.pointer, value.valid_size, value.buffer.clone())
                        for name, value in vars(scenario.observations).items()
                        if isinstance(value, CircularBuffer)
                    },
                })
            return result

        def policy(td):
            decisions.append(td.clone())
            return td.set(("agents", "action"), torch.zeros(2, 4, 2))

        monkeypatch.setattr(scenario, "reward", reward)
        monkeypatch.setattr(scenario, "done", done)
        if rollout_mode == "step":
            transition, decision = env.step_and_maybe_reset(policy(initial))
        else:
            rollout = env.rollout(
                max_steps=2, policy=policy, tensordict=initial, auto_reset=False,
                break_when_any_done=rollout_mode == "stop_early",
            )
            transition, decision = rollout[:, 0], decisions[1]

        physical = transition["next", "agents", "info"]
        torch.testing.assert_close(
            transition["next", "agents", "reward"],
            torch.stack(first_rewards, dim=1).unsqueeze(-1),
        )
        assert physical["is_collision_with_lanelets"][0].bool().all()
        assert physical["safety_margins"][0, 0, -1] > 0
        # The reset must change the car identity but never the physical label.
        assert physical["nod_ego_generation"][0, 0, 0] < snapshots[0]["generation"][0, 0]
        if enabled:
            torch.testing.assert_close(decision["agents", "info", "nod_world_pos"], snapshots[0]["pos"])
            assert not decision["agents", "info", "is_collision_with_lanelets"][0].bool().any()
            assert not torch.equal(decision["agents", "observation"][0, 0], transition["next", "agents", "observation"][0, 0])
            # Every car's neighbor geometry in this environment is rebuilt.
            if rollout_mode == "step":
                fresh = torch.stack([
                    scenario.observation(a, refresh_mask=torch.tensor([True, False]))
                    for a in scenario.world.agents
                ], dim=1)
                torch.testing.assert_close(decision["agents", "observation"][0], fresh[0])
        else:
            torch.testing.assert_close(decision["agents", "observation"], transition["next", "agents", "observation"])
        torch.testing.assert_close(decision["agents", "observation"][1], transition["next", "agents", "observation"][1])
        if rollout_mode == "step":
            for name, (pointer, size, previous) in snapshots[0]["buffers"].items():
                buffer = getattr(scenario.observations, name)
                assert (buffer.pointer, buffer.valid_size) == (pointer, size)
                torch.testing.assert_close(buffer.buffer[:, 1], previous[:, 1])
            # Refresh is idempotent, and whole-environment resets clear pending work.
            assert env._refresh_respawn_decision(decision) is decision
            env.reset()
            assert not scenario.respawn_observation_pending.any()
    finally:
        env.close()
