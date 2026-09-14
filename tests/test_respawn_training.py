"""Reward anchoring and task GAE across vehicle lifetimes; no policy training."""
import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.objectives.value.functional import generalized_advantage_estimate

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters, TransformedEnvCustom, prepare_task_episode_boundaries


@pytest.mark.parametrize("reset_step", [1, 2])
def test_gae_does_not_bootstrap_or_propagate_across_respawn(reset_step):
    shape = (1, 3, 2, 1)
    reward = torch.ones(shape)
    respawn = torch.zeros(shape, dtype=torch.bool)
    respawn[0, reset_step, 0] = True
    td = TensorDict({"next": {
        "done": torch.zeros(1, 3, 1, dtype=torch.bool),
        "terminated": torch.zeros(1, 3, 1, dtype=torch.bool),
        "agents": {"reward": reward, "info": {"task_respawn": respawn}},
    }}, batch_size=[1, 3])
    prepare_task_episode_boundaries(td, fix_respawn_training=True)
    done, terminal = (td["next", "agents", key] for key in ("done", "terminated"))
    assert not td["next", "done"].any()
    assert not done[..., 1, :].any()  # Neighbor cars are not made terminal.
    next_value = torch.zeros(shape)

    def estimate():
        return generalized_advantage_estimate(
            .9, 1., torch.zeros(shape), next_value, reward, done,
            terminated=terminal, time_dim=1,
        )[0]

    expected = estimate()
    next_value[0, reset_step, 0] = 10000.
    reward[0, reset_step + 1:, 0] = 10000.
    result = estimate()
    torch.testing.assert_close(result[:, :reset_step + 1, 0], expected[:, :reset_step + 1, 0])
    assert result[0, reset_step, 0, 0] == 1.


def test_timeout_still_bootstraps_and_legacy_needs_no_respawn_field():
    td = TensorDict({"next": {
        "done": torch.ones(1, 1, 1, dtype=torch.bool),
        "terminated": torch.zeros(1, 1, 1, dtype=torch.bool),
        "agents": {"reward": torch.ones(1, 1, 2, 1)},
    }}, batch_size=[1, 1])
    prepare_task_episode_boundaries(td)
    reward = td["next", "agents", "reward"]
    result, _ = generalized_advantage_estimate(
        .9, 1., torch.zeros_like(reward), torch.full_like(reward, 5.), reward,
        td["next", "agents", "done"], terminated=td["next", "agents", "terminated"],
        time_dim=1,
    )
    torch.testing.assert_close(result, torch.full_like(result, 5.5))


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("event", ["exit", "collision"])
def test_reset_reanchors_reward_and_preserves_physical_transition(monkeypatch, fixed, event):
    scenario = ScenarioRoadTraffic()
    scenario.parameters = Parameters(
        scenario_type="CPM_mixed", n_agents=4, max_steps=8,
        is_testing_mode=event == "collision", is_apply_mask=False, is_add_noise=False,
        is_using_nod_actor=False, is_using_nod_opinion=False,
        is_using_deadlock_critic=False, is_using_safety_value_shadow=True,
        refresh_respawn_observations=True, fix_respawn_training=fixed,
    )
    env = TransformedEnvCustom(VmasEnv(
        scenario=scenario, num_envs=2, continuous_actions=True,
        max_steps=8, device="cpu", n_agents=4,
    ))
    try:
        env.set_seed(23)
        initial = env.reset()
        original_reward, original_done = scenario.reward, scenario.done
        snapshot, reward_records, displacement = {}, [], []

        def reward(agent):
            # These are the same two positions used by the progress reward.
            if snapshot and agent is scenario.world.agents[0]:
                displacement.append(agent.state.pos[0] - scenario.state_buffer.get_latest()[0, 0, :2])
            result = original_reward(agent)
            if not snapshot:
                reward_records.append(result.clone())
                if agent is scenario.world.agents[-1]:
                    if event == "exit":
                        scenario.collisions.with_exit_segments[0, :2] = True
                    else:
                        scenario.collisions.with_lanelets[0, :2] = True
            return result

        def done():
            if not snapshot:
                snapshot.update(buffer=scenario.state_buffer.buffer.clone(),
                                pointer=scenario.state_buffer.pointer,
                                size=scenario.state_buffer.valid_size)
            return original_done()

        monkeypatch.setattr(scenario, "reward", reward)
        monkeypatch.setattr(scenario, "done", done)
        initial.set(("agents", "action"), torch.zeros(2, 4, 2))
        transition, decision = env.step_and_maybe_reset(initial)
        torch.testing.assert_close(transition["next", "agents", "reward"],
                                   torch.stack(reward_records, 1).unsqueeze(-1))
        buffer = scenario.state_buffer
        assert (buffer.pointer, buffer.valid_size) == (snapshot["pointer"], snapshot["size"])
        previous = snapshot["buffer"]
        slot = (buffer.pointer - 1) % buffer.buffer_size
        if fixed:
            expected = torch.zeros(2, 4, 1, dtype=torch.bool)
            expected[0, :2] = True
            assert torch.equal(transition["next", "agents", "info", "task_respawn"].bool(), expected)
            assert not decision["agents", "info", "task_respawn"].any()
            for agent_i in (0, 1):
                torch.testing.assert_close(buffer.get_latest()[0, agent_i, :2],
                                           scenario.world.agents[agent_i].state.pos[0])
            # Only the latest state of the two reset cars may change.
            unchanged = torch.ones_like(previous, dtype=torch.bool)
            unchanged[slot, 0, :2] = False
            torch.testing.assert_close(buffer.buffer[unchanged], previous[unchanged])
        else:
            torch.testing.assert_close(buffer.buffer, previous)
        if event == "collision":
            assert (transition["next", "agents", "info", "safety_margins"][0, :2, -1] > 0).all()
        if fixed:
            training_td = transition.clone()
            prepare_task_episode_boundaries(training_td, fix_respawn_training=True)
            assert torch.equal(training_td["next", "agents", "terminated"], expected)
            assert not training_td["next", "done"].any()
        respawn_position = scenario.world.agents[0].state.pos[0].clone()
        decision.set(("agents", "action"), torch.zeros(2, 4, 2))
        second, _ = env.step_and_maybe_reset(decision)
        physical_movement = second["next", "agents", "info", "nod_world_pos"][0, 0] - respawn_position
        if fixed:
            # Reset velocities can be nonzero: count only the real physical step.
            torch.testing.assert_close(displacement[0], physical_movement)
            assert displacement[0].norm() <= scenario.max_speed * scenario.world.dt
        else:
            assert (displacement[0] - physical_movement).norm() > .01
    finally:
        env.close()
