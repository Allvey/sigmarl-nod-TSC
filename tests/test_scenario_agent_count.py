"""NOD identity state must follow the resolved scenario vehicle count."""

import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters


@pytest.mark.parametrize("scenario_type,n_agents", [("CPM_mixed", 4), ("on_ramp_1", 8)])
def test_preconfigured_agent_count_supports_neighbors_and_resets(scenario_type, n_agents):
    scenario = ScenarioRoadTraffic()
    scenario.parameters = Parameters(
        scenario_type=scenario_type, n_agents=n_agents, is_testing_mode=True,
        is_apply_mask=False,
    )
    env = VmasEnv(
        scenario=scenario, num_envs=1, continuous_actions=True,
        max_steps=16, device="cpu", n_agents=n_agents,
    )
    try:
        assert scenario.nod_agent_generation.shape == (1, n_agents)
        env.rollout(max_steps=2, break_when_any_done=False)
        before = scenario.nod_agent_generation.clone()
        scenario.reset_world_at(env_index=0, agent_index=torch.tensor(n_agents - 1))
        expected = before.clone()
        expected[0, -1] += 1
        assert torch.equal(scenario.nod_agent_generation, expected)
        for agent in scenario.world.agents:
            info = scenario.info(agent)
            assert torch.isfinite(info["deadlock_state"]).all()
        env.reset()
    finally:
        env.close()
