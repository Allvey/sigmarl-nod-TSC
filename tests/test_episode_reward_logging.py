"""Completed environment returns stay independent of task-only respawns."""
import json

import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import completed_environment_reward, prepare_task_episode_boundaries


def rollout():
    return TensorDict({"next": {
        "done": torch.tensor([[[False], [True], [False]], [[True], [False], [False]]]),
        "terminated": torch.zeros(2, 3, 1, dtype=torch.bool),
        "agents": {
            "reward": torch.ones(2, 3, 2, 1),
            "episode_reward": torch.tensor([
                [[1000.], [2000.]], [[10.], [20.]], [[3000.], [4000.]],
                [[30.], [40.]], [[5000.], [6000.]], [[7000.], [8000.]],
            ]).reshape(2, 3, 2, 1),
            "info": {"task_respawn": torch.ones(2, 3, 2, 1, dtype=torch.bool)},
        },
    }}, batch_size=[2, 3])


def test_only_completed_environment_returns_affect_mean_and_ranking():
    td = rollout()
    before = td.clone()
    prepare_task_episode_boundaries(td, fix_respawn_training=True)
    task_done = td["next", "agents", "done"].clone()
    mean, count = completed_environment_reward(td)
    assert (mean, count) == (25., 2)
    assert completed_environment_reward(before) == (mean, count)
    assert torch.equal(td["next", "agents", "done"], task_done)
    assert torch.equal(td["next", "agents", "episode_reward"], before["next", "agents", "episode_reward"])
    # Arbitrarily large partial returns cannot make this beat a score of 30.
    partial = ~td["next", "done"].unsqueeze(-1).expand(2, 3, 2, 1)
    td["next", "agents", "episode_reward"][partial] = 1e9
    assert completed_environment_reward(td)[0] < 30.


def test_no_completed_episode_is_missing_even_when_every_car_respawns():
    td = rollout()
    td["next", "done"].zero_()
    prepare_task_episode_boundaries(td, fix_respawn_training=True)
    assert td["next", "agents", "done"].all()
    mean, count = completed_environment_reward(td)
    assert mean is None and count == 0
    assert json.dumps([mean], allow_nan=False) == "[null]"


@pytest.mark.parametrize("value", [0., -5.])
def test_zero_and_negative_returns_are_valid_completed_episodes(value):
    td = rollout()
    td["next", "agents", "episode_reward"].fill_(value)
    assert completed_environment_reward(td) == (value, 2)
