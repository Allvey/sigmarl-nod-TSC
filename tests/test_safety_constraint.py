import json

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs
from utilities.nod_marl.safety import SafetyCriticManager
from test_safety_critic import make_rollout


def test_constraint_reduces_action_risk_without_training_critic_or_states(tmp_path):
    p = Parameters(is_using_safety_constraint=True, safety_constraint_warmup_batches=1)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    td, observation, stored_action = make_rollout()
    assert not manager.constraint_ready
    manager.updates = manager.rollouts = 1
    # Known Q: future risk increases with speed; h=1 must never control actions.
    manager.model = torch.nn.Linear(manager.input_dim, 4)
    with torch.no_grad():
        manager.model.weight.zero_()
        manager.model.bias.zero_()
        manager.model.bias[0] = 100
        manager.model.weight[1:, 11] = 1  # first car's speed action
        manager.model.weight[1:, 24] = 1  # second car's speed action
    action = torch.nn.Parameter(torch.full_like(stored_action, 0.3))
    before = {k: v.clone() for k, v in manager.model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    loss = manager.actor_loss(td, action)
    loss.backward()
    assert action.grad[..., 0].gt(0).all()
    assert action.grad[..., 1].eq(0).all()
    assert observation.grad is None and stored_action.grad is None
    assert all(parameter.grad is None for parameter in manager.model.parameters())
    with torch.no_grad():
        action -= action.grad
    assert manager.actor_loss(td, action) < loss
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, manager.model.state_dict()[k]) for k, v in before.items())
    metrics = manager.finish_actor_update()
    assert metrics["actor_constraint_violation_rate"] == 1
    assert metrics["actor_constraint_risk"] < 1  # excludes the h=1 bias of 100
    assert metrics["actor_constraint_next_weight"] > metrics["actor_constraint_weight"]
    # A current violation cannot be undone by the action; empty masks stay finite.
    td["agents", "info", "safety_margins"].fill_(1)
    assert manager.actor_loss(td, action) == 0
    assert manager.finish_actor_update()["actor_constraint_eligible_fraction"] == 0
    td["agents", "info", "safety_margins"].fill_(-0.5)
    with torch.no_grad():
        action.fill_(-1)
    weight = manager.constraint_weight
    manager.actor_loss(td, action)
    manager.finish_actor_update()
    assert 0 <= manager.constraint_weight < weight


def test_constraint_configuration_and_legacy_default():
    assert not Parameters.from_dict({}).is_using_safety_constraint
    with pytest.raises(ValueError):
        Parameters(is_using_safety_constraint=True, is_using_safety_critic=False)
    with pytest.raises(ValueError):
        Parameters(is_using_safety_constraint=True, safety_horizons=[1])


def test_constraint_warmup_functional_ppo_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in [False, True]:
            p = Parameters.from_json("config.json")
            p.seed = 12345
            p.n_iters = p.num_epochs = 2
            p.frames_per_batch = 64
            p.total_frames = 128
            p.minibatch_size = p.safety_minibatch_size = 32
            p.num_vmas_envs = 4
            p.max_steps = p.nod_sequence_length = 16
            p.safety_num_epochs = 2
            p.is_using_safety_constraint = enabled
            p.safety_constraint_warmup_batches = 1
            p.safety_constraint_initial_weight = 0.1
            p.safety_constraint_margin = 1.0  # ensure an active penalty in this tiny run
            p.is_load_model = p.is_continue_train = False
            p.where_to_save = str(tmp_path / str(enabled)) + "/"
            env, _, _, _ = mappo_cavs(p)
            env.close()
        off = torch.load(tmp_path / "False/final_policy.pth")
        on = torch.load(tmp_path / "True/final_policy.pth")
        assert any(not torch.equal(off[k], on[k]) for k in off)
        # The penalty uses deterministic nominal actions without perturbing rollout RNG.
        data = [json.loads(next((tmp_path / str(e)).glob("reward*_data.json")).read_text())
                for e in [False, True]]
        assert data[0]["episode_reward_mean_list"] == data[1]["episode_reward_mean_list"]
        metrics = data[1]["safety_metrics_list"]
        assert metrics[0]["actor_constraint_active"] == 0
        assert metrics[1]["actor_constraint_active"] == 1
        assert metrics[1]["actor_constraint_loss"] > 0
        checkpoint = torch.load(tmp_path / "True/final_safety_critic.pth")
        assert checkpoint["rollouts"] == 2
        restored = SafetyCriticManager(p, 32, 4, 2, ("agents", "observation"))
        assert restored.load_if_available(tmp_path / "True/final_safety_critic.pth", True)
        assert restored.constraint_weight == checkpoint["constraint_weight"]
        assert restored.constraint_ready
        # Old sidecars load, but without a recorded warmup start conservatively.
        checkpoint.pop("rollouts")
        checkpoint.pop("constraint_weight")
        legacy = tmp_path / "legacy.pth"
        torch.save(checkpoint, legacy)
        old = SafetyCriticManager(p, 32, 4, 2, ("agents", "observation"))
        assert old.load_if_available(legacy, True)
        assert not old.constraint_ready
    finally:
        torch.set_num_threads(threads)
