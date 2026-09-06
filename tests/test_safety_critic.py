import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters, SaveData
from utilities.nod_marl.safety import (
    SafetyCriticManager,
    finite_horizon_targets,
    safety_margins,
)


def test_signed_margins_and_collision_override():
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.1, 0.0]]])
    clearance = torch.tensor([[0.05, 0.05], [0.005, 0.05]])
    collision = torch.tensor([[False, False], [False, True]])
    margins = safety_margins(pos, clearance, collision, 0.25, 0.01)
    assert (margins[0] <= 0).all()
    assert torch.allclose(margins[1, :, 0], torch.tensor([0.6, 0.6]))
    assert torch.isclose(margins[1, 0, 1], torch.tensor(0.5))
    assert margins[1, 1, 2] == 1
    single = safety_margins(
        pos[:1, :1], clearance[:1, :1], collision[:1, :1], 0.25, 0.01
    )
    assert torch.isfinite(single).all() and single[0, 0, 0] == -1


def test_targets_include_terminal_collision_without_next_episode_or_discount():
    current = torch.tensor([[-0.8, -0.4, -0.3, -0.9]])
    following = torch.tensor([[-0.4, 1.0, -0.9, -0.2]])
    done = torch.tensor([[False, True, False, False]])
    generation = torch.tensor([[[1, 1], [1, 1], [2, 2], [2, 2]]])
    target, mask = finite_horizon_targets(
        current, following, done, generation, generation, [1, 2, 4]
    )
    assert torch.equal(target[..., 0], current)
    assert torch.allclose(target[..., 1], torch.tensor([[-0.4, 1.0, -0.3, -0.2]]))
    assert target[0, 0, 2] == 1 and target[0, 1, 2] == 1
    assert mask[0, :2].all()
    assert not mask[0, 2:, 2].any()  # batch tail is censored, not called safe
    assert mask[0, -1, 1]  # one observed successor suffices for h=2


def test_targets_censor_single_agent_respawn():
    current = torch.tensor([[-0.8, -0.4, -0.3, -0.9]])
    following = torch.tensor([[-0.4, -0.3, 1.0, -0.2]])
    generation = torch.ones(1, 4, 2, dtype=torch.long)
    next_generation = generation.clone()
    next_generation[:, 1, 0] = 2
    target, mask = finite_horizon_targets(
        current,
        following,
        torch.zeros_like(current, dtype=torch.bool),
        generation,
        next_generation,
        [1, 2, 4],
    )
    assert not mask[0, 1].any()
    assert not mask[0, 0, 2]  # cannot propagate risk across the replaced vehicle
    assert mask[0, 0, 1] and target[0, 0, 1] == -0.4
    _, wrapped_mask = finite_horizon_targets(
        current,
        following,
        torch.zeros_like(current, dtype=torch.bool),
        generation.unsqueeze(-1),
        next_generation.unsqueeze(-1),
        [1, 2, 4],
    )
    assert torch.equal(mask, wrapped_mask)


def make_rollout():
    obs = torch.linspace(-1, 1, 48).reshape(2, 4, 2, 3).requires_grad_()
    action = torch.zeros(2, 4, 2, 2, requires_grad=True)
    margins = torch.full((2, 4, 2, 3), -0.5)
    following = margins.clone()
    following[:, -1, 0, 2] = 1
    done = torch.zeros(2, 4, 1, dtype=torch.bool)
    done[:, -1] = True
    data = {
        ("agents", "observation"): obs,
        ("agents", "action"): action,
        ("agents", "info", "pos"): torch.zeros(2, 4, 2, 2),
        ("agents", "info", "vel"): torch.zeros(2, 4, 2, 2),
        ("agents", "info", "rot"): torch.zeros(2, 4, 2, 1),
        ("agents", "info", "safety_margins"): margins,
        ("next", "agents", "info", "safety_margins"): following,
        ("next", "done"): done,
    }
    for prefix in [(), ("next",)]:
        data[prefix + ("agents", "info", "nod_ego_generation")] = torch.ones(
            2, 4, 2, dtype=torch.long
        )
    return TensorDict(data, batch_size=[2, 4]), obs, action


def test_independent_training_rng_gradients_checkpoint_and_old_json(tmp_path):
    p = Parameters(
        n_agents=2, safety_hidden_dim=16, safety_num_epochs=1, safety_minibatch_size=4
    )
    td, obs, action = make_rollout()
    rng = torch.random.get_rng_state().clone()
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    assert torch.equal(rng, torch.random.get_rng_state())
    before = {k: v.clone() for k, v in manager.model.state_dict().items()}
    metrics = manager.train_on_rollout(td)
    assert metrics["optimizer_updates"] == 2
    assert any(
        not torch.equal(before[k], v) for k, v in manager.model.state_dict().items()
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    assert obs.grad is None and action.grad is None
    assert metrics["unsafe_target_count"] > 0
    predicted = manager.predict(td)
    path = tmp_path / "safety.pth"
    torch.save(manager.checkpoint_state(), path)
    restored = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    assert restored.load_if_available(path, load_optimizer=True)
    assert torch.equal(predicted, restored.predict(td))
    assert restored.updates == manager.updates
    assert torch.equal(restored.generator.get_state(), manager.generator.get_state())
    manager.train_on_rollout(td)
    restored.train_on_rollout(td)
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in manager.model.state_dict().items()
    )
    assert not restored.load_if_available(tmp_path / "old_missing.pth")
    incompatible = SafetyCriticManager(p, 3, 3, 2, ("agents", "observation"))
    assert not incompatible.load_if_available(path)
    saved = SaveData.from_dict({"parameters": {"n_agents": 2}})
    assert saved.safety_metrics_list is None
    saved.safety_metrics_list = [metrics]
    assert SaveData.from_dict(saved.to_dict()).safety_metrics_list == [metrics]


def test_disabled_manager_does_not_require_safety_fields():
    p = Parameters(is_using_safety_critic=False)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    assert manager.train_on_rollout(TensorDict({}, batch_size=[])) == {"enabled": 0.0}
