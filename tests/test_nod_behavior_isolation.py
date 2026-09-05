from types import SimpleNamespace

import torch
from tensordict import TensorDict

from utilities.nod_marl.trainer import NODOpinionManager


def test_auxiliary_update_does_not_change_an_unrelated_actor():
    parameters = SimpleNamespace(
        device="cpu",
        dt=0.1,
        is_using_nod_opinion=True,
        nod_hidden_dim=8,
        nod_counterfactual_horizon=2,
    )
    manager = NODOpinionManager(parameters)
    actor = torch.nn.Linear(5, 2)
    actor_before = {
        key: value.detach().clone() for key, value in actor.state_dict().items()
    }

    batch, time, n_agents, k_neighbors = 1, 5, 2, 1
    pair_features = torch.randn(batch, time, n_agents, k_neighbors, 20)
    edge_mask = torch.ones(batch, time, n_agents, k_neighbors)
    neighbor_indices = torch.tensor([[[[1], [0]]] * time], dtype=torch.float32)
    generations = torch.ones(batch, time, n_agents)
    neighbor_generations = torch.ones(batch, time, n_agents, k_neighbors)
    positions = torch.zeros(batch, time, n_agents, 2)
    positions[..., 1, 0] = 1.0
    velocities = torch.zeros_like(positions)
    velocities[:, 0, 1, 0] = -1.0
    data = TensorDict(
        {
            ("agents", "info", "nod_pair_features"): pair_features,
            ("agents", "info", "nod_edge_mask"): edge_mask,
            ("agents", "info", "nod_neighbor_indices"): neighbor_indices,
            ("agents", "info", "nod_ego_generation"): generations,
            ("agents", "info", "nod_neighbor_generation"): neighbor_generations,
            ("agents", "info", "nod_world_pos"): positions,
            ("agents", "info", "nod_world_vel"): velocities,
        },
        batch_size=[batch, time],
    )

    metrics = manager.train_on_rollout(data)

    assert metrics["enabled"] == 1.0
    assert metrics["edge_count"] > 0
    assert metrics["curvature_min"] > 0
    for key, value in actor.state_dict().items():
        assert torch.equal(value, actor_before[key])


def test_manager_initialization_does_not_advance_global_rng():
    parameters = SimpleNamespace(
        device="cpu",
        dt=0.1,
        is_using_nod_opinion=True,
        nod_hidden_dim=8,
    )
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    NODOpinionManager(parameters)
    actual = torch.rand(5)
    assert torch.equal(actual, expected)


def test_v1_checkpoint_is_ignored_instead_of_breaking_training_interface():
    parameters = SimpleNamespace(
        device="cpu",
        dt=0.1,
        is_using_nod_opinion=True,
        nod_hidden_dim=8,
    )
    manager = NODOpinionManager(
        parameters, relation_feature_dim=16, action_dim=2
    )

    loaded = manager.load_checkpoint({"version": 1, "model": {}})

    assert not loaded
    assert "legacy NOD checkpoint ignored" in manager.last_load_info
    assert manager.online_state is None
