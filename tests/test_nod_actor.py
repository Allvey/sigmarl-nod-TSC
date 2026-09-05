from types import SimpleNamespace

import torch
from tensordict import TensorDict

from utilities.nod_marl.policy import (
    NODActorInputModule,
    NODMessageAggregator,
    NOD_ACTOR_ATTENTION_KEY,
    NOD_ACTOR_CONTEXT_READY_KEY,
    NOD_ACTOR_EDGE_CONTEXT_KEY,
    NOD_ACTOR_EDGE_MASK_KEY,
    NOD_ACTOR_MESSAGE_KEY,
    NOD_ACTOR_OBSERVATION_KEY,
)
from utilities.nod_marl.trainer import NODOpinionManager


def test_message_aggregation_is_permutation_invariant_and_zero_without_edges():
    torch.manual_seed(3)
    aggregator = NODMessageAggregator(context_dim=7, message_dim=5, hidden_dim=9)
    context = torch.randn(2, 3, 7)
    mask = torch.tensor([[True, False, True], [True, True, False]])

    message, _ = aggregator(context, mask)
    permutation = torch.tensor([2, 0, 1])
    permuted_message, _ = aggregator(
        context[:, permutation], mask[:, permutation]
    )
    empty_message, empty_attention = aggregator(
        context, torch.zeros_like(mask)
    )

    assert torch.allclose(message, permuted_message, atol=1e-6, rtol=1e-6)
    assert torch.equal(empty_message, torch.zeros_like(empty_message))
    assert torch.equal(empty_attention, torch.zeros_like(empty_attention))


def test_actor_input_reuses_detached_cache_and_trains_only_message_aggregator():
    class DummyNODManager:
        action_dim = 2
        online_context_dim = 7
        n_neighbors = 2

        def __init__(self):
            self.calls = 0

        def online_step(self, tensordict, topology_manager=None):
            self.calls += 1
            raise AssertionError("cached PPO input must not replay online NOD")

    class DummyTopologyManager:
        pass

    nod_manager = DummyNODManager()
    topology_manager = DummyTopologyManager()
    module = NODActorInputModule(
        observation_key=("agents", "observation"),
        base_observation_dim=5,
        topology_manager=topology_manager,
        nod_manager=nod_manager,
        message_dim=4,
        message_hidden_dim=8,
    )
    context = torch.randn(2, 3, 2, 7, requires_grad=True)
    observation = torch.randn(2, 3, 7)
    previous_velocity = torch.randn(2, 3)
    previous_steering = torch.randn(2, 3)
    data = TensorDict(
        {
            ("agents", "observation"): observation,
            ("agents", "info", "act_vel"): previous_velocity,
            ("agents", "info", "act_steer"): previous_steering,
            NOD_ACTOR_EDGE_CONTEXT_KEY: context,
            NOD_ACTOR_EDGE_MASK_KEY: torch.ones(2, 3, 2, dtype=torch.bool),
            NOD_ACTOR_CONTEXT_READY_KEY: torch.ones(
                2, 3, 1, dtype=torch.bool
            ),
            NOD_ACTOR_MESSAGE_KEY: torch.zeros(2, 3, 4),
            NOD_ACTOR_ATTENTION_KEY: torch.zeros(2, 3, 2),
            NOD_ACTOR_OBSERVATION_KEY: torch.zeros(2, 3, 11),
        },
        batch_size=[2],
    )

    module(data)
    actor_input = data.get(NOD_ACTOR_OBSERVATION_KEY)
    assert nod_manager.calls == 0
    assert actor_input.shape == (2, 3, 11)
    assert torch.equal(actor_input[..., :5], observation[..., :5])
    assert torch.equal(actor_input[..., -2], previous_velocity)
    assert torch.equal(actor_input[..., -1], previous_steering)

    data.get(NOD_ACTOR_MESSAGE_KEY).square().sum().backward()
    assert context.grad is None
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_online_opinion_state_resets_when_neighbor_generation_changes():
    parameters = SimpleNamespace(
        device="cpu",
        dt=0.1,
        n_agents=2,
        is_using_nod_opinion=True,
        nod_hidden_dim=8,
    )
    manager = NODOpinionManager(
        parameters, relation_feature_dim=7, action_dim=2
    )

    class TopologyStub:
        def encode_nod_inputs(self, tensordict, target_neighbor_indices):
            leading = target_neighbor_indices.shape
            return {
                "relation_features": torch.randn(*leading, 7),
                "predicted_actions": torch.randn(*leading, 2),
                "edge_probability": torch.full(leading, 0.75),
                "available": torch.ones(leading, dtype=torch.bool),
            }

    pair = torch.zeros(1, 2, 1, 20)
    pair[..., 6] = 0.4
    pair[..., 8] = 0.5
    pair[..., 9] = 1.0
    neighbor_generation = torch.ones(1, 2, 1, dtype=torch.long)
    data = TensorDict(
        {
            ("agents", "info", "nod_pair_features"): pair,
            ("agents", "info", "nod_edge_mask"): torch.ones(
                1, 2, 1, dtype=torch.bool
            ),
            ("agents", "info", "nod_neighbor_indices"): torch.tensor(
                [[[1], [0]]]
            ),
            ("agents", "info", "nod_ego_generation"): torch.ones(
                1, 2, dtype=torch.long
            ),
            ("agents", "info", "nod_neighbor_generation"): neighbor_generation,
        },
        batch_size=[1],
    )

    first = manager.online_step(data, topology_manager=TopologyStub())
    assert first["edge_context"].shape[-1] == manager.online_context_dim
    assert manager.online_state is not None
    assert manager.online_state["has_state"].all()

    neighbor_generation[0, 0, 0] = 2
    reset = manager.online_step(data, topology_manager=TopologyStub())
    assert reset["opinion"][0, 0, 0].item() == 0.0
    assert manager.online_state["neighbor_generation"][0, 0, 0].item() == 2
