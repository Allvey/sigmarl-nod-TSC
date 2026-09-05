import torch

from utilities.topology_module import TopologyActionPredictor, TopologyLearner


def test_action_predictor_reuses_detached_topology_latent_and_saves_head_only():
    learner = TopologyLearner(
        num_layers=2, d_latent=16, d_ego=5, d_nei=6, d_rel=4
    )
    predictor = TopologyActionPredictor(
        learner, action_dim=2, hidden_ratio=0.5, share_decoder=False
    )
    ego = torch.randn(3, 5)
    neighbors = torch.randn(3, 2, 6)
    relative = torch.randn(3, 2, 4)

    loss = predictor(ego, neighbors, relative).square().mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in learner.parameters())
    assert any(parameter.grad is not None for parameter in predictor.action_head.parameters())
    assert set(predictor.state_dict()) == {
        "action_head.mlp.0.weight",
        "action_head.mlp.0.bias",
        "action_head.mlp.2.weight",
        "action_head.mlp.2.bias",
    }


def test_legacy_action_predictor_checkpoint_loads_only_the_reusable_head():
    learner = TopologyLearner(
        num_layers=1, d_latent=8, d_ego=3, d_nei=4, d_rel=2
    )
    predictor = TopologyActionPredictor(learner, action_dim=2)
    legacy = {
        **{f"topology_learner.{key}": value for key, value in learner.state_dict().items()},
        **{key: value.clone() for key, value in predictor.state_dict().items()},
    }

    predictor.load_state_dict(legacy)

    for key, value in predictor.state_dict().items():
        assert torch.equal(value, legacy[key])
