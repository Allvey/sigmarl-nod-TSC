import torch

from utilities.nod_marl.counterfactual import build_counterfactual_labels


def _two_agent_rollout():
    positions = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ]
    )
    velocities = torch.zeros_like(positions)
    velocities[:, 0, 1, 0] = -1.0
    neighbor_indices = torch.tensor([[[[1], [0]]] * 4])
    edge_mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 4, 2, dtype=torch.long)
    return positions, velocities, generations, neighbor_indices, edge_mask


def test_counterfactual_label_rewards_observed_risk_mitigation():
    args = _two_agent_rollout()
    labels = build_counterfactual_labels(
        *args,
        horizon=2,
        dt=1.0,
        safe_distance=0.5,
        label_slope=12.0,
        label_margin=0.02,
    )
    assert labels["valid"][0, 0, 0, 0]
    assert labels["gap"][0, 0, 0, 0] > 0
    assert labels["label"][0, 0, 0, 0] > 0.5


def test_counterfactual_label_is_invalid_across_identity_reset():
    positions, velocities, generations, neighbor_indices, edge_mask = (
        _two_agent_rollout()
    )
    generations[:, 1:, 1] = 2
    labels = build_counterfactual_labels(
        positions,
        velocities,
        generations,
        neighbor_indices,
        edge_mask,
        horizon=2,
        dt=1.0,
        safe_distance=0.5,
        label_slope=12.0,
        label_margin=0.02,
    )
    assert not labels["valid"][0, 0, 0, 0]
