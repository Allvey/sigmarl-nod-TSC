import torch

from utilities.nod_marl.opinion import (
    NODOpinionModel,
    kl_objective,
    kl_proximal_update,
)


def test_kl_update_is_bounded_decreases_objective_and_has_finite_implicit_gradient():
    z_previous = torch.tensor([0.2], requires_grad=True)
    attention = torch.tensor([0.8], requires_grad=True)
    evidence = torch.tensor([0.5])
    mean_intercept = torch.tensor([0.0], requires_grad=True)
    slope = torch.tensor([0.5], requires_grad=True)
    variance = torch.tensor([0.2], requires_grad=True)
    kwargs = dict(
        bifurcation_gain=2.0,
        observation_weight=1.0,
        kl_weight=5.0,
    )
    updated = kl_proximal_update(
        z_previous,
        attention,
        evidence,
        mean_intercept,
        slope,
        variance,
        **kwargs,
    )
    objective_updated = kl_objective(
        updated,
        z_previous,
        attention,
        evidence,
        mean_intercept,
        slope,
        variance,
        **kwargs,
    )
    objective_previous = kl_objective(
        z_previous,
        z_previous,
        attention,
        evidence,
        mean_intercept,
        slope,
        variance,
        **kwargs,
    )

    assert updated.abs().item() < 1.0
    assert objective_updated.item() <= objective_previous.item() + 1e-6
    updated.sum().backward()
    for value in (z_previous, attention, mean_intercept, slope, variance):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_generation_change_resets_an_active_edge_to_neutral():
    model = NODOpinionModel(pair_feature_dim=20, hidden_dim=8)
    pair_features = torch.randn(1, 4, 2, 1, 20)
    edge_mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    ego_generation = torch.ones(1, 4, 2, dtype=torch.long)
    neighbor_generation = torch.ones(1, 4, 2, 1, dtype=torch.long)
    # The neighbor occupying slot 0 is replaced before frame 2.
    neighbor_generation[:, 2:, 0, 0] = 2

    outputs, _ = model.forward_sequence(
        pair_features, edge_mask, ego_generation, neighbor_generation
    )

    assert outputs["z"][0, 0, 0, 0].item() == 0.0
    assert outputs["z"][0, 2, 0, 0].item() == 0.0
    assert not outputs["learning_valid"][0, 0, 0, 0]
    assert outputs["learning_valid"][0, 1, 0, 0]
    assert not outputs["learning_valid"][0, 2, 0, 0]
    assert outputs["new_edges"].item() >= 3.0


def test_implicit_gradient_matches_finite_difference():
    kwargs = dict(
        bifurcation_gain=2.0,
        observation_weight=1.0,
        kl_weight=5.0,
        max_iterations=64,
        tolerance=1e-10,
    )

    def solve(evidence):
        return kl_proximal_update(
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([0.8], dtype=torch.float64),
            evidence,
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.2], dtype=torch.float64),
            **kwargs,
        )

    evidence = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    solve(evidence).sum().backward()
    step = 1e-5
    finite_difference = (
        solve(torch.tensor([0.5 + step], dtype=torch.float64))
        - solve(torch.tensor([0.5 - step], dtype=torch.float64))
    ) / (2.0 * step)
    assert torch.allclose(evidence.grad, finite_difference, atol=2e-5, rtol=2e-5)


def test_short_edge_gap_retains_identity_and_marks_resume():
    model = NODOpinionModel(pair_feature_dim=20, hidden_dim=8, retention_steps=2)
    pair_features = torch.randn(1, 4, 2, 1, 20)
    edge_mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    edge_mask[:, 1, 0, 0] = False
    generations = torch.ones(1, 4, 2, dtype=torch.long)
    neighbor_generations = torch.ones(1, 4, 2, 1, dtype=torch.long)

    outputs, _ = model.forward_sequence(
        pair_features, edge_mask, generations, neighbor_generations
    )

    assert outputs["resumed_edges"].item() == 1.0
    assert outputs["learning_valid"][0, 2, 0, 0]
