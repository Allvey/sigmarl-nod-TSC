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


def test_history_consumes_direct_physical_pair_features():
    model = NODOpinionModel(pair_feature_dim=20, hidden_dim=8)
    pair_a = torch.zeros(1, 2, 2, 1, 20)
    pair_b = pair_a.clone()
    pair_b[:, 1, :, :, 6] = 0.9
    pair_b[:, 1, :, :, 8] = 0.1
    edge_mask = torch.ones(1, 2, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 2, 2, dtype=torch.long)
    neighbor_generations = torch.ones(1, 2, 2, 1, dtype=torch.long)

    _, state_a = model.forward_sequence(
        pair_a,
        edge_mask,
        generations,
        neighbor_generations,
    )
    _, state_b = model.forward_sequence(
        pair_b,
        edge_mask,
        generations,
        neighbor_generations,
    )

    assert model.history.input_size == 20
    assert not torch.allclose(state_a["hidden"], state_b["hidden"])


def test_risk_attention_is_monotone_in_physical_risk_components():
    model = NODOpinionModel(pair_feature_dim=20, hidden_dim=8)
    safer = torch.zeros(1, 20)
    safer[..., 6] = 1.0
    safer[..., 8] = 1.0
    riskier = safer.clone()
    riskier[..., 6] = 0.2
    riskier[..., 8] = 0.2
    riskier[..., 9] = 1.0
    riskier[..., 15] = 0.8
    riskier[..., 16] = 0.8

    assert model.risk_attention(riskier).item() > model.risk_attention(safer).item()
    assert torch.all(model.risk_weights >= 0.0)


def test_history_can_be_disabled_without_removing_opinion_dynamics():
    model = NODOpinionModel(
        pair_feature_dim=20, hidden_dim=8, history_mode="none"
    )
    pair = torch.zeros(1, 2, 2, 1, 20)
    edge_mask = torch.ones(1, 2, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 2, 2, dtype=torch.long)
    neighbor_generations = torch.ones(1, 2, 2, 1, dtype=torch.long)

    outputs, _ = model.forward_sequence(
        pair,
        edge_mask,
        generations,
        neighbor_generations,
    )

    assert outputs["z"].shape == (1, 2, 2, 1)
    assert torch.isfinite(outputs["z"]).all()


def test_fixed_evidence_mapping_has_zero_intercept_and_unit_slope():
    model = NODOpinionModel(
        pair_feature_dim=20,
        hidden_dim=8,
        fixed_evidence_mapping=True,
    )
    hidden = torch.randn(2, 3, 4, 8)
    intercept, slope, variance = model.likelihood_parameters(hidden)

    assert torch.equal(intercept, torch.zeros_like(intercept))
    assert torch.equal(slope, torch.ones_like(slope))
    assert torch.isfinite(variance).all()
    assert (variance > 0.0).all()


def test_explicit_responsibility_mask_limits_where_opinion_evolves():
    model = NODOpinionModel(
        pair_feature_dim=20,
        hidden_dim=8,
        fixed_evidence_mapping=True,
    )
    pair = torch.zeros(1, 3, 2, 1, 20)
    edge_mask = torch.ones(1, 3, 2, 1, dtype=torch.bool)
    opinion_mask = torch.zeros_like(edge_mask)
    opinion_mask[:, 1:, 0, 0] = True
    evidence = torch.zeros_like(pair[..., 0])
    evidence[:, 1:, 0, 0] = 1.0
    generations = torch.ones(1, 3, 2, dtype=torch.long)
    neighbor_generations = torch.ones(1, 3, 2, 1, dtype=torch.long)

    outputs, _ = model.forward_sequence(
        pair,
        edge_mask,
        generations,
        neighbor_generations,
        opinion_mask=opinion_mask,
        evidence_override=evidence,
    )

    assert outputs["z"][0, 1, 0, 0] > 0.0
    assert (outputs["z"][~opinion_mask] == 0.0).all()
    assert torch.equal(outputs["opinion_active"], opinion_mask)
