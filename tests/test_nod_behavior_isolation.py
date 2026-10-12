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
    manager = NODOpinionManager(parameters)

    loaded = manager.load_checkpoint({"version": 1, "model": {}})

    assert not loaded
    assert "legacy NOD checkpoint ignored" in manager.last_load_info
    assert manager.online_state is None


def test_future_candidate_reuses_filter_but_not_old_output_semantics():
    common = dict(
        device="cpu",
        dt=0.1,
        is_using_nod_opinion=True,
        nod_hidden_dim=8,
    )
    behavior = NODOpinionManager(
        SimpleNamespace(**common, nod_label_mode="responsibility")
    )
    candidate = NODOpinionManager(
        SimpleNamespace(**common, nod_label_mode="future_mitigation")
    )
    old_checkpoint = behavior.checkpoint_state()
    assert old_checkpoint["version"] == 3
    assert candidate.checkpoint_state()["version"] == 4
    assert not candidate.load_checkpoint(old_checkpoint)
    candidate.initialize_future_from_behavior(old_checkpoint)
    for key, value in behavior.model.history.state_dict().items():
        torch.testing.assert_close(candidate.model.history.state_dict()[key], value)

    pair = torch.zeros(1, 3, 2, 1, 20)
    mask = torch.ones(1, 3, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 3, 2, dtype=torch.long)
    neighbor_generations = torch.ones(1, 3, 2, 1, dtype=torch.long)
    outputs, _ = candidate.model.forward_sequence(
        pair,
        mask,
        generations,
        neighbor_generations,
        opinion_mask=mask,
        evidence_override=torch.ones_like(pair[..., 0]),
    )
    labels = {"label": torch.ones_like(outputs["rho"]),
              "valid": torch.ones_like(mask)}
    loss, _, _, _, _, valid = candidate._losses(outputs, labels, mask)
    assert valid[:, 1:].all()
    loss.backward()
    assert candidate.model.future_head[-1].bias.grad.abs().sum() > 0
    assert not behavior.load_checkpoint(candidate.checkpoint_state())


def test_future_quantile_starts_from_frozen_stage_one_predictor():
    common = dict(device="cpu", dt=0.1, is_using_nod_opinion=True, nod_hidden_dim=8)
    stage_one = NODOpinionManager(
        SimpleNamespace(**common, nod_label_mode="future_mitigation")
    )
    stage_two = NODOpinionManager(
        SimpleNamespace(
            **common, nod_label_mode="future_mitigation",
            nod_predict_future_quantile=True, nod_quantile_level=0.1,
        )
    )
    stage_two.initialize_quantile_from_future(stage_one.checkpoint_state())
    assert stage_two.checkpoint_state()["version"] == 5
    assert all(
        parameter.requires_grad == name.startswith("future_quantile_head.")
        for name, parameter in stage_two.model.named_parameters()
    )
    pair = torch.zeros(1, 3, 2, 1, 20)
    mask = torch.ones(1, 3, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 3, 2, dtype=torch.long)
    neighbors = torch.ones(1, 3, 2, 1, dtype=torch.long)
    inputs = (pair, mask, generations, neighbors)
    baseline, _ = stage_one.model.forward_sequence(*inputs, opinion_mask=mask)
    candidate, _ = stage_two.model.forward_sequence(*inputs, opinion_mask=mask)
    torch.testing.assert_close(candidate["rho"], baseline["rho"])
    assert (candidate["future_quantile_raw"][:, 0] == 0).all()
    labels = {"label": torch.ones_like(candidate["rho"]), "valid": mask}
    loss, *_ = stage_two._losses(candidate, labels, mask)
    loss.backward()
    assert stage_two.model.future_quantile_head[-1].weight.grad is not None
    assert stage_two.model.future_head[-1].weight.grad is None


def test_fixed_variance_learned_evidence_warmstart_and_teacher_loss():
    common = dict(
        device="cpu", dt=0.1, is_using_nod_opinion=True,
        nod_hidden_dim=8, nod_label_mode="responsibility",
    )
    behavior = NODOpinionManager(SimpleNamespace(**common))
    candidate = NODOpinionManager(SimpleNamespace(
        **common, nod_variance_mode="fixed", nod_evidence_mode="learned",
    ))
    assert not candidate.load_checkpoint(behavior.checkpoint_state())
    candidate.initialize_ablation_from_behavior(behavior.checkpoint_state())
    for key, value in behavior.model.history.state_dict().items():
        torch.testing.assert_close(candidate.model.history.state_dict()[key], value)

    pair = torch.zeros(1, 3, 2, 1, 20)
    mask = torch.ones(1, 3, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 3, 2, dtype=torch.long)
    neighbors = torch.ones(1, 3, 2, 1, dtype=torch.long)
    outputs, _ = candidate.model.forward_sequence(
        pair, mask, generations, neighbors, opinion_mask=mask,
        evidence_override=-torch.ones_like(mask, dtype=torch.float32),
    )
    labels = {"label": torch.zeros_like(outputs["rho"]), "valid": mask}
    loss, *_ = candidate._losses(outputs, labels, mask)
    loss.backward()
    assert candidate.model.evidence_head.weight.grad is not None
    assert torch.isfinite(candidate.model.evidence_head.weight.grad).all()

    # A pending auto-calibrated learner must not be used as a frozen controller.
    assert not NODOpinionManager(SimpleNamespace(
        **common, nod_variance_mode="fixed", nod_evidence_mode="learned",
    )).load_checkpoint(candidate.checkpoint_state())
    candidate.model.fixed_variance.fill_(0.2)
    candidate._variance_needs_calibration = False
    loaded = NODOpinionManager(SimpleNamespace(
        **common, nod_variance_mode="fixed", nod_evidence_mode="learned",
    ))
    assert loaded.load_checkpoint(candidate.checkpoint_state())
    assert float(loaded.model.fixed_variance) == float(candidate.model.fixed_variance)


def test_scratch_candidate_keeps_explicit_variance_without_calibration():
    manager = NODOpinionManager(SimpleNamespace(
        device="cpu", dt=0.1, is_using_nod_opinion=True,
        nod_hidden_dim=8, nod_label_mode="responsibility",
        nod_variance_mode="fixed", nod_evidence_mode="learned",
        nod_fixed_variance_value=0.006738110911101103,
        nod_candidate_init_mode="scratch",
    ))
    assert not manager._variance_needs_calibration
    assert manager.checkpoint_state()["candidate_init_mode"] == "scratch"
    assert float(manager.model.fixed_variance) == torch.tensor(0.006738110911101103).item()
