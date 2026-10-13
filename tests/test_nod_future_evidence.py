"""Future evidence supervision, causal inference and checkpoint isolation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from utilities.helper_training import Parameters
from utilities.nod_marl.trainer import NODOpinionManager


def manager(**overrides):
    config = dict(
        device="cpu", dt=0.1, is_using_nod_opinion=True, nod_hidden_dim=8,
        nod_label_mode="responsibility", nod_variance_mode="fixed",
        nod_fixed_variance_value=1.0, nod_fixed_evidence_mapping=True,
        nod_evidence_mode="learned", nod_evidence_target="future",
        nod_nll_weight=0.0, nod_evidence_loss_weight=1.0,
    )
    config.update(overrides)
    return NODOpinionManager(SimpleNamespace(**config))


def inputs():
    pair = torch.zeros(1, 4, 2, 1, 20)
    mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    ego = torch.ones(1, 4, 2, dtype=torch.long)
    neighbor = torch.ones_like(mask, dtype=torch.long)
    return pair, mask, ego, neighbor


def test_future_evidence_loss_uses_future_labels_and_masks_window_tails():
    nod = manager()
    prediction = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    geometry = torch.full((3,), -1.0, requires_grad=True)
    mask = torch.ones(3, dtype=torch.bool)
    outputs = dict(
        variance=torch.ones(3), evidence=geometry, evidence_used=prediction,
        mean=torch.zeros(3), rho=torch.full((3,), 0.5), learning_valid=mask,
    )
    labels = dict(label=torch.tensor([1., 0., 1.]), valid=torch.tensor([True, True, False]))
    loss, nll, calibration, *_ = nod._losses(outputs, labels, mask)
    expected = F.binary_cross_entropy(torch.tensor([0.6, 0.3]), torch.tensor([1., 0.]))
    torch.testing.assert_close(loss, expected + calibration)
    assert nll.item() == 0
    loss.backward()
    assert prediction.grad[0] < 0 and prediction.grad[1] > 0
    assert prediction.grad[2] == 0
    assert geometry.grad is None
    changed = dict(outputs, evidence=torch.ones(3))
    torch.testing.assert_close(nod._losses(changed, labels, mask)[0], loss)
    labels["valid"] = torch.zeros_like(mask)
    assert nod._losses(outputs, labels, mask)[0].item() == 0


def test_future_prediction_drives_nod_instead_of_geometric_evidence():
    nod = manager()
    with torch.no_grad():
        nod.model.evidence_head[-1].bias.fill_(2.0)
    pair, mask, ego, neighbor = inputs()
    outputs, _ = nod.model.forward_sequence(
        pair, mask, ego, neighbor, opinion_mask=mask,
        evidence_override=-torch.ones_like(mask, dtype=torch.float32),
    )
    assert (outputs["evidence"] < 0).all()
    assert (outputs["evidence_used"] > 0).all()
    assert (outputs["z"][:, 1:] > 0).all()
    torch.testing.assert_close(outputs["z"], outputs["filter_z"])
    torch.testing.assert_close(outputs["variance"], torch.ones_like(outputs["variance"]))


def test_forecast_input_is_causal_and_includes_current_geometry():
    nod = manager()
    with torch.no_grad():
        nod.model.evidence_head[0].weight.zero_()
        nod.model.evidence_head[0].weight[:, -1].fill_(0.5)
        nod.model.evidence_head[0].bias.zero_()
        nod.model.evidence_head[-1].weight.fill_(0.2)
    pair, mask, ego, neighbor = inputs()
    evidence = -torch.ones_like(mask, dtype=torch.float32)
    baseline, _ = nod.model.forward_sequence(pair, mask, ego, neighbor, opinion_mask=mask, evidence_override=evidence)
    changed = evidence.clone()
    changed[:, 2:] = 1.0
    forecast, _ = nod.model.forward_sequence(pair, mask, ego, neighbor, opinion_mask=mask, evidence_override=changed)
    torch.testing.assert_close(baseline["evidence_used"][:, :2], forecast["evidence_used"][:, :2])
    torch.testing.assert_close(baseline["z"][:, :2], forecast["z"][:, :2])
    assert (forecast["evidence_used"][:, 2:] > baseline["evidence_used"][:, 2:]).all()


def test_warmstart_preserves_behavior_and_version_eight_cannot_mix_with_old_evidence():
    behavior = manager(nod_variance_mode="learned", nod_evidence_mode="geometric", nod_evidence_target="geometric", nod_nll_weight=1.0)
    before = {key: value.clone() for key, value in behavior.model.state_dict().items()}
    candidate = manager()
    candidate.initialize_ablation_from_behavior(behavior.checkpoint_state())
    for key, value in behavior.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    for key, value in behavior.model.history.state_dict().items():
        torch.testing.assert_close(candidate.model.history.state_dict()[key], value)
    torch.testing.assert_close(candidate.model.raw_risk_weights, behavior.model.raw_risk_weights)
    checkpoint = candidate.checkpoint_state()
    assert checkpoint["version"] == 8 and checkpoint["evidence_target"] == "future"
    assert manager().load_checkpoint(checkpoint)
    assert not manager(nod_evidence_target="geometric").load_checkpoint(checkpoint)
    assert not manager(nod_fixed_variance_value=0.5).load_checkpoint(checkpoint)


def test_future_losses_train_evidence_history_and_attention_but_not_variance_head():
    nod = manager()
    with torch.no_grad():
        nod.model.evidence_head[-1].weight.fill_(0.1)
        nod.model.evidence_head[-1].bias.fill_(0.5)
    pair, mask, ego, neighbor = inputs()
    outputs, _ = nod.model.forward_sequence(
        pair, mask, ego, neighbor, opinion_mask=mask,
        evidence_override=torch.ones_like(mask, dtype=torch.float32),
    )
    labels = dict(label=torch.zeros_like(outputs["rho"]), valid=mask)
    loss, *_ = nod._losses(outputs, labels, mask)
    loss.backward()
    assert nod.model.evidence_head[-1].weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in nod.model.history.parameters() if p.grad is not None) > 0
    assert nod.model.raw_risk_weights.grad.abs().sum() > 0
    assert all(p.grad is None for p in nod.model.likelihood_head.parameters())


def test_config_contract_and_matched_candidate_collection():
    future = Parameters.from_json("configs/ablations/config_dgppo_nod_future_evidence_candidate.json")
    variance = Parameters.from_json("configs/ablations/config_dgppo_nod_evidence_error_variance_candidate.json")
    control = Parameters.from_json("configs/ablations/config_dgppo_nod_future_evidence_control.json")
    for key in ("training_init_checkpoint", "seed", "n_iters", "frames_per_batch", "nod_counterfactual_horizon", "nod_num_epochs", "nod_lr", "nod_hidden_dim"):
        assert getattr(future, key) == getattr(variance, key)
    assert future.nod_training_mode == "candidate_only" and future.nod_freeze_training
    assert not future.dgppo_opinion_alpha
    assert control.nod_training_mode == "joint" and control.nod_freeze_training
    assert control.dgppo_alpha_gain == 1 and control.dgppo_opinion_deadzone == 0.1
    for change in (dict(nod_fixed_variance_value=None), dict(nod_nll_weight=1.0), dict(nod_evidence_mode="geometric"), dict(nod_label_mode="future_mitigation")):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(future.to_dict(), **change))
