"""Independent variance targets, masking, gradients and checkpoint semantics."""

import math
from types import SimpleNamespace

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.nod_marl.trainer import NODOpinionManager


def manager(**overrides):
    config = dict(
        device="cpu", dt=0.1, is_using_nod_opinion=True, nod_hidden_dim=8,
        nod_label_mode="responsibility", nod_variance_supervision="evidence_error",
        nod_min_log_sigma=-2.5, nod_max_log_sigma=math.log(2),
    )
    config.update(overrides)
    return NODOpinionManager(SimpleNamespace(**config))


def sequence(model):
    pair = torch.zeros(1, 3, 2, 1, 20)
    mask = torch.ones(1, 3, 2, 1, dtype=torch.bool)
    ego = torch.ones(1, 3, 2, dtype=torch.long)
    neighbor = torch.ones_like(mask, dtype=torch.long)
    outputs, _ = model.forward_sequence(
        pair, mask, ego, neighbor, opinion_mask=mask,
        evidence_override=torch.ones_like(mask, dtype=torch.float32),
    )
    return outputs, mask


def test_error_nll_uses_fixed_future_target_and_excludes_invalid_windows():
    nod = manager()
    variance = torch.full((3,), 0.25, requires_grad=True)
    evidence = torch.tensor([0., 1., -1.], requires_grad=True)
    posterior_mean = torch.tensor([0.8, 0.9, 0.9], requires_grad=True)
    label = torch.tensor([0.5, 0., 1.], requires_grad=True)
    mask = torch.ones(3, dtype=torch.bool)
    outputs = dict(
        variance=variance, evidence=evidence, mean=posterior_mean,
        rho=torch.full((3,), 0.5), learning_valid=mask,
    )
    labels = dict(label=label, valid=torch.tensor([True, True, False]))
    _, nll, *_ = nod._losses(outputs, labels, mask)
    expected = 0.5 * (
        torch.tensor([0., 4.]) / 0.25 + math.log(0.25) + math.log(2 * math.pi)
    ).mean()
    torch.testing.assert_close(nll, expected)
    nll.backward()
    assert variance.grad[0] > 0  # Zero error should reduce the scale.
    assert variance.grad[1] < 0  # Squared error 4 should increase it.
    assert variance.grad[2] == 0  # Incomplete/reset-crossing label is ignored.
    assert evidence.grad is None
    assert posterior_mean.grad is None
    assert label.grad is None


def test_opinion_calibration_and_variance_loss_have_separate_gradient_paths():
    nod = manager()
    outputs, mask = sequence(nod.model)
    labels = dict(label=torch.zeros_like(outputs["rho"]), valid=mask)
    _, nll, calibration, *_ = nod._losses(outputs, labels, mask)
    calibration.backward(retain_graph=True)
    assert nod.model.raw_risk_weights.grad.abs().sum() > 0
    assert all(p.grad is None for p in nod.model.likelihood_head.parameters())
    assert all(p.grad is None for p in nod.model.history.parameters())

    nod.model.zero_grad(set_to_none=True)
    nll.backward()
    assert nod.model.raw_risk_weights.grad is None
    assert nod.model.likelihood_head[-1].bias.grad[2].abs() > 0


def test_error_variance_initialization_preserves_behavior_and_resets_scale():
    behavior = manager(nod_variance_supervision="posterior", nod_max_log_sigma=0.0)
    original = {key: value.clone() for key, value in behavior.model.state_dict().items()}
    candidate = manager()
    assert not candidate.load_checkpoint(behavior.checkpoint_state())
    candidate.initialize_error_variance_from_behavior(behavior.checkpoint_state())
    for key, value in behavior.model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    for key, value in behavior.model.history.state_dict().items():
        torch.testing.assert_close(candidate.model.history.state_dict()[key], value)
    _, _, variance = candidate.model.likelihood_parameters(torch.randn(3, 8))
    torch.testing.assert_close(variance, torch.ones_like(variance))
    assert candidate.checkpoint_state()["version"] == 7
    assert manager().load_checkpoint(candidate.checkpoint_state())
    incompatible = manager(nod_max_log_sigma=math.log(3))
    assert not incompatible.load_checkpoint(candidate.checkpoint_state())
    assert not behavior.load_checkpoint(candidate.checkpoint_state())


def test_no_valid_future_labels_produce_zero_variance_loss():
    nod = manager()
    outputs, mask = sequence(nod.model)
    labels = dict(label=torch.ones_like(outputs["rho"]), valid=torch.zeros_like(mask))
    loss, nll, calibration, *_ = nod._losses(outputs, labels, mask)
    assert loss.item() == nll.item() == calibration.item() == 0.0


def test_independent_error_candidate_config_and_invalid_contracts():
    parameters = Parameters.from_json(
        "configs/ablations/config_dgppo_nod_evidence_error_variance_candidate.json"
    )
    assert parameters.nod_training_mode == "candidate_only"
    assert parameters.nod_freeze_training
    assert parameters.nod_evidence_mode == "geometric"
    assert parameters.nod_variance_supervision == "evidence_error"
    assert parameters.nod_max_log_sigma == math.log(2)
    assert not parameters.dgppo_opinion_alpha
    for change in (
        dict(nod_variance_mode="fixed"), dict(nod_evidence_mode="learned"),
        dict(nod_max_log_sigma=0.0), dict(nod_label_mode="future_mitigation"),
    ):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(parameters.to_dict(), **change))
