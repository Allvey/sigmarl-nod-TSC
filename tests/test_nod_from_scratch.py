"""Two-stage configuration and supervised/frozen predictor handoff, without simulation."""
from pathlib import Path
from types import SimpleNamespace

import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import _bind_frozen_nod
from utilities.nod_marl.trainer import NODOpinionManager


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/current/from_scratch"


def test_two_stage_lineage_and_joint_training_contract():
    first = Parameters.from_json(str(CONFIG_ROOT / "01_joint.json"))
    second = Parameters.from_json(str(CONFIG_ROOT / "02_ppo_beta.json"))
    assert {p.name for p in CONFIG_ROOT.glob("*.json")} == {"01_joint.json", "02_ppo_beta.json"}
    for name in ("training_init_checkpoint", "nod_candidate_init_checkpoint", "nod_control_checkpoint",
                 "nod_future_calibration_file"):
        assert not getattr(first, name)
    assert not first.is_load_model and not first.is_continue_train
    assert first.safety_training_mode == "scratch" and not first.nod_freeze_training
    assert first.nod_training_mode == "joint" and first.nod_update_interval == 1
    assert first.nod_evidence_gate_mode == "fixed" and first.lr == 2e-4
    assert second.training_init_checkpoint == first.where_to_save + "final"
    assert second.nod_freeze_training and second.safety_training_mode == "finetune"
    assert second.nod_training_mode == "joint" and second.nod_evidence_gate_mode == "ppo"
    assert second.nod_evidence_gate_target == "evidence"
    assert second.nod_evidence_gate_sequence_length == 8
    assert second.nod_evidence_gate_only_iters == 0
    assert (first.n_iters, second.n_iters) == (250, 70)
    assert first.total_frames + second.total_frames == 1310720
    for p in (first, second):
        assert p.nod_evidence_target == "future" and p.nod_evidence_mode == "learned"
        assert p.nod_variance_mode == "fixed" and p.nod_fixed_variance_value == 1.0
        assert p.dgppo_opinion_alpha and p.dgppo_alpha_gain == 1.0
        assert not p.is_add_noise and p.seed == 0
        assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_joint_predictor_receives_future_supervision_then_freezes_for_control():
    first = Parameters.from_json(str(CONFIG_ROOT / "01_joint.json"))
    nod = NODOpinionManager(first)
    with torch.no_grad():
        nod.model.evidence_head[-1].weight.fill_(0.1)
        nod.model.evidence_head[-1].bias.fill_(0.5)
    pair = torch.zeros(1, 4, 2, 1, 20)
    mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    outputs, _ = nod.model.forward_sequence(
        pair, mask, torch.ones(1, 4, 2, dtype=torch.long),
        torch.ones_like(mask, dtype=torch.long), opinion_mask=mask,
        evidence_override=torch.ones_like(mask, dtype=torch.float32),
    )
    loss, *_ = nod._losses(outputs, {"label": torch.zeros_like(outputs["rho"]), "valid": mask}, mask)
    loss.backward()
    assert nod.model.evidence_head[-1].weight.grad.abs().sum() > 0
    assert nod.model.raw_risk_weights.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in nod.model.history.parameters())
    assert all(p.grad is None for p in nod.model.likelihood_head.parameters())
    second = Parameters.from_json(str(CONFIG_ROOT / "02_ppo_beta.json"))
    frozen = NODOpinionManager(second)
    assert frozen.load_checkpoint(nod.checkpoint_state(), load_optimizer=False)
    value = SimpleNamespace(barrier_contract={})
    _bind_frozen_nod(frozen, value, second)
    assert not any(p.requires_grad for p in frozen.model.parameters())
    assert len(value.barrier_contract["nod_sha256"]) == 64
    for key, tensor in nod.model.state_dict().items():
        torch.testing.assert_close(frozen.model.state_dict()[key], tensor, rtol=0, atol=0)
