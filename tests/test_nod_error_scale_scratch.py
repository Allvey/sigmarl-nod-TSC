"""Error-scale joint training and frozen control handoff, without simulation."""
from pathlib import Path
from types import SimpleNamespace

import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import _bind_frozen_nod
from utilities.nod_marl.trainer import NODOpinionManager


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/current/error_scale_scratch"


def test_error_scale_scratch_has_no_pretrained_dependency_and_preserves_seed():
    first = Parameters.from_json(str(CONFIG_ROOT / "01_joint.json"))
    second = Parameters.from_json(str(CONFIG_ROOT / "02_control.json"))
    for key in ("training_init_checkpoint", "nod_candidate_init_checkpoint",
                "nod_control_checkpoint", "nod_future_calibration_file"):
        assert not getattr(first, key)
    assert not first.is_load_model and not first.is_continue_train
    assert first.safety_training_mode == "scratch" and not first.nod_freeze_training
    assert first.nod_training_mode == "joint" and first.nod_update_interval == 1
    assert first.lr == 2e-4 and first.nod_message_lr == 2e-4
    assert second.training_init_checkpoint == first.where_to_save + "final"
    assert second.safety_training_mode == "finetune" and second.nod_freeze_training
    assert second.safety_barrier_warmup_batches == 20
    assert second.lr == second.safety_finetune_lr == second.nod_message_lr == 5e-5
    assert second.safety_finetune_target_kl == 0.01
    assert (first.n_iters, second.n_iters) == (250, 70)
    assert first.total_frames + second.total_frames == 1310720
    for p in (first, second):
        assert p.seed == 5080868027432654403
        assert p.nod_label_mode == "responsibility" and p.nod_fixed_evidence_mapping
        assert p.nod_evidence_mode == p.nod_evidence_target == "geometric"
        assert p.nod_variance_mode == "learned"
        assert p.nod_variance_supervision == "evidence_error"
        assert p.nod_nll_weight == p.nod_calibration_weight == 1
        assert p.nod_evidence_gate_mode == "fixed"
        assert p.dgppo_opinion_alpha and p.dgppo_alpha_gain == 1
        assert p.dgppo_opinion_deadzone == 0.1 and p.dgppo_task_mode == "gated"
        assert not p.is_add_noise
        assert "nod_error_scale_scratch/seed5080868027432654403/" in p.where_to_save
        assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_joint_error_scale_learns_then_hands_off_exact_frozen_nod():
    first = Parameters.from_json(str(CONFIG_ROOT / "01_joint.json"))
    nod = NODOpinionManager(first)
    pair = torch.zeros(1, 4, 2, 1, 20)
    mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    outputs, _ = nod.model.forward_sequence(
        pair, mask, torch.ones(1, 4, 2, dtype=torch.long),
        torch.ones_like(mask, dtype=torch.long), opinion_mask=mask,
        evidence_override=torch.ones_like(mask, dtype=torch.float32),
    )
    loss, *_ = nod._losses(
        outputs, {"label": torch.zeros_like(outputs["rho"]), "valid": mask}, mask,
    )
    loss.backward()
    assert nod.model.likelihood_head[-1].bias.grad[2].abs() > 0
    assert nod.model.raw_risk_weights.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in nod.model.history.parameters())
    second = Parameters.from_json(str(CONFIG_ROOT / "02_control.json"))
    frozen = NODOpinionManager(second)
    assert frozen.load_checkpoint(nod.checkpoint_state(), load_optimizer=False)
    value = SimpleNamespace(barrier_contract={})
    _bind_frozen_nod(frozen, value, second)
    assert not any(p.requires_grad for p in frozen.model.parameters())
    assert len(value.barrier_contract["nod_sha256"]) == 64
    for key, tensor in nod.model.state_dict().items():
        torch.testing.assert_close(frozen.model.state_dict()[key], tensor, rtol=0, atol=0)
