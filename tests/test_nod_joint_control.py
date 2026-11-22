"""Co-adaptation gradients and checkpoint/score alignment without simulation."""

import json

import pytest
import torch
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss

from test_nod_evidence_gate import make_actor
from test_nod_evidence_gate_sequence import sequence_data
from utilities.helper_training import (
    Parameters, SaveData, find_the_highest_reward_among_all_models,
    delete_files_with_lower_mean_reward,
)
from utilities.nod_marl.checkpointing import snapshot_state, write_scored_checkpoint


CONFIG = 'configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_joint_control.json'


def test_joint_ppo_updates_message_and_attention_but_not_forecaster():
    torch.manual_seed(42)
    actor, module, manager, _ = make_actor(sequence_length=8, target='attention')
    data, _ = sequence_data(manager, module)
    with torch.no_grad():
        actor(data)
    critic = TensorDictModule(torch.nn.Linear(5, 1),
                             in_keys=[('agents', 'observation')],
                             out_keys=[('agents', 'state_value')])
    loss = ClipPPOLoss(actor=actor, critic=critic, normalize_advantage=False)
    loss.set_keys(action=('agents', 'action'), sample_log_prob=('agents', 'sample_log_prob'),
                  value=('agents', 'state_value'), advantage=('agents', 'advantage'),
                  value_target=('agents', 'value_target'))
    data.set(('agents', 'advantage'), torch.randn(1, 5, 2, 1))
    data.set(('agents', 'value_target'), torch.ones(1, 5, 2, 1))
    gate = list(module.evidence_gate.parameters())
    message = list(module.aggregator.parameters())
    ids = {id(p) for p in gate + message}
    optimizer = torch.optim.Adam([
        dict(params=[p for p in loss.parameters() if id(p) not in ids], lr=1e-5),
        dict(params=message, lr=3e-5), dict(params=gate, lr=5e-6),
    ])
    before_gate, before_message = snapshot_state(gate), snapshot_state(message)
    forecast = snapshot_state(manager.model.state_dict())
    losses = loss(data)
    (losses['loss_objective'] + losses['loss_critic']).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in message)
    assert all(p.grad is None for p in manager.model.parameters())
    optimizer.step()
    assert any(not torch.equal(p.detach(), before) for p, before in zip(gate, before_gate))
    assert any(not torch.equal(p.detach(), before) for p, before in zip(message, before_message))
    for key, value in manager.model.state_dict().items():
        torch.testing.assert_close(value, forecast[key], rtol=0, atol=0)


def test_scored_bundle_preserves_weights_buffers_and_optimizer_state_before_update(tmp_path):
    _, module, manager, _ = make_actor(sequence_length=8, target='attention')
    module.evidence_gate.training_batches.fill_(3)
    critic = torch.nn.Linear(2, 1)
    value = {'model': {'weight': torch.tensor([.4])}, 'rollouts': 20,
             'optimizer': {'state': {0: {'exp_avg': torch.tensor([.1])}}}}
    snapshot = snapshot_state(dict(policy=module.state_dict(), critic=critic.state_dict(),
                                   nod=manager.checkpoint_state(), safety_value=value))
    metadata = {'checkpoint_selection': {'raw_reward': 7.123, 'policy_update_batches': 3,
                                        'weights_timing': 'before_current_rollout_optimization'}}
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.add_(1.)
        module.evidence_gate.training_batches.add_(1)
        critic.weight.add_(1.)
        value['model']['weight'].add_(1.)
        value['optimizer']['state'][0]['exp_avg'].add_(1.)
    write_scored_checkpoint(tmp_path, 'reward7.12', snapshot, metadata)
    restored = torch.load(tmp_path/'reward7.12_policy.pth', map_location='cpu')
    assert restored['evidence_gate.training_batches'].item() == 3
    torch.testing.assert_close(restored['evidence_gate.net.2.bias'], torch.zeros(1))
    assert not torch.equal(torch.load(tmp_path/'reward7.12_critic.pth')['weight'], critic.weight)
    saved_value = torch.load(tmp_path/'reward7.12_safety_value.pth')
    torch.testing.assert_close(saved_value['model']['weight'], torch.tensor([.4]))
    torch.testing.assert_close(saved_value['optimizer']['state'][0]['exp_avg'], torch.tensor([.1]))
    assert json.loads((tmp_path/'reward7.12_data.json').read_text()) == metadata
    assert (tmp_path/'reward7.12_nod.pth').is_file()


def test_periodic_bundles_load_by_reward_and_survive_best_checkpoint_cleanup(tmp_path):
    snapshot = dict(policy={}, critic={}, nod={}, safety_value={})
    write_scored_checkpoint(tmp_path, 'reward7.10', snapshot, {})
    periodic = tmp_path/'iterations'/'iteration0025'
    write_scored_checkpoint(periodic, 'reward6.50', snapshot, {})
    write_scored_checkpoint(tmp_path, 'reward7.20', snapshot, {})
    p = Parameters.from_json(CONFIG)
    p.where_to_save = str(tmp_path)+'/'
    p.episode_reward_intermediate = 7.20
    delete_files_with_lower_mean_reward(p)
    assert find_the_highest_reward_among_all_models(str(tmp_path)) == 7.20
    assert find_the_highest_reward_among_all_models(str(periodic)) == 6.50
    assert (periodic/'reward6.50_policy.pth').is_file()
    assert not (tmp_path/'reward7.10_policy.pth').exists()


def test_new_configuration_and_checkpoint_provenance_roundtrip():
    p = Parameters.from_json(CONFIG)
    assert p.nod_evidence_gate_target == 'attention' and p.nod_freeze_training
    assert p.nod_evidence_gate_only_iters == 0
    assert p.nod_evidence_gate_sequence_length == 8
    assert (p.lr, p.nod_message_lr, p.nod_evidence_gate_lr) == (1e-5, 3e-5, 5e-6)
    # The original profile intentionally forces 60 epochs; this branch must
    # use explicit settings for its reduced 30-epoch update budget to survive.
    assert p.ppo_training_profile == 'current'
    assert p.num_epochs == 30 and p.safety_finetune_target_kl == .005
    assert p.training_checkpoint_preupdate and p.training_checkpoint_interval == 5
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    provenance = dict(scored_iteration=26, policy_update_batches=5, raw_reward=7.123)
    data = SaveData(p, checkpoint_selection=provenance)
    assert SaveData.from_dict(data.to_dict()).checkpoint_selection == provenance
    assert 'checkpoint_selection' not in SaveData(Parameters()).to_dict()
    for change in (dict(training_checkpoint_interval=-1),
                   dict(training_checkpoint_preupdate=False),
                   dict(nod_evidence_gate_target='evidence'), dict(is_continue_train=True)):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(p.to_dict(), **change))
