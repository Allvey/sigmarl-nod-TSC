"""Profile checks with synthetic tensors; no training or environment rollout."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.dgppo import prepare_dgppo_advantage
from utilities.nod_marl.safety_value import SafetyValueManager


def test_profiles_resolve_and_round_trip():
    old = Parameters.from_json('configs/archive/dgppo_history/config_dgppo_minimal.json')
    assert old.ppo_training_profile == 'current'
    assert (old.num_epochs, old.lr, old.lmbda, old.clip_epsilon) == (15, .0003, .95, .25)
    original = Parameters.from_dict(dict(old.to_dict(), ppo_training_profile='original'))
    assert (original.num_epochs, original.lr, original.lmbda, original.clip_epsilon) == (60, .0002, .9, .2)
    assert Parameters.from_dict(original.to_dict()).to_dict() == original.to_dict()
    assert Parameters(num_epochs=7, lr=.0001).num_epochs == 7
    with pytest.raises(ValueError, match='ppo_training_profile'):
        Parameters(ppo_training_profile='unknown')


def test_matched_configs_and_checkpoint_metadata():
    task = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_task_only.json')
    safe = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo.json')
    assert {k for k in task.to_dict() if task.to_dict()[k] != safe.to_dict()[k]} == {'dgppo_weight', 'where_to_save'}
    assert task.dgppo_weight == 0 and safe.dgppo_weight == 1
    assert not task.is_load_model and not safe.is_load_model
    old = SafetyValueManager(Parameters.from_json('configs/archive/dgppo_history/config_dgppo_minimal.json'), 10, ('agents', 'observation'))
    new = SafetyValueManager(safe, 10, ('agents', 'observation'))
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    assert 'ppo_training_profile' not in old.barrier_contract
    assert new.barrier_contract['normalization'] == 'raw_task_GAE_including_warmup'
    assert new.barrier_contract['task_ppo']['num_epochs'] == 60


@pytest.mark.parametrize('profile', ['current', 'original'])
@pytest.mark.parametrize('task_mode', ['gated', 'additive'])
@pytest.mark.parametrize('phase', ['warmup', 'weight_zero', 'disabled', 'active_safe', 'active_unsafe'])
def test_advantage_scaling_is_consistent_across_phases(profile, task_mode, phase):
    p = Parameters.from_dict(dict(Parameters.from_json('configs/archive/dgppo_history/config_dgppo_minimal.json').to_dict(),
                                  ppo_training_profile=profile, dgppo_task_mode=task_mode))
    p.dgppo_weight = 0. if phase == 'weight_zero' else 1.
    p.is_using_safety_constraint = phase != 'disabled'
    g = torch.full((1, 3, 1, 3), -.5)
    if phase == 'active_unsafe':
        g[0, 0, 0, 0] = .4
    state = dict(g=g, valid=torch.ones_like(g, dtype=torch.bool),
                 ego_gen=torch.zeros_like(g, dtype=torch.long), other_gen=torch.zeros_like(g, dtype=torch.long))
    manager = SimpleNamespace(parameters=p, enabled=True, rollouts=0,
                              barrier_fit_batches=0 if phase == 'warmup' else 20,
                              state=lambda td: state, model=lambda s: s['g'])
    raw = torch.tensor([2., 4., 6.]).reshape(1, 3, 1, 1)
    td = TensorDict({'advantage': raw.clone(),
                     'next': TensorDict({'done': torch.zeros(1, 3, 1, dtype=torch.bool)}, [1, 3])}, [1, 3])
    metrics = prepare_dgppo_advantage(manager, td, 'advantage')
    expected = raw.clone() if profile == 'original' else (raw - raw.mean(1, keepdim=True)) / (
        raw.std(1, unbiased=False, keepdim=True) + 1e-8)
    if phase == 'active_unsafe':
        retained = expected[0, 0, 0, 0].item() if task_mode == 'additive' else 0.
        expected[0, 0, 0, 0] = retained - (p.dgppo_alpha * .4 + p.dgppo_eps)
    torch.testing.assert_close(td['advantage'], expected)
    torch.testing.assert_close(td['agents', 'barrier_task_advantage'], raw)
    if profile == 'original':
        assert metrics['task_normalization_delta_abs'] == 0


def test_additive_config_and_contract():
    old = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo_finetune.json')
    new = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo_additive.json')
    assert old.dgppo_task_mode == 'gated' and new.dgppo_task_mode == 'additive'
    assert {k for k in old.to_dict() if old.to_dict()[k] != new.to_dict()[k]} == {
        'dgppo_task_mode', 'where_to_save'}
    assert Parameters.from_dict(new.to_dict()).dgppo_task_mode == 'additive'
    with pytest.raises(ValueError, match='dgppo_task_mode'):
        Parameters(dgppo_task_mode='invalid')
    a = SafetyValueManager(old, 10, ('agents', 'observation'))
    b = SafetyValueManager(new, 10, ('agents', 'observation'))
    assert a.contract == b.contract and a.loss_contract == b.loss_contract
    assert 'dgppo_task_mode' not in a.barrier_contract
    assert b.barrier_contract['advantage'] == 'full_task_minus_risk_rate'
