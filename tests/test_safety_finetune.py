"""Offline fine-tuning checks; no simulation or model training."""
from types import SimpleNamespace

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.nod_marl.finetune import actor_warmup_frozen, approximate_policy_kl
from utilities.nod_marl.safety_value import SafetyValueManager


def test_finetune_profile_and_phase_switch():
    p = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo_finetune.json')
    assert p.lr == p.safety_finetune_lr == 5e-5
    assert p.num_epochs == 60 and p.ppo_training_profile == 'original'
    assert p.n_iters == 70 and p.safety_barrier_warmup_batches == 20
    assert actor_warmup_frozen(p, SimpleNamespace(barrier_fit_batches=19))
    assert not actor_warmup_frozen(p, SimpleNamespace(barrier_fit_batches=20))
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    old = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo.json')
    assert old.safety_training_mode == 'scratch' and old.lr == .0002
    assert not actor_warmup_frozen(old, SimpleNamespace(barrier_fit_batches=0))


@pytest.mark.parametrize('change', [dict(training_init_checkpoint=None), dict(dgppo_weight=0),
                                  dict(safety_barrier_warmup_batches=0), dict(is_prb=True),
                                  dict(safety_finetune_target_kl=0), dict(safety_finetune_lr=float('nan'))])
def test_invalid_finetuning_configuration(change):
    p = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo_finetune.json')
    with pytest.raises(ValueError):
        Parameters.from_dict(dict(p.to_dict(), **change))


def test_sampled_kl_matches_categorical_expectation_and_detaches():
    old = torch.tensor([.8, .2], dtype=torch.float64)
    new = torch.tensor([.6, .4], dtype=torch.float64)
    # Five actions sampled with exactly the old-policy frequencies.
    ids = torch.tensor([0, 0, 0, 0, 1])
    log_new = new.log()[ids].requires_grad_()
    kl = approximate_policy_kl(log_new, old.log()[ids].unsqueeze(-1))
    torch.testing.assert_close(kl, (old * (old.log() - new.log())).sum())
    assert not kl.requires_grad and kl > .01
    assert approximate_policy_kl(old.log(), old.log()) == 0
    with pytest.raises(ValueError):
        approximate_policy_kl(torch.zeros(3), torch.zeros(2))


def test_value_target_unchanged_but_finetune_warmup_contract_distinct():
    p = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo_finetune.json')
    old = SafetyValueManager(Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_task_only.json'), 10, ('agents','observation'))
    new = SafetyValueManager(p, 10, ('agents','observation'))
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    assert new.barrier_contract['actor_warmup'] == 'frozen_until_value_fit_batches'
    assert new.barrier_contract != old.barrier_contract
