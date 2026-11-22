"""Original shared-A dynamics, causal replay and policy semantic isolation."""

import pytest
import torch

from test_nod_evidence_gate import make_actor, cached_data
from utilities.helper_training import Parameters
from utilities.mappo_cavs import _load_policy_checkpoint
from utilities.nod_marl.opinion import control_attentions, kl_proximal_update
from utilities.nod_marl.policy import (
    NOD_ACTOR_EDGE_CONTEXT_KEY, NOD_EVIDENCE_GATE_CONTEXT_KEY, NOD_ACTOR_MESSAGE_KEY,
)


CONFIG = 'configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_sequence_control.json'


def test_shared_attention_preserves_original_equation_and_full_implicit_gradient():
    attention = torch.tensor([.6], dtype=torch.float64, requires_grad=True)
    base = torch.tensor([.2], dtype=torch.float64)
    current, evidence = control_attentions(base, attention, torch.tensor([True]), 'attention')
    assert current is evidence
    kwargs = dict(bifurcation_gain=2., observation_weight=1., kl_weight=5.,
                  max_iterations=64, tolerance=0.)
    previous = torch.tensor([.1], dtype=torch.float64)
    e = torch.tensor([.8], dtype=torch.float64)
    root = kl_proximal_update(previous, current, e, torch.zeros_like(e),
                              torch.ones_like(e), torch.ones_like(e),
                              evidence_attention=evidence, **kwargs)
    original = kl_proximal_update(previous, attention, e, torch.zeros_like(e),
                                  torch.ones_like(e), torch.ones_like(e), **kwargs)
    torch.testing.assert_close(root, original, rtol=0, atol=0)
    gradient = torch.autograd.grad(root.sum(), attention)[0]
    curvature = 5./(1.-root.square()) + 1. - attention*2.*(1.-torch.tanh(2.*root).square()) + attention
    expected = (torch.tanh(2.*root) + e - root) / curvature
    torch.testing.assert_close(gradient, expected)
    shift = 1e-5
    def solve(a):
        return kl_proximal_update(previous, a, e, torch.zeros_like(e),
                                  torch.ones_like(e), torch.ones_like(e), **kwargs)
    numeric = (solve(attention.detach()+shift) - solve(attention.detach()-shift))/(2*shift)
    torch.testing.assert_close(gradient, numeric, rtol=1e-5, atol=1e-7)


def test_invalid_interactions_keep_physical_prior():
    prior = torch.tensor([.2, .7])
    attention, evidence = control_attentions(prior, torch.tensor([.5, 0.]),
                                            torch.tensor([True, False]), 'attention')
    torch.testing.assert_close(attention, torch.tensor([.5, .7]))
    assert attention is evidence


def test_online_model_uses_learned_attention_in_both_terms_and_actor_replays_it():
    _, module, manager, _ = make_actor(target='attention')
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(.7)
    data, out = cached_data(manager, module)
    valid = out['learning_valid']
    assert valid.any()
    torch.testing.assert_close(out['attention'][valid], out['evidence_attention'][valid])
    assert (out['attention'][valid] - out['gate_context'][..., -5][valid]).abs().max() > .01
    reference, _ = module.aggregator(data[NOD_ACTOR_EDGE_CONTEXT_KEY],
                                    data['agents', 'info', 'nod_actor_edge_mask'])
    cache = data[NOD_ACTOR_EDGE_CONTEXT_KEY].clone()
    module(data)
    torch.testing.assert_close(data[NOD_ACTOR_MESSAGE_KEY], reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(data[NOD_ACTOR_EDGE_CONTEXT_KEY], cache, rtol=0, atol=0)
    assert manager.online_state is None
    # Isolate the Actor's A coordinate by ignoring z in this probe aggregator.
    class AttentionProbe(torch.nn.Module):
        def forward(self, context, mask):
            return context[..., -2].mean(-1, keepdim=True).expand(*context.shape[:-2], 4), mask.float()
    module.aggregator = AttentionProbe()
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(-.4)
    module(data)
    gate_context = data[NOD_EVIDENCE_GATE_CONTEXT_KEY]
    weight = module.evidence_gate(gate_context)
    expected, _ = control_attentions(gate_context[..., -5], weight, gate_context[..., -1], 'attention')
    torch.testing.assert_close(data[NOD_ACTOR_MESSAGE_KEY][..., 0], expected.mean(-1))
    torch.testing.assert_close(data[NOD_ACTOR_EDGE_CONTEXT_KEY], cache, rtol=0, atol=0)


def test_unified_attention_policy_is_not_interchangeable_with_beta(tmp_path):
    fixed, _, _, _ = make_actor('fixed')
    unified, module, _, p = make_actor(sequence_length=8, gate_only_iters=10, target='attention')
    p.training_init_checkpoint = str(tmp_path/'source')
    path = p.training_init_checkpoint+'_policy.pth'
    torch.save(fixed.state_dict(), path)
    assert _load_policy_checkpoint(path, unified, p, actor_base_observation_dim=5,
                                   use_nod_actor=True) == 'gate_initialized'
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, unified.state_dict()[key], rtol=0, atol=0)
    module.evidence_gate.training_batches.fill_(10)
    torch.save(unified.state_dict(), path)
    restored, restored_module, _, _ = make_actor(sequence_length=8, gate_only_iters=10, target='attention')
    p.is_load_model = True
    assert _load_policy_checkpoint(path, restored, p, actor_base_observation_dim=5,
                                   use_nod_actor=True) == 'loaded'
    assert restored_module.evidence_gate.training_batches.item() == 10
    beta, _, _, _ = make_actor(sequence_length=8, gate_only_iters=10)
    with pytest.raises(RuntimeError):
        _load_policy_checkpoint(path, beta, p, actor_base_observation_dim=5, use_nod_actor=True)
    torch.save(beta.state_dict(), path)
    with pytest.raises(RuntimeError):
        _load_policy_checkpoint(path, unified, p, actor_base_observation_dim=5, use_nod_actor=True)


def test_new_config_matches_split_branch_except_weight_role_and_output():
    p = Parameters.from_json(CONFIG)
    old = Parameters.from_json('configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json')
    a, b = p.to_dict(), old.to_dict()
    differences = {key for key in a if a[key] != b[key]}
    assert differences == {'nod_evidence_gate_target', 'where_to_save'}
    assert p.nod_evidence_gate_target == 'attention'
    assert p.nod_freeze_training and p.nod_fixed_variance_value == 1.
    assert p.nod_evidence_gate_sequence_length == 8 and p.nod_evidence_gate_only_iters == 10
    assert Parameters.from_dict(a).to_dict() == a
    for change in (dict(nod_evidence_gate_target='invalid'), dict(nod_evidence_gate_mode='fixed')):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(a, **change))
