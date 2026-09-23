"""Offline checks for opinion-conditioned DGPPO; no environment rollout."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.dgppo import (
    constraint_alpha,
    opinion_alpha,
    dgppo_advantage,
    prepare_dgppo_advantage,
)
from utilities.nod_marl.safety_value import SafetyValueManager


def params():
    return Parameters.from_json('config_dgppo_nod_opinion_finetune.json')


def sample():
    ids = torch.tensor([[[[1, 2], [0, 2], [0, 1]]]])
    generation = torch.tensor([2, 4, 6])
    context = torch.zeros(1, 1, 3, 2, 72)
    context[..., -1] = torch.tensor([[-1., 1.], [0., .5], [-.5, 0.]])
    info = TensorDict(dict(nod_actor_edge_context=context,
        nod_actor_edge_mask=torch.ones_like(ids, dtype=torch.bool),
        nod_edge_mask=torch.ones_like(ids, dtype=torch.bool),
        nod_actor_context_ready=torch.ones(1, 1, 3, 1, dtype=torch.bool),
        nod_neighbor_indices=ids, nod_neighbor_generation=generation[ids]), [1, 1, 3])
    g = -torch.ones(1, 1, 3, 5)
    ego = generation.view(1, 1, 3, 1).expand_as(g)
    other = torch.cat([generation.view(1, 1, 1, 3).expand(1, 1, 3, 3),
                       generation.view(1, 1, 3, 1).expand(1, 1, 3, 2)], -1)
    valid = torch.ones_like(g, dtype=torch.bool)
    valid[..., :3] &= ~torch.eye(3, dtype=torch.bool)
    state = dict(g=g, ego_gen=ego, other_gen=other, valid=valid)
    td = TensorDict({'agents': TensorDict({'info': info}, [1, 1, 3]),
        'advantage': torch.ones(1, 1, 3, 1),
        'next': TensorDict({'done': torch.zeros(1, 1, 1, dtype=torch.bool)}, [1, 1])}, [1, 1])
    return td, state


def mapped(td, state, value=None, span=5.):
    return opinion_alpha(td, state, state['g'] if value is None else value, alpha=10., span=span)


def advantage(td, state, nxt, alpha):
    return dgppo_advantage(td['advantage'], state, state, state['g'], nxt,
                          dt=.05, alpha=alpha, eps=.01, weight=1.)


def test_config_roundtrip_and_matched_control():
    p = params()
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    control = Parameters.from_json('config_dgppo_nod_fixed_control_finetune.json')
    diff = {k for k, v in p.to_dict().items() if control.to_dict()[k] != v}
    assert diff == {'dgppo_opinion_alpha', 'where_to_save'}
    assert p.nod_freeze_training and p.training_init_checkpoint.endswith('reward7.33')
    assert not Parameters.from_json('config_dgppo_nod_fixed_finetune.json').dgppo_opinion_alpha


def test_candidate_only_and_gain1_control_configs_are_staged():
    candidate = Parameters.from_json('config_dgppo_nod_candidate_only.json')
    assert candidate.nod_training_mode == 'candidate_only'
    assert candidate.nod_freeze_training
    assert not candidate.dgppo_opinion_alpha
    assert candidate.nod_update_interval == 1
    control = Parameters.from_json('config_dgppo_nod_gain1_control_finetune.json')
    assert control.nod_training_mode == 'joint'
    assert control.nod_freeze_training
    assert control.dgppo_opinion_alpha
    assert control.dgppo_alpha_gain == 1
    assert control.dgppo_opinion_deadzone == pytest.approx(.1)
    assert control.nod_actor_opinion_mode == 'online'
    fixed = Parameters.from_json('config_dgppo_nod_ablation_fixed_alpha.json')
    neutral = Parameters.from_json('config_dgppo_nod_ablation_neutral_z.json')
    assert not fixed.dgppo_opinion_alpha
    assert fixed.nod_actor_opinion_mode == 'online'
    assert not neutral.dgppo_opinion_alpha
    assert neutral.nod_actor_opinion_mode == 'neutral'
    ignored = {'where_to_save', 'dgppo_opinion_alpha', 'nod_actor_opinion_mode'}
    assert {key for key, value in control.to_dict().items()
            if fixed.to_dict()[key] != value} <= ignored
    assert {key for key, value in control.to_dict().items()
            if neutral.to_dict()[key] != value} <= ignored
    road = Parameters.from_json('config_dgppo_nod_gain1_road_safe_finetune.json')
    assert road.n_iters == 30 and road.safety_barrier_warmup_batches == 10
    assert road.dgppo_road_alpha_safe == pytest.approx(7.5)
    assert road.dgppo_road_alpha_recovery == pytest.approx(15.)
    assert road.training_init_checkpoint.endswith('reward7.18')


def test_candidate_only_requires_frozen_initialized_behavior_nod():
    base = params().to_dict()
    for change in (
        dict(nod_freeze_training=False),
        dict(training_init_checkpoint=None),
        dict(nod_observation_mode='legacy_paths'),
    ):
        with pytest.raises(ValueError):
            Parameters.from_dict(
                dict(base, nod_training_mode='candidate_only', **change)
            )


@pytest.mark.parametrize('change', [dict(dgppo_alpha_span=-1.), dict(dgppo_alpha_span=float('nan')),
    dict(dgppo_alpha_span=10.), dict(dgppo_alpha_span=9.9, dt=.06),
    dict(dgppo_opinion_alpha='yes'), dict(is_using_safety_constraint=False),
    dict(is_using_nod_actor=False), dict(is_using_safety_value_shadow=False)])
def test_invalid_configuration(change):
    with pytest.raises(ValueError): Parameters.from_dict(dict(params().to_dict(), **change))


def test_alignment_permutation_and_only_current_cache():
    td, state = sample()
    alpha, info = mapped(td, state)
    torch.testing.assert_close(alpha[0, 0, 0], torch.tensor([10., 5., 15., 10., 10.]))
    assert info['applied'].sum() == 6
    permuted = td.clone(); cache = permuted['agents', 'info']
    for key in ['nod_neighbor_indices', 'nod_neighbor_generation', 'nod_actor_edge_mask', 'nod_edge_mask']:
        cache[key] = cache[key].flip(-1)
    cache['nod_actor_edge_context'] = cache['nod_actor_edge_context'].flip(-2)
    permuted['next', 'agents', 'info', 'nod_actor_edge_context'] = torch.full_like(cache['nod_actor_edge_context'], 999.)
    torch.testing.assert_close(mapped(permuted, state)[0], alpha)


@pytest.mark.parametrize('missing', ['ready', 'generation', 'nan', 'range', 'mask', 'physical_mask',
                                    'duplicate', 'index', 'absent', 'all_info'])
def test_unusable_opinion_falls_back_to_baseline(missing):
    td, state = sample(); info = td['agents', 'info']
    if missing == 'ready': info['nod_actor_context_ready'].zero_()
    elif missing == 'generation': info['nod_neighbor_generation'] += 1
    elif missing in ('nan', 'range'): info['nod_actor_edge_context'][..., -1] = float('nan') if missing == 'nan' else 2.
    elif missing == 'mask': info['nod_actor_edge_mask'].zero_()
    elif missing == 'physical_mask': info['nod_edge_mask'].zero_()
    elif missing == 'duplicate': info['nod_neighbor_indices'].fill_(1)
    elif missing == 'index': info['nod_neighbor_indices'].fill_(-1)
    elif missing == 'absent': info.del_('nod_actor_edge_context')
    else: td['agents'].del_('info')
    alpha, details = mapped(td, state)
    assert not details['applied'].any()
    torch.testing.assert_close(alpha, torch.full_like(alpha, 10.))


@pytest.mark.parametrize('g,v', [(-1.,0.), (-1.,.2), (.1,-1.), (.1,.2), (float('nan'),-1.), (-1.,float('nan'))])
def test_unsafe_or_nonfinite_pair_cannot_be_relaxed(g, v):
    td, state = sample(); value = state['g'].clone()
    state['g'][..., 0, 2] = g; value[..., 0, 2] = v
    alpha, details = mapped(td, state, value)
    assert alpha[..., 0, 2] == 10 and not details['applied'][..., 0, 2]
    assert (alpha[..., -2:] == 10).all()
    assert alpha[..., 0, 1] == 5  # Another pair remains independent.


def test_neutral_and_zero_span_exactly_recover_fixed_advantage():
    td, state = sample(); nxt = state['g'] + .5
    torch.testing.assert_close(advantage(td, state, nxt, mapped(td,state,span=0.)[0])[0],
                               advantage(td,state,nxt,10.)[0], rtol=0,atol=0)
    td['agents','info','nod_actor_edge_context'][..., -1] = 0.
    torch.testing.assert_close(advantage(td,state,nxt,mapped(td,state)[0])[0],
                               advantage(td,state,nxt,10.)[0],rtol=0,atol=0)


def test_opinion_changes_pair_gate_and_actor_gradient_without_nod_gradient():
    td, state = sample(); nxt = state['g'].clone(); nxt[...,0,1] = -.5
    results = []
    for z in [-1.,1.]:
        context = td['agents','info','nod_actor_edge_context'].clone().detach()
        context[..., -1] = 0.; context[...,0,0,-1] = z
        td['agents','info','nod_actor_edge_context'] = context.requires_grad_()
        alpha, _ = mapped(td,state)
        adjusted, info = advantage(td,state,nxt,alpha)
        assert not alpha.requires_grad and not adjusted.requires_grad
        loc = torch.zeros(1,1,3,1,requires_grad=True)
        log_prob = torch.distributions.Normal(loc,1.).log_prob(torch.ones_like(loc))
        ratio = (log_prob - log_prob.detach()).exp()
        (-(ratio * adjusted).sum()).backward()  # PPO surrogate at ratio=1.
        assert context.grad is None
        results.append((adjusted,info,loc.grad.clone()))
    low,high = results
    assert low[1]['violation'][...,0] and not high[1]['violation'][...,0]
    assert low[0][...,0,:] < 0 and high[0][...,0,:] > 0
    assert low[2][...,0,:] > 0 and high[2][...,0,:] < 0
    assert (low[1]['delta'] != high[1]['delta']).sum() == 1
    torch.testing.assert_close(low[0][...,1:,:],high[0][...,1:,:])


def test_prepare_reports_final_advantage_effect_and_dominant_road_suppression():
    for dominant_road in [False,True]:
        td, state = sample(); following = {k:v.clone() for k,v in state.items()}
        following['g'][...,0,1] = -.5
        if dominant_road: following['g'][...,0,-2] = 1.
        manager = SimpleNamespace(parameters=params(),enabled=True,barrier_fit_batches=99,rollouts=1,
            state=lambda x: following if x.get('done',default=None) is not None else state,
            model=lambda s:s['g'])
        metrics = prepare_dgppo_advantage(manager,td,'advantage')
        assert metrics['opinion_c_delta_abs'] > 0
        assert (metrics['opinion_advantage_changed_count'] > 0) == (not dominant_road)
        assert (metrics['opinion_advantage_delta_abs'] > 0) == (not dominant_road)
        assert metrics['opinion_alpha_min'] == 5 and metrics['opinion_alpha_max'] == 15


def test_new_contract_reuses_value_weights_and_restarts_warmup(tmp_path):
    old = SafetyValueManager(Parameters.from_json('config_dgppo_nod_fixed_finetune.json'),5,('agents','observation'))
    new = SafetyValueManager(params(),5,('agents','observation'))
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    old.barrier_fit_batches = 70
    path=tmp_path/'value.pth'; torch.save(old.checkpoint_state(),path)
    assert new.load_if_available(path,load_optimizer=False)
    assert new.barrier_fit_batches == 0 and new.barrier_contract['opinion_controls_alpha']
    for a,b in zip(old.model.parameters(),new.model.parameters()): torch.testing.assert_close(a,b)


def test_warmup_keeps_task_and_reports_inactive_opinion_effect():
    td,state=sample(); original=td['advantage'].clone()
    manager=SimpleNamespace(parameters=params(),enabled=True,barrier_fit_batches=0,rollouts=1)
    metrics=prepare_dgppo_advantage(manager,td,'advantage')
    torch.testing.assert_close(td['advantage'],original,rtol=0,atol=0)
    assert metrics['opinion_alpha_enabled'] == 1 and metrics['opinion_advantage_changed_count'] == 0
    assert not metrics['barrier_ready']


def test_respawned_pair_cannot_contribute_opinion_penalty():
    td,state=sample(); nxt=state['g'].clone(); nxt[...,0,1]=-.5
    following={k:v.clone() for k,v in state.items()}
    following['other_gen'][...,0,1]+=1
    alpha,_=mapped(td,state)
    changed,info=dgppo_advantage(td['advantage'],state,following,state['g'],nxt,
                               dt=.05,alpha=alpha,eps=.01,weight=1.)
    fixed,_=dgppo_advantage(td['advantage'],state,following,state['g'],nxt,
                           dt=.05,alpha=10.,eps=.01,weight=1.)
    assert not info['valid'][...,0,1]
    torch.testing.assert_close(changed,fixed,rtol=0,atol=0)


@pytest.mark.parametrize('gain', [0., -1., float('nan'), float('inf'), True])
def test_invalid_opinion_gain(gain):
    with pytest.raises(ValueError):
        Parameters.from_dict(dict(params().to_dict(), dgppo_alpha_gain=gain))


@pytest.mark.parametrize('deadzone', [-.1, 1., float('nan'), float('inf')])
def test_invalid_opinion_deadzone(deadzone):
    with pytest.raises(ValueError):
        Parameters.from_dict(
            dict(params().to_dict(), dgppo_opinion_deadzone=deadzone)
        )


@pytest.mark.parametrize('mode', ['zero', '', None, 1])
def test_invalid_actor_opinion_mode(mode):
    with pytest.raises(ValueError):
        Parameters.from_dict(dict(params().to_dict(), nod_actor_opinion_mode=mode))


@pytest.mark.parametrize('safe,recovery', [
    (0., 15.), (10.1, 15.), (7.5, 9.9), (7.5, 20.),
    (float('nan'), 15.), (7.5, float('inf')), (True, 15.),
])
def test_invalid_road_alpha(safe, recovery):
    with pytest.raises(ValueError):
        Parameters.from_dict(dict(
            params().to_dict(),
            dgppo_road_alpha_safe=safe,
            dgppo_road_alpha_recovery=recovery,
        ))


def test_road_alpha_is_strict_when_safe_and_strong_during_recovery():
    value = -torch.ones(1, 1, 2, 4)
    value[..., 1, -2] = .2
    alpha = constraint_alpha(
        value, alpha=10., road_safe_alpha=7.5, road_recovery_alpha=15.
    )
    assert alpha[..., 0, -2].item() == pytest.approx(7.5)
    assert alpha[..., 1, -2].item() == pytest.approx(15.)
    assert (alpha[..., :-2] == 10).all() and (alpha[..., -1] == 10).all()


def test_road_alpha_contract_reuses_value_and_restarts_barrier_warmup(tmp_path):
    old_p = Parameters.from_json('config_dgppo_nod_gain1_control_finetune.json')
    new_p = Parameters.from_json('config_dgppo_nod_gain1_road_safe_finetune.json')
    old = SafetyValueManager(old_p, 5, ('agents', 'observation'))
    new = SafetyValueManager(new_p, 5, ('agents', 'observation'))
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    assert new.barrier_contract['road_alpha_safe'] == pytest.approx(7.5)
    assert new.barrier_contract['road_alpha_recovery'] == pytest.approx(15.)
    old.barrier_fit_batches = 70
    path = tmp_path / 'value.pth'
    torch.save(old.checkpoint_state(), path)
    assert new.load_if_available(path, load_optimizer=False)
    assert new.barrier_fit_batches == 0
    for a, b in zip(old.model.parameters(), new.model.parameters()):
        torch.testing.assert_close(a, b)


def test_opinion_deadzone_uses_fixed_alpha_without_hiding_available_opinion():
    td, state = sample()
    context = td['agents', 'info', 'nod_actor_edge_context']
    context[..., -1] = 0.0
    context[..., 0, 0, -1] = 0.08
    context[..., 0, 1, -1] = 0.12
    alpha, details = opinion_alpha(
        td,
        state,
        state['g'],
        alpha=10.0,
        span=5.0,
        gain=1.0,
        deadzone=0.1,
    )
    assert details['available'][..., 0, 1]
    assert not details['applied'][..., 0, 1]
    assert alpha[..., 0, 1] == 10.0
    assert details['applied'][..., 0, 2]
    assert alpha[..., 0, 2] == pytest.approx(10.6)


def test_explicit_interaction_mask_prevents_stale_opinion_alpha():
    td, state = sample()
    active = torch.ones_like(
        td['agents', 'info', 'nod_actor_edge_mask'], dtype=torch.bool
    )
    active[..., 0, 0] = False
    td['agents', 'info', 'nod_opinion_active'] = active
    alpha, details = mapped(td, state)
    assert not details['available'][..., 0, 1]
    assert not details['applied'][..., 0, 1]
    assert alpha[..., 0, 1] == 10.0


def test_gain_amplifies_weak_opinions_but_preserves_bounds_and_fallbacks():
    td, state = sample()
    context = td['agents', 'info', 'nod_actor_edge_context']
    context[..., -1] = torch.tensor([[-.072, .072], [-.8, .8], [0., .1]])
    original = context.clone()
    unit, _ = opinion_alpha(td, state, state['g'], alpha=10., span=5., gain=1.)
    torch.testing.assert_close(unit, mapped(td, state)[0], rtol=0, atol=0)
    amplified, info = opinion_alpha(td, state, state['g'], alpha=10., span=5., gain=2.)
    torch.testing.assert_close(amplified[0, 0, 0], torch.tensor([10., 9.28, 10.72, 10., 10.]))
    assert amplified[0, 0, 1, 0] == 5 and amplified[0, 0, 1, 2] == 15
    assert amplified[0, 0, 2, 0] == 10
    assert amplified.min() >= 5 and amplified.max() <= 15
    torch.testing.assert_close(context, original, rtol=0, atol=0)
    assert not info['mapped_opinions'].requires_grad
    state['g'][..., 0, 2] = .1
    td['agents', 'info', 'nod_actor_context_ready'][..., 1, :] = False
    fallback, _ = opinion_alpha(td, state, state['g'], alpha=10., span=5., gain=2.)
    assert fallback[..., 0, 2] == 10
    assert (fallback[..., 1, :] == 10).all()


def test_gain_reaches_prepared_advantage_and_diagnostics():
    td, state = sample()
    td['agents', 'info', 'nod_actor_edge_context'][..., -1] = 0.
    td['agents', 'info', 'nod_actor_edge_context'][..., 0, 0, -1] = -.072
    following = {k: v.clone() for k, v in state.items()}
    following['g'][..., 0, 1] = -.52  # C=-.04 at gain=1; C=.32 at gain=2.
    p = params()
    manager = SimpleNamespace(parameters=p, enabled=True, barrier_fit_batches=99, rollouts=1,
        state=lambda x: following if x.get('done', default=None) is not None else state,
        model=lambda s: s['g'])
    baseline = td.clone()
    prepare_dgppo_advantage(manager, baseline, 'advantage')
    p.dgppo_alpha_gain = 2.
    metrics = prepare_dgppo_advantage(manager, td, 'advantage')
    assert baseline['advantage'][..., 0, :] > 0 and td['advantage'][..., 0, :] < 0
    assert metrics['opinion_alpha_gain'] == 2 and metrics['opinion_advantage_changed_count'] == 1
    assert metrics['opinion_mapped_z_mean'] == pytest.approx(2 * metrics['opinion_z_mean'])
    assert metrics['opinion_alpha_saturation_rate'] == 0


def test_gain_config_and_checkpoint_contract(tmp_path):
    p = Parameters.from_json('config_dgppo_nod_opinion_gain2_finetune.json')
    assert p.dgppo_alpha_gain == 2 and p.nod_freeze_training
    assert p.training_init_checkpoint.endswith('reward6.98')
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    old_p = params()
    assert old_p.dgppo_alpha_gain == 1
    old = SafetyValueManager(old_p, 5, ('agents', 'observation'))
    new = SafetyValueManager(p, 5, ('agents', 'observation'))
    assert 'alpha_gain' not in old.barrier_contract  # Old gain=1 contracts still match.
    assert new.barrier_contract['alpha_gain'] == 2
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    old.barrier_fit_batches = 50
    path = tmp_path / 'value.pth'; torch.save(old.checkpoint_state(), path)
    assert new.load_if_available(path, load_optimizer=True)
    assert new.barrier_fit_batches == 0
    for a, b in zip(old.model.parameters(), new.model.parameters()): torch.testing.assert_close(a, b)
    same = SafetyValueManager(old_p, 5, ('agents', 'observation'))
    assert same.load_if_available(path, load_optimizer=True)
    assert same.barrier_fit_batches == 50
