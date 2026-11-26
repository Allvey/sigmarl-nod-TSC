"""Numerical checks of opt-in progressive control; no simulation."""
import pytest
import torch
from utilities.helper_training import Parameters
from utilities.nod_marl.dgppo import dgppo_advantage, opinion_alpha
from utilities.nod_marl.safety_value import SafetyValueManager
from tests.test_dgppo_opinion import sample


def run(task=2., rate=0., unsafe=False, weight=1.):
    td, state = sample()
    value = state['g'].clone()
    nxt = value + .05 * (10 * value.abs() + rate)
    if unsafe:
        state['g'][..., -2] = .1
    return dgppo_advantage(torch.full_like(td['advantage'], task), state, state,
        value, nxt, dt=.05, alpha=10., eps=.01, weight=weight,
        task_mode='progressive', progressive_scale=1.)


def test_progressive_retention_and_negative_advantages():
    safe, a = run(rate=-1.)
    mild, b = run(rate=1.)
    severe, c = run(rate=20.)
    assert (a['task_retention'] == 1).all()
    assert ((b['task_retention'] > 0) & (b['task_retention'] < 1)).all()
    assert (c['task_retention'] == 0).all()
    assert (safe > mild).all() and (mild > severe).all()
    negative, info = run(task=-2., rate=1.)
    assert torch.allclose(negative, -2. - info['penalty'])
    unsafe, info = run(rate=-1., unsafe=True)
    assert (info['task_retention'] == 0).all()
    assert (unsafe <= 0).all()


def test_zero_weight_and_unknown_successors_fall_back_to_task():
    result, _ = run(rate=20., weight=0.)
    assert (result == 2).all()
    td, state = sample()
    following = dict(state, valid=torch.zeros_like(state['valid']))
    result, _ = dgppo_advantage(td['advantage'], state, following, state['g'],
        state['g'], dt=.05, alpha=10., eps=.01, weight=1., task_mode='progressive')
    assert torch.equal(result, td['advantage'])


def test_continuous_deadzone_has_no_jump_and_keeps_endpoints():
    td, state = sample()
    context = td['agents', 'info', 'nod_actor_edge_context']
    for z, expected in [(0.,10.), (.1,10.), (.100001,10.+5e-6/.9),
                        (.55,12.5), (-.55,7.5), (1.,15.), (-1.,5.)]:
        context[..., -1] = z
        alpha, _ = opinion_alpha(td, state, state['g'], alpha=10., span=5.,
            deadzone=.1, deadzone_mode='continuous')
        assert float(alpha[0,0,0,1]) == pytest.approx(expected, abs=2e-6)
        assert (alpha[..., -2:] == 10).all()


def test_configuration_provenance_and_legacy_defaults():
    p = Parameters.from_json('configs/current/config_dgppo_nod_progressive_control.json')
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    assert p.nod_freeze_training and p.training_init_checkpoint.endswith('reward7.38')
    m = SafetyValueManager(p, 5, ('agents','observation'))
    assert m.barrier_contract['dgppo_task_mode'] == 'progressive'
    assert m.barrier_contract['opinion_deadzone_mode'] == 'continuous'
    old = Parameters.from_json('configs/current/error_scale_repeat/02_control.json')
    assert old.dgppo_task_mode == 'gated' and old.dgppo_opinion_deadzone_mode == 'hard'
    for key, value in [('dgppo_progressive_scale',0), ('dgppo_progressive_scale',float('nan')),
                       ('dgppo_opinion_deadzone_mode','invalid')]:
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(p.to_dict(), **{key:value}))
