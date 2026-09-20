"""Test-only mixed control semantics, without simulation or training."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from utilities.testing_rule_policy import assign_rule_vehicles, rule_command, TestingRulePolicy as MixedPolicy


def test_rule_fraction_assignment_has_exact_counts_and_preserves_actor():
    import random
    weights={'yielding': .25, 'moderate': .5, 'non_yielding': .25}
    global_rng=random.getstate()
    assigned=assign_rule_vehicles(8, .5, weights, seed=123, actor_index=0)
    assert random.getstate()==global_rng
    assert len(assigned)==4 and 0 not in assigned
    assert [list(assigned.values()).count(name) for name in weights]==[1,2,1]
    assert assigned==assign_rule_vehicles(8, .5, weights, seed=123, actor_index=0)
    assert assign_rule_vehicles(8, 0, weights, seed=123)=={}


def test_rule_fraction_rounding_and_full_rule_population():
    weights={'yielding': .25, 'moderate': .5, 'non_yielding': .25}
    assigned=assign_rule_vehicles(8, .3, weights, seed=123)
    assert len(assigned)==2  # 30% of eight rounds to two slots (25% actual).
    assert list(assigned.values()).count('moderate')==1
    all_rule=assign_rule_vehicles(3, 1, weights, seed=123, actor_index=None)
    assert set(all_rule)=={0,1,2}


@pytest.mark.parametrize('n_agents,fraction,weights,seed,actor_index',[
    (8, -.1, {'yielding': .25, 'moderate': .5, 'non_yielding': .25}, 123, 0),
    (8, 1.1, {'yielding': .25, 'moderate': .5, 'non_yielding': .25}, 123, 0),
    (8, 1, {'yielding': .25, 'moderate': .5, 'non_yielding': .25}, 123, 0),
    (8, .5, {'yielding': .25, 'moderate': .5, 'non_yielding': .24}, 123, 0),
    (8, .5, {'yielding': .25, 'moderate': .75}, 123, 0),
    (8, .5, {'yielding': .25, 'moderate': .5, 'non_yielding': .25}, 123, 8),
])
def test_rule_fraction_rejects_invalid_setup(n_agents,fraction,weights,seed,actor_index):
    with pytest.raises(ValueError):
        assign_rule_vehicles(n_agents,fraction,weights,seed=seed,actor_index=actor_index)


def command(profile, distance=.5, neighbor_speed=-.6, was_yielding=False):
    return rule_command(torch.zeros(1,2), torch.zeros(1), torch.tensor([.6]),
        torch.tensor([[[0.,0.],[.5,0.],[1.5,0.]]]),
        torch.tensor([[[distance,0.]]]), torch.tensor([[[neighbor_speed,0.]]]),
        profile=profile,cruise_speed=.6,steering_limit=.6,wheelbase=.16,
        sensing_range=.8,dt=.05,was_yielding=torch.tensor([was_yielding]))


def test_profiles_react_to_conflict_but_not_outside_sensing():
    for profile in ['yielding','moderate']:
        action,yielding,_=command(profile)
        assert yielding.all() and action[0,0] < .6
    action,yielding,_=command('non_yielding')
    assert not yielding.any() and action[0,0] == pytest.approx(.6)
    for profile in ['yielding','moderate','non_yielding']:
        action,yielding,_=command(profile,distance=2.)
        assert not yielding.any() and action[0,0] == pytest.approx(.6)


def test_earlier_yielding_and_release_hysteresis():
    # Slow closing motion: proactive profile reacts earlier.
    assert command('yielding',distance=.7,neighbor_speed=.3)[1].all()
    assert not command('moderate',distance=.7,neighbor_speed=.3)[1].any()
    assert not command('moderate',distance=.23,neighbor_speed=.6)[1].any()
    assert not command('moderate',distance=.23,neighbor_speed=.6,was_yielding=True)[1].any()  # parallel motion must release


def test_turn_command_bounded_and_stopped_vehicle_checks_candidate_motion():
    for side in [-1.,1.]:
        action,yielding,_=rule_command(torch.zeros(1,2),torch.zeros(1),torch.zeros(1),
            torch.tensor([[[.1,side*.2],[.2,side*.4]]]),
            torch.empty(1,0,2),torch.empty(1,0,2),profile='moderate',cruise_speed=.6,
            steering_limit=.6,wheelbase=.16,sensing_range=.8,dt=.05,
            was_yielding=torch.zeros(1,dtype=torch.bool))
        assert 0 < action[0,0] <= .061 and 0 < side*action[0,1] <= .601
        assert not yielding.any()


class DummyActor(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def forward(self,td):
        self.calls+=1
        td['agents','action']=torch.full((1,3,2),.123)
        td['agents','sample_log_prob']=torch.zeros(1,3)
        return td


def scenario():
    agents=[]
    for x in [0.,.5,2.]:
        agents.append(SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[x,0.]]),
            vel=torch.tensor([[.6,0.]]),rot=torch.zeros(1,1)),
            dynamics=SimpleNamespace(l_f=.08,l_r=.08)))
    return SimpleNamespace(parameters=SimpleNamespace(is_testing_mode=True,is_continue_train=False,
        is_using_prioritized_marl=False,nod_sensing_range=.8,dt=.05),max_speed=1.,
        max_steering_angle=.6,n_agents=3,world=SimpleNamespace(agents=agents),
        nod_agent_generation=torch.zeros(1,3,dtype=torch.long),
        ref_paths_agent_related=SimpleNamespace(n_points_long_term=torch.full((1,3),2),is_loop=torch.zeros(1,3,dtype=torch.bool),long_term=torch.tensor(
            [[[[.1,0.],[.3,0.]],[[.6,0.],[.8,0.]],[[2.1,0.],[2.3,0.]]]])))


def test_wrapper_changes_only_rule_slots_and_blocks_training():
    s=scenario(); actor=DummyActor(); wrapper=MixedPolicy(actor,s,{1:'moderate'})
    td=wrapper(TensorDict({},[1]))
    assert actor.calls==1
    assert (td['agents','action'][:,[0,2]]==.123).all()
    assert td['agents','rule_controlled'].flatten().tolist()==[False,True,False]
    assert torch.isnan(td['agents','sample_log_prob'][:,1]).all()
    assert torch.isfinite(td['agents','sample_log_prob'][:,[0,2]]).all()
    assert 'A2: moderate' in wrapper.overlay_lines()[1]
    # Generation changes clear the yielding latch before the next command.
    wrapper.previous[1]=(torch.zeros(1,dtype=torch.long),torch.ones(1,dtype=torch.bool),torch.zeros(1))
    s.nod_agent_generation[:,1]+=1
    wrapper(TensorDict({},[1])); assert not wrapper.previous[1][1].any()
    s.parameters.is_testing_mode=False
    with pytest.raises(RuntimeError): wrapper(TensorDict({},[1]))
    with pytest.raises(ValueError): MixedPolicy(actor,s,{1:'moderate'})


def test_multiple_rule_slots_share_one_actor_call_and_keep_other_slots_intact(tmp_path):
    import csv
    actor=DummyActor()
    wrapper=MixedPolicy(actor,scenario(),{0:'yielding',1:'moderate'})
    td=wrapper(TensorDict({},[1]))
    assert actor.calls==1
    assert td['agents','rule_controlled'][0,:,0].tolist()==[True,True,False]
    assert torch.isnan(td['agents','sample_log_prob'][0,:2]).all()
    assert torch.isfinite(td['agents','sample_log_prob'][0,2])
    assert (td['agents','action'][0,2]==.123).all()
    td['agents','info','nod_ego_generation']=torch.zeros(1,3,dtype=torch.long)
    td['next','agents','info','testing_road_contact']=torch.zeros(1,3,dtype=torch.bool)
    td['next','agents','info','is_collision_with_agents']=torch.zeros(1,3,dtype=torch.bool)
    td['next','agents','info','testing_route_error']=torch.zeros(1,3)
    td['next','agents','info','nod_world_vel']=torch.zeros(1,3,2)
    csv_path=tmp_path/'rules.csv'
    wrapper.save_diagnostics(torch.stack([td],1),csv_path)
    rows=list(csv.DictReader(open(csv_path)))
    assert [row['agent'] for row in rows]==['1','2']


@pytest.mark.parametrize('vehicles',[{9:'moderate'},{1:'unknown'}])
def test_invalid_setup(vehicles):
    with pytest.raises(ValueError): MixedPolicy(DummyActor(),scenario(),vehicles)


def test_route_interpolation_uses_arc_length_and_ignores_padding():
    from utilities.testing_rule_policy import route_geometry, sample_route
    path=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[999.,999.]]])
    route=route_geometry(torch.tensor([[.9,0.]]),path,torch.tensor([3]))
    point,_=sample_route(route,torch.tensor([[.3]]))
    torch.testing.assert_close(point,torch.tensor([[[1.,.2]]]))
    closed=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]]])
    route=route_geometry(torch.tensor([[0.,.1]]),closed,loops=torch.tensor([True]))
    point,_=sample_route(route,torch.tensor([[.3]]))
    torch.testing.assert_close(point,torch.tensor([[[.2,0.]]]),rtol=0,atol=1e-6)


def test_centre_tracking_matches_a_circular_bicycle_path():
    radius=.5; lr=.08; length=.16
    beta=torch.asin(torch.tensor(lr/radius))
    theta=torch.linspace(0,2*torch.pi,501)
    path=torch.stack([radius*theta.cos(),radius*theta.sin()],-1)[None]
    expected=torch.atan(torch.tensor(length/lr)*beta.tan())
    action,_,_=rule_command(torch.tensor([[radius,0.]]), (torch.pi/2-beta).reshape(1),
        torch.tensor([.3]),path,torch.empty(1,0,2),torch.empty(1,0,2),
        profile='non_yielding',cruise_speed=.6,steering_limit=.6,wheelbase=length,
        rear_length=lr,sensing_range=.8,dt=.05,was_yielding=torch.tensor([False]),
        loops=torch.tensor([True]))
    assert float(action[0,1]) == pytest.approx(float(expected),abs=.002)


def local_command(neighbor_pos,neighbor_vel,profile='moderate',speed=.6,path=None):
    if path is None:path=torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]])
    return rule_command(torch.zeros(1,2),torch.zeros(1),torch.tensor([speed]),path,
        torch.tensor([[neighbor_pos]]),torch.tensor([[neighbor_vel]]),
        profile=profile,cruise_speed=.6,steering_limit=.6,wheelbase=.16,
        sensing_range=.8,dt=.05,was_yielding=torch.tensor([True]),return_details=True)


def test_side_and_rear_neighbors_do_not_lock_a_stopped_car():
    for position,velocity in [([0.,.15],[0.,0.]),([-.15,0.],[.4,0.]),([.18,0.],[.8,0.])]:
        action,yielding,_,details=local_command(position,velocity,speed=0.)
        assert not yielding.any() and action[0,0]>0
        assert details['blocker'][0] == -1


def test_stationary_obstacle_ahead_remains_a_reason_to_stop():
    _,yielding,target,details=local_command([.25,0.],[0.,0.],speed=0.)
    assert yielding.all() and target[0] == 0
    assert details['blocker'][0] == 0 and details['reason'][0] == 2


def test_approaching_rear_vehicle_is_counted_for_stopping_risk():
    _, yielding, target, details = local_command([-.16,0.],[.7,0.],speed=0.)
    assert not yielding.any() and target[0] > 0
    assert details['rear_risk'][0] and not details['all_blocked'][0]


def test_all_blocked_compares_risk_instead_of_automatically_stopping():
    route=torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]])
    _, yielding, target, details=rule_command(torch.zeros(1,2),torch.zeros(1),
        torch.tensor([.2]),route,torch.tensor([[[.3,0.],[-.16,0.]]]),
        torch.tensor([[[-.3,0.],[.7,0.]]]),profile='moderate',cruise_speed=.6,
        steering_limit=.6,wheelbase=.16,sensing_range=.8,dt=.05,
        was_yielding=torch.tensor([True]),return_details=True)
    assert details['all_blocked'][0] and details['rear_risk'][0]
    assert target[0] > 0 and details['reason'][0] == 4


def test_stopped_vehicle_restarts_without_extra_wait_and_neighbor_filter_resets():
    s=scenario()
    s.world.agents[1].state.vel.zero_()
    s.world.agents[0].state.pos[:,0]=-2
    s.world.agents[0].state.vel.zero_()
    wrapper=MixedPolicy(DummyActor(),s,{1:'moderate'})
    wrapper.previous[1]=(torch.zeros(1,dtype=torch.long),
                         torch.ones(1,dtype=torch.bool),torch.zeros(1))
    first=wrapper(TensorDict({},[1]))
    assert first['agents','action'][0,1,0] > 0
    # A newly accelerating neighbor is filtered, but not across its respawn.
    s.world.agents[0].state.pos[:,0]=.35
    s.world.agents[0].state.vel[:,0]=.7
    wrapper(TensorDict({},[1]))
    assert wrapper.neighbor_history[1][1][0,0,0] == pytest.approx(.65*.7)
    s.nod_agent_generation[:,0]+=1
    wrapper(TensorDict({},[1]))
    assert wrapper.neighbor_history[1][1][0,0,0] == pytest.approx(.7)


def test_crossing_traffic_can_be_handled_by_slowing_instead_of_stopping():
    _,yielding,target,details=local_command([.4,-.35],[0.,.6])
    assert yielding.all() and 0 < target[0] < .6
    assert details['reason'][0] == 1


def test_curved_route_can_avoid_a_false_straight_line_conflict():
    path=torch.tensor([[[0.,0.],[.1,0.],[.2,.1],[.2,.3],[.2,1.]]])
    _,yielding,_,_=local_command([.65,0.],[0.,0.],speed=0.,path=path)
    assert not yielding.any()


def test_diagnostic_log_keeps_terminal_contacts_per_vehicle_and_generation(tmp_path):
    import csv
    s=scenario();wrapper=MixedPolicy(DummyActor(),s,{1:'moderate'})
    frames=[]
    for generation,contact in [(0,True),(0,True),(1,True),(1,False)]:
        td=wrapper(TensorDict({},[1]))
        td['agents','info','nod_ego_generation']=torch.full((1,3),generation)
        td['next','agents','info','testing_road_contact']=torch.tensor([[True,contact,True]])
        td['next','agents','info','is_collision_with_agents']=torch.zeros(1,3,dtype=torch.bool)
        td['next','agents','info','testing_route_error']=torch.zeros(1,3)
        td['next','agents','info','nod_world_vel']=torch.zeros(1,3,2)
        frames.append(td)
    path=tmp_path/'diagnostics.csv';wrapper.save_diagnostics(torch.stack(frames,1),path)
    rows=list(csv.DictReader(open(path)))
    assert [r['road_events'] for r in rows] == ['1','1','2','2']
    assert [r['road_contact'] for r in rows] == ['1','1','1','0']
    assert rows[-1]['vehicle_contact_events']=='0' and rows[-1]['agent']=='2'
