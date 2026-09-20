"""Real VMAS dynamics regressions for centralized test traffic."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vmas.simulator.core import Agent, Box, World

from utilities.kinematic_bicycle import KinematicBicycle
from utilities.testing_rule_coordinator import RuleCoordinator, bicycle_step, swept_conflict


def make_scene(positions, headings, paths):
    world = World(batch_dim=1, device='cpu', dt=.05)
    for i, (position, heading) in enumerate(zip(positions, headings)):
        agent = Agent(name=str(i), shape=Box(length=.16, width=.08), collide=False,
                      max_speed=1., u_range=[1., .6],
                      dynamics=KinematicBicycle(world, .08, .08, .08, .6))
        world.add_agent(agent)
        agent.set_pos(torch.tensor(position), batch_index=0)
        agent.set_rot(torch.tensor([heading]), batch_index=0)
        agent.set_vel(torch.zeros(2), batch_index=0)
        agent.state.ang_vel.zero_()
    return SimpleNamespace(world=world, parameters=SimpleNamespace(dt=.05),
        max_speed=1., max_steering_angle=.6,
        nod_agent_generation=torch.zeros(1, len(positions), dtype=torch.long),
        ref_paths_agent_related=SimpleNamespace(
            long_term=torch.tensor([paths], dtype=torch.float32),
            n_points_long_term=torch.full((1, len(paths)), len(paths[0])),
            is_loop=torch.zeros(1, len(paths), dtype=torch.bool)))


def poses(scene):
    return np.array([np.r_[a.state.pos[0].numpy(), a.state.rot[0, 0].item()]
                     for a in scene.world.agents])


def advance(scene, commands):
    for i, agent in enumerate(scene.world.agents):
        agent.action.u = commands[:, i]
        agent.dynamics.process_action()
    scene.world.step()


@pytest.mark.parametrize('steering', [0., .4, -.4])
def test_prediction_matches_real_world_with_drag_and_velocity_limit(steering):
    s=make_scene([[0., 0.]], [.3], [[[-1.,0.],[2.,0.]]])
    a=s.world.agents[0]
    a.state.vel[:] = torch.tensor([[.7, .15]])
    a.state.ang_vel[:] = .2
    for speed in [.9, .3, 0., 0.]:
        before=poses(s)[0]
        expected, vel, angular=bicycle_step(before, a.state.vel[0].numpy().copy(),
            float(a.state.ang_vel[0,0]), np.array([speed,steering]),
            dt=.05, lf=.08, lr=.08, drag=s.world._drag, max_speed=1.)
        advance(s, torch.tensor([[[speed,steering]]]))
        np.testing.assert_allclose(poses(s)[0], expected, atol=1e-6)
        np.testing.assert_allclose(a.state.vel[0].numpy(), vel, atol=1e-6)
        assert float(a.state.ang_vel[0,0]) == pytest.approx(angular, abs=1e-6)


def test_swept_check_detects_crossing_between_endpoints_and_containment():
    a=np.array([[-.3,0.,0.],[.3,0.,0.]])
    b=np.array([[0.,-.3,np.pi/2],[0.,.3,np.pi/2]])
    assert swept_conflict(a,b,(.16,.08),(.16,.08),margin=0)
    assert swept_conflict(np.zeros((2,3)),np.zeros((2,3)),(.2,.1),(.1,.05),margin=0)
    assert not swept_conflict(a,b+np.array([2.,0.,0.]),(.16,.08),(.16,.08))


@pytest.mark.parametrize('kind', ['crossing', 'following'])
def test_two_rules_both_pass_without_contact_or_mutual_wait(kind):
    if kind=='crossing':
        s=make_scene([[-.45,0.],[0.,-.45]], [0.,np.pi/2],
                     [[[-1.,0.],[2.,0.]],[[0.,-1.],[0.,2.]]])
    else:
        s=make_scene([[-.45,0.],[-.15,0.]], [0.,0.],
                     [[[-1.,0.],[2.,0.]], [[-1.,0.],[2.,0.]]])
    coordinator=RuleCoordinator(s,{0:'non_yielding',1:'non_yielding'})
    for _ in range(120):
        previous=poses(s)
        commands, _, _, _, infeasible=coordinator.coordinate(torch.tensor([[[.6,0.],[.6,0.]]]))
        assert not infeasible.any()
        advance(s,commands)
        current=poses(s)
        assert not swept_conflict(np.stack([previous[0],current[0]]),
                                  np.stack([previous[1],current[1]]),(.16,.08),(.16,.08),margin=0)
    assert current[0,0]>.5
    assert current[1,1 if kind=='crossing' else 0]>.5


def test_generation_change_discards_old_grant_and_wait_age():
    s=make_scene([[-.45,0.],[0.,-.45]], [0.,np.pi/2],
                 [[[-1.,0.],[2.,0.]],[[0.,-1.],[0.,2.]]])
    coordinator=RuleCoordinator(s,{0:'moderate',1:'moderate'})
    coordinator.coordinate(torch.tensor([[[.6,0.],[.6,0.]]]))
    coordinator.ages[0,0]=100.
    s.nod_agent_generation[0,0]+=1
    _,_,_,waiting,_=coordinator.coordinate(torch.zeros(1,2,2))
    assert waiting[0,0]==0


def test_crossing_grant_is_exclusive_and_both_queues_eventually_clear():
    s=make_scene([[-.45,0.],[0.,-.45]], [0.,np.pi/2],
                 [[[-1.,0.],[2.,0.]],[[0.,-1.],[0.,2.]]])
    s.ref_paths_agent_related.path_id=torch.tensor([[0,1]])
    coordinator=RuleCoordinator(s,{0:'non_yielding',1:'yielding'})
    coordinator.zones={0: {0:(.75,1.25),1:(.75,1.25)}}
    for _ in range(120):
        commands,_,_,_,infeasible=coordinator.coordinate(torch.tensor([[[.6,0.],[.6,0.]]]))
        assert not infeasible.any()
        assert len(coordinator.owners[0,0]) <= 1
        advance(s,commands)
        current=poses(s)
        assert int(abs(current[0,0])<.24)+int(abs(current[1,1])<.24) <= 1
    assert current[0,0]>.5 and current[1,1]>.5


def test_blocked_exit_does_not_grant_entry_and_releases_when_clear():
    s=make_scene([[-.4,0.],[.35,0.]], [0.,0.],
                 [[[-1.,0.],[2.,0.]], [[-1.,0.],[2.,0.]]])
    s.ref_paths_agent_related.path_id=torch.tensor([[0,0]])
    coordinator=RuleCoordinator(s,{0:'moderate',1:'moderate'})
    coordinator.zones={0:{0:(.75,1.25)}}
    cars={i:coordinator._vehicle(i,0) for i in [0,1]}
    initial={i:(poses(s)[i],np.zeros(2),0.) for i in [0,1]}
    coordinator._assign_crossing(0,cars,initial)
    assert not coordinator.owners[0,0]
    assert cars[0]['stop_at']<.75
    initial[1][0][0]=.8
    coordinator._assign_crossing(0,cars,initial)
    assert list(coordinator.owners[0,0])==[0]
