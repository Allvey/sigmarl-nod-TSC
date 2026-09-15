import math
import random
import numpy as np
import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv
from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters, TransformedEnvCustom
from utilities.diagnose_tracking import (project_path, fork_env, frame, step_physical,
                                        rng_snapshot, replay_rng, remove_traffic, paired_replay)


def test_path_errors_are_signed_and_heading_wraps():
    line=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[2.,0.]])
    result=project_path(torch.tensor([.5,.2]),torch.tensor(2*math.pi+.1),torch.tensor([.3,-.4]),line)
    assert result['lateral_error_m']==pytest.approx(.2)
    assert result['normal_velocity_mps']==pytest.approx(-.4)
    assert result['heading_error_deg']==pytest.approx(.1*180/math.pi,abs=1e-4)
    assert result['route_s_m']==pytest.approx(.5)
    with pytest.raises(ValueError):
        project_path(torch.zeros(2),torch.tensor(0.),torch.zeros(2),torch.zeros(3,2))


def test_replay_rng_does_not_change_main_stream():
    state=rng_snapshot()
    with replay_rng(state):
        expected=(torch.rand(2),random.random(),np.random.rand())
    with replay_rng(state):
        got=(torch.rand(2),random.random(),np.random.rand())
    assert torch.equal(expected[0],got[0]) and expected[1:]==got[1:]
    assert torch.equal(state[0],torch.get_rng_state())


@torch.no_grad()
def test_paired_rollout_reproduces_original_and_removes_only_other_vehicles():
    s=ScenarioRoadTraffic()
    s.parameters=Parameters(n_agents=4,scenario_type='CPM_mixed',max_steps=32,is_testing_mode=True,
        is_using_nod_actor=False,is_using_nod_opinion=False,is_using_deadlock_critic=False,
        is_apply_mask=False,is_add_noise=False,refresh_respawn_observations=True)
    env=TransformedEnvCustom(VmasEnv(scenario=s,num_envs=1,continuous_actions=True,max_steps=32,device='cpu',n_agents=4))
    def policy(td):
        return td.set(('agents','action'),torch.zeros(1,4,2))
    snapshot=None;isolated=None
    try:
        env.set_seed(23);td=env.reset()
        snapshot=dict(env=fork_env(env),decision=td.clone(),rng=rng_snapshot())
        refs=[]
        for _ in range(3):
            _,td,physical,_=step_physical(env,policy,td)
            refs.append(physical[0])
        pair=paired_replay(snapshot,policy,0,3,refs)
        assert pair['traffic']['completed_steps']==3
        assert pair['isolated']['completed_steps']==3
        assert not pair['isolated']['agent_collision']
        isolated=fork_env(snapshot['env'])
        original=frame(snapshot['env'].base_env.scenario,0)
        decision=remove_traffic(isolated,snapshot['decision'],0)
        sc=isolated.base_env.scenario
        assert frame(sc,0)['nearest_neighbor_m'] is None
        assert frame(sc,0)['x_m']==original['x_m']
        assert frame(snapshot['env'].base_env.scenario,0)['nearest_neighbor_m'] is not None
        neighbors=decision['agents','observation'][0,0,-22:].reshape(2,11)
        assert (neighbors[:,:8]==1).all() and (neighbors[:,8:10]==0).all() and (neighbors[:,10]==1).all()
        parked=torch.stack([a.state.pos.clone() for a in sc.world.agents[1:]])
        step_physical(isolated,policy,decision,only_agent=0)
        torch.testing.assert_close(parked,torch.stack([a.state.pos for a in sc.world.agents[1:]]))
    finally:
        if isolated:isolated.close()
        if snapshot:snapshot['env'].close()
        env.close()
