"""Manual integration check: runs small VMAS/PPO jobs in pytest temp folders.

Run separately from the offline test_dgppo_nod.py checks.
"""
import json

import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


def small_job(p, output):
    p.num_epochs = 1
    p.n_iters = 3; p.frames_per_batch = 64; p.total_frames = 192
    p.minibatch_size = 32; p.num_vmas_envs = 4; p.max_steps = 16
    p.safety_value_num_envs = 4; p.safety_value_rollout_steps = 16
    p.safety_value_minibatch_size = 32; p.safety_barrier_warmup_batches = 1
    p.nod_update_interval = 1; p.nod_num_epochs = 1
    p.nod_sequence_length = 8; p.nod_counterfactual_horizon = 2
    p.where_to_save = str(output) + '/'
    return p


def test_base_to_local_nod_finetune_and_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    env = None
    try:
        # Make an ordinary non-NOD checkpoint without relying on user's models.
        base = small_job(Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo.json'), tmp_path / 'base')
        env, *_ = mappo_cavs(base); env.close(); env = None
        p = Parameters.from_json('configs/archive/nod_history/config_dgppo_nod_fixed_finetune.json')
        p.training_init_checkpoint = str(tmp_path / 'base' / 'final')
        p = small_job(p, tmp_path / 'nod')
        env, *_ = mappo_cavs(p)
        assert env.scenario.nod_manager.observation_mode == 'local_kinematics'
        env.close(); env = None
        saved = json.loads(next((tmp_path / 'nod').glob('reward*_data.json')).read_text())
        ms = saved['safety_value_metrics_list']
        assert ms[0]['finetune_actor_frozen'] == 1.
        assert any(m['finetune_ppo_updates'] > 0 for m in ms)
        ns = saved['nod_metrics_list']
        assert all(m['actor_context_ready_ratio'] == 1. for m in ns)
        assert any(m['optimizer_updates'] > 0 for m in ns)
        assert torch.load(tmp_path / 'nod' / 'final_nod.pth')['observation_mode'] == 'local_kinematics'
        p.is_load_model = p.is_load_final_model = True
        p.is_continue_train = False
        env, policy, *_ = mappo_cavs(p)
        assert env.scenario.nod_manager.last_load_info == 'loaded'
        env.rollout(2, policy, break_when_any_done=False)
    finally:
        if env is not None: env.close()
        torch.set_num_threads(threads)
