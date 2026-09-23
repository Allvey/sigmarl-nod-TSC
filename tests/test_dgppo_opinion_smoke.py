"""Manual integration check using the completed step-1 checkpoint (read only).

Runs two tiny fine-tuning jobs in pytest temp directories, then loads inference.
"""
import json
from pathlib import Path

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


def test_frozen_nod_fixed_and_opinion_finetune(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    env = None
    try:
        for mode in ['fixed_control', 'opinion']:
            p = Parameters.from_json(
                f'configs/archive/nod_history/config_dgppo_nod_{mode}_finetune.json'
            )
            source = Path(p.training_init_checkpoint + '_nod.pth')
            if not source.is_file(): pytest.skip('Requires the completed step-1 reward7.33 checkpoint')
            initial = torch.load(source, map_location='cpu')['model']
            p.num_epochs = 1; p.n_iters = 3
            p.frames_per_batch = 64; p.total_frames = 192; p.minibatch_size = 32
            p.num_vmas_envs = 4; p.max_steps = 16
            p.safety_value_num_envs = 4; p.safety_value_rollout_steps = 16
            p.safety_value_minibatch_size = 32; p.safety_barrier_warmup_batches = 1
            out = tmp_path / mode; p.where_to_save = str(out) + '/'
            env, *_ = mappo_cavs(p)
            assert all(not w.requires_grad for w in env.scenario.nod_manager.model.parameters())
            env.close(); env = None
            final = torch.load(out / 'final_nod.pth', map_location='cpu')['model']
            assert all(torch.equal(initial[k], final[k]) for k in initial)
            saved = json.loads(next(out.glob('reward*_data.json')).read_text())
            ns, ms = saved['nod_metrics_list'], saved['safety_value_metrics_list']
            assert all(m['training_frozen'] and m['optimizer_updates'] == 0 for m in ns)
            assert all(m['actor_context_ready_ratio'] == 1 for m in ns)
            assert ms[0]['finetune_actor_frozen'] == 1
            assert any(m['finetune_ppo_updates'] > 0 for m in ms)
            if mode == 'opinion':
                assert all(m['opinion_alpha_enabled'] == 1 for m in ms)
                assert any(m['barrier_ready'] for m in ms)
                for m in ms:
                    if m['opinion_pair_count']:
                        assert 5 <= m['opinion_alpha_min'] <= m['opinion_alpha_max'] <= 15
                # A tiny physical rollout need not contain an active pair gate.
                # Offline tests separately prove nonzero advantage/gradient effects.
                print('Opinion diagnostic:', {k:v for k,v in ms[-1].items() if k.startswith('opinion_')})
            p.is_load_model = p.is_load_final_model = True
            p.is_continue_train = False
            env, policy, *_ = mappo_cavs(p)
            assert env.scenario.nod_manager.last_load_info == 'loaded'
            env.rollout(2, policy, break_when_any_done=False)
            env.close(); env = None
    finally:
        if env is not None: env.close()
        torch.set_num_threads(threads)
