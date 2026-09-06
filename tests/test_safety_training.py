"""Small end-to-end check: adding Safety must not change PPO's trajectory."""
import json

import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


def test_short_training_isolation_and_checkpoint_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in [False, True]:
            p = Parameters.from_json("config.json")
            p.seed = 12345
            p.n_iters = 2
            p.num_epochs = 2
            p.frames_per_batch = 64
            p.minibatch_size = 32
            p.num_vmas_envs = 4
            p.max_steps = 16
            p.total_frames = 128
            p.safety_num_epochs = 2
            p.safety_minibatch_size = 32
            p.nod_sequence_length = 16
            p.is_load_model = False
            p.is_continue_train = False
            p.is_save_intermediate_model = True
            p.where_to_save = str(tmp_path / str(enabled)) + "/"
            p.is_using_safety_critic = enabled
            env, _, _, _ = mappo_cavs(p)
            env.close()
        for name in ["policy", "critic"]:
            before = torch.load(tmp_path / f"False/final_{name}.pth")
            after = torch.load(tmp_path / f"True/final_{name}.pth")
            assert all(torch.equal(before[k], after[k]) for k in before)
        before_nod = torch.load(tmp_path / "False/final_nod.pth")["model"]
        after_nod = torch.load(tmp_path / "True/final_nod.pth")["model"]
        assert all(torch.equal(before_nod[k], after_nod[k]) for k in before_nod)
        data = json.loads(
            next((tmp_path / "True").glob("reward*_data.json")).read_text()
        )
        assert len(data["safety_metrics_list"]) == 2
        assert all(m["optimizer_updates"] > 0 for m in data["safety_metrics_list"])
        assert (tmp_path / "True/final_safety_critic.pth").exists()
        assert list((tmp_path / "True").glob("reward*_safety_critic.pth"))
        for final in [False, True]:
            p.is_load_model = True
            p.is_load_final_model = final
            env, _, _, _ = mappo_cavs(p)
            assert env.scenario.safety_manager.updates > 0
            env.close()
        previous_updates = torch.load(tmp_path / "True/final_safety_critic.pth")[
            "updates"
        ]
        p.is_continue_train = True
        p.n_iters = 1
        p.total_frames = p.frames_per_batch
        env, _, _, _ = mappo_cavs(p)
        assert env.scenario.safety_manager.updates > previous_updates
        env.close()
        # The same loader must accept legacy policy/NOD files with no safety sidecar.
        p.is_continue_train = False
        p.where_to_save = str(tmp_path / "False") + "/"
        env, _, _, _ = mappo_cavs(p)
        assert env.scenario.safety_manager.updates == 0
        env.close()
    finally:
        torch.set_num_threads(old_threads)
