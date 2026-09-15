"""Final-model evaluation must not reuse best-model caches or labels."""
import pytest
import torch
from utilities.evaluation_base import Evaluation
from utilities.helper_training import Parameters


@pytest.mark.parametrize('use_final', [False, True])
def test_selected_checkpoint_controls_cache_and_logged_identity(tmp_path, use_final, monkeypatch):
    def no_rollout(**kwargs):
        raise AssertionError('Cache selection test must not construct a rollout')
    monkeypatch.setattr('utilities.evaluation_base.mappo_cavs', no_rollout)
    for suffix in ('policy', 'critic', 'safety_value'):
        (tmp_path / f'final_{suffix}.pth').touch()
    (tmp_path / 'reward5.60_policy.pth').touch()
    for prefix, value in [('reward5.60', 10.), ('final', 20.)]:
        torch.save(torch.tensor(value), tmp_path / f'{prefix}_out_td_CPM_mixed_respawn_refresh.pth')
    evaluator = Evaluation(
        model_paths=[str(tmp_path) + '/'], scenario_type='CPM_mixed',
        load_final_models=[use_final], refresh_respawn_observations=[True],
        expected_checkpoint_names=['final' if use_final else 'reward5.60'],
        render_titles=['test'], is_render=False,
    )
    evaluator.model_i = 0
    evaluator.model_i_path = str(tmp_path) + '/'
    evaluator.parameters = Parameters()
    evaluator._adjust_parameters()
    assert evaluator.parameters.is_load_final_model is use_final
    assert evaluator._get_simulation_outputs().item() == (20. if use_final else 10.)
    assert f"checkpoint={'final' if use_final else 'reward5.60'}," in evaluator.model_run_details[0]


def test_missing_final_checkpoint_never_falls_back_to_best(tmp_path):
    (tmp_path / 'reward5.60_policy.pth').touch()
    evaluator = Evaluation(model_paths=[str(tmp_path) + '/'], load_final_models=[True], render_titles=['test'])
    evaluator.model_i = 0
    evaluator.model_i_path = str(tmp_path) + '/'
    evaluator.parameters = Parameters()
    evaluator._adjust_parameters()
    with pytest.raises(FileNotFoundError, match='final_policy'):
        evaluator._get_simulation_outputs()
