import torch
from utilities.diagnose_safety_stability import observed_window, score_predictions


def states():
    def one():
        g = torch.full((1, 4, 1, 3), -1.)
        return dict(g=g, valid=torch.ones_like(g, dtype=torch.bool),
                    ego_gen=torch.zeros_like(g, dtype=torch.long),
                    other_gen=torch.zeros_like(g, dtype=torch.long))
    return one(), one(), torch.zeros(1, 4, 1, dtype=torch.bool)


def test_physical_window_sees_future_event_but_stops_at_horizon():
    c, n, done = states()
    n['g'][0, 2, 0, 1] = 1.
    observed, steps, valid = observed_window(c, n, done, 2)
    assert observed[0, 0, 0, 1] == -1
    assert observed[0, 1, 0, 1] == 1
    assert steps[0, :, 0, 1].tolist() == [2, 2, 2, 1]
    assert valid.all()


def test_window_does_not_cross_respawn_or_environment_end():
    c, n, done = states()
    n['g'][0, 2, 0, 1] = 1.
    c['ego_gen'][:, 2:] = 1
    n['ego_gen'][:, 2:] = 1
    observed, steps, _ = observed_window(c, n, done, 4)
    assert observed[0, 0, 0, 1] == -1
    assert steps[0, 0, 0, 1] == 2
    done[0, 0] = True
    _, steps, _ = observed_window(c, n, done, 4)
    assert steps[0, 0, 0, 1] == 1


def test_comparison_uses_common_labels_and_excludes_short_safe_suffixes():
    c, n, done = states()
    predictions = dict(best=torch.full_like(c['g'], -.5), final=torch.full_like(c['g'], .5))
    report = score_predictions(c, n, predictions, predictions, done, 3, .05, 10.)
    for head in ('pair', 'road', 'collision'):
        assert report['best'][head]['complete_window_safe'] == 2
        assert report['final'][head]['complete_window_safe'] == 2
        assert report['best'][head]['observed_safe_alarm_rate'] == 0
        assert report['final'][head]['observed_safe_alarm_rate'] == 1
        assert report['final'][head]['unsafe_recall'] is None
    assert report['best']['any_gate_rate'] == 0
    assert report['final']['any_gate_rate'] == 1


def test_loaded_adam_moments_are_preserved_but_config_controls_lr(tmp_path):
    from utilities.helper_training import Parameters
    from utilities.nod_marl.safety_value import SafetyValueManager
    old_params = Parameters.from_json('config_dgppo_v2_respawn_training.json')
    old = SafetyValueManager(old_params, 32, ('agents', 'observation'))
    old.optimizer.zero_grad()
    sum(p.sum() for p in old.model.parameters()).backward()
    old.optimizer.step()
    path = tmp_path / 'value.pth'
    torch.save(old.checkpoint_state(), path)
    new_params = Parameters.from_dict(dict(old_params.to_dict(), safety_value_lr=.0003))
    new = SafetyValueManager(new_params, 32, ('agents', 'observation'))
    assert new.load_if_available(path, load_optimizer=True)
    assert all(g['lr'] == .0003 for g in new.optimizer.param_groups)
    for a, b in zip(old.model.parameters(), new.model.parameters()):
        torch.testing.assert_close(a, b)
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(old.optimizer.state[a][key], new.optimizer.state[b][key])


def test_duration_change_preserves_warmup_only_without_schedule(tmp_path):
    from utilities.helper_training import Parameters
    from utilities.nod_marl.safety_value import SafetyValueManager
    for schedule in (False, True):
        p = Parameters.from_dict(dict(Parameters.from_json('config_dgppo_v2_respawn_training.json').to_dict(),
                                      n_iters=500, dgppo_schedule=schedule))
        old = SafetyValueManager(p, 32, ('agents', 'observation'))
        old.barrier_fit_batches = 309
        path = tmp_path / f'value_{schedule}.pth'
        torch.save(old.checkpoint_state(), path)
        new_p = Parameters.from_dict(dict(p.to_dict(), n_iters=50, safety_value_lr=.0003))
        new = SafetyValueManager(new_p, 32, ('agents', 'observation'))
        assert new.load_if_available(path, load_optimizer=True)
        assert new.barrier_fit_batches == (0 if schedule else 309)
        assert new.optimizer.param_groups[0]['lr'] == .0003
        # Actual constraint changes still require warmup even without scheduling.
        changed_p = Parameters.from_dict(dict(new_p.to_dict(), dgppo_alpha=p.dgppo_alpha + 1))
        changed = SafetyValueManager(changed_p, 32, ('agents', 'observation'))
        assert changed.load_if_available(path, load_optimizer=True)
        assert changed.barrier_fit_batches == 0


def test_contract_normalization_does_not_mutate_or_ignore_schedule_toggle():
    from utilities.nod_marl.safety_value import equivalent_barrier_contract
    saved = dict(mode='dgppo', schedule=False, schedule_iters=500)
    current = dict(mode='dgppo', schedule=False, schedule_iters=50)
    assert equivalent_barrier_contract(saved, current)
    assert saved['schedule_iters'] == 500 and current['schedule_iters'] == 50
    assert not equivalent_barrier_contract(saved, dict(current, schedule=True))
    assert not equivalent_barrier_contract(None, current)
