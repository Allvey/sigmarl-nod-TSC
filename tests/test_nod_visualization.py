"""Offline checks for truthful, read-only opinion/alpha display."""
from types import SimpleNamespace

import pytest
import torch

from test_dgppo_opinion import sample, params
from utilities.nod_marl.visualization import opinion_alpha_lines


def fixture():
    td, state = sample()
    td = td.squeeze(1)
    state = {k: v.squeeze(1) for k, v in state.items()}
    p = params()
    manager = SimpleNamespace(parameters=p, enabled=True, last_load_info='loaded shadow Value',
                              state=lambda _: state, model=lambda s: s['g'])
    return td, state, manager


def test_rows_use_neighbor_identity_and_do_not_mutate_cached_context():
    td, state, manager = fixture()
    before = td.clone()
    lines = opinion_alpha_lines(td, manager, agent_index=0, decision_time=.25)
    assert 'Agent 1' in lines[0] and 't=0.25s' in lines[1]
    assert 'A1 -> A2  z=-1.000  alpha=5.00  [opinion]' in lines
    assert 'A1 -> A3  z=+1.000  alpha=15.00  [opinion]' in lines
    for key in before.keys(True, True):
        torch.testing.assert_close(td[key], before[key], rtol=0, atol=0)
    # Select another fixed vehicle, not its index in the neighbor list.
    assert any('A2 -> A1  z=+0.000  alpha=10.00' in x
               for x in opinion_alpha_lines(td, manager, agent_index=1))


def test_actual_fallback_differs_from_unconditional_alpha_mapping():
    td, state, manager = fixture()
    state['g'][0, 0, 2] = .2  # Positive z must not display alpha=15 here.
    lines = opinion_alpha_lines(td, manager)
    assert 'A1 -> A3  z=+1.000  alpha=10.00  [non-safe: fixed]' in lines
    td['agents', 'info', 'nod_actor_context_ready'].zero_()
    assert any('z=N/A  alpha=10.00  [missing: fixed]' in x
               for x in opinion_alpha_lines(td, manager))


def test_missing_model_does_not_display_random_value_as_real_alpha():
    td, state, manager = fixture()
    manager.last_load_info = 'missing sidecar; fresh shadow Value'
    manager.model = lambda _: (_ for _ in ()).throw(AssertionError('Must not call fresh Value'))
    lines = opinion_alpha_lines(td, manager)
    assert any('z=+1.000  alpha=N/A  [Value unavailable]' in x for x in lines)
    manager.parameters.dgppo_opinion_alpha = False
    assert any('alpha=10.00  [fixed mode]' in x for x in opinion_alpha_lines(td, manager))


def test_empty_neighbors_and_invalid_selection_are_explicit():
    td, state, manager = fixture()
    state['valid'][..., :3] = False
    assert opinion_alpha_lines(td, manager)[-1] == 'No visible NOD neighbors'
    assert '0..2' in opinion_alpha_lines(td, manager, agent_index=9)[-1]


def test_display_uses_saved_gain_without_amplifying_displayed_raw_opinion():
    td, state, manager = fixture()
    td['agents', 'info', 'nod_actor_edge_context'][..., 0, 0, -1] = -.072
    manager.parameters.dgppo_alpha_gain = 2.
    lines = opinion_alpha_lines(td, manager)
    assert 'A1 -> A2  z=-0.072  alpha=9.28  [opinion]' in lines
    manager.parameters.dgppo_alpha_gain = 1.
    assert 'A1 -> A2  z=-0.072  alpha=9.64  [opinion]' in opinion_alpha_lines(td, manager)


def test_display_marks_deadzone_fallback():
    td, state, manager = fixture()
    td['agents', 'info', 'nod_actor_edge_context'][..., 0, 0, -1] = 0.08
    manager.parameters.dgppo_opinion_deadzone = 0.1
    lines = opinion_alpha_lines(td, manager)
    assert 'A1 -> A2  z=+0.080  alpha=10.00  [deadzone: fixed]' in lines


@pytest.mark.parametrize('target', ['evidence', 'attention'])
def test_ppo_gate_display_uses_collected_neighbor_slots_without_mutation(target):
    td, state, manager = fixture()
    manager.parameters.nod_evidence_gate_mode = 'ppo'
    manager.parameters.nod_evidence_gate_target = target
    gate = torch.zeros(1, 3, 2, 75)
    gate[..., -5] = .6
    gate[..., -4] = -.4
    gate[..., -1] = 1.
    td.set(('agents', 'info', 'nod_evidence_gate_context'), gate)
    td.set(('agents', 'info', 'nod_evidence_weight'), torch.tensor([[[.2, .8], [.3, .7], [.4, .6]]]))
    before = td.clone()
    lines = opinion_alpha_lines(td, manager)
    detail = ('e=-0.400  A_RL=0.200  A0=0.600' if target == 'attention'
              else 'e=-0.400  beta=0.200  A=0.600')
    weight_label = 'A_RL' if target == 'attention' else 'beta'
    assert any('A1 -> A2' in row and detail in row for row in lines)
    assert any('A1 -> A3' in row and f'{weight_label}=0.800' in row for row in lines)
    for key in before.keys(True, True):
        torch.testing.assert_close(td[key], before[key], rtol=0, atol=0)
    td['agents', 'info', 'nod_evidence_gate_context'][..., -1] = 0.
    assert all(f'{weight_label}=N/A' in row for row in opinion_alpha_lines(td, manager)[1:])
