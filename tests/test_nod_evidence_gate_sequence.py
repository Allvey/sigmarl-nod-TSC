"""Short-window credit assignment, reset boundaries and gate-only PPO updates."""

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss

from test_nod_evidence_gate import make_actor, cached_data
from utilities.helper_training import Parameters
from utilities.nod_marl.gate_training import ordered_gate_windows, freeze_actor_body_gradients
from utilities.nod_marl.policy import (
    replay_evidence_sequence, NOD_ACTOR_EDGE_CONTEXT_KEY, NOD_ACTOR_MESSAGE_KEY,
    NOD_EVIDENCE_GATE_CONTEXT_KEY,
)


def sequence_data(manager, module):
    data, outputs = cached_data(manager, module)
    data = data.unsqueeze(0)
    data.set(('agents', 'info', 'nod_neighbor_indices'),
             torch.tensor([[[[1], [0]]] * 5]))
    data.set(('agents', 'info', 'nod_ego_generation'), torch.ones(1, 5, 2, dtype=torch.long))
    generation = torch.ones(1, 5, 2, 1, dtype=torch.long)
    generation[:, 4, 1] = 2
    data.set(('agents', 'info', 'nod_neighbor_generation'), generation)
    data.set(('next', 'done'), torch.zeros(1, 5, 1, dtype=torch.bool))
    return data, outputs


def test_windows_never_mix_environments_or_change_time_order():
    stamps = torch.arange(32).reshape(2, 16)
    data = TensorDict({'stamp': stamps, 'nested': {'value': stamps + 100}}, batch_size=[2, 16])
    windows = ordered_gate_windows(data, 8)
    assert windows.batch_size == torch.Size([4, 8])
    torch.testing.assert_close(windows['stamp'], torch.stack([
        stamps[0, :8], stamps[1, :8], stamps[0, 8:], stamps[1, 8:]
    ]))
    assert (windows['stamp'][:, 1:] - windows['stamp'][:, :-1]).eq(1).all()
    torch.testing.assert_close(windows['nested', 'value'], windows['stamp'] + 100)
    with pytest.raises(ValueError):
        ordered_gate_windows(data, 6)


@pytest.mark.parametrize('target', ['evidence', 'attention'])
def test_late_opinion_has_gradient_to_earlier_weights_and_resets_cut_it(target):
    class TimeGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.full((4,), -.2, dtype=torch.float64))

        def forward(self, context):
            return torch.sigmoid(self.logits[context[..., 0].long()])

    gate = TimeGate()
    gate.target = target
    context = torch.zeros(1, 4, 1, 1, 19, dtype=torch.float64)
    context[0, :, 0, 0, 0] = torch.arange(4)
    context[..., -5] = .6
    context[..., -4] = .8
    context[..., -2:] = 1.
    z = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
    continuation = torch.ones_like(z, dtype=torch.bool)
    continuation[:, 0] = False
    kwargs = dict(bifurcation_gain=2., observation_weight=1., kl_weight=5.,
                  max_iterations=64, tolerance=0.)
    opinions, _ = replay_evidence_sequence(context, z, continuation, gate, kwargs)
    gradient = torch.autograd.grad(opinions[:, -1].sum(), gate.logits)[0]
    assert (gradient > 0).all()
    continuation[:, 2] = False
    reset_opinions, _ = replay_evidence_sequence(context, z, continuation, gate, kwargs)
    reset_gradient = torch.autograd.grad(reset_opinions[:, -1].sum(), gate.logits)[0]
    assert reset_gradient[:2].eq(0.).all()
    assert (reset_gradient[2:] > 0).all()
    changed = context.clone()
    changed[:, -1, ..., -4] = -.8
    causal, _ = replay_evidence_sequence(changed, z, continuation, gate, kwargs)
    torch.testing.assert_close(causal[:, :-1], reset_opinions[:, :-1])


@pytest.mark.parametrize('target', ['evidence', 'attention'])
def test_sequence_replay_reproduces_collection_and_honors_done_and_identity(target):
    _, module, manager, _ = make_actor(sequence_length=8, target=target)
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(.7)
    data, _ = sequence_data(manager, module)
    before = data.clone()
    reference, _ = module.aggregator(data[NOD_ACTOR_EDGE_CONTEXT_KEY],
                                    data['agents', 'info', 'nod_actor_edge_mask'])
    module(data)
    torch.testing.assert_close(data[NOD_ACTOR_MESSAGE_KEY], reference, atol=1e-6, rtol=1e-6)
    assert manager.online_state is None
    torch.testing.assert_close(data[NOD_ACTOR_EDGE_CONTEXT_KEY], before[NOD_ACTOR_EDGE_CONTEXT_KEY])
    torch.testing.assert_close(data[NOD_EVIDENCE_GATE_CONTEXT_KEY], before[NOD_EVIDENCE_GATE_CONTEXT_KEY])
    gate = data[NOD_EVIDENCE_GATE_CONTEXT_KEY]
    continuation = module._sequence_continuation(data, gate, data[NOD_ACTOR_EDGE_CONTEXT_KEY][..., -1])
    assert not continuation[:, 0].any()
    assert continuation[:, 2].all()
    assert not continuation[:, 3, 0].any()  # No interaction.
    assert not continuation[:, 4, 1].any()  # Neighbor identity replaced.
    data['next', 'done'][:, 1] = True
    assert not module._sequence_continuation(data, gate, data[NOD_ACTOR_EDGE_CONTEXT_KEY][..., -1])[:, 2].any()
    data['next', 'done'].zero_()
    data['agents', 'info', 'nod_ego_generation'][:, 2] = 9
    assert not module._sequence_continuation(data, gate, data[NOD_ACTOR_EDGE_CONTEXT_KEY][..., -1])[:, 2].any()


@pytest.mark.parametrize('target', ['evidence', 'attention'])
def test_gate_only_ppo_preserves_actor_body_but_trains_gate_and_critic(target):
    torch.manual_seed(42)
    actor, module, manager, _ = make_actor(sequence_length=8, gate_only_iters=10, target=target)
    data, _ = sequence_data(manager, module)
    with torch.no_grad():
        actor(data)
    critic = TensorDictModule(torch.nn.Linear(5, 1),
                             in_keys=[('agents', 'observation')],
                             out_keys=[('agents', 'state_value')])
    loss = ClipPPOLoss(actor=actor, critic=critic, normalize_advantage=False)
    loss.set_keys(action=('agents', 'action'), sample_log_prob=('agents', 'sample_log_prob'),
                  value=('agents', 'state_value'), advantage=('agents', 'advantage'),
                  value_target=('agents', 'value_target'))
    data.set(('agents', 'advantage'), torch.randn(1, 5, 2, 1))
    data.set(('agents', 'value_target'), torch.ones(1, 5, 2, 1))
    gate_ids = {id(p) for p in module.evidence_gate.parameters()}
    body = [p for p in actor.parameters() if id(p) not in gate_ids]
    assert body
    snapshots = [p.detach().clone() for p in body]
    forecast = {k: v.clone() for k, v in manager.model.state_dict().items()}
    critic_before = [p.detach().clone() for p in critic.parameters()]
    gate_before = [p.detach().clone() for p in module.evidence_gate.parameters()]
    optim = torch.optim.Adam(loss.parameters(), lr=1e-3)
    losses = loss(data)
    (losses['loss_objective'] + losses['loss_critic']).backward()
    freeze_actor_body_gradients(body, True)
    assert all(p.grad is None for p in body)
    assert module.evidence_gate.net[-1].weight.grad.abs().sum() > 0
    optim.step()
    for p, previous in zip(body, snapshots):
        torch.testing.assert_close(p, previous, rtol=0, atol=0)
    assert any(not torch.equal(p.detach(), old) for p, old in zip(critic.parameters(), critic_before))
    assert any(not torch.equal(p.detach(), old) for p, old in zip(module.evidence_gate.parameters(), gate_before))
    for key, value in manager.model.state_dict().items():
        torch.testing.assert_close(value, forecast[key], rtol=0, atol=0)
    # Phase progress is paired with the corresponding policy snapshot.
    module.evidence_gate.training_batches.fill_(10)
    restored, restored_module, _, _ = make_actor(sequence_length=8, gate_only_iters=10, target=target)
    restored.load_state_dict(actor.state_dict())
    assert restored_module.evidence_gate.training_batches.item() == 10
    # Joint training no longer masks the actor gradients.
    optim.zero_grad()
    losses = loss(data)
    losses['loss_objective'].backward()
    freeze_actor_body_gradients(body, False)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in body)


def test_staged_config_pins_trained_fixed_policy_and_keeps_forecast_contract():
    p = Parameters.from_json('configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json')
    assert p.training_init_checkpoint.endswith('dgppo_nod_future_evidence_control/reward7.08')
    assert p.nod_evidence_gate_sequence_length == 8 and p.nod_evidence_gate_only_iters == 10
    assert p.nod_freeze_training and p.nod_fixed_variance_value == 1.
    assert p.nod_evidence_gate_lr == pytest.approx(5 * p.lr)
    assert p.nod_evidence_gate_lr == pytest.approx(5 * p.nod_message_lr)
    assert p.safety_finetune_target_kl == .01
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    for change in (dict(nod_evidence_gate_sequence_length=7), dict(nod_evidence_gate_only_iters=50),
                   dict(nod_evidence_gate_sequence_length=0), dict(nod_evidence_gate_mode='fixed'),
                   dict(max_steps=120)):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(p.to_dict(), **change))
