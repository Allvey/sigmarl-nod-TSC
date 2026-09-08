import json

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs
from utilities.nod_marl.safety import SafetyCriticManager
from test_safety_critic import make_rollout


def test_constraint_reduces_action_risk_without_training_critic_or_states(tmp_path):
    p = Parameters(is_using_safety_constraint=True, safety_constraint_warmup_batches=1)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    td, observation, stored_action = make_rollout()
    assert not manager.constraint_ready
    manager.updates = manager.rollouts = 1
    manager.reliability_history = [[100, 64, 0, 0]]
    # Known Q: future risk increases with speed; h=1 must never control actions.
    manager.model = torch.nn.Linear(manager.input_dim, 4)
    with torch.no_grad():
        manager.model.weight.zero_()
        manager.model.bias.zero_()
        manager.model.bias[0] = 100
        manager.model.weight[1:, 11] = 1  # first car's speed action
        manager.model.weight[1:, 24] = 1  # second car's speed action
    action = torch.nn.Parameter(torch.full_like(stored_action, 0.3))
    before = {k: v.clone() for k, v in manager.model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    loss = manager.actor_loss(td, action)
    loss.backward()
    assert action.grad[..., 0].gt(0).all()
    assert action.grad[..., 1].eq(0).all()
    assert observation.grad is None and stored_action.grad is None
    assert all(parameter.grad is None for parameter in manager.model.parameters())
    with torch.no_grad():
        action -= action.grad
    assert manager.actor_loss(td, action) < loss
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, manager.model.state_dict()[k]) for k, v in before.items())
    metrics = manager.finish_actor_update()
    assert metrics["actor_constraint_violation_rate"] == 1
    assert metrics["actor_constraint_risk"] < 1  # excludes the h=1 bias of 100
    assert metrics["actor_constraint_next_weight"] > metrics["actor_constraint_weight"]
    # A current violation cannot be undone by the action; empty masks stay finite.
    td["agents", "info", "safety_margins"].fill_(1)
    assert manager.actor_loss(td, action) == 0
    assert manager.finish_actor_update()["actor_constraint_eligible_fraction"] == 0
    td["agents", "info", "safety_margins"].fill_(-0.5)
    with torch.no_grad():
        action.fill_(-1)
    weight = manager.constraint_weight
    manager.actor_loss(td, action)
    manager.finish_actor_update()
    assert manager.constraint_weight == weight  # zero budget: safe negatives do not decay lambda
    p.safety_constraint_risk_budget = 0.02
    manager.actor_loss(td, action)
    manager.finish_actor_update()
    assert 0 <= manager.constraint_weight < weight


def test_constraint_configuration_and_legacy_default():
    assert not Parameters.from_dict({}).is_using_safety_constraint
    with pytest.raises(ValueError):
        Parameters(is_using_safety_constraint=True, is_using_safety_critic=False)
    with pytest.raises(ValueError):
        Parameters(is_using_safety_constraint=True, safety_horizons=[1])
    for invalid in [dict(safety_gate_min_unsafe=0), dict(safety_gate_window=0),
                    dict(safety_gate_min_recall=1.1), dict(safety_gate_max_underestimate=-1),
                    dict(safety_constraint_risk_budget=-1), dict(safety_constraint_dual_lr=float("inf"))]:
        with pytest.raises(ValueError):
            Parameters(**invalid)


def test_reliability_gate_uses_future_valid_safe_states_and_can_close_again():
    p = Parameters(is_using_safety_constraint=True, safety_constraint_warmup_batches=1,
                   safety_gate_window=2, safety_gate_min_unsafe=4)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    manager.updates = manager.rollouts = 1
    target = torch.ones(4, 4)
    valid = torch.ones(4, 4, dtype=torch.bool)
    current = torch.full((4,), -0.5)
    good = target.clone()
    good[:, 0] = -100  # current-state head is irrelevant to the gate
    manager._record_reliability(good, target, valid, current)
    assert manager.constraint_ready
    manager._record_reliability(-target, target, valid, current)
    assert manager.constraint_gate()["reason"] == 4
    manager.reliability_history = []
    manager._record_reliability(target * 0.8, target, valid, current)
    assert manager.constraint_gate()["reason"] == 5  # recalled, but underestimated
    manager._record_reliability(good, target, valid, current)
    manager._record_reliability(good, target, valid, current)
    assert manager.constraint_ready  # old failures age out
    manager._record_reliability(good, target, valid, -current)
    manager._record_reliability(good, target, ~valid, current)
    assert manager.constraint_gate()["reason"] == 3
    assert manager.constraint_gate()["valid_count"] == 0
    manager.rollouts = 0
    assert manager.constraint_gate()["reason"] == 2


def test_negative_signed_mean_cannot_cancel_positive_risk_or_trap_zero_weight():
    p = Parameters(is_using_safety_constraint=True, safety_constraint_warmup_batches=1)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    manager.updates = manager.rollouts = 1
    manager.reliability_history = [[100, 64, 0, 0]]
    manager.constraint_weight = 0
    # Reproduces the old failure: signed mean -0.25 despite 30% violations.
    manager._constraint_totals = [-25.0, 15.0, 30, 100, 100]
    first = manager.finish_actor_update()
    assert first["actor_constraint_signed_risk"] < 0
    assert first["actor_constraint_positive_risk"] == 0.15
    assert first["actor_constraint_next_weight"] > 0
    assert first["actor_constraint_active"] == 0  # applied weight was zero
    for _ in range(200):
        manager._constraint_totals = [-25.0, 15.0, 30, 100, 100]
        metrics = manager.finish_actor_update()
    assert metrics["actor_constraint_active"] == 1
    assert manager.constraint_weight == p.safety_constraint_max_weight


def test_gate_records_predictions_before_fitting_and_clears_critic_gradients(monkeypatch):
    p = Parameters(safety_hidden_dim=16, safety_num_epochs=1, safety_minibatch_size=4)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    td, _, _ = make_rollout()
    before = manager.predict(td).reshape(-1, 4).clone()
    original = manager._record_reliability
    recorded = []
    def record(prediction, target, valid, current):
        recorded.append(prediction.clone())
        original(prediction, target, valid, current)
    monkeypatch.setattr(manager, "_record_reliability", record)
    manager.train_on_rollout(td)
    assert torch.equal(recorded[0], before)
    assert not torch.equal(manager.predict(td).reshape(-1, 4), before)
    assert all(v.grad is None for v in manager.model.parameters())


def test_safety_penalty_reaches_actor_and_message_aggregator():
    from utilities.nod_marl.policy import NODMessageAggregator
    p = Parameters(is_using_safety_constraint=True, safety_constraint_warmup_batches=1)
    manager = SafetyCriticManager(p, 3, 2, 2, ("agents", "observation"))
    manager.updates = manager.rollouts = 1
    manager.reliability_history = [[100, 64, 0, 0]]
    td, _, _ = make_rollout()
    aggregator = NODMessageAggregator(context_dim=7, message_dim=5, hidden_dim=9)
    context = torch.randn(2, 4, 2, 2, 7, requires_grad=True)
    message, _ = aggregator(context.detach(), torch.ones(2, 4, 2, 2, dtype=torch.bool))
    actor = torch.nn.Linear(5, 2)
    action = actor(message).tanh()
    manager.model = torch.nn.Linear(manager.input_dim, 4)
    with torch.no_grad():
        manager.model.weight.zero_()
        manager.model.bias.fill_(10)
        manager.model.weight[1:, 11] = 1
        manager.model.weight[1:, 24] = 1
    manager.actor_loss(td, action).backward()
    for module in [actor, aggregator]:
        grads = [v.grad for v in module.parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert context.grad is None
    assert all(v.grad is None for v in manager.model.parameters())


def test_constraint_warmup_functional_ppo_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in [False, True]:
            p = Parameters.from_json("config.json")
            p.safety_control_mode = 'legacy_q'
            p.seed = 12345
            p.n_iters = p.num_epochs = 2
            p.frames_per_batch = 64
            p.total_frames = 128
            p.minibatch_size = p.safety_minibatch_size = 32
            p.num_vmas_envs = 4
            p.max_steps = p.nod_sequence_length = 16
            p.safety_num_epochs = 2
            p.is_using_safety_constraint = enabled
            p.safety_constraint_warmup_batches = 1
            # Tiny wiring test only; production thresholds remain strict in config.
            p.safety_horizons = [1, 4]
            p.safety_gate_min_unsafe = 1
            p.safety_gate_min_recall = 0
            p.safety_gate_max_underestimate = 1
            p.safety_constraint_initial_weight = 0.1
            p.safety_constraint_margin = 1.0  # ensure an active penalty in this tiny run
            p.is_load_model = p.is_continue_train = False
            p.where_to_save = str(tmp_path / str(enabled)) + "/"
            env, _, _, _ = mappo_cavs(p)
            env.close()
        off = torch.load(tmp_path / "False/final_policy.pth")
        on = torch.load(tmp_path / "True/final_policy.pth")
        assert any(not torch.equal(off[k], on[k]) for k in off)
        # The penalty uses deterministic nominal actions without perturbing rollout RNG.
        data = [json.loads(next((tmp_path / str(e)).glob("reward*_data.json")).read_text())
                for e in [False, True]]
        assert data[0]["episode_reward_mean_list"] == data[1]["episode_reward_mean_list"]
        metrics = data[1]["safety_metrics_list"]
        assert metrics[0]["actor_constraint_active"] == 0
        assert metrics[1]["actor_constraint_active"] == 1
        assert metrics[1]["actor_constraint_loss"] > 0
        assert not (tmp_path / "True/final_deadlock_critic.pth").exists()
        assert all(m == {"enabled": 0.0} for m in data[1]["deadlock_metrics_list"])
        checkpoint = torch.load(tmp_path / "True/final_safety_critic.pth")
        assert checkpoint["rollouts"] == 2
        restored = SafetyCriticManager(p, 32, 4, 2, ("agents", "observation"))
        assert restored.load_if_available(tmp_path / "True/final_safety_critic.pth", True)
        assert restored.constraint_weight == checkpoint["constraint_weight"]
        assert restored.reliability_history == checkpoint["reliability_history"]
        assert restored.rollouts == 2
        # Old sidecars load, but without a recorded warmup start conservatively.
        checkpoint.pop("rollouts")
        checkpoint.pop("constraint_weight")
        checkpoint.pop("constraint_contract")
        checkpoint.pop("reliability_history")
        legacy = tmp_path / "legacy.pth"
        torch.save(checkpoint, legacy)
        old = SafetyCriticManager(p, 32, 4, 2, ("agents", "observation"))
        assert old.load_if_available(legacy, True)
        assert not old.constraint_ready
        assert old.rollouts == 0 and not old.reliability_history
        assert old.constraint_weight == p.safety_constraint_initial_weight
        p.safety_gate_min_recall = 0.95
        changed = SafetyCriticManager(p, 32, 4, 2, ("agents", "observation"))
        assert changed.load_if_available(tmp_path / "True/final_safety_critic.pth", True)
        assert changed.rollouts == 0 and not changed.constraint_ready
    finally:
        torch.set_num_threads(threads)
