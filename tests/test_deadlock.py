import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters, SaveData
from utilities.nod_marl.deadlock import (
    DEADLOCK_STATE_DIM,
    DeadlockTracker,
    DeadlockCriticManager,
    conflict_components,
    forward_escape_available,
)


def scene(count=2):
    pos = torch.zeros(1, count, 2)
    pos[0, :, 0] = torch.arange(count) * 0.4
    paths = pos.unsqueeze(-2) + torch.tensor([[[[0.1, 0.0], [0.4, 0.0], [0.8, 0.0]]]])
    return dict(
        positions=pos,
        velocities=torch.zeros_like(pos),
        yaws=torch.zeros(1, count),
        paths=paths,
        clearance=torch.full((1, count), 0.05),
        collision=torch.zeros(1, count, dtype=torch.bool),
        generations=torch.ones(1, count, dtype=torch.long),
        forward_allowed=torch.ones(1, count, dtype=torch.bool),
    )


def run_stopped(tracker, data, steps=80):
    output = None
    for t in range(steps):
        output = tracker.update(**data, steps=torch.tensor([t]))
    return output


def test_sustained_group_deadlock_and_idempotent_frame():
    p = Parameters()
    tracker = DeadlockTracker(p)
    data = scene()
    out = run_stopped(tracker, data, 50)
    assert (out["deadlock_margin"] <= 0).all()  # under 1s window + 2s persistence
    for t in range(50, 80):
        out = tracker.update(**data, steps=torch.tensor([t]))
    assert (out["deadlock_margin"] > 0).all()
    assert out["deadlock_state"].shape == (1, 2, DEADLOCK_STATE_DIM)
    before = tracker.eligible_time.clone()
    tracker.update(**data, steps=torch.tensor([79]))
    assert torch.equal(before, tracker.eligible_time)
    # Physical progress immediately breaks the persistent stalled-group label.
    data["velocities"][0, 1, 0] = 0.2
    data["positions"][0, 1, 0] += 0.03
    out = tracker.update(**data, steps=torch.tensor([80]))
    assert (out["deadlock_margin"] <= 0).all()


def test_red_light_normal_waiting_and_unestablished_escape_are_not_deadlock():
    for case in [
        "red",
        "moving_neighbor",
        "too_close",
        "boundary",
        "isolated",
        "different_corridors",
    ]:
        data = scene(1 if case == "isolated" else 2)
        if case == "red":
            data["forward_allowed"][:] = False
        if case == "moving_neighbor":
            data["velocities"][0, 1, 0] = 0.2
        if case == "too_close":
            data["positions"][0, 1, 0] = 0.1
        if case == "boundary":
            data["clearance"][:] = 0.005
        if case == "different_corridors":
            data["positions"][0, 1, 1] = 0.3
            data["paths"][0, 1, :, 1] = 0.3
        out = run_stopped(DeadlockTracker(Parameters()), data)
        assert (out["deadlock_margin"] <= 0).all(), case


def test_generation_reset_clears_history_and_group_timer():
    tracker = DeadlockTracker(Parameters())
    data = scene()
    assert (run_stopped(tracker, data)["deadlock_margin"] > 0).all()
    tracker.reset(0, 1)
    assert tracker.eligible_time.eq(0).all()
    data["generations"][0, 1] += 1
    data["positions"][0, 1, 0] = 0.5  # respawn teleport is not route progress
    out = tracker.update(**data, steps=torch.tensor([80]))
    assert (out["deadlock_margin"] <= 0).all()
    assert tracker.history[0, 1].eq(0).all()
    assert tracker.low_time[0, 1] == 0
    assert tracker.age[0, 1] == 0


def test_forward_probe_checks_swept_separation_not_just_endpoint():
    pos = torch.tensor([[[0.0, 0.0], [0.3, 0.1]]])
    vel = torch.tensor([[[0.0, 0.0], [-1.0, 0.0]]])
    good = forward_escape_available(
        pos,
        vel,
        torch.zeros(1, 2),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        torch.full((1, 2), 1.0),
        distance=0.1,
        speed=0.1,
        safe_distance=0.25,
        boundary_margin=0.01,
        min_progress=0.01,
    )
    assert not good[0, 0]  # other car crosses the ego's swept path


def test_deadlock_critic_positive_targets_rng_gradient_isolation_and_checkpoint(
    tmp_path,
):
    p = Parameters(
        deadlock_hidden_dim=16, deadlock_num_epochs=1, deadlock_minibatch_size=4
    )
    shape = (1, 8, 2)
    obs = torch.zeros(*shape, 3, requires_grad=True)
    action = torch.zeros(*shape, 2, requires_grad=True)
    data = {
        ("agents", "observation"): obs,
        ("agents", "action"): action,
        ("agents", "info", "pos"): torch.zeros(*shape, 2),
        ("agents", "info", "vel"): torch.zeros(*shape, 2),
        ("agents", "info", "rot"): torch.zeros(*shape, 1),
        ("agents", "info", "deadlock_state"): torch.zeros(*shape, DEADLOCK_STATE_DIM),
        ("agents", "info", "deadlock_margin"): torch.linspace(-1, 0.4, 8)
        .view(1, 8, 1, 1)
        .expand(*shape, 1),
        ("next", "agents", "info", "deadlock_margin"): torch.linspace(-0.8, 0.6, 8)
        .view(1, 8, 1, 1)
        .expand(*shape, 1),
        ("agents", "info", "deadlock_eligible_seconds"): torch.ones(*shape, 1),
        ("next", "agents", "info", "deadlock_onset"): torch.zeros(
            *shape, 1, dtype=torch.bool
        ),
        ("next", "done"): torch.tensor([[[False]] * 7 + [[True]]]),
    }
    for prefix in [(), ("next",)]:
        data[prefix + ("agents", "info", "nod_ego_generation")] = torch.ones(
            *shape, 1, dtype=torch.long
        )
    td = TensorDict(data, batch_size=[1, 8])
    rng = torch.random.get_rng_state().clone()
    manager = DeadlockCriticManager(p, 3, 2, 2, ("agents", "observation"))
    before = {k: v.clone() for k, v in manager.model.state_dict().items()}
    metrics = manager.train_on_rollout(td)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert obs.grad is None and action.grad is None
    assert metrics["deadlock_target_count"] > 0 and metrics["optimizer_updates"] == 2
    assert any(
        not torch.equal(before[k], v) for k, v in manager.model.state_dict().items()
    )
    path = tmp_path / "deadlock.pth"
    torch.save(manager.checkpoint_state(), path)
    restored = DeadlockCriticManager(p, 3, 2, 2, ("agents", "observation"))
    assert restored.load_if_available(path, load_optimizer=True)
    assert torch.equal(restored.predict(td), manager.predict(td))
    manager.train_on_rollout(td)
    restored.train_on_rollout(td)
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in manager.model.state_dict().items()
    )
    p.deadlock_duration_seconds = 3
    incompatible = DeadlockCriticManager(p, 3, 2, 2, ("agents", "observation"))
    assert not incompatible.load_if_available(path)
    assert SaveData.from_dict({"parameters": {}}).deadlock_metrics_list is None
