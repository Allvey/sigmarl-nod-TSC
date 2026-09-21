import torch

from utilities.nod_marl.counterfactual import build_counterfactual_labels


def _two_agent_rollout():
    positions = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ]
    )
    velocities = torch.zeros_like(positions)
    velocities[:, 0, 1, 0] = -1.0
    neighbor_indices = torch.tensor([[[[1], [0]]] * 4])
    edge_mask = torch.ones(1, 4, 2, 1, dtype=torch.bool)
    generations = torch.ones(1, 4, 2, dtype=torch.long)
    return positions, velocities, generations, neighbor_indices, edge_mask


def test_counterfactual_label_rewards_observed_risk_mitigation():
    args = _two_agent_rollout()
    labels = build_counterfactual_labels(
        *args,
        horizon=2,
        dt=1.0,
        safe_distance=0.5,
        label_slope=12.0,
        label_margin=0.02,
    )
    assert labels["valid"][0, 0, 0, 0]
    assert labels["gap"][0, 0, 0, 0] > 0
    assert labels["label"][0, 0, 0, 0] > 0.5


def test_counterfactual_label_is_invalid_across_identity_reset():
    positions, velocities, generations, neighbor_indices, edge_mask = (
        _two_agent_rollout()
    )
    generations[:, 1:, 1] = 2
    labels = build_counterfactual_labels(
        positions,
        velocities,
        generations,
        neighbor_indices,
        edge_mask,
        horizon=2,
        dt=1.0,
        safe_distance=0.5,
        label_slope=12.0,
        label_margin=0.02,
    )
    assert not labels["valid"][0, 0, 0, 0]


def _interaction_labels(args, **overrides):
    settings = dict(horizon=2, dt=1., safe_distance=.5, label_slope=12.,
                    label_margin=.02, mode="interaction", reference_seconds=3.)
    settings.update(overrides)
    return build_counterfactual_labels(*args, **settings)


def test_stopped_neighbor_keeps_yield_credit_after_braking_has_finished():
    args = _two_agent_rollout()
    result = _interaction_labels(args)
    # At t=1 the observed speed is already zero. The legacy reference sees
    # identical actual and stationary futures, losing the braking contribution.
    legacy = _interaction_labels(args, mode="instantaneous")
    assert legacy['gap'][0, 1, 0, 0] == 0
    assert result['valid'][0, 1, 0, 0]
    assert result['reference_active'][0, 1, 0, 0]
    assert result['label'][0, 1, 0, 0] > .5


def test_stationary_vehicle_without_prior_conflict_is_neutral():
    args = _two_agent_rollout()
    args[1].zero_()
    result = _interaction_labels(args)
    assert not result['reference_active'].any()
    assert (result['label'][result['valid']] == .5).all()


def test_unchanged_motion_during_conflict_has_neutral_label():
    args = _two_agent_rollout()
    args[0][0, :, 1, 0] = torch.tensor([1., .6, .2, -.2])
    args[1][0, :, 1, 0] = -.4
    result = _interaction_labels(args)
    assert result['reference_active'][0, 0, 0, 0]
    assert result['label'][0, 0, 0, 0] == .5


def test_motion_that_increases_conflict_risk_gets_negative_label():
    args = _two_agent_rollout()
    args[0][0, :, 1, 0] = torch.tensor([1., .7, .1, .05])
    args[1][0, 0, 1, 0] = -.4
    result = _interaction_labels(args)
    assert result['label'][0, 0, 0, 0] < .5


def test_reference_releases_after_ego_has_passed():
    args = _two_agent_rollout()
    args[0][0, 1:, 0, 0] = 2.
    result = _interaction_labels(args)
    assert not result['reference_active'][0, 1, 0, 0]
    assert result['label'][0, 1, 0, 0] == .5


def test_timeout_discards_reference_without_immediate_rearming():
    args = _two_agent_rollout()
    args[1][0, :, 1, 0] = -.4  # Moving and still in conflict at expiry.
    result = _interaction_labels(args, reference_seconds=.5)
    assert result['reference_active'][0, 0, 0, 0]
    assert not result['reference_active'][0, 1:, 0, 0].any()
    assert result['label'][0, 1, 0, 0] == .5


def test_turn_discards_stale_straight_line_reference():
    args = _two_agent_rollout()
    args[1][0, 1:, 1, 1] = 1.
    result = _interaction_labels(args)
    assert not result['reference_active'][0, 1, 0, 0]
    assert result['label'][0, 1, 0, 0] == .5


def test_reset_and_visibility_gap_do_not_transfer_yield_credit():
    for kind in ['generation', 'visibility']:
        args = _two_agent_rollout()
        if kind == 'generation':
            args[2][0, 1:, 1] = 2
        else:
            args[4][0, 1, 0, 0] = False
        result = _interaction_labels(args)
        assert not result['reference_active'][0, 1:, 0, 0].any()
        if kind == 'generation':
            assert not result['valid'][0, 0, 0, 0]


def test_reference_selection_does_not_use_future_motion():
    args = _two_agent_rollout()
    original = _interaction_labels(args)
    args[0][:, 2:] += 4.
    args[1][:, 2:] += 3.
    changed = _interaction_labels(args)
    assert torch.equal(original['reference_active'][:, :2], changed['reference_active'][:, :2])
    assert torch.equal(original['reference_age'][:, :2], changed['reference_age'][:, :2])
