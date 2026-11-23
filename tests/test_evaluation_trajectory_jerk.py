import torch

from utilities.evaluation_base import _compute_trajectory_jerk_rms


def test_trajectory_jerk_ignores_respawns_and_preserves_other_vehicles():
    velocities = torch.zeros(1, 6, 2, 2)
    velocities[0, :, 0, 0] = torch.tensor([0, 1, 4, 100, 101, 104])
    velocities[0, :, 1, 1] = torch.tensor([0, 3, 6, 9, 12, 15])
    generations = torch.zeros(1, 6, 2, 1)
    generations[0, 3:, 0] = 1

    rms = _compute_trajectory_jerk_rms(velocities, generations, dt=0.5)

    # Two valid samples of 8 m/s³ and four samples of zero jerk.
    torch.testing.assert_close(rms, torch.tensor([(2 * 8**2 / 6) ** 0.5]))


def test_trajectory_jerk_is_missing_without_a_continuous_three_frame_window():
    velocities = torch.zeros(1, 3, 1, 2)
    generations = torch.tensor([[[0], [0], [1]]])

    assert _compute_trajectory_jerk_rms(velocities, generations, dt=0.05).isnan().all()
    assert _compute_trajectory_jerk_rms(
        velocities[:, :2], generations[:, :2], dt=0.05
    ).isnan().all()
