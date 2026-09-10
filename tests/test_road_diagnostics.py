from utilities.diagnose_road_safety import history_indices


def test_collision_history_stops_at_respawn():
    assert list(history_indices([0, 0, 1, 1, 1], 4, 20)) == [2, 3, 4]
    assert list(history_indices([0, 0, 1], 2, 20)) == [2]


def test_collision_history_respects_window_and_episode_start():
    assert list(history_indices([3] * 30, 29, 20)) == list(range(10, 30))
    assert list(history_indices([3] * 30, 2, 20)) == [0, 1, 2]
