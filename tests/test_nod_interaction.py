import torch

from utilities.nod_marl.interaction import (
    CONFLICT_VALID,
    NOD_PAIR_FEATURE_DIM,
    VISIBLE,
    build_directed_interactions,
)


def test_directed_candidates_have_stable_global_identity_and_explicit_masks():
    positions = torch.tensor(
        [[[0.0, 0.0], [0.35, 0.0], [0.0, 0.4], [2.0, 2.0]]]
    )
    velocities = torch.tensor(
        [[[0.2, 0.0], [-0.2, 0.0], [0.0, -0.2], [0.0, 0.0]]]
    )
    yaws = torch.zeros(1, 4)
    offsets = torch.tensor([0.1, 0.2, 0.3]).view(1, 1, 3, 1)
    paths = positions.unsqueeze(2).repeat(1, 1, 3, 1)
    paths[..., 0] += offsets[..., 0]

    result = build_directed_interactions(
        positions,
        velocities,
        yaws,
        paths,
        ego_index=0,
        sensing_range=0.8,
        interaction_distance=0.48,
        ttc_limit=2.0,
        conflict_radius=0.08,
        max_speed=1.0,
    )

    assert result["features"].shape == (1, 3, NOD_PAIR_FEATURE_DIM)
    assert torch.equal(result["neighbor_indices"], torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(result["features"]).all()
    assert result["edge_mask"].dtype == torch.bool
    assert result["conflict_valid"].dtype == torch.bool
    assert result["features"][0, 2, VISIBLE].item() == 0.0
    assert result["features"][0, 2, CONFLICT_VALID].item() == 0.0
    assert not result["edge_mask"][0, 2]


def test_pair_features_are_directed_in_ego_coordinates():
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    velocities = torch.zeros_like(positions)
    paths = positions.unsqueeze(2).repeat(1, 1, 2, 1)
    result_0 = build_directed_interactions(
        positions,
        velocities,
        torch.zeros(1, 2),
        paths,
        ego_index=0,
        sensing_range=2.0,
        interaction_distance=2.0,
        ttc_limit=2.0,
        conflict_radius=0.05,
        max_speed=1.0,
    )
    result_1 = build_directed_interactions(
        positions,
        velocities,
        torch.zeros(1, 2),
        paths,
        ego_index=1,
        sensing_range=2.0,
        interaction_distance=2.0,
        ttc_limit=2.0,
        conflict_radius=0.05,
        max_speed=1.0,
    )
    assert result_0["features"][0, 0, 0] > 0
    assert result_1["features"][0, 0, 0] < 0


def test_segment_intersection_is_detected_between_sparse_path_points():
    positions = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])
    velocities = torch.zeros_like(positions)
    paths = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])

    result = build_directed_interactions(
        positions,
        velocities,
        torch.zeros(1, 2),
        paths,
        ego_index=0,
        sensing_range=3.0,
        interaction_distance=0.1,
        ttc_limit=2.0,
        conflict_radius=0.01,
        max_speed=1.0,
    )

    assert result["path_intersects"][0, 0]
    assert result["conflict_valid"][0, 0]
    assert result["edge_mask"][0, 0]
