import torch

from utilities.leader_selection import build_soft_label_leader_gate


def test_soft_label_leader_gate_suppresses_weak_leaders():
    p_final = torch.tensor(
        [
            [
                [0.0, 0.59, 0.61],
                [0.41, 0.0, 0.70],
                [0.39, 0.30, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    selected_neighbors = torch.tensor([[[1, 2], [0, 2], [0, 1]]])

    gate = build_soft_label_leader_gate(
        p_final=p_final,
        selected_neighbor_indices=selected_neighbors,
        leader_margin=0.1,
    )

    expected = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]])
    assert torch.equal(gate, expected)


def test_soft_label_leader_gate_ignores_invalid_neighbor_slots():
    p_final = torch.tensor(
        [
            [
                [0.0, 0.75],
                [0.25, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    selected_neighbors = torch.tensor([[[1, -1], [0, -1]]])

    gate = build_soft_label_leader_gate(
        p_final=p_final,
        selected_neighbor_indices=selected_neighbors,
        leader_margin=0.1,
    )

    expected = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert torch.equal(gate, expected)
