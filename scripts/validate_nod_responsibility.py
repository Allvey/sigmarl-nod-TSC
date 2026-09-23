"""Deterministic two-vehicle acceptance cases for responsibility opinions.

Run from the repository root:

    python -m scripts.validate_nod_responsibility
    python -m scripts.validate_nod_responsibility --checkpoint \
        outputs/dgppo_nod_candidate_only/final_nod.pth
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from utilities.nod_marl.counterfactual import build_responsibility_evidence
from utilities.nod_marl.interaction import (
    APPROACH_CONFIDENCE,
    CONFLICT_VALID,
    DISTANCE,
    ETA_GAP,
    OVERLAP_RISK,
    TTC,
)
from utilities.nod_marl.opinion import NODOpinionModel


def _rollout(kind: str):
    steps = 5
    dt = 0.25
    positions = torch.zeros(1, steps, 2, 2)
    velocities = torch.zeros_like(positions)
    for time_index in range(steps):
        progress = dt * time_index
        positions[0, time_index, 0] = torch.tensor([-1.0 + progress, 0.0])
        positions[0, time_index, 1] = torch.tensor([0.0, -1.0 + progress])
        velocities[0, time_index, 0] = torch.tensor([1.0, 0.0])
        velocities[0, time_index, 1] = torch.tensor([0.0, 1.0])

    if kind == "partial_yield":
        for time_index in range(1, steps):
            # 0.88 of the original speed produces roughly half of the required
            # risk reduction in this symmetric crossing geometry.
            progress = 0.88 * dt * time_index
            positions[0, time_index, 1] = torch.tensor([0.0, -1.0 + progress])
            velocities[0, time_index, 1] = torch.tensor([0.0, 0.88])
    elif kind == "full_yield":
        positions[0, 1:, 1] = torch.tensor([0.0, -1.0])
        velocities[0, 1:, 1] = 0.0
    elif kind == "ego_only_yield":
        positions[0, 1:, 0] = torch.tensor([-1.0, 0.0])
        velocities[0, 1:, 0] = 0.0
    elif kind == "no_conflict":
        for time_index in range(steps):
            positions[0, time_index, 1] = torch.tensor(
                [0.0, 1.0 + dt * time_index]
            )
    elif kind != "keep_speed":
        raise ValueError(f"Unknown case: {kind}")

    generations = torch.ones(1, steps, 2, dtype=torch.long)
    neighbor_indices = torch.tensor([[[[1], [0]]] * steps])
    visible = torch.ones(1, steps, 2, 1, dtype=torch.bool)
    return positions, velocities, generations, neighbor_indices, visible, dt


def _model(checkpoint: Optional[str]) -> NODOpinionModel:
    model = NODOpinionModel(
        pair_feature_dim=20,
        hidden_dim=64,
        fixed_evidence_mapping=True,
    )
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(saved.get("model", saved))
    model.eval()
    return model


@torch.no_grad()
def evaluate_case(model: NODOpinionModel, kind: str):
    positions, velocities, generations, ids, visible, dt = _rollout(kind)
    responsibility, _ = build_responsibility_evidence(
        positions,
        velocities,
        generations,
        ids,
        visible,
        dt=dt,
        lookahead=2.0,
        safe_distance=0.25,
        max_age=3.0,
    )
    pair = torch.zeros(1, positions.shape[1], 2, 1, 20)
    pair[..., TTC] = 0.1
    pair[..., DISTANCE] = 0.1
    pair[..., ETA_GAP] = 0.0
    pair[..., OVERLAP_RISK] = 1.0
    pair[..., APPROACH_CONFIDENCE] = 1.0
    pair[..., CONFLICT_VALID] = 1.0
    neighbor_generations = torch.ones_like(ids)
    outputs, _ = model.forward_sequence(
        pair,
        visible,
        generations,
        neighbor_generations,
        opinion_mask=responsibility["active"],
        evidence_override=responsibility["evidence"],
    )
    index = 1
    return {
        "active": bool(responsibility["active"][0, index, 0, 0]),
        "q": float(responsibility["responsibility"][0, index, 0, 0]),
        "e": float(responsibility["evidence"][0, index, 0, 0]),
        "z": float(outputs["z"][0, index, 0, 0]),
        "cleared": not bool(responsibility["active"][0, -1, 0, 0]),
        "final_z": float(outputs["z"][0, -1, 0, 0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    model = _model(args.checkpoint)
    cases = [
        "keep_speed",
        "partial_yield",
        "full_yield",
        "ego_only_yield",
        "no_conflict",
    ]
    result = {name: evaluate_case(model, name) for name in cases}
    print("case             active       q        e        z")
    for name in cases:
        item = result[name]
        print(
            f"{name:16s} {str(item['active']):>6s} "
            f"{item['q']:+8.3f} {item['e']:+8.3f} {item['z']:+8.3f}"
        )

    assert result["keep_speed"]["z"] < 0.0
    assert result["ego_only_yield"]["z"] < 0.0
    assert result["full_yield"]["z"] > result["partial_yield"]["z"]
    assert result["partial_yield"]["z"] > result["keep_speed"]["z"]
    assert not result["no_conflict"]["active"]
    assert result["no_conflict"]["z"] == 0.0
    assert result["keep_speed"]["cleared"]
    assert result["keep_speed"]["final_z"] == 0.0
    print("PASS: controlled responsibility ordering is valid")


if __name__ == "__main__":
    main()
