"""Keep rollout scores paired with the weights that collected them."""

import copy
import json
from pathlib import Path

import torch


def snapshot_state(value):
    """Detach all tensors, including buffers and optimizer state, onto CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = snapshot_state(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [snapshot_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_state(item) for item in value)
    return copy.deepcopy(value)


def write_scored_checkpoint(directory, prefix, snapshot, data):
    """Write a complete scored bundle using existing reward-prefix conventions.

    Periodic bundles belong in their own iteration subdirectories so existing
    evaluation can select them without confusing the root's best checkpoint.
    This function never queries live modules after their optimizer updates.
    """
    if not {"policy", "critic", "nod", "safety_value"}.issubset(snapshot):
        raise ValueError("Scored checkpoint requires policy, critic, NOD and Safety Value")
    if Path(prefix).name != prefix or not prefix:
        raise ValueError("Checkpoint prefix must be a filename component")
    directory = Path(directory)
    metadata = json.dumps(data, indent=4, allow_nan=False)
    directory.mkdir(parents=True, exist_ok=True)
    for name, state in snapshot.items():
        torch.save(state, directory / f"{prefix}_{name}.pth")
    with (directory / f"{prefix}_data.json").open("w") as handle:
        handle.write(metadata)
