"""Geometry checks for small training-only translations from a reference path."""
import torch


def safe_lateral_position(position, yaw, offset, *, width, length, left, right,
                          is_loop, other_positions, min_distance, clearance):
    """Return a candidate or None; never mutate the supplied state.

    Starting from a center-line spawn, the entire swept vehicle rectangle must
    avoid both road boundaries (and non-loop end caps), including tangencies.
    Checking the sweep also prevents jumping completely across a boundary.
    Other vehicles retain the scenario's conservative center-distance clearance.
    """
    tangent = torch.stack((yaw.cos(), yaw.sin())).reshape(2)
    normal = torch.stack((-tangent[1], tangent[0]))
    candidate = position + offset * normal
    if other_positions.numel() and bool(
            ((other_positions - candidate).square().sum(-1) < min_distance ** 2).any()):
        return None

    polylines = [left, right]
    if not is_loop:
        polylines += [torch.stack((left[0], right[0])),
                      torch.stack((left[-1], right[-1]))]
    starts = torch.cat([line[:-1] for line in polylines])
    ends = torch.cat([line[1:] for line in polylines])
    basis = torch.stack((tangent, normal), dim=1)
    center = position + offset * normal / 2
    starts, ends = (starts - center) @ basis, (ends - center) @ basis
    half_size = position.new_tensor([length / 2 + clearance,
                                     width / 2 + abs(float(offset)) / 2 + clearance])
    # Segment/box slab test, inclusive of endpoints and parallel tangencies.
    delta = ends - starts
    parallel = delta.abs() < 1e-12
    denominator = torch.where(parallel, torch.ones_like(delta), delta)
    t1, t2 = (-half_size - starts) / denominator, (half_size - starts) / denominator
    enter = torch.where(parallel, -torch.inf, torch.minimum(t1, t2)).amax(-1).clamp_min(0)
    leave = torch.where(parallel, torch.inf, torch.maximum(t1, t2)).amin(-1).clamp_max(1)
    outside_parallel = (parallel & (starts.abs() > half_size)).any(-1)
    if bool(((enter <= leave) & ~outside_parallel).any()):
        return None
    return candidate
