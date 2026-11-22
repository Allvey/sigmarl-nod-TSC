"""Ordered PPO windows and staged evidence-gate optimization."""

import torch


def ordered_gate_windows(rollout, sequence_length):
    """Pack [environment,time] data without mixing environments or padding.

    Shuffle these windows, never the time steps inside a window. The collected
    previous opinion at each window start is the detached TBPTT boundary state.
    """
    if rollout.ndim != 2 or sequence_length < 2:
        raise ValueError("Gate sequence replay requires [environment,time] data and length >= 2")
    time_steps = rollout.shape[1]
    if time_steps % sequence_length:
        raise ValueError("Gate sequence length must divide the collected time dimension")
    return torch.cat([
        rollout[:, start:start + sequence_length]
        for start in range(0, time_steps, sequence_length)
    ], dim=0)


def freeze_actor_body_gradients(actor_body_parameters, gate_only):
    """Keep fixed actor layers differentiable in their inputs but skip updates.

    Setting only their parameter gradients to None also prevents Adam momentum
    from moving them. The gate gradient can still pass through the fixed layers.
    Critic parameters are not part of this list and keep their own gradients.
    """
    if gate_only:
        for parameter in actor_body_parameters:
            parameter.grad = None
