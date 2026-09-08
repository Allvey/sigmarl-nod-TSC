"""Stage 8B: detached fixed-kappa advantages on actual stochastic transitions.

Warmup enables an experiment, not a reliability certificate. No z, Q-action
gradient, or task Critic target is used/modified here.
"""
import torch

from .safety_value import same_entities, VALUE_HEADS


@torch.no_grad()
def fixed_barrier_advantage(task, current, following, value, next_value,
                            *, kappa, road_kappa, nu, strength):
    finite = (torch.isfinite(value) & torch.isfinite(next_value)
              & torch.isfinite(current['g']) & torch.isfinite(following['g']))
    valid = current['valid'] & following['valid'] & same_entities(current, following) & finite
    # All currently required constraints must have usable successors. Unknown
    # constraints cause task fallback, never an all-safe classification.
    eligible = current['valid'].any(-1) & (~current['valid'] | valid).all(-1)
    coefficients = torch.full_like(value, kappa)
    coefficients[..., -2] = road_kappa
    # Already unsafe states use fixed non-increase recovery, with no relaxation.
    recovery = (value > 0) | (current['g'] > 0)
    coefficients = torch.where(recovery, 0., coefficients)
    delta = next_value - (1 - coefficients) * value
    valid &= torch.isfinite(delta)
    eligible &= (~current['valid'] | valid).all(-1)
    delta = torch.where(valid, delta, 0.)
    penalty = delta.clamp_min(0).amax(-1, keepdim=True)
    violates = penalty > 0
    # beta=1 recovers the planned hard task mask; beta=0 is exact task PPO.
    # A small beta intentionally softens both the task mask and penalty.
    mixed = task * (1 - strength * violates.to(task.dtype)) - strength * nu * penalty
    adjusted = torch.where(eligible.unsqueeze(-1), mixed, task).detach()
    return adjusted, dict(delta=delta, valid=valid, eligible=eligible,
                          penalty=penalty, recovery=recovery & valid)


@torch.no_grad()
def prepare_barrier_advantage(manager, td, advantage_key):
    """Once per rollout, before PPO epochs; Value is held fixed for both states."""
    p = manager.parameters
    enabled = (p.safety_control_mode == 'barrier_fixed' and p.is_using_safety_constraint
               and manager.enabled)
    ready = enabled and manager.barrier_fit_batches >= p.safety_barrier_warmup_batches and manager.updates > 0
    metrics = dict(barrier_enabled=float(enabled), barrier_ready=float(ready), barrier_active=0.,
                   barrier_gate_reason=0. if ready else (2. if enabled else 1.),
                   barrier_fit_batches=float(manager.barrier_fit_batches),
                   barrier_strength=p.safety_barrier_strength, barrier_nu=p.safety_barrier_nu,
                   barrier_kappa=p.safety_barrier_kappa, barrier_road_kappa=p.safety_barrier_road_kappa,
                   barrier_eligible_count=0., barrier_fallback_count=0.,
                   barrier_violation_rate=0., barrier_advantage_delta_abs=0.)
    if enabled:
        # Stable replay-buffer schema across the warmup -> active transition.
        td.set(('agents', 'barrier_task_advantage'), td.get(advantage_key).detach().clone())
    if not ready or p.safety_barrier_strength == 0:
        return metrics
    current, following = manager.state(td), manager.state(td.get('next'))
    value, next_value = manager.model(current), manager.model(following)
    # Same terminal convention as 8A: observed terminal physics, no imagined
    # continuation or bridge to the next episode. Collector cuts are not done.
    done = td.get(('next', 'done')).bool().reshape(*td.batch_size, 1, 1)
    next_value = torch.where(done, following['g'], next_value)
    task = td.get(advantage_key).detach()
    adjusted, info = fixed_barrier_advantage(
        task, current, following, value, next_value,
        kappa=p.safety_barrier_kappa, road_kappa=p.safety_barrier_road_kappa,
        nu=p.safety_barrier_nu, strength=p.safety_barrier_strength)
    td.set(advantage_key, adjusted)
    eligible = info['eligible']
    violation = info['penalty'].squeeze(-1) > 0
    def mean(x):
        return float(x.float().mean()) if x.numel() else 0.
    metrics.update(barrier_active=float(bool((adjusted != task).any())),
                   barrier_eligible_count=float(eligible.sum()),
                   barrier_fallback_count=float((~eligible).sum()),
                   barrier_violation_rate=mean(violation[eligible]),
                   barrier_positive_mean=mean(info['penalty'].squeeze(-1)[eligible]),
                   barrier_advantage_delta_abs=mean((adjusted - task).abs()),
                   barrier_task_retention_mean=mean(1 - p.safety_barrier_strength * violation[eligible].float()))
    for name, selection in VALUE_HEADS:
        mask = info['valid'][..., selection] & eligible.unsqueeze(-1)
        c = info['delta'][..., selection][mask]
        metrics.update({f'barrier_{name}_count': float(mask.sum()),
                        f'barrier_{name}_positive_rate': mean(c > 0),
                        f'barrier_{name}_positive_mean': mean(c.clamp_min(0)),
                        f'barrier_{name}_recovery_count': float(info['recovery'][..., selection][mask].sum())})
    return metrics
