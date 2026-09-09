"""Stages 8B/9: detached barrier advantages on actual stochastic transitions.

Warmup enables an experiment, not a reliability certificate. Cached z may
adjust pair kappa; no gradient flows through z, Value or the task targets.
"""
import torch

from .safety_value import same_entities, VALUE_HEADS


@torch.no_grad()
def aligned_opinion_kappa(td, current, *, minimum, maximum):
    """Cached pre-action z (last context coordinate), aligned by world slot.

    Missing/invalid opinions use minimum, never masquerade as neutral z=0.
    Neighbor generations are verified against the current world identities;
    successor continuity remains the barrier's separate validity check.
    """
    n = current['g'].shape[-2]
    shape = current['g'][..., :n].shape
    kappas = current['g'].new_full(shape, minimum)
    opinions = torch.zeros_like(kappas)
    available = torch.zeros_like(kappas, dtype=torch.bool)
    info = td.get(('agents', 'info'))
    keys = ['nod_actor_edge_context', 'nod_actor_edge_mask', 'nod_actor_context_ready',
            'nod_neighbor_indices', 'nod_neighbor_generation', 'nod_edge_mask']
    context, mask, ready, ids, generations, physical_mask = [info.get(k, default=None) for k in keys]
    if any(x is None for x in (context, mask, ready, ids, generations, physical_mask)):
        return kappas, available, opinions
    ids = ids.long()
    z = context[..., -1].detach()
    if z.shape != ids.shape or mask.shape != ids.shape or generations.shape != ids.shape:
        raise ValueError('Opinion cache and neighbor identities have incompatible shapes')
    safe_ids = ids.clamp(0, n - 1)
    valid_id = (ids >= 0) & (ids < n)
    counts = torch.zeros_like(kappas, dtype=torch.long)
    counts.scatter_add_(-1, safe_ids, valid_id.long())
    unique = counts.gather(-1, safe_ids) == 1
    expected = current['other_gen'][..., :n].gather(-1, safe_ids)
    valid = (valid_id & unique & mask.bool() & physical_mask.bool() & ready.bool()
             & torch.isfinite(z) & (z >= -1) & (z <= 1) & (generations.long() == expected))
    # scatter_add avoids an invalid clamped index overwriting a valid opinion.
    opinions.scatter_add_(-1, safe_ids, torch.where(valid, z, 0.))
    available.scatter_add_(-1, safe_ids, valid)
    available &= current['valid'][..., :n]
    available &= ~torch.eye(n, dtype=torch.bool, device=kappas.device)
    mapped = (minimum + maximum) / 2 + (maximum - minimum) / 2 * opinions
    kappas = torch.where(available, mapped, kappas)
    return kappas, available, opinions


@torch.no_grad()
def fixed_barrier_advantage(task, current, following, value, next_value,
                            *, kappa, road_kappa, nu, strength, pair_kappa=None):
    finite = (torch.isfinite(value) & torch.isfinite(next_value)
              & torch.isfinite(current['g']) & torch.isfinite(following['g']))
    valid = current['valid'] & following['valid'] & same_entities(current, following) & finite
    # All currently required constraints must have usable successors. Unknown
    # constraints cause task fallback, never an all-safe classification.
    eligible = current['valid'].any(-1) & (~current['valid'] | valid).all(-1)
    coefficients = torch.full_like(value, kappa)
    coefficients[..., -2] = road_kappa
    if pair_kappa is not None:
        coefficients[..., :-2] = pair_kappa.detach()
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
    opinion_mode = p.safety_control_mode == 'barrier_opinion'
    enabled = (p.safety_control_mode in {'barrier_fixed', 'barrier_opinion'} and p.is_using_safety_constraint
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
    pair_kappa = None
    if opinion_mode:
        pair_kappa, opinion_available, opinions = aligned_opinion_kappa(
            td, current, minimum=p.safety_barrier_kappa_min, maximum=p.safety_barrier_kappa_max)
    adjusted, info = fixed_barrier_advantage(
        task, current, following, value, next_value,
        kappa=p.safety_barrier_kappa, road_kappa=p.safety_barrier_road_kappa,
        nu=p.safety_barrier_nu, strength=p.safety_barrier_strength, pair_kappa=pair_kappa)
    td.set(advantage_key, adjusted)
    eligible = info['eligible']
    violation = info['penalty'].squeeze(-1) > 0
    def mean(x):
        return float(x.float().mean()) if x.numel() else 0.
    if opinion_mode:
        fixed, fixed_info = fixed_barrier_advantage(
            task, current, following, value, next_value,
            kappa=p.safety_barrier_kappa, road_kappa=p.safety_barrier_road_kappa,
            nu=p.safety_barrier_nu, strength=p.safety_barrier_strength)
        pair_mask = (info['valid'][..., :-2] & eligible.unsqueeze(-1)
                     & ~info['recovery'][..., :-2])
        available = pair_mask & opinion_available
        delta_diff = info['delta'][..., :-2] - fixed_info['delta'][..., :-2]
        flips = (info['delta'][..., :-2] > 0) != (fixed_info['delta'][..., :-2] > 0)
        argmax = info['delta'].clamp_min(0).argmax(-1)
        metrics.update(
            opinion_pair_count=float(pair_mask.sum()), opinion_valid_count=float(available.sum()),
            opinion_missing_count=float((pair_mask & ~opinion_available).sum()),
            opinion_z_mean=mean(opinions[available]),
            opinion_kappa_mean=mean(pair_kappa[pair_mask]),
            opinion_kappa_min=float(pair_kappa[pair_mask].min()) if pair_mask.any() else 0.,
            opinion_kappa_max=float(pair_kappa[pair_mask].max()) if pair_mask.any() else 0.,
            opinion_c_delta_abs=mean(delta_diff[pair_mask].abs()),
            opinion_pair_flip_rate=mean(flips[pair_mask]),
            opinion_available_pair_flip_rate=mean(flips[available]),
            opinion_agent_flip_rate=mean((violation != (fixed_info['penalty'].squeeze(-1) > 0))[eligible]),
            opinion_advantage_delta_abs=mean((adjusted - fixed).abs()),
            opinion_advantage_changed_count=float((adjusted != fixed).sum()),
            opinion_pair_controls_max_rate=mean((argmax < value.shape[-2])[eligible & violation]),
        )
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
