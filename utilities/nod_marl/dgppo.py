"""DGPPO-style safety updates with the existing local PairSafetyValue.

Reference: MIT-REALM/dgppo, algo/utils.py and algo/dgppo.py.
Upstream copyright (c) 2025 REALM; MIT license in DGPPO_LICENSE.txt.
This is a PyTorch adaptation, not the full graph/recurrent DGPPO algorithm.
Identity masks and observed terminal boundaries adapt it to respawning VMAS cars.
"""
import torch

from .safety_value import same_entities, discounted_max_targets
from .barrier import aligned_opinions


@torch.no_grad()
def opinion_alpha(td, current, value, *, alpha, span, gain=1.0, deadzone=0.0):
    """Adjust only safe, identity-aligned pair heads using pre-action opinions."""
    available, opinions = aligned_opinions(td, current)
    n = opinions.shape[-1]
    g, v = current['g'][..., :n], value[..., :n]
    safe = current['valid'][..., :n] & torch.isfinite(g) & torch.isfinite(v) & (g <= 0) & (v < 0)
    applied = available & safe & (opinions.abs() > float(deadzone))
    mapped_opinions = (gain * opinions).clamp(-1., 1.)
    coefficients = alpha.clone() if isinstance(alpha, torch.Tensor) else torch.full_like(value, alpha)
    pair_base = coefficients[..., :n].clone()
    coefficients[..., :n] = torch.where(
        applied, pair_base + span * mapped_opinions, pair_base
    )
    return coefficients, dict(available=available, opinions=opinions, safe=safe, applied=applied,
                             mapped_opinions=mapped_opinions)


@torch.no_grad()
def constraint_alpha(value, *, alpha, road_safe_alpha, road_recovery_alpha):
    """Use a stricter safe-side road rate and a stronger unsafe recovery rate."""
    if road_safe_alpha == alpha and road_recovery_alpha == alpha:
        return alpha
    coefficients = torch.full_like(value, alpha)
    road_value = value[..., -2]
    coefficients[..., -2] = torch.where(
        road_value > 0,
        road_value.new_tensor(road_recovery_alpha),
        road_value.new_tensor(road_safe_alpha),
    )
    return coefficients


@torch.no_grad()
def dgppo_targets(current, following, next_value, done, gamma, gae_lambda):
    """Exact lambda mixture of nonlinear n-step max backups (not max of a mixture).

    On a continuous trajectory this matches compute_dec_ocp_gae's default
    discount_to_max=True. Missing identities end the available horizon, using
    the preceding transition's own bootstrap. True terminals use observed g.
    Shapes: [environment, time, agent, constraint].
    """
    g, gn = current['g'], following['g']
    terminal = done.bool().reshape(*g.shape[:2], 1, 1).expand_as(g)
    valid = (current['valid'] & following['valid'] & same_entities(current, following)
             & torch.isfinite(g) & torch.isfinite(gn) & torch.isfinite(next_value))
    connected = torch.zeros_like(valid)
    connected[:, :-1] = (
        valid[:, :-1] & valid[:, 1:] & ~terminal[:, :-1]
        & (following['ego_gen'][:, :-1] == current['ego_gen'][:, 1:])
        & (following['other_gen'][:, :-1] == current['other_gen'][:, 1:]))
    lengths = torch.ones_like(g, dtype=torch.long)
    for t in reversed(range(g.shape[1] - 1)):
        lengths[:, t] += torch.where(connected[:, t], lengths[:, t + 1], 0)
    physical_valid = current['valid'] & torch.isfinite(g)
    worst = g.masked_fill(~physical_valid, -torch.inf).amax(-1, keepdim=True)
    worst = torch.where(physical_valid.any(-1, keepdim=True), worst, 0.)
    bootstrap = torch.where(terminal, gn, next_value)
    backup = torch.maximum(g, (1 - gamma) * worst + gamma * bootstrap)
    target = torch.zeros_like(g)
    for horizon in range(1, int(lengths.max()) + 1):
        weight = g.new_tensor(gae_lambda ** (horizon - 1))
        weight = torch.where(lengths == horizon, weight, weight * (1 - gae_lambda))
        target += torch.where(valid & (lengths >= horizon), weight * backup, 0.)
        tail = bootstrap.clone()
        tail[:, :-1] = torch.where(connected[:, :-1], backup[:, 1:], tail[:, :-1])
        backup = torch.maximum(g, (1 - gamma) * worst + gamma * tail)
    # Diagnostics use the actual observed suffix, independently of lambda targets.
    _, _, observed, steps = discounted_max_targets(current, following, next_value, done, gamma)
    return target, valid, observed, steps


@torch.no_grad()
def dgppo_advantage(task, current, following, value, next_value, *, dt, alpha, eps, weight,
                    task_mode="gated"):
    """Task scaling is resolved upstream; never re-center the mixed advantage."""
    if task_mode not in {"gated", "additive"}:
        raise ValueError("task_mode must be 'gated' or 'additive'")
    valid = (current['valid'] & following['valid'] & same_entities(current, following)
             & torch.isfinite(value) & torch.isfinite(next_value)
             & torch.isfinite(current['g']) & torch.isfinite(following['g']))
    delta = (next_value - value) / dt + alpha * value
    valid &= torch.isfinite(delta)
    delta = torch.where(valid, delta, 0.)
    violation = (valid & (delta > 0)).any(-1)
    complete = current['valid'].any(-1) & (~current['valid'] | valid).all(-1)
    # Missing neighbors cannot erase another known violation. Unknown-only
    # transitions fall back to task PPO and are never labelled certified-safe.
    eligible = complete | violation
    penalty = torch.where(valid, (delta + eps).clamp_min(0), 0.).amax(-1, keepdim=True)
    retained_task = task if task_mode == "additive" else torch.where(violation.unsqueeze(-1), 0., task)
    mixed = retained_task - weight * penalty
    adjusted = torch.where(eligible.unsqueeze(-1), mixed, task) if weight > 0 else task
    return adjusted.detach(), dict(delta=delta, valid=valid, eligible=eligible,
                                   penalty=penalty, violation=violation, complete=complete,
                                   recovery=(value > 0) & valid)


@torch.no_grad()
def prepare_dgppo_advantage(manager, td, advantage_key):
    p = manager.parameters
    raw_task = td.get(advantage_key).detach()
    td.set(('agents', 'barrier_task_advantage'), raw_task.clone())
    td.set(('agents', 'barrier_violation_mask'), torch.zeros_like(raw_task, dtype=torch.bool))
    enabled = manager.enabled and p.is_using_safety_constraint
    ready = enabled and manager.barrier_fit_batches >= p.safety_barrier_warmup_batches
    progress = manager.rollouts / max(p.n_iters, 1)
    weight = p.dgppo_weight * (2 ** (int(progress >= .5) + int(progress >= .75)) if p.dgppo_schedule else 1)
    metrics = dict(barrier_enabled=float(enabled), barrier_ready=float(ready), barrier_active=0.,
                   barrier_dgppo=1., barrier_weight=weight, barrier_fit_batches=manager.barrier_fit_batches,
                   barrier_task_additive=float(p.dgppo_task_mode == "additive"),
                   road_alpha_safe=p.dgppo_road_alpha_safe,
                   road_alpha_recovery=p.dgppo_road_alpha_recovery)
    opinion_mode = getattr(p, 'dgppo_opinion_alpha', False)
    if opinion_mode:
        metrics.update(opinion_alpha_enabled=1., opinion_alpha_gain=p.dgppo_alpha_gain,
                       opinion_alpha_deadzone=p.dgppo_opinion_deadzone, **dict.fromkeys([
            'opinion_pair_count', 'opinion_valid_count', 'opinion_missing_count',
            'opinion_z_mean', 'opinion_alpha_mean', 'opinion_alpha_min', 'opinion_alpha_max',
            'opinion_c_delta_abs', 'opinion_pair_flip_rate', 'opinion_agent_flip_rate',
            'opinion_advantage_delta_abs', 'opinion_advantage_changed_count',
            'opinion_advantage_changed_rate', 'opinion_mapped_z_mean',
            'opinion_alpha_saturation_rate'], 0.))
    # Use the same task scaling in task-only, warmup and constrained runs.
    # Original PPO uses raw GAE; current keeps the existing time normalization.
    task = raw_task if p.ppo_training_profile == 'original' else (
        (raw_task - raw_task.mean(1, keepdim=True))
        / (raw_task.std(1, unbiased=False, keepdim=True) + 1e-8))
    td.set(advantage_key, task)
    metrics['task_normalization_delta_abs'] = float((task - raw_task).abs().mean())
    if not ready or weight == 0:
        return metrics
    current, following = manager.state(td), manager.state(td.get('next'))
    value, nxt = manager.model(current), manager.model(following)
    terminal = td.get(('next', 'done')).bool().reshape(*td.batch_size, 1, 1)
    nxt = torch.where(terminal, following['g'], nxt)
    fixed_alpha = constraint_alpha(
        value,
        alpha=p.dgppo_alpha,
        road_safe_alpha=p.dgppo_road_alpha_safe,
        road_recovery_alpha=p.dgppo_road_alpha_recovery,
    )
    alpha = fixed_alpha
    if opinion_mode:
        alpha, opinion_info = opinion_alpha(td, current, value, alpha=alpha,
                                            span=p.dgppo_alpha_span, gain=p.dgppo_alpha_gain,
                                            deadzone=p.dgppo_opinion_deadzone)
    # Scaling has already been applied above; do not normalize the mixed result.
    adjusted, info = dgppo_advantage(task, current, following, value, nxt,
                                    dt=p.dt, alpha=alpha, eps=p.dgppo_eps, weight=weight,
                                    task_mode=p.dgppo_task_mode)
    td.set(advantage_key, adjusted)
    td.set(('agents', 'barrier_violation_mask'), info['violation'].unsqueeze(-1))
    def mean(x):
        return float(x.float().mean()) if x.numel() else 0.
    if opinion_mode:
        # Same physical predictions and task advantages: isolates the effect
        # of opinions, including suppression by a dominant road/collision head.
        fixed, fixed_info = dgppo_advantage(
            task, current, following, value, nxt, dt=p.dt, alpha=fixed_alpha,
            eps=p.dgppo_eps, weight=weight, task_mode=p.dgppo_task_mode)
        n = opinion_info['opinions'].shape[-1]
        pair = info['valid'][..., :n] & info['eligible'].unsqueeze(-1) & opinion_info['safe']
        available = pair & opinion_info['available']
        coefficients = alpha[..., :n][pair]
        flips = (info['delta'][..., :n] > 0) != (fixed_info['delta'][..., :n] > 0)
        changed = adjusted.squeeze(-1) != fixed.squeeze(-1)
        metrics.update(
            opinion_pair_count=float(pair.sum()), opinion_valid_count=float(available.sum()),
            opinion_missing_count=float((pair & ~opinion_info['available']).sum()),
            opinion_z_mean=mean(opinion_info['opinions'][available]),
            opinion_mapped_z_mean=mean(opinion_info['mapped_opinions'][available]),
            opinion_alpha_saturation_rate=mean(opinion_info['mapped_opinions'][available].abs() >= 1.),
            opinion_alpha_mean=mean(coefficients),
            opinion_alpha_min=float(coefficients.min()) if coefficients.numel() else 0.,
            opinion_alpha_max=float(coefficients.max()) if coefficients.numel() else 0.,
            opinion_c_delta_abs=mean((info['delta'][..., :n] - fixed_info['delta'][..., :n])[pair].abs()),
            opinion_pair_flip_rate=mean(flips[pair]),
            opinion_agent_flip_rate=mean((info['violation'] != fixed_info['violation'])[info['eligible']]),
            opinion_advantage_delta_abs=mean((adjusted - fixed).abs()),
            opinion_advantage_changed_count=float(changed.sum()),
            opinion_advantage_changed_rate=mean(changed[info['eligible']]))
    metrics.update(barrier_active=float(bool((adjusted != raw_task).any())),
                   barrier_eligible_count=float(info['eligible'].sum()),
                   barrier_fallback_count=float((~info['eligible']).sum()),
                   barrier_violation_rate=mean(info['violation'][info['eligible']]),
                   barrier_unsafe_count=float(info['violation'].sum()),
                   barrier_unsafe_positive_advantage_rate=mean((adjusted.squeeze(-1) > 0)[info['violation']]),
                   barrier_positive_mean=mean(info['penalty'][info['eligible']]),
                   barrier_advantage_delta_abs=mean((adjusted - raw_task).abs()),
                   barrier_task_retention_mean=mean(
                       (torch.ones_like(info['violation']) if p.dgppo_task_mode == "additive"
                        else ~info['violation'])[info['eligible']]),
                   barrier_observed_violation_count=float((following['valid'] & (following['g'] > 0)).sum()))
    return metrics
