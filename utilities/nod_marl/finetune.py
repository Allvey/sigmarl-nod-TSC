"""Opt-in safety fine-tuning controls; no changes to DGPPO advantages."""
import torch


def actor_warmup_frozen(parameters, manager):
    return (parameters.safety_training_mode == 'finetune'
            and manager.barrier_fit_batches < parameters.safety_barrier_warmup_batches)


@torch.no_grad()
def approximate_policy_kl(new_log_prob, rollout_log_prob):
    """Nonnegative sampled KL(old || new) estimator, averaged over agent actions.

    Log-probabilities must already sum across action dimensions. The sample
    estimate is not a hard bound on policy divergence or cumulative drift.
    """
    if new_log_prob.numel() != rollout_log_prob.numel():
        raise ValueError('Action log-probability shapes do not match')
    ratio = new_log_prob.detach().reshape(-1) - rollout_log_prob.detach().reshape(-1)
    if not torch.isfinite(ratio).all():
        raise ValueError('Non-finite action log-probabilities during KL check')
    return (torch.expm1(ratio) - ratio).clamp_min(0).mean()
