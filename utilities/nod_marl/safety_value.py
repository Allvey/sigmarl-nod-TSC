"""Stage 8A: identity-aligned safety State Value, with no Actor loss.

Pair channels use world agent slots (not nearest-neighbor ranks). Generation
and visibility masks censor transitions; disappearing cars are never safe zeros.
The discounted max target is an approximation, not a safety certificate.
"""

import copy
import random
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .interaction import NOD_PAIR_FEATURE_DIM

VALUE_HEADS = (("pair", slice(0, -2)), ("road", slice(-2, -1)), ("collision", slice(-1, None)))
UNDERESTIMATE_MARGIN = 0.05


@torch.no_grad()
def positive_class_weights(target, valid, cap):
    """Rollout-level weights: scarce positive targets get at most `cap`."""
    weights = []
    for _, selection in VALUE_HEADS:
        y = target[..., selection][valid[..., selection]]
        positives = int((y > 0).sum())
        weights.append(min(cap, max(1., (y.numel() - positives) / positives))
                       if positives else 1.)
    return weights


def balanced_value_loss(prediction, target, valid, positive_weights, underestimate_weight):
    """Equal head means, bounded class/underestimation weighting within heads.

    Targets are unchanged. Missing heads are omitted; single-class heads still
    train on real samples. The effective weight is <= class cap * under weight.
    """
    losses = []
    for index, (_, selection) in enumerate(VALUE_HEADS):
        mask = valid[..., selection]
        if not mask.any():
            continue
        pred, y = prediction[..., selection][mask], target[..., selection][mask].detach()
        positive = y > 0
        weights = torch.where(positive, y.new_tensor(positive_weights[index]), y.new_tensor(1.))
        low = positive & (pred.detach() < y - UNDERESTIMATE_MARGIN)
        weights = weights * torch.where(low, y.new_tensor(underestimate_weight), y.new_tensor(1.))
        losses.append((weights * F.smooth_l1_loss(pred, y, reduction="none")).sum() / weights.sum())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.


@torch.no_grad()
def value_head_metrics(prediction, target, observed, current_g, valid):
    """Pre-fit diagnostics; observed-safe is only a finite-window proxy."""
    def mean(values):
        return float(values.float().mean()) if values.numel() else 0.

    result = {}
    for name, selection in VALUE_HEADS:
        mask = valid[..., selection]
        pred, y, obs, g = (x[..., selection][mask] for x in (prediction, target, observed, current_g))
        positive, unsafe = y > 0, obs > 0
        warning = (g <= 0) & unsafe
        mismatch = unsafe & ~positive
        err = (pred - y).abs()
        metrics = dict(
            samples=float(mask.sum()), observed_unsafe=float(unsafe.sum()),
            unsafe_recall=mean(pred[unsafe] > 0),
            underestimate_rate=mean(pred < obs - UNDERESTIMATE_MARGIN),
            worst_underestimate=float((obs - pred).clamp_min(0).max()) if pred.numel() else 0.,
            target_positive_count=float(positive.sum()), target_nonpositive_count=float((~positive).sum()),
            target_positive_ratio=mean(positive), target_positive_mae=mean(err[positive]),
            target_nonpositive_mae=mean(err[~positive]), target_positive_recall=mean(pred[positive] > 0),
            observed_unsafe_target_nonpositive_count=float(mismatch.sum()),
            observed_unsafe_target_nonpositive_ratio=mean(mismatch[unsafe]),
            early_warning_count=float(warning.sum()), early_warning_recall=mean(pred[warning] > 0),
            early_warning_missed_count=float((warning & (pred <= 0)).sum()),
            early_warning_underestimate_rate=mean(pred[warning] < obs[warning] - UNDERESTIMATE_MARGIN),
            observed_safe_count=float((~unsafe).sum()), observed_safe_positive_rate=mean(pred[~unsafe] > 0),
        )
        result.update({name + "_" + k: v for k, v in metrics.items()})
    return result


@contextmanager
def isolated_rng(state):
    """Persist a private stream, restoring all process RNGs even on exceptions."""
    outer_python, outer_numpy = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            if "torch" in state:
                torch.set_rng_state(state["torch"].cpu())
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
                if devices and "cuda" in state:
                    torch.cuda.set_rng_state_all(state["cuda"])
            else:
                torch.manual_seed(state["seed"])
                random.seed(state["seed"])
                np.random.seed(state["seed"] % (2**32))
            yield
        finally:
            state.update(torch=torch.get_rng_state(), python=random.getstate(),
                         numpy=np.random.get_state())
            if devices:
                state["cuda"] = torch.cuda.get_rng_state_all()
            random.setstate(outer_python)
            np.random.set_state(outer_numpy)


def value_state(td, observation_key, safe_distance, *, local_sensing_range=None):
    """Build current-state-only features and dimensionless physical labels."""
    info = td.get(("agents", "info"))
    pair = info.get("nod_pair_features").detach()
    ids = info.get("nod_neighbor_indices").long()
    edge = info.get("nod_edge_mask").bool()
    if local_sensing_range is not None:
        # Existing NOD conflict features use neighbors' reference paths. The
        # no-communication DGPPO branch keeps only observable kinematics.
        pair = pair.clone()
        pair[..., 9:16] = 0.
        edge = pair[..., 17] > 0.5
    n = pair.shape[-3]
    if bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError("Safety Value neighbor identity is out of bounds")
    dense_pair = pair.new_zeros(*pair.shape[:-2], n, pair.shape[-1])
    dense_pair.scatter_(-2, ids.unsqueeze(-1).expand_as(pair), pair)
    mask = torch.zeros(*pair.shape[:-2], n, device=pair.device, dtype=torch.bool)
    mask.scatter_(-1, ids, edge)
    mask &= ~torch.eye(n, device=pair.device, dtype=torch.bool)
    pos = info.get("nod_world_pos").detach()
    distance = (torch.cdist(pos, pos) if local_sensing_range is None
                else dense_pair[..., 6] * local_sensing_range)
    g_pair = ((safe_distance - distance) / safe_distance).clamp_min(-1)
    margins = info.get("safety_margins").detach()
    # Road and unattributed actual collision remain separate, non-opinion heads.
    g = torch.cat([g_pair, margins[..., 1:3]], dim=-1)
    valid = torch.cat([mask, torch.ones_like(margins[..., 1:3], dtype=torch.bool)], -1)
    generation = info.get("nod_ego_generation").long().reshape(*pos.shape[:-1])
    ego_gen = generation.unsqueeze(-1).expand_as(g)
    other_gen = torch.cat([
        generation.unsqueeze(-2).expand(*generation.shape[:-1], n, n),
        generation.unsqueeze(-1).expand(*generation.shape, 2),
    ], -1)
    node = torch.cat([td.get(observation_key).detach(), margins[..., 1:3]], -1)
    return dict(pair=dense_pair, node=node, mask=mask, valid=valid, g=g,
                ego_gen=ego_gen, other_gen=other_gen)


def same_entities(left, right):
    return ((left["ego_gen"] == right["ego_gen"])
            & (left["other_gen"] == right["other_gen"]))


def discounted_max_targets(current, following, next_value, done, gamma):
    """Ordered [env,time,agent,constraint] multi-step max backup.

    Missing successor identities invalidate the current target. A valid step
    before a censored suffix uses its own successor bootstrap, not that suffix.
    True terminal physics is retained; collector cuts bootstrap without reset.
    """
    g, gn = current["g"], following["g"]
    terminal = done.bool().reshape(*g.shape[:2], 1, 1).expand_as(g)
    valid = current["valid"] & following["valid"] & same_entities(current, following)
    targets, observed = torch.zeros_like(g), torch.zeros_like(g)
    observed_steps = torch.zeros_like(g)
    for t in reversed(range(g.shape[1])):
        tail = torch.maximum(gn[:, t], next_value[:, t])
        physical_tail = gn[:, t]
        count = torch.ones_like(physical_tail)
        if t + 1 < g.shape[1]:
            connected = (valid[:, t + 1]
                         & (following["ego_gen"][:, t] == current["ego_gen"][:, t + 1])
                         & (following["other_gen"][:, t] == current["other_gen"][:, t + 1])
                         & ~terminal[:, t])
            tail = torch.where(connected, targets[:, t + 1], tail)
            physical_tail = torch.where(connected, observed[:, t + 1], physical_tail)
            count = torch.where(connected, observed_steps[:, t + 1], count)
        tail = torch.where(terminal[:, t], gn[:, t], tail)
        physical_tail = torch.where(terminal[:, t], gn[:, t], physical_tail)
        count = torch.where(terminal[:, t], torch.ones_like(count), count)
        targets[:, t] = torch.maximum(g[:, t], (1 - gamma) * g[:, t] + gamma * tail)
        observed[:, t] = torch.maximum(g[:, t], physical_tail)
        observed_steps[:, t] = count + 1
    return targets.detach(), valid, observed, observed_steps


class PairSafetyValue(nn.Module):
    """Shared pair encoder + local masked mean; independent of vehicle count."""

    def __init__(self, observation_dim, width):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(NOD_PAIR_FEATURE_DIM, width), nn.Tanh())
        self.node = nn.Sequential(nn.Linear(observation_dim + 2, width), nn.Tanh())
        self.pair_head = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.local_head = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 2))

    def forward(self, state):
        edge = self.edge(state["pair"])
        mask = state["mask"].unsqueeze(-1)
        pooled = (edge * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        node = self.node(state["node"])
        context = torch.cat([node, pooled], -1)
        pair_value = self.pair_head(torch.cat([
            edge, context.unsqueeze(-2).expand(*edge.shape[:-1], context.shape[-1])
        ], -1)).squeeze(-1)
        return torch.cat([pair_value, self.local_head(context)], -1)


class SafetyStartBuffer:
    """Physical episode starts, never old actions/targets or recurrent states.

    Reserve ordinary environments for unconditioned pre-fit diagnostics. Mine
    at most one start per danger type and episode, from identity-contiguous
    trajectories, rather than the scenario's globally reset state history.
    """

    def __init__(self, parameters):
        from utilities.helper_scenario import InitialStateBuffer
        p = parameters
        self.lookback = p.safety_value_start_lookback
        count = (max(1, round(p.safety_value_num_envs * p.safety_value_challenging_fraction))
                 if p.safety_value_challenging_fraction else 0)
        self.normal_envs = max(1, p.safety_value_num_envs - count)
        self.active = torch.zeros(p.safety_value_num_envs, device=p.device, dtype=torch.bool)
        self.pools = {name: InitialStateBuffer(buffer=torch.zeros(
            p.safety_value_start_buffer_size, p.n_agents, 9, device=p.device))
            for name in ("collision", "road")}
        self.contract = dict(version=1, scenario=p.scenario_type, agents=p.n_agents,
                             dt=p.dt, lookback=self.lookback, capacity=p.safety_value_start_buffer_size,
                             scenario_probabilities=list(p.cpm_scenario_probabilities))

    def sample(self, env_index):
        self.active[env_index] = False
        pools = [pool for pool in self.pools.values() if pool.valid_size]
        if env_index < self.normal_envs or not pools:
            return None
        # Equal choice of populated danger types prevents one type crowding out the other.
        pool = pools[int(torch.randint(len(pools), ()))]
        self.active[env_index] = True
        return pool.get_random().clone()

    @torch.no_grad()
    def update(self, td):
        info, nxt = td.get(("agents", "info")), td.get(("next", "agents", "info"))
        states, gen, ng = (info["safety_reset_state"], info["nod_ego_generation"],
                           nxt["nod_ego_generation"])
        safe = (info["safety_margins"] <= 0).flatten(2).all(-1)
        finite = torch.isfinite(states).flatten(2).all(-1)
        done = td.get(("next", "done")).reshape(*td.batch_size).bool()
        hazards = dict(collision=(nxt["safety_margins"][..., 2] > 0).any(-1),
                       road=(nxt["safety_margins"][..., 1] > 0).any(-1))
        metrics = {}
        for name, hazard in hazards.items():
            added, seen = 0, set()
            for env, t in hazard.nonzero().tolist():
                episode = (env, *gen[env, t].reshape(-1).tolist())
                if episode in seen:
                    continue
                # A short episode may provide fewer than lookback steps.
                start = max(0, t + 1 - self.lookback)
                identity = gen[env, t]
                eligible = ((gen[env, start:t+1] == identity).flatten(1).all(-1)
                            & safe[env, start:t+1] & finite[env, start:t+1])
                choices = eligible.nonzero().flatten()
                if not choices.numel():
                    continue
                start += int(choices[0])
                if (not (gen[env, start:t+1] == identity).all()
                        or not (ng[env, start:t+1] == identity).all()
                        or done[env, start:t].any()):
                    continue
                self.pools[name].add(states[env, start].detach())
                seen.add(episode)
                added += 1
            metrics[name + "_starts_added"] = float(added)
            metrics[name + "_start_buffer_size"] = float(self.pools[name].valid_size)
        return metrics

    def state_dict(self):
        return dict(contract=self.contract, pools={name: dict(
            buffer=pool.buffer.detach().clone(), pointer=pool.pointer, valid_size=pool.valid_size)
            for name, pool in self.pools.items()})

    def load_state_dict(self, state):
        if not state or state.get("contract") != self.contract:
            return False
        for name, pool in self.pools.items():
            saved = state["pools"][name]
            pool.buffer.copy_(saved["buffer"])
            pool.pointer, pool.valid_size = saved["pointer"], saved["valid_size"]
        return True


class SafetyValueManager:
    """Independent shadow learner; never registered with the PPO optimizer."""

    def __init__(self, parameters, observation_dim, observation_key):
        self.enabled = bool(getattr(parameters, "is_using_safety_value_shadow", False))
        self.parameters, self.observation_key = parameters, observation_key
        self.dgppo = parameters.safety_control_mode == "dgppo"
        self.updates = self.rollouts = self.frames = 0
        self.sampler_state = {"seed": int(parameters.seed or 0) ^ 0x3856414C}
        self.start_buffer_state = None
        self.barrier_fit_batches = 0
        self.barrier_contract = dict(
            version=1, mode=parameters.safety_control_mode,
            enabled=parameters.is_using_safety_constraint,
            kappa=parameters.safety_barrier_kappa, road_kappa=parameters.safety_barrier_road_kappa,
            nu=parameters.safety_barrier_nu, strength=parameters.safety_barrier_strength,
            warmup=parameters.safety_barrier_warmup_batches,
            gate="warmup_only_experiment", recovery="nonincrease_value",
            advantage="soft_task_mask_minus_max_positive", normalization="unchanged_task_GAE",
            nod_frozen=parameters.nod_freeze_training,
        )
        if parameters.safety_control_mode == 'barrier_opinion':
            self.barrier_contract.update(kappa_min=parameters.safety_barrier_kappa_min,
                                         kappa_max=parameters.safety_barrier_kappa_max,
                                         opinion="cached_preaction_z_world_slot_generation",
                                         missing_opinion="kappa_min")
        if parameters.safety_control_mode in {'barrier_fixed', 'barrier_opinion', 'dgppo'} and (
                not self.enabled or parameters.is_using_prioritized_marl):
            raise ValueError("Stage 8B requires Safety Value and non-prioritized MARL")
        self.contract = dict(
            version=1, mode="state_value_shadow", observation_dim=observation_dim,
            width=parameters.safety_value_hidden_dim, pair_dim=NOD_PAIR_FEATURE_DIM,
            gamma=parameters.safety_value_gamma, target_tau=parameters.safety_value_target_tau,
            safe_distance=parameters.safety_safe_distance,
            boundary_margin=parameters.safety_boundary_margin, dt=parameters.dt,
            identity="world_slot_and_generation", heads="pair_distance,road,collision",
            sensing_range=parameters.nod_sensing_range,
            interaction_distance=parameters.nod_interaction_distance,
            conflict_radius=parameters.nod_conflict_radius, ttc_limit=parameters.nod_ttc_limit,
        )
        if parameters.use_navigation_boundary:
            self.contract.update(boundary_semantics="navigation_corridor_footprint_v1",
                                 boundary_observation="navigation_distance")
        self.loss_contract = ({"mode": "legacy"} if parameters.safety_value_loss_mode == "legacy" else dict(
            mode="balanced", positive_weight_cap=parameters.safety_value_positive_weight_cap,
            underestimate_weight=parameters.safety_value_underestimate_weight,
            underestimate_margin=UNDERESTIMATE_MARGIN, reduction="equal_head_weighted_mean",
            class_counts="valid_rollout_targets",
        ))
        if self.dgppo:
            self.contract.update(mode="dgppo_minimal_value", target="max_lambda_cross_head",
                                 gae_lambda=parameters.dgppo_lambda, target_tau=None,
                                 features="local_kinematics_no_neighbor_path", mask="sensing_range")
            self.loss_contract = dict(mode="mse", reduction="half_mean_valid", bootstrap="preupdate_model")
            self.barrier_contract.update(mode="dgppo", alpha=parameters.dgppo_alpha,
                eps=parameters.dgppo_eps, weight=parameters.dgppo_weight,
                schedule=parameters.dgppo_schedule, schedule_iters=parameters.n_iters, dt=parameters.dt,
                recovery="positive_value_contraction", advantage="feasible_task_minus_risk_rate",
                normalization="per_env_agent_time_including_warmup", partial_successor="known_violation_retained")
            if parameters.dgppo_task_mode == "additive":
                self.barrier_contract.update(dgppo_task_mode="additive",
                                             advantage="full_task_minus_risk_rate")
            if parameters.ppo_training_profile == "original":
                self.barrier_contract.update(
                    normalization="raw_task_GAE_including_warmup", ppo_training_profile="original",
                    task_ppo=dict(num_epochs=parameters.num_epochs, lr=parameters.lr,
                                  lmbda=parameters.lmbda, clip_epsilon=parameters.clip_epsilon))
        if parameters.safety_training_mode == 'finetune':
            self.barrier_contract.update(safety_training_mode='finetune',
                                         actor_warmup='frozen_until_value_fit_batches',
                                         finetune_lr=parameters.safety_finetune_lr,
                                         target_kl=parameters.safety_finetune_target_kl)
        self.model = self.target = self.optimizer = None
        self.last_load_info = "disabled" if not self.enabled else "fresh shadow Value"
        if not self.enabled:
            return
        with isolated_rng({"seed": int(parameters.seed or 0) ^ 0x38564E45}):
            self.model = PairSafetyValue(observation_dim, parameters.safety_value_hidden_dim).to(parameters.device)
        self.target = None if self.dgppo else copy.deepcopy(self.model).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=parameters.safety_value_lr)
        self.generator = torch.Generator(device="cpu").manual_seed(int(parameters.seed or 0) ^ 0x38564F50)

    def state(self, td):
        return value_state(td, self.observation_key, self.parameters.safety_safe_distance,
                           local_sensing_range=self.parameters.nod_sensing_range if self.dgppo else None)

    @torch.no_grad()
    def predict(self, td):
        state = self.state(td)
        return self.model(state), state["valid"]

    def train_on_rollout(self, td):
        if not self.enabled:
            return {}
        current, following = self.state(td), self.state(td.get("next"))
        with torch.no_grad():
            before = self.model(current)
            next_value = (self.model if self.dgppo else self.target)(following)
            if self.dgppo:
                from .dgppo import dgppo_targets
                y, valid, observed, steps = dgppo_targets(
                    current, following, next_value, td.get(("next", "done")),
                    self.parameters.safety_value_gamma, self.parameters.dgppo_lambda)
            else:
                y, valid, observed, steps = discounted_max_targets(
                    current, following, next_value, td.get(("next", "done")), self.parameters.safety_value_gamma)
        if not all(torch.isfinite(x).all() for x in (current["pair"], current["node"], y, before)):
            raise ValueError("Non-finite Stage-8A Safety State Value data")
        metrics = {"shadow_only": 1., "actor_updates": 0., "valid_samples": float(valid.sum()),
                   "invalid_ratio": float((~valid).float().mean()), "optimizer_updates": 0.}
        # All reported errors use pre-fit predictions. Observed suffix max is
        # a finite-window diagnostic, not ground truth for the infinite future.
        metrics.update(value_head_metrics(before, y, observed, current["g"], valid))
        # Ordinary lanes never use stored starts, including after auto-reset.
        # These are pre-fit training diagnostics, not a held-out test set.
        normal = td.get("safety_normal_env", default=None)
        if normal is not None:
            for name, selection in (("normal", normal), ("challenge", ~normal)):
                group_valid = valid & selection[..., None, None]
                metrics.update({name + "_" + k: v for k, v in value_head_metrics(
                    before, y, observed, current["g"], group_valid).items()})
        positive_weights = positive_class_weights(y, valid, self.parameters.safety_value_positive_weight_cap)
        balanced = self.parameters.safety_value_loss_mode == "balanced"
        metrics["balanced_loss"] = float(balanced)
        if self.dgppo:
            metrics["dgppo_target"] = 1.
        for (name, _), weight in zip(VALUE_HEADS, positive_weights):
            metrics[name + "_positive_class_weight"] = weight if balanced else 1.
        if valid.any():
            metrics.update(value_mean=float(before[valid].mean()),
                           value_zero_ratio=float((before[valid].abs() < 1e-3).float().mean()),
                           prefit_target_mae=float((before[valid] - y[valid]).abs().mean()),
                           observed_steps_mean=float(steps[valid].mean()),
                           next_value_delta_abs=float((next_value - before)[valid].abs().mean()))
        batch = {k: v.flatten(0, 1) for k, v in current.items()}
        target, masks = y.flatten(0, 1), valid.flatten(0, 1)
        losses = []
        for _ in range(self.parameters.safety_value_num_epochs):
            order = torch.randperm(target.shape[0], generator=self.generator)
            for ids in order.split(self.parameters.safety_value_minibatch_size):
                ids = ids.to(target.device)
                mask = masks[ids]
                if not mask.any():
                    continue
                pred = self.model({k: v[ids] for k, v in batch.items()})
                if self.dgppo:
                    loss = 0.5 * F.mse_loss(pred[mask], target[ids][mask])
                else:
                    loss = (balanced_value_loss(pred, target[ids], mask, positive_weights,
                                                self.parameters.safety_value_underestimate_weight)
                            if balanced else F.smooth_l1_loss(pred[mask], target[ids][mask]))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                losses.append(float(loss.detach()))
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for dest, src in zip(self.target.parameters() if self.target is not None else [], self.model.parameters()):
                dest.lerp_(src, self.parameters.safety_value_target_tau)
        self.rollouts += 1
        if losses:
            self.barrier_fit_batches += 1
        self.frames += td.numel()
        metrics.update(optimizer_updates=float(len(losses)), training_loss=float(np.mean(losses)) if losses else 0.,
                       deterministic_frames=float(td.numel()), total_deterministic_frames=float(self.frames))
        return metrics

    def checkpoint_state(self):
        if not self.enabled:
            return None
        return dict(contract=self.contract, loss_contract=self.loss_contract,
                    model=self.model.state_dict(), target=self.target.state_dict() if self.target is not None else None,
                    optimizer=self.optimizer.state_dict(), updates=self.updates, rollouts=self.rollouts,
                    frames=self.frames, generator=self.generator.get_state(), sampler_state=self.sampler_state,
                    start_buffer_state=self.start_buffer_state,
                    barrier_contract=self.barrier_contract, barrier_fit_batches=self.barrier_fit_batches)

    def load_if_available(self, path, *, load_optimizer=False):
        if not self.enabled:
            return False
        if not Path(path).is_file():
            self.last_load_info = "missing sidecar; fresh shadow Value (no safety control)"
            print("[INFO] Safety Value:", self.last_load_info)
            return False
        checkpoint = torch.load(path, map_location=self.parameters.device)
        if checkpoint.get("contract") != self.contract:
            self.last_load_info = "incompatible State Value contract; fresh shadow Value (old Q is not migrated)"
            print("[WARN] Safety Value:", self.last_load_info)
            return False
        self.model.load_state_dict(checkpoint["model"])
        if self.target is not None:
            self.target.load_state_dict(checkpoint["target"])
        self.updates, self.rollouts, self.frames = (checkpoint[k] for k in ("updates", "rollouts", "frames"))
        loss_changed = checkpoint.get("loss_contract", {"mode": "legacy"}) != self.loss_contract
        barrier_changed = checkpoint.get('barrier_contract') != self.barrier_contract
        self.barrier_fit_batches = (checkpoint.get('barrier_fit_batches', 0)
                                   if load_optimizer and not barrier_changed and not loss_changed else 0)
        if load_optimizer:
            if not loss_changed:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                # Retain compatible Value/target weights; old Adam moments were
                # fitted with another loss. Do not silently reuse them.
                self.optimizer.state.clear()
            self.generator.set_state(checkpoint["generator"].cpu())
            self.sampler_state = checkpoint["sampler_state"]
            self.start_buffer_state = checkpoint.get("start_buffer_state")
        self.last_load_info = "loaded shadow Value; not used for action selection"
        if load_optimizer and loss_changed:
            self.last_load_info += "; optimizer reset after loss contract change"
        if load_optimizer and (barrier_changed or loss_changed) and self.parameters.safety_control_mode in {'barrier_fixed', 'barrier_opinion', 'dgppo'}:
            self.last_load_info += "; barrier starts fresh warmup (Value weights retained)"
        print("[INFO] Safety Value:", self.last_load_info)
        return True


class DeterministicSafetySampler:
    """Separate environment, policy copy, NOD state and persistent RNG stream.

    Each batch starts fresh, after syncing policy/NOD weights. Thus no histories
    generated with old opinion weights are carried into a new batch or resume.
    """

    def __init__(self, parameters, policy, nod_manager, manager):
        from torchrl.envs import RewardSum
        from torchrl.envs.libs.vmas import VmasEnv
        from scenarios.road_traffic import ScenarioRoadTraffic
        from utilities.helper_training import TransformedEnvCustom
        from .trainer import NODOpinionManager
        from .policy import NODActorInputModule

        if parameters.is_using_prioritized_marl:
            raise ValueError("Stage 8A shadow sampling currently requires is_using_prioritized_marl=false")
        self.manager = manager
        self.source_policy, self.source_nod = policy, nod_manager
        with isolated_rng(manager.sampler_state):
            p = copy.deepcopy(parameters)
            p.is_using_deadlock_critic = False
            # Keep the original PPO/testing initial-state settings untouched.
            p.is_challenging_initial_state_buffer = False
            p.num_vmas_envs = parameters.safety_value_num_envs
            scenario = ScenarioRoadTraffic()
            scenario.parameters = p
            self.starts = SafetyStartBuffer(p) if p.safety_value_challenging_fraction else None
            if self.starts is not None:
                scenario.safety_start_buffer = self.starts
                restored = self.starts.load_state_dict(manager.start_buffer_state)
                if manager.start_buffer_state is not None and not restored:
                    print("[INFO] Safety Value: incompatible start buffer; collecting fresh starts")
            base = VmasEnv(scenario=scenario, num_envs=p.num_vmas_envs,
                           continuous_actions=True, max_steps=p.max_steps,
                           device=p.device, n_agents=p.n_agents)
            self.env = TransformedEnvCustom(base, RewardSum(
                in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
            self.nod = NODOpinionManager(p)
            scenario.nod_manager = self.nod
            self.policy = copy.deepcopy(policy)
            for module in self.policy.modules():
                if isinstance(module, NODActorInputModule):
                    object.__setattr__(module, "_nod_manager_ref", weakref.ref(self.nod))
            self.policy.requires_grad_(False)
        self.steps = parameters.safety_value_rollout_steps
        self.last_metrics = {}

    @torch.no_grad()
    def collect(self):
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        with isolated_rng(self.manager.sampler_state), set_exploration_type(ExplorationType.MODE):
            self.policy.load_state_dict(self.source_policy.state_dict())
            self.nod.model.load_state_dict(self.source_nod.model.state_dict())
            self.nod.reset_online_state()
            start = time.monotonic()
            rollout = self.env.rollout(self.steps, self.policy, break_when_any_done=False)
            self.last_metrics = {}
            if self.starts is not None:
                normal = (torch.arange(rollout.shape[0], device=rollout.device)
                          < self.starts.normal_envs)
                rollout.set("safety_normal_env", normal[:, None].expand(*rollout.batch_size))
                self.last_metrics.update(self.starts.update(rollout))
                active = rollout.get(("agents", "info", "safety_challenging_start"))
                self.last_metrics.update(challenging_start_frame_ratio=float(active.float().mean()),
                                         normal_env_count=float(normal.sum()),
                                         challenge_env_count=float((~normal).sum()))
                self.manager.start_buffer_state = self.starts.state_dict()
            return rollout, time.monotonic() - start

    def close(self):
        with isolated_rng(self.manager.sampler_state):
            self.env.close()
