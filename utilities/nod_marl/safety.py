"""Finite-horizon safety prediction and an optional Stage-7 Actor penalty."""

from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call
import torch.nn.functional as F


def safety_margins(
    positions, boundary_clearance, collision, safe_distance, boundary_margin
):
    """Per-agent dimensionless [center-distance, boundary, collision] margins.

    Non-positive means satisfied. Distance is explicitly center-to-center;
    boundary clearance already accounts for the vehicle footprint in VMAS.
    Clipping only the safe side at -1 preserves every positive violation.
    """
    if safe_distance <= 0 or boundary_margin <= 0:
        raise ValueError("Safety distance and boundary margin must be positive")
    n = positions.shape[-2]
    distance = torch.cdist(positions, positions)
    distance = distance.masked_fill(
        torch.eye(n, device=positions.device, dtype=torch.bool), torch.inf
    )
    distance_margin = ((safe_distance - distance.amin(-1)) / safe_distance).clamp_min(
        -1
    )
    road_margin = ((boundary_margin - boundary_clearance) / boundary_margin).clamp_min(
        -1
    )
    collision_margin = torch.where(collision.bool(), 1.0, -1.0)
    return torch.stack((distance_margin, road_margin, collision_margin), -1)


def finite_horizon_targets(
    current, following, done, generations, next_generations, horizons
):
    """Max of h states starting at s_t, including terminal s_{t+1} for h>=2.

    Inputs have [environment, time] shape (generations adds an agent axis).
    A batch cut is NOT a terminal state: incomplete horizons are masked.
    Identity changes are censored rather than labelled using respawned cars.
    """
    if (
        current.ndim != 2
        or following.shape != current.shape
        or done.shape != current.shape
    ):
        raise ValueError("Safety targets require ordered [environment, time] tensors")
    if not horizons or min(horizons) < 1:
        raise ValueError("Safety horizons must be positive")
    # VMAS may wrap scalar per-agent info in a final singleton dimension.
    generations = generations.reshape(*current.shape, -1)
    next_generations = next_generations.reshape(*current.shape, -1)
    stable = (generations == next_generations).all(-1)
    connected = torch.zeros_like(done, dtype=torch.bool)
    connected[:, :-1] = (next_generations[:, :-1] == generations[:, 1:]).all(-1)
    previous, previous_valid = current, stable
    targets, masks = {}, {}
    for horizon in range(1, max(horizons) + 1):
        if horizon > 1:
            # One successor state is available even at the collector's tail.
            future = following.clone()
            future_valid = (
                done.bool().clone()
                if horizon > 2
                else torch.ones_like(done, dtype=torch.bool)
            )
            if horizon > 2:
                future[:, :-1] = previous[:, 1:]
                future_valid[:, :-1] = previous_valid[:, 1:] & connected[:, :-1]
            # Absorb the observed terminal violation; never use the next episode.
            future = torch.where(done, following, future)
            future_valid = torch.where(
                done, torch.ones_like(future_valid), future_valid
            )
            previous = torch.maximum(current, future)
            previous_valid = stable & future_valid
        if horizon in horizons:
            targets[horizon], masks[horizon] = previous, previous_valid
    return torch.stack([targets[h] for h in horizons], -1), torch.stack(
        [masks[h] for h in horizons], -1
    )


class SafetyCriticManager:
    """Centralized joint-action Q, supervised on fresh ordered rollouts only."""

    def __init__(
        self,
        parameters,
        observation_dim,
        n_agents,
        action_dim,
        observation_key,
        *,
        kind="safety",
        context_key="safety_margins",
        context_dim=3,
        margin_key="safety_margins",
    ):
        # Stage 6 reuses the finite-horizon training mechanics, never weights or RNG.
        self.kind = kind
        self.context_key = context_key
        self.margin_key = margin_key
        self.enabled = bool(getattr(parameters, f"is_using_{kind}_critic", True))
        self.parameters = parameters
        self.observation_key = observation_key
        self.horizons = tuple(getattr(parameters, f"{kind}_horizons"))
        self.num_epochs = getattr(parameters, f"{kind}_num_epochs")
        self.minibatch_size = getattr(parameters, f"{kind}_minibatch_size")
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or min(self.horizons) < 1
        ):
            raise ValueError(
                "safety_horizons must be increasing, unique positive integers"
            )
        self.input_dim = n_agents * (observation_dim + 5 + context_dim + action_dim)
        self.metadata = {
            "version": 1,
            "input_dim": self.input_dim,
            "horizons": self.horizons,
            "hidden_dim": getattr(parameters, f"{kind}_hidden_dim"),
            "safe_distance": parameters.safety_safe_distance,
            "boundary_margin": parameters.safety_boundary_margin,
        }
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            width = self.metadata["hidden_dim"]
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, len(self.horizons)),
            ).to(parameters.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=getattr(parameters, f"{kind}_lr")
        )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(
            int(getattr(parameters, "seed", 0) or 0)
            ^ (0x53414645 if kind == "safety" else 0x44454144)
        )
        self.updates = 0
        self.rollouts = 0
        self.constraint_enabled = (
            kind == "safety" and parameters.is_using_safety_constraint
        )
        self.constraint_weight = parameters.safety_constraint_initial_weight
        self._constraint_totals = [0.0] * 5
        self.reliability_history = []

    def features(self, td, action=None):
        # Current information and actual action only: no next-state leakage.
        parts = [td.get(self.observation_key)]
        for key in ("pos", "vel", "rot", self.context_key):
            parts.append(td.get(("agents", "info", key)))
        parts = [part.detach() for part in parts]
        parts.append(td.get(("agents", "action")).detach() if action is None else action)
        return torch.cat(parts, -1).flatten(-2)

    @property
    def constraint_ready(self):
        return self.constraint_gate()["reason"] == 0

    def constraint_gate(self):
        """Gate from recent pre-fit predictions on currently safe states.

        Reasons: 0 ready, 1 disabled, 2 warmup, 3 too few unsafe targets,
        4 low recall, 5 excessive underestimation. Counts are state targets,
        not independent collision events.
        """
        count, unsafe, missed, low = (
            sum(row[i] for row in self.reliability_history) for i in range(4)
        )
        recall = (unsafe - missed) / max(1, unsafe)
        underestimate = low / max(1, count)
        p = self.parameters
        reason = 0
        if not self.enabled or not self.constraint_enabled:
            reason = 1
        elif self.updates == 0 or self.rollouts < p.safety_constraint_warmup_batches:
            reason = 2
        elif unsafe < p.safety_gate_min_unsafe:
            reason = 3
        elif recall < p.safety_gate_min_recall:
            reason = 4
        elif underestimate > p.safety_gate_max_underestimate:
            reason = 5
        return dict(reason=reason, valid_count=count, unsafe_count=unsafe,
                    recall=recall, underestimate_rate=underestimate)

    def _record_reliability(self, prediction, target, valid, current):
        if self.kind != "safety":
            return
        future = [i for i, h in enumerate(self.horizons) if h > 1]
        if not future:
            return
        selected = valid[:, future].all(-1) & (current.reshape(-1) <= 0)
        predicted = prediction[:, future].amax(-1)[selected]
        actual = target[:, future].amax(-1)[selected]
        unsafe = actual > 0
        self.reliability_history.append([
            actual.numel(), int(unsafe.sum()),
            int((unsafe & (predicted <= 0)).sum()),
            int((predicted < actual - 0.05).sum()),
        ])
        self.reliability_history = self.reliability_history[-self.parameters.safety_gate_window:]

    def actor_loss(self, td, action):
        """Differentiate only through current actions, using a frozen safety Q.

        h=1 is action independent. Already-violating states cannot satisfy a
        max-over-states target, so this first version constrains safe states only.
        """
        if not self.constraint_ready:
            return action.sum() * 0
        prediction = functional_call(
            self.model, {k: v.detach() for k, v in self.model.named_parameters()},
            (self.features(td, action),),
        )
        future = [i for i, h in enumerate(self.horizons) if h > 1]
        risk = prediction[..., future].amax(-1) + self.parameters.safety_constraint_margin
        eligible = td.get(("agents", "info", self.margin_key)).detach().amax((-2, -1)) <= 0
        selected = risk[eligible]
        penalty = selected.relu().sum() / eligible.sum().clamp_min(1)
        values = [selected.detach().sum().item(), selected.detach().relu().sum().item(),
                  (selected.detach() > 0).sum().item(), selected.numel(), risk.numel()]
        self._constraint_totals = [a + b for a, b in zip(self._constraint_totals, values)]
        return self.constraint_weight * penalty

    def finish_actor_update(self):
        """Update from mean positive violation: safe negatives cannot cancel risk."""
        risk_sum, penalty_sum, violations, count, total = self._constraint_totals
        weight = self.constraint_weight
        positive_risk = penalty_sum / max(1, count)
        dual_signal = positive_risk - self.parameters.safety_constraint_risk_budget
        if self.constraint_ready and count:
            self.constraint_weight = min(
                self.parameters.safety_constraint_max_weight,
                max(0.0, weight + self.parameters.safety_constraint_dual_lr * dual_signal),
            )
        self._constraint_totals = [0.0] * 5
        return {
            "actor_constraint_ready": float(self.constraint_ready),
            "actor_constraint_active": float(self.constraint_ready and weight > 0 and penalty_sum > 0),
            "actor_constraint_weight": weight,
            "actor_constraint_next_weight": self.constraint_weight,
            "actor_constraint_loss": weight * penalty_sum / max(1, count),
            "actor_constraint_risk": risk_sum / max(1, count),
            "actor_constraint_signed_risk": risk_sum / max(1, count),
            "actor_constraint_positive_risk": positive_risk,
            "actor_constraint_dual_signal": dual_signal if count else 0.0,
            "actor_constraint_sample_count": count,
            "actor_constraint_weight_at_cap": float(weight >= self.parameters.safety_constraint_max_weight),
            "actor_constraint_violation_rate": violations / max(1, count),
            "actor_constraint_eligible_fraction": count / max(1, total),
            **{f"constraint_gate_{key}": value for key, value in self.constraint_gate().items()},
        }

    @torch.no_grad()
    def predict(self, td):
        return self.model(self.features(td))

    def train_on_rollout(self, td):
        if not self.enabled:
            return {"enabled": 0.0}
        if td.ndim != 2:
            raise ValueError("Safety training expects [environment, time] rollout")
        current = td.get(("agents", "info", self.margin_key)).detach().amax((-2, -1))
        following = (
            td.get(("next", "agents", "info", self.margin_key)).detach().amax((-2, -1))
        )
        targets, valid = finite_horizon_targets(
            current,
            following,
            td.get(("next", "done")).squeeze(-1).bool(),
            td.get(("agents", "info", "nod_ego_generation")),
            td.get(("next", "agents", "info", "nod_ego_generation")),
            self.horizons,
        )
        x = self.features(td).reshape(-1, self.input_dim)
        y, mask = targets.reshape(-1, len(self.horizons)), valid.reshape(
            -1, len(self.horizons)
        )
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Non-finite Safety Critic features or targets")
        # Pre-update metrics measure prediction on this fresh batch, not fit error.
        with torch.no_grad():
            prediction = self.model(x)
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite Safety Critic predictions")
            self._record_reliability(prediction, y, mask, current)
            error = prediction - y
            unsafe = (y > 0) & mask
            missed = unsafe & (prediction <= 0)
            metrics = {
                "enabled": 1.0,
                "instant_violation_rate": float((current > 0).float().mean()),
                "instant_violation_count": int((current > 0).sum()),
                "transition_count": current.numel(),
                "valid_target_ratio": float(mask.float().mean()),
                "prediction_mae": float(error[mask].abs().mean())
                if mask.any()
                else 0.0,
                "unsafe_target_count": int(unsafe.sum()),
                "missed_unsafe_count": int(missed.sum()),
                "unsafe_recall": float(((prediction > 0) & unsafe).sum() / unsafe.sum())
                if unsafe.any()
                else 0.0,
                "underestimate_rate": float((error[mask] < -0.05).float().mean())
                if mask.any()
                else 0.0,
                "worst_underestimate": float((-error[mask]).clamp_min(0).max())
                if mask.any()
                else 0.0,
            }
            if self.kind == "deadlock":
                positive_predictions = (prediction > 0) & mask
                metrics["positive_prediction_count"] = int(positive_predictions.sum())
                metrics["false_positive_count"] = int(
                    (positive_predictions & ~unsafe).sum()
                )
                metrics["precision"] = (
                    float(
                        (positive_predictions & unsafe).sum()
                        / positive_predictions.sum()
                    )
                    if positive_predictions.any()
                    else 0.0
                )
            for index, horizon in enumerate(self.horizons):
                selected = mask[:, index]
                metrics[f"h{horizon}/valid_count"] = int(selected.sum())
                metrics[f"h{horizon}/mae"] = (
                    float(error[selected, index].abs().mean())
                    if selected.any()
                    else 0.0
                )
                metrics[f"h{horizon}/target_mean"] = (
                    float(y[selected, index].mean()) if selected.any() else 0.0
                )
                if self.kind == "deadlock":
                    positives = unsafe[:, index]
                    metrics[f"h{horizon}/unsafe_count"] = int(positives.sum())
                    metrics[f"h{horizon}/missed_count"] = int(missed[:, index].sum())
                    metrics[f"h{horizon}/false_positive_count"] = int(
                        ((prediction[:, index] > 0) & selected & ~positives).sum()
                    )
                    metrics[f"h{horizon}/unsafe_recall"] = (
                        float(
                            ((prediction[:, index] > 0) & positives).sum()
                            / positives.sum()
                        )
                        if positives.any()
                        else 0.0
                    )
        rows = mask.any(-1).nonzero().squeeze(-1)
        losses = []
        for _ in range(self.num_epochs):
            order = torch.randperm(rows.numel(), generator=self.generator).to(
                rows.device
            )
            for indices in rows[order].split(self.minibatch_size):
                if indices.numel() == 0:
                    continue
                prediction = self.model(x[indices])
                weights = torch.where(prediction.detach() < y[indices], 2.0, 1.0)
                loss = (
                    F.smooth_l1_loss(prediction, y[indices], reduction="none") * weights
                )[mask[indices]].mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, error_if_nonfinite=True
                )
                self.optimizer.step()
                self.updates += 1
                losses.append(float(loss.detach()))
        metrics["training_loss"] = sum(losses) / max(1, len(losses))
        metrics["optimizer_updates"] = len(losses)
        if losses:
            self.rollouts += 1
        # Clear fitting gradients so Actor-phase isolation can be asserted.
        self.optimizer.zero_grad(set_to_none=True)
        return metrics

    def _constraint_contract(self):
        names = ("safety_constraint_warmup_batches", "safety_constraint_initial_weight",
                 "safety_constraint_max_weight", "safety_constraint_dual_lr",
                 "safety_constraint_margin", "safety_constraint_risk_budget",
                 "safety_gate_window", "safety_gate_min_unsafe", "safety_gate_min_recall",
                 "safety_gate_max_underestimate")
        return {"version": 2, **{name: getattr(self.parameters, name) for name in names}}

    def checkpoint_state(self):
        return {
            **self.metadata,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "updates": self.updates,
            "rollouts": self.rollouts,
            **({"constraint_weight": self.constraint_weight,
                "constraint_contract": self._constraint_contract(),
                "reliability_history": [row[:] for row in self.reliability_history]}
               if self.kind == "safety" else {}),
        }

    def load_if_available(self, path, load_optimizer=False):
        if not self.enabled:
            return False
        if not Path(path).exists():
            print(
                f"[INFO] No {self.kind} Critic checkpoint at {path}; initialized independently."
            )
            return False
        checkpoint = torch.load(path, map_location=self.parameters.device)
        if any(checkpoint.get(key) != value for key, value in self.metadata.items()):
            print(
                f"[WARN] Incompatible {self.kind} Critic checkpoint {path}; initialized independently."
            )
            return False
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.generator.set_state(checkpoint["generator"].cpu())
        self.updates = checkpoint.get("updates", 0)
        self.rollouts = checkpoint.get("rollouts", 0)
        self.reliability_history = []
        self._constraint_totals = [0.0] * 5
        if load_optimizer and self.constraint_enabled:
            self.constraint_weight = self.parameters.safety_constraint_initial_weight
            if (checkpoint.get("constraint_contract") == self._constraint_contract()
                    and all(key in checkpoint for key in
                            ("reliability_history", "rollouts", "constraint_weight"))):
                self.reliability_history = [row[:] for row in checkpoint.get("reliability_history", [])]
                self.constraint_weight = min(
                    self.parameters.safety_constraint_max_weight,
                    max(0.0, checkpoint.get("constraint_weight", self.constraint_weight)),
                )
            else:
                self.rollouts = 0
                print("[INFO] Safety constraint state changed or missing; restarting reliability warmup.")
        print(f"[INFO] Loaded {self.kind} Critic: {path}")
        return True
