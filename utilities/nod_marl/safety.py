"""Stage 5: independent, finite-horizon safety prediction (no Actor loss)."""

from pathlib import Path

import torch
from torch import nn
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

    def features(self, td):
        # Current information and actual action only: no next-state leakage.
        parts = [td.get(self.observation_key)]
        for key in ("pos", "vel", "rot", self.context_key):
            parts.append(td.get(("agents", "info", key)))
        parts.append(td.get(("agents", "action")))
        return torch.cat(parts, -1).flatten(-2).detach()

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
        return metrics

    def checkpoint_state(self):
        return {
            **self.metadata,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "updates": self.updates,
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
        print(f"[INFO] Loaded {self.kind} Critic: {path}")
        return True
