"""Per-environment contact events, independent of rendering frequency."""
import torch


class CollisionCounter:
    def __init__(self, batch_dim, n_agents, device):
        self.vehicle = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self.road = torch.zeros_like(self.vehicle)
        self._pairs = torch.zeros(batch_dim, n_agents, n_agents, dtype=torch.bool, device=device)
        self._road = torch.zeros(batch_dim, n_agents, dtype=torch.bool, device=device)

    def update(self, pairs, road):
        # An unordered vehicle pair counts once; persistent contact is one event.
        pairs = torch.triu(pairs.bool() | pairs.transpose(-1, -2).bool(), diagonal=1)
        road = road.bool()
        self.vehicle += (pairs & ~self._pairs).sum(dim=(-2, -1))
        self.road += (road & ~self._road).sum(dim=-1)
        self._pairs.copy_(pairs)
        self._road.copy_(road)

    def reset(self, env_index, agent_index=None):
        if agent_index is None:
            self.vehicle[env_index] = 0
            self.road[env_index] = 0
            self._pairs[env_index] = False
            self._road[env_index] = False
        else:
            # Respawning a car ends its contacts but preserves the session totals.
            self._pairs[env_index, agent_index, :] = False
            self._pairs[env_index, :, agent_index] = False
            self._road[env_index, agent_index] = False
