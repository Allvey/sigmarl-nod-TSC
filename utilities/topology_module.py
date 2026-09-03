"""
This module implements the TopologyLearner, a neural network module for learning
topological relationships between agents in a multi-agent system.

The implementation is inspired by the concepts presented in:
- "Behavior and Topology-Aware Interaction Learning for Multi-Agent Systems" (BeTopNet)
- "VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation"

The core components are:
- TopoDecoderLayer: A single layer for updating relational features.
- TopoDecoder: A stack of TopoDecoderLayers to deeply encode relationships.
- TopologyHead: An MLP head to predict edge logits from relational features.
- TopologyLearner: The main module that encapsulates the entire process.
"""

import torch
from torch import nn
import math
from .topology_labels import (
    generate_e_labels_from_refs,
    generate_e_labels_with_corridor,
)


class TopoDecoderLayer(nn.Module):
    """
    A single layer for the Topology Decoder. It updates the relational features
    by combining ego query, neighbor semantics, relative features, and previous
    relational features.
    """

    def __init__(self, d_latent: int, d_ego: int, d_nei: int, d_rel: int):
        super().__init__()
        self.d_latent = d_latent
        self.mlp = nn.Sequential(
            nn.Linear(d_ego + d_nei + d_rel + d_latent, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, q_ego, s_neighbors, r_relative, q_R_in):
        B, K, _ = s_neighbors.shape
        q_ego_broadcasted = q_ego.unsqueeze(1).expand(-1, K, -1)

        combined_features = torch.cat(
            [q_ego_broadcasted, s_neighbors, r_relative, q_R_in], dim=-1
        )

        q_R_update = self.mlp(combined_features)
        q_R_out = q_R_in + q_R_update  # Residual connection
        return q_R_out


class TopoDecoder(nn.Module):
    """
    A stack of TopoDecoderLayers to deeply encode relationships between agents.
    """

    def __init__(
        self, num_layers: int, d_latent: int, d_ego: int, d_nei: int, d_rel: int
    ):
        super().__init__()
        self.initial_mapper = nn.Linear(d_rel, d_latent)
        self.layers = nn.ModuleList(
            [TopoDecoderLayer(d_latent, d_ego, d_nei, d_rel) for _ in range(num_layers)]
        )

    def forward(self, ego_observation, neighbors_observation, relative_features):
        q_R = self.initial_mapper(relative_features)

        for layer in self.layers:
            q_R = layer(ego_observation, neighbors_observation, relative_features, q_R)

        return q_R


class TopologyHead(nn.Module):
    """
    An MLP head to predict edge logits from the final relational features.
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.ReLU(),
            nn.Linear(d_latent // 2, 1),
        )

    def forward(self, q_R_final):
        return self.mlp(q_R_final)


class TopologyLearner(nn.Module):
    """
    The main module that encapsulates the TopoDecoder and TopologyHead to learn
    and predict topological relationships (edge logits).
    """

    def __init__(
        self, num_layers: int, d_latent: int, d_ego: int, d_nei: int, d_rel: int
    ):
        super().__init__()
        self.decoder = TopoDecoder(num_layers, d_latent, d_ego, d_nei, d_rel)
        self.head = TopologyHead(d_latent)

    def forward(self, ego_observation, neighbors_observation, relative_features):
        q_R_final = self.decoder(
            ego_observation, neighbors_observation, relative_features
        )
        edge_logits = self.head(q_R_final)
        return edge_logits


class NeighborActionHead(nn.Module):
    """
    A lightweight MLP head that maps the relational latent `q_R_final`
    to per-neighbor action predictions.

    This keeps changes minimal by reusing the decoder's output directly
    without modifying the existing `TopologyLearner`.
    """

    def __init__(self, d_latent: int, action_dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_size = max(1, int(d_latent * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, q_R_final: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_R_final: Tensor of shape [B, K, d_latent]
        Returns:
            Tensor of shape [B, K, action_dim]
        """
        return self.mlp(q_R_final)


class TopologyActionPredictor(nn.Module):
    """
    一个轻量包装器，用于基于拓扑结构的关系表征进行邻居动作预测。

    支持两种模式：
    - 参数共享（share_decoder=True）：复用已有 `TopologyLearner.decoder` 的参数与结构。
    - 参数独立（share_decoder=False）：新建一个同结构的 `TopoDecoder`，独立训练，避免与拓扑网络相互干扰。

    最小侵入：不改动 `TopologyLearner`，仅复用其结构信息来构造独立分支。
    """

    def __init__(
        self,
        topology_learner: TopologyLearner,
        action_dim: int,
        hidden_ratio: float = 0.5,
        share_decoder: bool = False,
    ):
        super().__init__()
        self.topology_learner = topology_learner
        self._share_decoder = share_decoder

        # 从拓扑头推断 d_latent，保证与拓扑结构的潜在维度一致
        d_latent = topology_learner.head.mlp[0].in_features
        self._d_latent = d_latent
        self.action_head = NeighborActionHead(
            d_latent=d_latent, action_dim=action_dim, hidden_ratio=hidden_ratio
        )

        # 若选择参数独立，则记录结构信息，解码器在首次前向时按输入维度“惰性初始化”
        if not self._share_decoder:
            dec = topology_learner.decoder
            self._num_layers = len(dec.layers)
            # d_rel 可由 initial_mapper 的输入维度得到；d_latent 已上面推断
            self._independent_decoder = None  # 将在 forward 中依据输入维度构建

    def forward(
        self,
        ego_observation: torch.Tensor,
        neighbors_observation: torch.Tensor,
        relative_features: torch.Tensor,
        return_edges: bool = False,
    ):
        """
        Args:
            ego_observation: Tensor [B, D_ego]
            neighbors_observation: Tensor [B, K, D_nei]
            relative_features: Tensor [B, K, D_rel]
            return_edges: If True, also returns edge logits from the topology head.

        Returns:
            If return_edges is False: action_pred [B, K, A]
            If return_edges is True: (action_pred [B, K, A], edge_logits [B, K, 1])
        """
        if self._share_decoder:
            q_R_final = self.topology_learner.decoder(
                ego_observation, neighbors_observation, relative_features
            )
            action_pred = self.action_head(q_R_final)
            if return_edges:
                edge_logits = self.topology_learner.head(q_R_final)
                return action_pred, edge_logits
            return action_pred

        # 参数独立：首次调用时按输入维度构建一个同结构的新解码器
        if self._independent_decoder is None:
            d_ego = ego_observation.shape[-1]
            d_nei = neighbors_observation.shape[-1]
            d_rel = relative_features.shape[-1]
            self._independent_decoder = TopoDecoder(
                num_layers=self._num_layers,
                d_latent=self._d_latent,
                d_ego=d_ego,
                d_nei=d_nei,
                d_rel=d_rel,
            ).to(ego_observation.device)

        q_R_final_ind = self._independent_decoder(
            ego_observation, neighbors_observation, relative_features
        )
        action_pred = self.action_head(q_R_final_ind)

        if return_edges:
            # 为避免干扰，边缘 logits 始终使用原拓扑网络路径计算
            q_R_topo = self.topology_learner.decoder(
                ego_observation, neighbors_observation, relative_features
            )
            edge_logits = self.topology_learner.head(q_R_topo)
            return action_pred, edge_logits
        return action_pred


class TopologyManager:
    def __init__(self, parameters, scenario):
        self.parameters = parameters
        self.scenario = scenario
        self.learner = None
        self.optim = None
        self.action_predictor = None
        self.action_optim = None
        self._num_layers = 2
        self._d_latent = 128
        self.action_dim = int(getattr(parameters, "topology_action_dim", 2) or 2)

    def ensure_initialized(
        self, ego_observation, neighbors_flat, relative_features, k_neighbors: int
    ):
        d_ego = int(ego_observation.shape[-1])
        d_nei = (
            int(neighbors_flat.shape[-1] // k_neighbors)
            if neighbors_flat is not None
            else 0
        )
        d_rel = int(relative_features.shape[-1])
        if self.learner is None:
            self.learner = TopologyLearner(
                num_layers=self._num_layers,
                d_latent=self._d_latent,
                d_ego=d_ego,
                d_nei=d_nei,
                d_rel=d_rel,
            ).to(self.parameters.device)
            self.optim = torch.optim.Adam(
                self.learner.parameters(), lr=self.parameters.lr
            )
            try:
                self.scenario.topology_learner = self.learner
            except Exception:
                pass
        if self.action_predictor is None:
            self.action_predictor = TopologyActionPredictor(
                topology_learner=self.learner,
                action_dim=self.action_dim,
                hidden_ratio=float(
                    getattr(self.parameters, "topology_action_hidden_ratio", 0.5)
                ),
                share_decoder=False,
            ).to(self.parameters.device)
            self.action_optim = torch.optim.Adam(
                self.action_predictor.parameters(),
                lr=float(
                    getattr(self.parameters, "lr_action_predictor", self.parameters.lr)
                ),
            )
            try:
                self.scenario.topology_action_predictor = self.action_predictor
            except Exception:
                pass

    def generate_labels(
        self,
        ref_local_flat: torch.Tensor,
        ref_neighbors_flat: torch.Tensor,
        neighbors_distance: torch.Tensor,
        neighbors_mask_distance: torch.Tensor,
        k_neighbors: int,
        n_points_short_term: int,
    ) -> torch.Tensor:
        use_corridor_labels = bool(
            getattr(self.parameters, "topology_use_corridor_labels", True)
        )
        if use_corridor_labels:
            agent_width_m = float(getattr(self.scenario, "agent_width", 0.2))
            corridor_buffer_m = float(
                getattr(self.parameters, "topology_corridor_buffer_m", 0.4)
            )
            pos_world_norm = getattr(self.scenario.normalizers, "pos_world", 1.0)
            if isinstance(pos_world_norm, torch.Tensor):
                pos_world_norm = pos_world_norm.to(ref_local_flat.device)
            return generate_e_labels_with_corridor(
                ref_local_flat,
                ref_neighbors_flat,
                neighbors_distance,
                neighbors_mask_distance,
                distance_threshold=float(self.scenario.thresholds.distance_mask_agents),
                k_neighbors=k_neighbors,
                n_points_short_term=n_points_short_term,
                pos_world_normalizer=pos_world_norm,
                corridor_agent_width=agent_width_m,
                corridor_buffer=corridor_buffer_m,
                use_intersection=bool(
                    getattr(self.parameters, "topology_use_intersection", False)
                ),
                use_corridor=bool(
                    getattr(self.parameters, "topology_use_corridor", True)
                ),
                use_mask=False,
            )
        return generate_e_labels_from_refs(
            ref_local_flat,
            ref_neighbors_flat,
            neighbors_distance,
            neighbors_mask_distance,
            distance_threshold=float(self.scenario.thresholds.distance_mask_agents),
            k_neighbors=k_neighbors,
            n_points_short_term=n_points_short_term,
            use_mask=False,
        )

    def compute_bce(
        self,
        ego_b: torch.Tensor,
        nei_b: torch.Tensor,
        rel_b: torch.Tensor,
        e_labels: torch.Tensor,
    ):
        edge_logits = self.learner(ego_b, nei_b, rel_b).squeeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            edge_logits, e_labels
        )
        return bce, edge_logits

    def zero_grad(self):
        if self.optim is not None:
            self.optim.zero_grad()

    def step(self, topo_weight: float):
        if self.optim is not None and float(topo_weight) > 0.0:
            self.optim.step()

    def load_state_dict(self, state_dict):
        if self.learner is not None:
            self.learner.load_state_dict(state_dict)

    def train_action_predictor(self, mini_batch_data) -> torch.Tensor:
        if (self.action_predictor is None) or (self.action_optim is None):
            return None
        ego_obs = mini_batch_data.get(("agents", "info", "ego_observation"))
        neighbors_flat = mini_batch_data.get(
            ("agents", "info", "topology_neighbors_observation_flat"), default=None
        )
        relative_feats = mini_batch_data.get(
            ("agents", "info", "topology_relative_features"), default=None
        )
        idx_neighbors = mini_batch_data.get(
            ("agents", "info", "topology_neighbors_indices"), default=None
        )
        if (
            (neighbors_flat is None)
            or (relative_feats is None)
            or (idx_neighbors is None)
        ):
            neighbors_flat = mini_batch_data.get(
                ("agents", "info", "neighbors_observation_flat")
            )
            relative_feats = mini_batch_data.get(
                ("agents", "info", "relative_features")
            )
            idx_neighbors = mini_batch_data.get(("agents", "info", "neighbors_indices"))
        k_neighbors = int(
            getattr(
                self.parameters,
                "n_topology_nearing_agents_observed",
                getattr(self.parameters, "n_nearing_agents_observed", 2),
            )
            or 2
        )
        d_ego = int(ego_obs.shape[-1])
        d_nei = (
            int(neighbors_flat.shape[-1] // k_neighbors)
            if neighbors_flat is not None
            else 0
        )
        d_rel = int(relative_feats.shape[-1])
        sample_shape = (
            neighbors_flat.shape[:-1]
            if neighbors_flat is not None
            else ego_obs.shape[:-1]
        )
        b_total = (
            int(math.prod(sample_shape))
            if len(sample_shape) > 0
            else int(ego_obs.shape[0])
        )
        ego_b = ego_obs.contiguous().view(b_total, d_ego)
        nei_b = (
            neighbors_flat.contiguous().view(b_total, k_neighbors, d_nei)
            if neighbors_flat is not None
            else None
        )
        rel_b = relative_feats.contiguous().view(b_total, k_neighbors, d_rel)
        labels_b_ap = self.generate_action_labels(mini_batch_data)
        assert nei_b is not None
        pred_ap = self.action_predictor(ego_b, nei_b, rel_b)
        action_loss = torch.nn.functional.mse_loss(pred_ap, labels_b_ap)
        self.action_optim.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.action_predictor.parameters(), self.parameters.max_grad_norm
        )
        self.action_optim.step()
        return action_loss.detach()

    def generate_action_labels(self, td: torch.Tensor) -> torch.Tensor:
        neighbors_flat = td.get(
            ("agents", "info", "topology_neighbors_observation_flat"), default=None
        )
        ego_obs = td.get(("agents", "info", "ego_observation"))
        k_neighbors = int(
            getattr(
                self.parameters,
                "n_topology_nearing_agents_observed",
                getattr(self.parameters, "n_nearing_agents_observed", 2),
            )
            or 2
        )
        sample_shape = (
            neighbors_flat.shape[:-1]
            if neighbors_flat is not None
            else ego_obs.shape[:-1]
        )
        b_total = (
            int(math.prod(sample_shape))
            if len(sample_shape) > 0
            else int(ego_obs.shape[0])
        )
        act_vel = td.get(("agents", "info", "act_vel"))
        act_steer = td.get(("agents", "info", "act_steer"))
        if act_vel.dim() == 1 and act_vel.numel() == b_total:
            act_vel = act_vel.view(*sample_shape)
        elif (
            act_vel.dim() == len(sample_shape) + 1
            and act_vel.shape[-1] == 1
            and act_vel.numel() == b_total
        ):
            act_vel = act_vel.squeeze(-1).view(*sample_shape)
        if act_steer.dim() == 1 and act_steer.numel() == b_total:
            act_steer = act_steer.view(*sample_shape)
        elif (
            act_steer.dim() == len(sample_shape) + 1
            and act_steer.shape[-1] == 1
            and act_steer.numel() == b_total
        ):
            act_steer = act_steer.squeeze(-1).view(*sample_shape)
        idx_neighbors = td.get(
            ("agents", "info", "topology_neighbors_indices"), default=None
        )
        if idx_neighbors is None:
            idx_neighbors = td.get(("agents", "info", "neighbors_indices"))
        idx_exp = idx_neighbors.to(dtype=torch.long, device=act_vel.device)
        if idx_exp.dim() == 2 and list(idx_exp.shape) == [
            int(b_total),
            int(k_neighbors),
        ]:
            idx_exp = idx_exp.view(*sample_shape, k_neighbors)
        k_local = int(idx_exp.shape[-1])
        act_vel_exp = act_vel.unsqueeze(-1).expand(*sample_shape, k_local)
        act_steer_exp = act_steer.unsqueeze(-1).expand(*sample_shape, k_local)
        vel_nei = torch.gather(act_vel_exp, dim=-2, index=idx_exp)
        steer_nei = torch.gather(act_steer_exp, dim=-2, index=idx_exp)
        labels_ap = torch.stack([vel_nei, steer_nei], dim=-1)
        labels_b_ap = labels_ap.contiguous().view(b_total, k_neighbors, self.action_dim)
        return labels_b_ap
