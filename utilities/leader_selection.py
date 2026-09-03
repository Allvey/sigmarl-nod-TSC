import torch


def get_soft_label_leader_threshold(parameters=None) -> float:
    leader_margin = 0.0
    if parameters is not None:
        leader_margin = float(getattr(parameters, "soft_label_leader_margin", 0.0) or 0.0)
    return 0.5 + leader_margin


def build_soft_label_leader_gate(
    p_final: torch.Tensor,
    selected_neighbor_indices: torch.Tensor,
    leader_margin: float = 0.0,
) -> torch.Tensor:
    """Return a [B, N, K] gate for neighbors that strongly precede ego agents."""
    if selected_neighbor_indices is None:
        raise ValueError("selected_neighbor_indices is required")

    B, N, _ = p_final.shape
    K = int(selected_neighbor_indices.shape[-1])
    device = p_final.device

    sel_idx = selected_neighbor_indices.to(device=device, dtype=torch.long)
    valid_mask = sel_idx.ge(0)
    sel_idx_clamped = sel_idx.clamp(min=0, max=N - 1)

    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, K)
    ego_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)
    leader_probs = p_final[batch_idx, sel_idx_clamped, ego_idx]

    threshold = 0.5 + float(leader_margin)
    gate = leader_probs.gt(threshold).to(dtype=p_final.dtype)
    return gate * valid_mask.to(dtype=p_final.dtype)
