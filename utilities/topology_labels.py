import torch
import numpy as np
import math
from typing import Tuple, Union

from .interX_original import interX


def _reshape_refs(
    ref_local_flat: torch.Tensor,
    ref_neighbors_flat: torch.Tensor,
    k_neighbors: int,
    n_points_short_term: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将扁平化的参考路径重塑为结构化张量，便于线段相交计算。

    - ref_local_flat: [B, T*2]
    - ref_neighbors_flat: [B, K*T*2]

    返回：
    - ref_local: [B, T, 2]
    - ref_neighbors: [B, K, T, 2]
    """
    B = ref_local_flat.shape[0]
    ref_local = ref_local_flat.view(B, n_points_short_term, 2)
    ref_neighbors = ref_neighbors_flat.view(B, k_neighbors, n_points_short_term, 2)
    return ref_local, ref_neighbors


def generate_e_labels_from_refs(
    ref_local_flat: torch.Tensor,
    ref_neighbors_flat: torch.Tensor,
    neighbors_distance: torch.Tensor,
    neighbors_mask_distance: torch.Tensor,
    distance_threshold: float,
    k_neighbors: int,
    n_points_short_term: int,
    use_mask: bool = True,
) -> torch.Tensor:
    """
    基于短期参考路径生成二值边标签 e_ij。

    定义：若自车短期参考与邻居 j 的短期参考在自车坐标系下存在任意一次线段相交，且邻居距离不超过 distance_threshold，则 e_ij=1，否则为 0。

    输入：
    - ref_local_flat: [B, T*2] 自车短期参考（展平）
    - ref_neighbors_flat: [B, K*T*2] 邻居短期参考（展平）
    - neighbors_distance: [B, K] 邻居距离（米）
    - neighbors_mask_distance: [B, K] 距离掩码（True 表示被距离门限屏蔽）
    - distance_threshold: 距离门限（米）
    - k_neighbors: K 邻居数
    - n_points_short_term: T 短期参考点数

    输出：
    - e_labels: [B, K] 浮点 {0,1}
    """
    device = ref_local_flat.device
    B = ref_local_flat.shape[0]
    ref_local, ref_neighbors = _reshape_refs(
        ref_local_flat, ref_neighbors_flat, k_neighbors, n_points_short_term
    )

    # 转为 numpy 以调用 interX（其实现基于 numpy），按批与邻居迭代
    e_labels = torch.zeros((B, k_neighbors), dtype=torch.float32, device=device)

    for b in range(B):
        L1 = ref_local[b].T.detach().cpu().numpy()  # shape [2, T]
        for j in range(k_neighbors):
            if use_mask and neighbors_mask_distance[b, j]:
                # 已被场景距离门限屏蔽，跳过
                continue
            if float(neighbors_distance[b, j].item()) > distance_threshold:
                continue

            L2 = ref_neighbors[b, j].T.detach().cpu().numpy()  # [2, T]
            try:
                # interX 返回是否存在线段相交；异常时安全回退为 False
                has_intersection = interX(L1, L2, is_return_points=False)
            except Exception:
                has_intersection = False
            if has_intersection:
                e_labels[b, j] = 1.0

    return e_labels


def generate_soft_labels_full_graph(
    short_term_all_agents: torch.Tensor,
    sigma: float = 1.0,
    tau: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    device = short_term_all_agents.device
    B, A, T, _ = short_term_all_agents.shape
    if T < 2:
        raise ValueError("At least two short-term path points are required.")

    # Transform every agent j into every ego agent i's local frame in one
    # broadcasted operation: [B, ego_i, agent_j, T, xy].
    vec = short_term_all_agents[:, :, 1] - short_term_all_agents[:, :, 0]
    theta = torch.atan2(vec[..., 1], vec[..., 0])
    c = torch.cos(theta)
    s = torch.sin(theta)
    origin = short_term_all_agents[:, :, 0]
    delta = (
        short_term_all_agents[:, None, :, :, :]
        - origin[:, :, None, None, :]
    )
    y_in_ego = (
        -s[:, :, None, None] * delta[..., 0]
        + c[:, :, None, None] * delta[..., 1]
    )
    own_y = torch.diagonal(y_in_ego, dim1=1, dim2=2).permute(0, 2, 1)
    d_ij = (own_y[:, :, None, :] - y_in_ego).abs().amin(dim=-1)
    d_ji = d_ij.transpose(1, 2)
    temperature = max(float(sigma) * float(tau), float(eps))
    P = torch.sigmoid((d_ji - d_ij) / temperature).to(torch.float32)
    diagonal = torch.arange(A, device=device)
    P[:, diagonal, diagonal] = 0.0
    return P


def _build_priority_order(P_single: torch.Tensor) -> list:
    A = int(P_single.shape[0])
    s = torch.zeros((A,), dtype=torch.float32, device=P_single.device)
    for i in range(A):
        s[i] = P_single[:, i].sum() - P_single[i, :].sum()
    order = torch.argsort(s, descending=True).tolist()
    return order


def break_cycles_and_build_priority_forest(
    P_full: torch.Tensor,
    eps_neutralize: float = 0.02,
) -> Tuple[torch.Tensor, list, list]:
    B, A, _ = P_full.shape
    P_out = P_full.clone()
    forests = []
    cycles_flags = []
    for b in range(B):
        P_b = P_out[b]
        order = _build_priority_order(P_b)
        pos = {order[k]: k for k in range(A)}
        for i in range(A):
            for j in range(A):
                if i == j:
                    continue
                pij = float(P_b[i, j].item())
                if pij > 0.5:
                    if pos[j] > pos[i]:
                        P_b[i, j] = max(0.0, 0.5 - eps_neutralize)
                        P_b[j, i] = max(0.0, 0.5 - eps_neutralize)
        edges = []
        parents = [-1 for _ in range(A)]
        for i in range(A):
            cand = [
                (j, float(P_b[i, j].item()))
                for j in range(A)
                if j != i and pos[j] < pos[i] and float(P_b[i, j].item()) > 0.5
            ]
            if len(cand) > 0:
                j_best, _ = max(cand, key=lambda x: x[1])
                parents[i] = j_best
                edges.append((j_best, i, float(P_b[i, j_best].item())))
        adj = [[] for _ in range(A)]
        for (u, v, _) in edges:
            adj[u].append(v)
        visited = [0] * A
        stack = [0] * A

        def _dfs(u):
            visited[u] = 1
            stack[u] = 1
            for v in adj[u]:
                if visited[v] == 0:
                    if _dfs(v):
                        return True
                elif stack[v] == 1:
                    return True
            stack[u] = 0
            return False

        has_cycle = False
        for n in range(A):
            if visited[n] == 0:
                if _dfs(n):
                    has_cycle = True
                    break
        forests.append(edges)
        cycles_flags.append(has_cycle)
    return P_out, forests, cycles_flags


def break_cycles_min_cost(
    P_full: torch.Tensor,
    eps_neutralize: float = 0.02,
) -> Tuple[torch.Tensor, list]:
    B, A, _ = P_full.shape
    P_out = P_full.clone()
    removed = []
    for b in range(B):
        P_b = P_out[b]

        def _build_adj():
            adj = [[] for _ in range(A)]
            for j in range(A):
                for i in range(A):
                    if i == j:
                        continue
                    if float(P_b[i, j].item()) > 0.5:
                        adj[j].append(i)
            return adj

        while True:
            adj = _build_adj()
            visited = [0] * A
            stack = [0] * A
            path = []
            cycle_nodes = None

            def _dfs(u):
                nonlocal cycle_nodes
                visited[u] = 1
                stack[u] = 1
                path.append(u)
                for v in adj[u]:
                    if visited[v] == 0:
                        _dfs(v)
                        if cycle_nodes is not None:
                            return
                    elif stack[v] == 1:
                        idx = 0
                        for k in range(len(path)):
                            if path[k] == v:
                                idx = k
                                break
                        cycle_nodes = path[idx:].copy()
                        return
                stack[u] = 0
                path.pop()

            for s in range(A):
                if visited[s] == 0 and cycle_nodes is None:
                    _dfs(s)
            if cycle_nodes is None:
                break
            min_w = 1e9
            min_edge = None
            L = len(cycle_nodes)
            for k in range(L):
                u = cycle_nodes[k]
                v = cycle_nodes[(k + 1) % L]
                w = float(P_b[v, u].item())
                if w < min_w:
                    min_w = w
                    min_edge = (u, v, w)
            if min_edge is None:
                break
            u, v, w = min_edge
            P_b[v, u] = max(0.0, 0.5 - float(eps_neutralize))
            P_b[u, v] = max(0.0, 0.5 - float(eps_neutralize))
            removed.append((b, u, v, w))
    return P_out, removed


def enforce_transitivity(
    P_full: torch.Tensor,
    eps_neutralize: float = 0.02,
    gamma: float = 0.5,
    delta: float = 1e-3,
) -> torch.Tensor:
    B, A, _ = P_full.shape
    P_out = P_full.clone()
    for b in range(B):
        P_b = P_out[b]
        W = torch.zeros((A, A), dtype=torch.float32, device=P_b.device)
        for i in range(A):
            for j in range(A):
                if i == j:
                    continue
                pij = float(P_b[i, j].item())
                if pij > 0.5:
                    W[j, i] = pij
        for k in range(A):
            for u in range(A):
                if u == k:
                    continue
                for v in range(A):
                    if v == k or v == u:
                        continue
                    via = min(float(W[u, k].item()), float(W[k, v].item()))
                    if via > float(W[u, v].item()):
                        W[u, v] = via
        for i in range(A):
            for j in range(i + 1, A):
                win_ji = float(W[j, i].item())
                win_ij = float(W[i, j].item())
                if max(win_ji, win_ij) <= 0.5:
                    continue
                if win_ji >= win_ij:
                    li = i
                    wj = j
                    win = win_ji
                else:
                    li = j
                    wj = i
                    win = win_ij
                sum_ij = float(P_b[li, wj].item()) + float(P_b[wj, li].item())
                neutral_mass = max(0.0, 1.0 - sum_ij)
                target = 0.5 + float(gamma) * (win - 0.5)
                lower_bound = max(
                    float(P_b[li, wj].item()),
                    target,
                    0.5 + float(eps_neutralize) - neutral_mass,
                )
                upper_bound = 1.0 - neutral_mass
                new_pij = min(lower_bound, upper_bound)
                if abs(new_pij - 0.5) < float(delta):
                    new_pij = 0.5 + float(delta)
                max_pji = 0.5 - float(eps_neutralize)
                new_pji = max(0.0, upper_bound - new_pij)
                if new_pji > max_pji:
                    new_pji = max_pji
                    new_pij = max(0.0, upper_bound - new_pji)
                P_b[li, wj] = float(new_pij)
                P_b[wj, li] = float(new_pji)
    return P_out


def complete_total_order(
    P_full: torch.Tensor,
    eps_neutralize: float = 0.02,
    gamma: float = 0.5,
    delta: float = 1e-3,
) -> torch.Tensor:
    B, A, _ = P_full.shape
    P_out = P_full.clone()
    for b in range(B):
        P_b = P_out[b]
        adj = [[] for _ in range(A)]
        indeg = [0] * A
        thr = 0.5 + float(eps_neutralize) + float(delta)
        for i in range(A):
            for j in range(A):
                if i == j:
                    continue
                pij = float(P_b[i, j].item())
                if pij > thr:
                    adj[i].append(j)
                    indeg[j] += 1
        q = sorted([i for i in range(A) if indeg[i] == 0])
        order = []
        while q:
            u = q.pop(0)
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    # keep ascending index for tie-break
                    idxs = q + [v]
                    q = sorted(idxs)
        # fallback linear order if graph empty
        if len(order) < A:
            rest = [i for i in range(A) if i not in order]
            order += sorted(rest)
        pos = {order[k]: k for k in range(A)}
        W = torch.zeros((A, A), dtype=torch.float32, device=P_b.device)
        for i in range(A):
            for j in range(A):
                if i == j:
                    continue
                pij = float(P_b[i, j].item())
                if pij > 0.5:
                    W[j, i] = pij
        for k in range(A):
            for u in range(A):
                if u == k:
                    continue
                for v in range(A):
                    if v == k or v == u:
                        continue
                    via = min(float(W[u, k].item()), float(W[k, v].item()))
                    if via > float(W[u, v].item()):
                        W[u, v] = via
        for r in range(A):
            for s in range(r + 1, A):
                i = order[r]
                j = order[s]
                win = float(W[j, i].item())
                target = 0.5 + float(gamma) * (max(win, 0.5) - 0.5)
                sum_ij = float(P_b[i, j].item()) + float(P_b[j, i].item())
                neutral_mass = max(0.0, 1.0 - sum_ij)
                lower_bound = max(
                    float(P_b[i, j].item()),
                    target,
                    0.5 + float(eps_neutralize) - neutral_mass,
                )
                upper_bound = 1.0 - neutral_mass
                new_pij = min(lower_bound, upper_bound)
                if abs(new_pij - 0.5) < float(delta):
                    new_pij = 0.5 + float(delta)
                max_pji = 0.5 - float(eps_neutralize)
                new_pji = max(0.0, upper_bound - new_pij)
                if new_pji > max_pji:
                    new_pji = max_pji
                    new_pij = max(0.0, upper_bound - new_pji)
                P_b[i, j] = float(new_pij)
                P_b[j, i] = float(new_pji)
    return P_out


def generate_e_labels_with_corridor(
    ref_local_flat: torch.Tensor,
    ref_neighbors_flat: torch.Tensor,
    neighbors_distance: torch.Tensor,
    neighbors_mask_distance: torch.Tensor,
    distance_threshold: float,
    k_neighbors: int,
    n_points_short_term: int,
    pos_world_normalizer: Union[float, torch.Tensor],
    corridor_agent_width: float,
    corridor_buffer: float = 0.0,
    use_intersection: bool = True,
    use_corridor: bool = True,
    use_mask: bool = True,
    max_time_lag_steps: int = None,
) -> torch.Tensor:
    """
    结合“线段相交”与“管道宽度近邻”生成二值边标签。

    定义：若满足以下任一条件且邻居距离不超过 distance_threshold，则 e_ij=1：
    - use_intersection: 自车短期折线与邻居短期折线存在线段相交；
    - use_corridor: 两条折线的最小中心线距离 <= corridor_threshold，其中
      corridor_threshold = corridor_agent_width + corridor_buffer（假设两车等宽，带宽为两车半宽之和+缓冲）。

    说明：ref_* 输入通常是按 pos_world_normalizer 归一化后的坐标，本函数内会还原到米单位再做距离计算。

    输入形状与含义与 generate_e_labels_from_refs 相同，新增：
    - pos_world_normalizer: 坐标归一化常数（将坐标乘以该值恢复米单位）
    - corridor_agent_width: 车辆宽度（米）
    - corridor_buffer: 走廊缓冲（米），默认 0
    - use_intersection/use_corridor: 是否启用各判定项
    - max_time_lag_steps: 允许的最大时间差（步数），仅在走廊距离判定时生效；
      若为 None 则比较所有点对；例如设为 2 表示 |t_ego - t_nei| <= 2 的点对才参与。

    输出：
    - e_labels: [B, K] 浮点 {0,1}
    """
    device = ref_local_flat.device
    B = ref_local_flat.shape[0]
    ref_local, ref_neighbors = _reshape_refs(
        ref_local_flat, ref_neighbors_flat, k_neighbors, n_points_short_term
    )

    # 距离与掩码联合可行性：仅在可行且距离未超过阈值时参与判定
    dist_ok = neighbors_distance <= distance_threshold
    if use_mask:
        # 鲁棒处理：确保掩码为布尔类型
        neighbors_mask_distance = neighbors_mask_distance.bool()
        dist_ok = dist_ok & (~neighbors_mask_distance)

    e_labels = torch.zeros((B, k_neighbors), dtype=torch.float32, device=device)
    corridor_threshold = corridor_agent_width + corridor_buffer

    # 走廊近邻（批量矢量化）：计算每对 (b,j) 折线的最短中心线距离
    if use_corridor:
        # 将坐标按 pos_world_normalizer 还原到米单位，支持标量或 [2] 向量广播
        ref_local_m = ref_local * pos_world_normalizer  # [B, T, 2]
        ref_neighbors_m = ref_neighbors * pos_world_normalizer  # [B, K, T, 2]

        # 展平到 [B*K, T, 2] 后一次性计算 cdist，避免 Python 双重 for 循环
        local_m_pairs = (
            ref_local_m.unsqueeze(1)
            .expand(B, k_neighbors, n_points_short_term, 2)
            .contiguous()
        )
        local_m_pairs = local_m_pairs.view(B * k_neighbors, n_points_short_term, 2)
        nei_m_pairs = ref_neighbors_m.view(B * k_neighbors, n_points_short_term, 2)

        dmat = torch.cdist(local_m_pairs, nei_m_pairs)  # [B*K, T, T]

        # 若限定最大时间差，仅保留 |t_ego - t_nei| <= max_time_lag_steps 的点对
        if (max_time_lag_steps is not None) and (max_time_lag_steps >= 0):
            t_idx = torch.arange(n_points_short_term, device=device)
            lag_mask = (t_idx.view(-1, 1) - t_idx.view(1, -1)).abs() <= int(
                max_time_lag_steps
            )  # [T, T]
            lag_mask = lag_mask.unsqueeze(0).expand(
                B * k_neighbors, -1, -1
            )  # [B*K, T, T]
            large_val = torch.tensor(1e6, device=device, dtype=dmat.dtype)
            dmat = torch.where(lag_mask, dmat, large_val)
        dmin_pairs = dmat.min(dim=-1).values.min(dim=-1).values  # [B*K]
        dmin = dmin_pairs.view(B, k_neighbors)

        corridor_pos = (dmin <= corridor_threshold) & dist_ok
        e_labels = corridor_pos.float()

        # 若仅用走廊判定，可提前返回
        if not use_intersection:
            return e_labels

    # 对未被走廊匹配为正的样本，再做线段相交判定（减少 interX 调用次数）
    if use_intersection:
        # 候选集合：满足 dist_ok 且尚未为正的 (b,j)
        candidate_mask = dist_ok & (e_labels == 0)
        idxs = candidate_mask.nonzero(as_tuple=False)  # [N, 2] -> (b, j)

        for idx in idxs:
            b, j = int(idx[0].item()), int(idx[1].item())
            L1 = ref_local[b].T.detach().cpu().numpy()  # [2, T]
            L2 = ref_neighbors[b, j].T.detach().cpu().numpy()  # [2, T]
            try:
                is_inter = bool(interX(L1, L2, is_return_points=False))
            except Exception:
                is_inter = False
            if is_inter:
                e_labels[b, j] = 1.0

    return e_labels
