# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from utilities.helper_training import Parameters, SaveData
from utilities.topology_labels import (
    generate_e_labels_from_refs,
    generate_e_labels_with_corridor,
)
import torch
import os

from vmas.simulator.utils import save_video
import json

from utilities.mappo_cavs import mappo_cavs

from utilities.constants import SCENARIOS

path = "outputs/testing_random/"  # Adjust parameters therein

try:
    path_to_json_file = next(
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")
    )  # Find the first json file in the folder
    # Load parameters from the saved json file
    with open(path_to_json_file, "r") as file:
        data = json.load(file)
        saved_data = SaveData.from_dict(data)
        parameters = saved_data.parameters

        # Adjust parameters
        parameters.is_testing_mode = True
        parameters.is_real_time_rendering = True
        parameters.is_save_eval_results = False
        parameters.is_load_model = True
        parameters.is_load_final_model = False
        parameters.is_load_out_td = False
        parameters.max_steps = 1200  # 1200 -> 1 min
        if parameters.is_load_out_td:
            parameters.num_vmas_envs = 32
        else:
            parameters.num_vmas_envs = 1

        parameters.scenario_type = (
            # "intersection_2"
            # "roundabout_1"
            # "CPM_entire"
            # "CPM_mixed"  
            "on_ramp_1"
            # roundabout_1, intersection_1/2/3, CPM_mixed
        )
        parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]

        parameters.is_save_simulation_video = True
        parameters.is_visualize_short_term_path = False
        parameters.is_visualize_lane_boundary = False
        parameters.is_visualize_extra_info = True
        parameters.is_visualize_observed_neighbors = False
        # 固定展示邻居的智能体索引（0-based）。可按需修改。
        parameters.visualize_observed_neighbors_agent_index = 0
        # 关闭“未来三个位置点”可视化
        parameters.is_visualize_future_three_points = False
        parameters.is_visualize_agent_trajectory = True
        parameters.agent_trajectory_len = 25
        # 放慢测试渲染速度，便于观察（倍数：>1 越慢）。
        parameters.render_pause_scale = 1.0
        parameters.is_print_agent_speed = True
        parameters.print_speed_interval = 1
        parameters.is_save_agent_speed = True
        parameters.agent_speed_log_path = os.path.join(path, "agent_speeds.csv")
        parameters.agent_speed_log_interval = 1
        parameters.is_save_action_prediction_error = True
        parameters.action_prediction_error_log_path = os.path.join(
            path, "action_prediction_errors.csv"
        )
        parameters.action_prediction_error_summary_path = os.path.join(
            path, "action_prediction_error_summary.json"
        )
        parameters.action_prediction_error_log_interval = 1

        env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)

        os.makedirs(path, exist_ok=True)
        speed_log_f = None
        if getattr(parameters, "is_save_agent_speed", False):
            speed_log_path = getattr(
                parameters, "agent_speed_log_path", None
            ) or os.path.join(path, "agent_speeds.csv")
            speed_log_f = open(speed_log_path, "w", encoding="utf-8", buffering=1)
            speed_log_f.write(
                "step,t_sec,"
                + ",".join([f"agent_{i+1}_speed" for i in range(parameters.n_agents)])
                + "\n"
            )

        action_pred_log_f = None
        action_pred_stats = {
            "count": 0,
            "sum_abs_vel": 0.0,
            "sum_abs_steer": 0.0,
            "sum_sq_vel": 0.0,
            "sum_sq_steer": 0.0,
            "sum_l2": 0.0,
            "max_abs_vel": 0.0,
            "max_abs_steer": 0.0,
            "warned_missing_predictor": False,
        }
        if getattr(parameters, "is_save_action_prediction_error", False):
            action_pred_log_path = getattr(
                parameters, "action_prediction_error_log_path", None
            ) or os.path.join(path, "action_prediction_errors.csv")
            action_pred_log_f = open(
                action_pred_log_path, "w", encoding="utf-8", buffering=1
            )
            action_pred_log_f.write(
                "step,t_sec,ego_agent,neighbor_agent,"
                "pred_vel_norm,pred_steer_norm,gt_vel_norm,gt_steer_norm,"
                "abs_vel_norm,abs_steer_norm,l2_norm\n"
            )

        def update_action_prediction_error(env, td, step_val):
            predictor = getattr(env.scenario, "topology_action_predictor", None)
            if predictor is None or not getattr(
                env.scenario.parameters, "is_topology_action_predictor_loaded", False
            ):
                if not action_pred_stats["warned_missing_predictor"]:
                    print(
                        "[ActionPred] Skipped: no trained topology action predictor checkpoint loaded."
                    )
                    action_pred_stats["warned_missing_predictor"] = True
                return

            ego_obs = td.get(("agents", "info", "ego_observation"), default=None)
            neighbors_flat = td.get(
                ("agents", "info", "topology_neighbors_observation_flat"),
                default=None,
            )
            relative_feats = td.get(
                ("agents", "info", "topology_relative_features"), default=None
            )
            neighbor_idx = td.get(
                ("agents", "info", "topology_neighbors_indices"), default=None
            )
            neighbor_mask = td.get(
                ("agents", "info", "topology_neighbors_mask_distance"), default=None
            )

            if (
                neighbors_flat is None
                or relative_feats is None
                or neighbor_idx is None
            ):
                neighbors_flat = td.get(
                    ("agents", "info", "neighbors_observation_flat"), default=None
                )
                relative_feats = td.get(
                    ("agents", "info", "relative_features"), default=None
                )
                neighbor_idx = td.get(
                    ("agents", "info", "neighbors_indices"), default=None
                )
                neighbor_mask = td.get(
                    ("agents", "info", "neighbors_mask_distance"), default=None
                )

            act_vel = td.get(("agents", "info", "act_vel"), default=None)
            act_steer = td.get(("agents", "info", "act_steer"), default=None)
            if (
                ego_obs is None
                or neighbors_flat is None
                or relative_feats is None
                or neighbor_idx is None
                or act_vel is None
                or act_steer is None
            ):
                return

            B, N, D_ego = ego_obs.shape
            K = int(neighbor_idx.shape[-1])
            D_nei = int(neighbors_flat.shape[-1] // K)
            d_rel = int(relative_feats.shape[-1])
            ego_b = ego_obs.contiguous().view(B * N, D_ego)
            nei_b = neighbors_flat.contiguous().view(B * N, K, D_nei)
            rel_b = relative_feats.contiguous().view(B * N, K, d_rel)

            with torch.no_grad():
                pred = predictor(ego_b, nei_b, rel_b).view(B, N, K, -1)

            act_vel = act_vel.squeeze(-1) if act_vel.dim() == 3 else act_vel
            act_steer = act_steer.squeeze(-1) if act_steer.dim() == 3 else act_steer
            idx = neighbor_idx.to(dtype=torch.long, device=act_vel.device)
            valid = idx.ge(0)
            if neighbor_mask is not None:
                valid = valid & (~neighbor_mask.to(device=act_vel.device).bool())
            idx_clamped = idx.clamp(min=0, max=N - 1)

            act_vel_all = act_vel.unsqueeze(1).expand(B, N, N)
            act_steer_all = act_steer.unsqueeze(1).expand(B, N, N)
            gt_vel = torch.gather(act_vel_all, dim=2, index=idx_clamped)
            gt_steer = torch.gather(act_steer_all, dim=2, index=idx_clamped)
            gt = torch.stack([gt_vel, gt_steer], dim=-1)

            err = pred[..., :2] - gt
            abs_err = err.abs()
            l2_err = torch.linalg.norm(err, dim=-1)
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                return

            abs_vel_valid = abs_err[..., 0][valid]
            abs_steer_valid = abs_err[..., 1][valid]
            l2_valid = l2_err[valid]
            action_pred_stats["count"] += valid_count
            action_pred_stats["sum_abs_vel"] += float(abs_vel_valid.sum().item())
            action_pred_stats["sum_abs_steer"] += float(abs_steer_valid.sum().item())
            action_pred_stats["sum_sq_vel"] += float(
                (err[..., 0][valid] ** 2).sum().item()
            )
            action_pred_stats["sum_sq_steer"] += float(
                (err[..., 1][valid] ** 2).sum().item()
            )
            action_pred_stats["sum_l2"] += float(l2_valid.sum().item())
            action_pred_stats["max_abs_vel"] = max(
                action_pred_stats["max_abs_vel"], float(abs_vel_valid.max().item())
            )
            action_pred_stats["max_abs_steer"] = max(
                action_pred_stats["max_abs_steer"],
                float(abs_steer_valid.max().item()),
            )

            interval = int(
                getattr(parameters, "action_prediction_error_log_interval", 1)
            )
            interval = max(1, interval)
            should_log = (step_val is None) or (step_val % interval == 0)
            if action_pred_log_f is not None and should_log:
                t_sec = (
                    float(step_val) * float(parameters.dt)
                    if step_val is not None
                    else ""
                )
                step_out = step_val if step_val is not None else ""
                valid_cpu = valid.detach().cpu()
                pred_cpu = pred.detach().cpu()
                gt_cpu = gt.detach().cpu()
                abs_cpu = abs_err.detach().cpu()
                l2_cpu = l2_err.detach().cpu()
                idx_cpu = idx.detach().cpu()
                for ego_i in range(N):
                    for k_i in range(K):
                        if not bool(valid_cpu[0, ego_i, k_i].item()):
                            continue
                        action_pred_log_f.write(
                            f"{step_out},{t_sec},{ego_i+1},"
                            f"{int(idx_cpu[0, ego_i, k_i].item())+1},"
                            f"{float(pred_cpu[0, ego_i, k_i, 0].item()):.6f},"
                            f"{float(pred_cpu[0, ego_i, k_i, 1].item()):.6f},"
                            f"{float(gt_cpu[0, ego_i, k_i, 0].item()):.6f},"
                            f"{float(gt_cpu[0, ego_i, k_i, 1].item()):.6f},"
                            f"{float(abs_cpu[0, ego_i, k_i, 0].item()):.6f},"
                            f"{float(abs_cpu[0, ego_i, k_i, 1].item()):.6f},"
                            f"{float(l2_cpu[0, ego_i, k_i].item()):.6f}\n"
                        )

            count = max(1, action_pred_stats["count"])
            if should_log:
                print(
                    "[ActionPred] "
                    f"count={action_pred_stats['count']}, "
                    f"MAE(v,steer)=("
                    f"{action_pred_stats['sum_abs_vel']/count:.4f}, "
                    f"{action_pred_stats['sum_abs_steer']/count:.4f}), "
                    f"RMSE(v,steer)=("
                    f"{(action_pred_stats['sum_sq_vel']/count) ** 0.5:.4f}, "
                    f"{(action_pred_stats['sum_sq_steer']/count) ** 0.5:.4f})"
                )

        def render_and_print_topology(env, td):
            step_val = None
            if (
                getattr(env, "scenario", None) is not None
                and hasattr(env.scenario, "timer")
                and hasattr(env.scenario.timer, "step")
            ):
                step_val = env.scenario.timer.step[0]
                step_val = (
                    int(step_val.item())
                    if isinstance(step_val, torch.Tensor)
                    else int(step_val)
                )

            try:
                update_action_prediction_error(env, td, step_val)
            except Exception as e:
                print(f"[ActionPred] Skipped due to error: {e}")

            try:
                topo = getattr(env.scenario, "topology_learner", None)
                if topo is not None:
                    K = env.scenario.parameters.n_nearing_agents_observed
                    agent_i = 0  # 智能体1（1-based 显示，0-based 索引）

                    ego = td.get(("agents", "info", "ego_observation"))
                    nei_flat = td.get(("agents", "info", "neighbors_observation_flat"))
                    rel = td.get(("agents", "info", "relative_features"))
                    mask_dist = td.get(("agents", "info", "neighbors_mask_distance"))
                    nei_dist = td.get(("agents", "info", "neighbors_distance"))
                    ref_local = td.get(("agents", "info", "ref_local"))
                    ref_neighbors_local = td.get(
                        ("agents", "info", "ref_neighbors_local")
                    )

                    if (
                        ego is not None
                        and nei_flat is not None
                        and rel is not None
                        and mask_dist is not None
                    ):
                        ego_a = ego[0, agent_i]  # [D_ego]
                        nei_flat_a = nei_flat[0, agent_i]  # [K*D_nei]
                        D_nei = (
                            nei_flat_a.shape[0] // K if nei_flat_a.numel() > 0 else 0
                        )
                        if D_nei > 0:
                            nei_a = nei_flat_a.view(K, D_nei).unsqueeze(
                                0
                            )  # [1,K,D_nei]
                            rel_a = rel[0, agent_i].unsqueeze(0)  # [1,K,d_rel]
                            ego_b = ego_a.unsqueeze(0)  # [1,D_ego]

                            with torch.no_grad():
                                edge_logits = topo(ego_b, nei_a, rel_a)  # [1,K,1]
                                edge_probs = edge_logits.squeeze(-1).squeeze(0)  # [K]
                                mask_k = mask_dist[0, agent_i].bool()
                                valid_mask = (~mask_k).bool()
                                edge_probs_valid = edge_probs[valid_mask]
                                nei_idx = (
                                    env.scenario.observations.nearing_agents_indices[
                                        :, agent_i
                                    ][0]
                                )  # [K]
                                nei_idx_valid = nei_idx[valid_mask]

                            # 计算真实标签 e_ij（若需要的输入齐全）
                            gt_vals_valid = None
                            try:
                                if (
                                    nei_dist is not None
                                    and ref_local is not None
                                    and ref_neighbors_local is not None
                                ):
                                    B = 1
                                    T_short = ref_local.shape[-1] // 2
                                    distance_thresh = float(
                                        env.scenario.thresholds.distance_mask_agents
                                    )
                                    pos_world_norm = (
                                        env.scenario.normalizers.pos_world
                                    )  # [2] 张量，按轴还原坐标到米
                                    agent_width = float(
                                        getattr(env.scenario, "agent_width", 0.2)
                                    )
                                    corridor_buffer = 0.4  # 可按需调整缓冲（米）
                                    e_labels = generate_e_labels_with_corridor(
                                        ref_local[0, agent_i].unsqueeze(0),
                                        ref_neighbors_local[0, agent_i].unsqueeze(0),
                                        nei_dist[0, agent_i].unsqueeze(0),
                                        mask_dist[0, agent_i].unsqueeze(0),
                                        distance_thresh,
                                        K,
                                        T_short,
                                        pos_world_norm,
                                        agent_width,
                                        corridor_buffer,
                                        use_intersection=True,
                                        use_corridor=True,
                                        max_time_lag_steps=2,
                                    )  # [1,K]
                                    gt_vals_valid = e_labels.squeeze(0)[valid_mask]
                            except Exception as e_gt:
                                gt_vals_valid = None
                                print(f"[Topo] GT labels compute error: {e_gt}")

                            if edge_probs_valid.numel() > 0:
                                k_out = min(K, edge_probs_valid.numel())
                                topk_vals, topk_idx = torch.topk(
                                    edge_probs_valid, k=k_out
                                )
                                top_neighbor_ids = nei_idx_valid[topk_idx]
                                if (
                                    gt_vals_valid is not None
                                    and gt_vals_valid.numel()
                                    == edge_probs_valid.numel()
                                ):
                                    gt_top = gt_vals_valid[topk_idx]
                                    pairs_str = ", ".join(
                                        [
                                            f"{int(id.item())+1}:{float(val.item()):.3f}|{int(gt.item())}"
                                            for id, val, gt in zip(
                                                top_neighbor_ids, topk_vals, gt_top
                                            )
                                        ]
                                    )
                                    print(
                                        f"[Topo] Agent 1 top-{k_out} (prob|gt): {pairs_str}"
                                    )
                                else:
                                    pairs_str = ", ".join(
                                        [
                                            f"{int(id.item())+1}:{float(val.item()):.3f}"
                                            for id, val in zip(
                                                top_neighbor_ids, topk_vals
                                            )
                                        ]
                                    )
                                    print(
                                        f"[Topo] Agent 1 top-{k_out} (prob): {pairs_str}"
                                    )
                            else:
                                print("[Topo] Agent 1 top-K (prob): 无有效邻居")
            except Exception as e:
                print(f"[Topo] Skipped topology print due to error: {e}")

            try:
                if getattr(parameters, "is_print_agent_speed", False):
                    interval = int(getattr(parameters, "print_speed_interval", 1))
                    interval = max(1, interval)

                    should_print = (step_val is None) or (step_val % interval == 0)
                    if should_print:
                        vel = td.get(("agents", "info", "vel"), default=None)
                        if vel is None and getattr(env, "scenario", None) is not None:
                            vel = torch.stack(
                                [a.state.vel for a in env.scenario.world.agents], dim=1
                            )
                        if isinstance(vel, torch.Tensor):
                            vel_env = vel[0] if vel.dim() == 3 else vel
                            speed = vel_env.norm(dim=-1)
                            ego_i = int(
                                getattr(
                                    parameters,
                                    "visualize_observed_neighbors_agent_index",
                                    0,
                                )
                            )
                            ego_i = max(0, min(int(speed.shape[0]) - 1, ego_i))
                            t_sec = (
                                float(step_val) * float(parameters.dt)
                                if step_val is not None
                                else None
                            )
                            if t_sec is None:
                                prefix = "[Speed]"
                            else:
                                prefix = f"[Speed] n={step_val}, t={t_sec:.2f}s"
                            print(
                                f"{prefix}, ego={ego_i+1}, v={float(speed[ego_i].item()):.3f} m/s, "
                                f"mean={float(speed.mean().item()):.3f}, max={float(speed.max().item()):.3f}"
                            )
            except Exception as e:
                print(f"[Speed] Skipped speed print due to error: {e}")

            try:
                if speed_log_f is not None:
                    interval = int(getattr(parameters, "agent_speed_log_interval", 1))
                    interval = max(1, interval)

                    should_log = (step_val is None) or (step_val % interval == 0)
                    if should_log:
                        vel = td.get(("agents", "info", "vel"), default=None)
                        if vel is None and getattr(env, "scenario", None) is not None:
                            vel = torch.stack(
                                [a.state.vel for a in env.scenario.world.agents], dim=1
                            )
                        if isinstance(vel, torch.Tensor):
                            vel_env = vel[0] if vel.dim() == 3 else vel
                            speed = vel_env.norm(dim=-1)
                            t_sec = (
                                float(step_val) * float(parameters.dt)
                                if step_val is not None
                                else ""
                            )
                            step_out = step_val if step_val is not None else ""
                            speed_list = [f"{float(v.item()):.6f}" for v in speed]
                            speed_log_f.write(
                                f"{step_out},{t_sec}," + ",".join(speed_list) + "\n"
                            )
            except Exception as e:
                print(f"[SpeedLog] Skipped speed logging due to error: {e}")

            return env.render(mode="rgb_array", visualize_when_rgb=True)

        try:
            rollout_result = env.rollout(
                max_steps=parameters.max_steps - 1,
                policy=policy,
                priority_module=priority_module,
                callback=render_and_print_topology,  # 同步渲染 + 概率版 Top-K 输出
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=parameters.is_save_simulation_video,
            )
        finally:
            if speed_log_f is not None:
                speed_log_f.close()
            if action_pred_log_f is not None:
                action_pred_log_f.close()

        # 兼容返回值：部分版本仅返回 out_td
        if isinstance(rollout_result, tuple) and len(rollout_result) == 2:
            out_td, frame_list = rollout_result
        else:
            out_td = rollout_result
            frame_list = []
        if len(frame_list) > 0:
            save_video(os.path.join(path, "video"), frame_list, fps=1 / parameters.dt)
        if action_pred_stats["count"] > 0:
            count = action_pred_stats["count"]
            summary = {
                "count": count,
                "mae_vel_norm": action_pred_stats["sum_abs_vel"] / count,
                "mae_steer_norm": action_pred_stats["sum_abs_steer"] / count,
                "rmse_vel_norm": (action_pred_stats["sum_sq_vel"] / count) ** 0.5,
                "rmse_steer_norm": (action_pred_stats["sum_sq_steer"] / count) ** 0.5,
                "mean_l2_norm": action_pred_stats["sum_l2"] / count,
                "max_abs_vel_norm": action_pred_stats["max_abs_vel"],
                "max_abs_steer_norm": action_pred_stats["max_abs_steer"],
            }
            summary_path = getattr(
                parameters, "action_prediction_error_summary_path", None
            ) or os.path.join(path, "action_prediction_error_summary.json")
            with open(summary_path, "w", encoding="utf-8") as summary_f:
                json.dump(summary, summary_f, indent=4)
            print(f"[ActionPred] Summary saved to {summary_path}: {summary}")
except StopIteration:
    raise FileNotFoundError("No json file found.")
