# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from utilities.helper_training import Parameters, SaveData
import torch
import os

from vmas.simulator.utils import save_video
import json

from utilities.mappo_cavs import mappo_cavs

from utilities.constants import SCENARIOS
from utilities.nod_marl.visualization import opinion_alpha_lines
from utilities.testing_rule_policy import TestingRulePolicy, assign_rule_vehicles
from utilities.testing_rule_coordinator import CONTROLLER_VERSION

path = "outputs/dgppo_nod_opinion_gain2_finetune/"  # Match the current from-scratch training output.

# 比例模式：8辆车、比例0.5时分配4辆规则车；车辆1留给Actor观察NOD。
# 设为None则使用下方的手动映射；0表示全部Actor，1需要把保留索引设为None。
rule_fraction = 0.5
rule_profile_weights = {"yielding": 0.25, "moderate": 0.5, "non_yielding": 0.25}
rule_assignment_seed = 123  # 仅用于控制类型分配，不改变环境随机种子。
reserved_actor_index = 0  # 0-based；设为None可选择所有车辆。
manual_rule_vehicles = {1: "moderate"}  # 仅在rule_fraction=None时使用。
rule_cruise_speed = 1.0  # m/s，所有类型共用，转弯时适度减速。

try:
    path_to_json_file = next(
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")
    )  # Find the first json file in the folder
    # Load parameters from the saved json file
    with open(path_to_json_file, "r") as file:
        data = json.load(file)
        saved_data = SaveData.from_dict(data)
        parameters = saved_data.parameters
        # Moved checkpoint folders must not load models from the old JSON path.
        parameters.where_to_save = os.path.join(path, "")

        # Adjust parameters
        # Safety-only rollout, including when loading older training JSON files.
        parameters.is_using_deadlock_critic = False
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
            "intersection_2"
            # "roundabout_1"
            # "CPM_entire"
            # "CPM_mixed"
            # "on_ramp_1"
            # roundabout_1, intersection_1/2/3, CPM_mixed
        )
        parameters.n_agents = SCENARIOS[parameters.scenario_type]["n_agents"]
        rule_vehicles = (assign_rule_vehicles(
            parameters.n_agents, rule_fraction, rule_profile_weights,
            seed=rule_assignment_seed, actor_index=reserved_actor_index)
            if rule_fraction is not None else dict(manual_rule_vehicles))
        if rule_fraction is None:
            test_output = os.path.join(path, "rule_vehicle_visualization", CONTROLLER_VERSION) if rule_vehicles else path
        else:
            mix = "_".join(f"{name}_{rule_profile_weights[name]:g}" for name in rule_profile_weights)
            run_name = (f"{parameters.scenario_type}_fraction_{rule_fraction:g}_{mix}"
                        f"_seed{rule_assignment_seed}_actor{reserved_actor_index}_{CONTROLLER_VERSION}").replace(".", "p")
            test_output = os.path.join(path, "rule_vehicle_visualization", run_name)
        displayed_roles = {i + 1: profile for i, profile in rule_vehicles.items()}
        print(f"[Rule vehicles] {len(rule_vehicles)}/{parameters.n_agents} "
              f"({len(rule_vehicles) / parameters.n_agents:.1%}): "
              f"{displayed_roles}")
        print(f"[Test output] {os.path.abspath(test_output)}")

        parameters.is_save_simulation_video = True
        parameters.is_visualize_short_term_path = False
        parameters.is_visualize_lane_boundary = False
        parameters.is_visualize_extra_info = True
        parameters.is_visualize_observed_neighbors = False
        # 固定展示邻居的智能体索引（0-based）。可按需修改。
        parameters.visualize_observed_neighbors_agent_index = 0
        # 显示该车对可见邻居的 z 和训练规则 alpha；0 对应画面中的车辆 1。
        parameters.is_visualize_nod_alpha = True
        parameters.is_visualize_agent_id = True
        # 关闭“未来三个位置点”可视化
        parameters.is_visualize_future_three_points = False
        parameters.is_visualize_agent_trajectory = True
        parameters.agent_trajectory_len = 25
        # 放慢测试渲染速度，便于观察（倍数：>1 越慢）。
        parameters.render_pause_scale = 1.0
        parameters.is_print_agent_speed = True
        parameters.print_speed_interval = 1
        parameters.is_save_agent_speed = True
        parameters.agent_speed_log_path = os.path.join(test_output, "agent_speeds.csv")
        parameters.agent_speed_log_interval = 1
        parameters.dgppo_alpha_gain = 2.0
        env, policy, priority_module, parameters = mappo_cavs(parameters=parameters)
        if rule_vehicles:
            policy = TestingRulePolicy(policy, env.scenario, rule_vehicles,
                                       cruise_speed=rule_cruise_speed)

        os.makedirs(test_output, exist_ok=True)
        if rule_fraction is not None or rule_vehicles:
            with open(os.path.join(test_output, "rule_setup.json"), "w") as setup_file:
                json.dump(dict(scenario=parameters.scenario_type, vehicles=rule_vehicles,
                               mode="fraction" if rule_fraction is not None else "manual",
                               requested_fraction=rule_fraction,
                               actual_fraction=len(rule_vehicles) / parameters.n_agents,
                               profile_weights=rule_profile_weights if rule_fraction is not None else None,
                               profile_counts={name: list(rule_vehicles.values()).count(name)
                                               for name in rule_profile_weights} if rule_fraction is not None else None,
                               assignment_seed=rule_assignment_seed if rule_fraction is not None else None,
                               reserved_actor_index=reserved_actor_index if rule_fraction is not None else None,
                               controller_version=CONTROLLER_VERSION,
                               cruise_speed=rule_cruise_speed, seed=parameters.seed,
                               checkpoint="final" if parameters.is_load_final_model else parameters.model_name),
                          setup_file, indent=2)
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

        def render_and_log(env, td):
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

            if parameters.is_visualize_nod_alpha:
                # Nonstop rollout supplies the transition root (pre-action
                # context); the rendered world has already advanced one step.
                env.scenario.nod_alpha_overlay = {
                    0: opinion_alpha_lines(
                        td, getattr(env.scenario, "safety_value_manager", None),
                        agent_index=int(parameters.visualize_observed_neighbors_agent_index),
                        env_index=0,
                        decision_time=(max(0, step_val - 1) * parameters.dt
                                       if step_val is not None else None))
                }
            if rule_vehicles:
                env.scenario.testing_rule_overlay = {0: policy.overlay_lines(0)}
            return env.render(mode="rgb_array", visualize_when_rgb=True)

        try:
            rollout_result = env.rollout(
                max_steps=parameters.max_steps - 1,
                policy=policy,
                priority_module=priority_module,
                callback=render_and_log,
                auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=parameters.is_save_simulation_video,
            )
        finally:
            if speed_log_f is not None:
                speed_log_f.close()

        # 兼容返回值：部分版本仅返回 out_td
        if isinstance(rollout_result, tuple) and len(rollout_result) == 2:
            out_td, frame_list = rollout_result
        else:
            out_td = rollout_result
            frame_list = []
        if rule_vehicles:
            policy.save_diagnostics(out_td, os.path.join(test_output, 'rule_diagnostics.csv'))
        if len(frame_list) > 0:
            save_video(os.path.join(test_output, "video"), frame_list, fps=1 / parameters.dt)
except StopIteration:
    raise FileNotFoundError("No json file found.")
