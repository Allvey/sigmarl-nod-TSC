# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Add project root to system path
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from datetime import datetime
import torch
import numpy as np
from termcolor import colored, cprint

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib

# Use Type 1 fonts (vector fonts) for IEEE paper submission
matplotlib.rcParams["pdf.fonttype"] = 42  # Use Type 42 (TrueType) fonts

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]

from vmas.simulator.utils import save_video

# Scientific plotting
import scienceplots  # Do not remove (https://github.com/garrettj403/SciencePlots)

plt.rcParams.update(
    {"figure.dpi": "100"}
)  # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(
    ["science", "ieee"]
)  # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

import time
import json

from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import (
    SaveData,
    find_the_highest_reward_among_all_models,
    get_model_name,
)

from utilities.constants import SCENARIOS, AGENTS

from utilities.colors import Color, colors


class TimedPolicy:
    """
    Wrap a policy callable and measure only its forward-call wall time.
    Environment stepping, rendering, logging, and metric computation are excluded.
    """

    def __init__(self, policy):
        self.policy = policy
        self.total_time_s = 0.0
        self.num_calls = 0
        self.latencies_s = []
        self._sync_cuda = self._uses_cuda(policy)

    @staticmethod
    def _uses_cuda(policy):
        try:
            return any(param.is_cuda for param in policy.parameters())
        except AttributeError:
            return torch.cuda.is_available()

    def __call__(self, tensordict):
        if self._sync_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        out = self.policy(tensordict)

        if self._sync_cuda:
            torch.cuda.synchronize()

        latency_s = time.perf_counter() - start
        self.total_time_s += latency_s
        self.latencies_s.append(latency_s)
        self.num_calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self.policy, name)


class Evaluation:
    """
    A class for handling the evaluation of simulation outputs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Evaluation object with model paths and titles for rendering.

        Args:
        model_paths (list): List of paths to the model directories.
        where_to_save_eva_results (str): Where to save evaluation results.
        render_titles (list): Titles for rendering the plots.
        video_names (list): Video names.
        """
        self.scenario_type = kwargs.pop("scenario_type", None)

        self.agent_width = AGENTS["width"]

        self.max_speed = AGENTS["max_speed_achievable"]  # Maximum achievable speed

        self.model_paths = kwargs.pop("model_paths", None)

        self.where_to_save_eva_results = kwargs.pop(
            "where_to_save_eva_results", "outputs"
        )

        self.is_show_different_collisions = kwargs.pop(
            "is_show_different_collisions", True
        )

        self.collision_efficiency_penalty = kwargs.pop(
            "collision_efficiency_penalty", 2.0
        )
        self.collision_smoothness_penalty = kwargs.pop(
            "collision_smoothness_penalty", 1.0
        )

        self.x_tick_label_rotation = kwargs.pop("x_tick_label_rotation", 0)

        self.fig_sizes = kwargs.pop(
            "fig_sizes",
            {
                "episode_reward": (3.8, 4.3),
                "collision_rate": (3.5, 2.0),
                "centerline_deviation": (3.5, 2.0),
                "average_speed": (3.5, 2.0),
                "smoothness": (3.5, 2.0),
            },
        )

        self.y_limits = kwargs.pop(
            "y_limits",
            {
                "episode_reward": [-2, 10],
                "collision_rate": [0, 100],
                "centerline_deviation": [0, 100],
                "average_speed": [0, 100],
                "smoothness": [0, 100],
            },
        )

        self.smoothness_beta = kwargs.pop("smoothness_beta", 0.5)
        self.u_lon_max = kwargs.pop("u_lon_max", 1.0)
        self.u_steer_max = kwargs.pop("u_steer_max", 1.0)

        self.where_to_save_logging = kwargs.pop("where_to_save_logging", "log.txt")

        self.fitst_model_index = kwargs.pop("fitst_model_index", 1)

        self.legends = kwargs.pop("legends", None)
        self.render_titles = kwargs.pop("render_titles", None)
        self.num_simulations_per_model = kwargs.pop("num_simulations_per_model", 32)
        self.num_agents = kwargs.pop("num_agents", 15)
        self.simulation_steps = kwargs.pop("simulation_steps", 15)
        self.is_measure_policy_inference_time = kwargs.pop(
            "is_measure_policy_inference_time", False
        )

        self.is_render = kwargs.pop("is_render", True)
        self.is_save_simulation_video = kwargs.pop("is_save_simulation_video", False)
        self.video_names = kwargs.pop("video_names", None)

        self.idx_our = next(
            (i for i, path in enumerate(self.model_paths) if "our" in path), -1
        )  # Index of our model

        if self.idx_our in {-1, None}:
            self.idx_our = kwargs.pop("idx_our", 0)

        print(f"Index of our model: {self.idx_our}")

        self.parameters = None  # This will be set when loading model parameters

        self.saved_data = None

        self.num_models = len(self.model_paths)

        self.x_ticks = kwargs.pop(
            "x_ticks", [f"$M_{{{idx}}}$" for idx in range(0, self.num_models)]
        )

        self.labels = [m.split("/")[-2] for m in self.model_paths]

    def _load_parameters(self):
        """
        Loads parameters from a JSON file located in the model path.
        """
        try:
            path_to_json_file = next(
                os.path.join(self.model_i_path, file)
                for file in os.listdir(self.model_i_path)
                if file.endswith(".json")
            )  # Find the first json file in the folder
            # Load parameters from the saved json file
            with open(path_to_json_file, "r") as file:
                data = json.load(file)
                self.saved_data = SaveData.from_dict(data)
                self.parameters = self.saved_data.parameters
        except StopIteration:
            raise FileNotFoundError("No json file found.")

    def _adjust_parameters(self):
        # Adjust parameters
        self.parameters.scenario_type = self.scenario_type
        self.parameters.is_testing_mode = True
        self.parameters.is_real_time_rendering = False
        self.parameters.is_save_eval_results = True
        self.parameters.is_load_model = True
        self.parameters.is_load_final_model = False
        self.parameters.is_load_out_td = True
        self.parameters.n_agents = self.num_agents
        self.parameters.max_steps = self.simulation_steps

        self.parameters.is_save_simulation_video = self.is_save_simulation_video

        if self.is_render or self.parameters.is_save_simulation_video:
            # Only one env is needed if video should be shown or saved
            self.parameters.num_vmas_envs = 1
            self.parameters.is_load_out_td = False
        else:
            # Otherwise run multiple parallel envs and average the evaluation results
            self.parameters.num_vmas_envs = self.num_simulations_per_model
        self.parameters.frames_per_batch = (
            self.parameters.max_steps * self.parameters.num_vmas_envs
        )

        # The two parameters below are only used in training. Set them to False so that they do not effect testing
        self.parameters.is_prb = False
        self.parameters.is_challenging_initial_state_buffer = False

        self.parameters.is_visualize_short_term_path = False
        self.parameters.is_visualize_lane_boundary = False
        self.parameters.is_visualize_extra_info = False
        self.parameters.render_title = self.render_titles[self.model_i]

    def _evaluate_model_i(self):
        """
        Evaluate a model.
        """
        out_td = self._get_simulation_outputs()

        positions = out_td["agents", "info", "pos"]
        velocities = out_td["agents", "info", "vel"]
        is_collision_with_agents = out_td[
            "agents", "info", "is_collision_with_agents"
        ].bool()
        is_collision_with_lanelets = out_td[
            "agents", "info", "is_collision_with_lanelets"
        ].bool()
        distance_ref = out_td["agents", "info", "distance_ref"]

        is_collide = is_collision_with_agents | is_collision_with_lanelets

        num_steps = positions.shape[1]

        v_norm = velocities.norm(dim=-1)

        ever_collide = is_collide.squeeze(-1).any(dim=1)
        safe_mask = ~ever_collide

        safe_counts_agents = safe_mask.sum(dim=-1)
        has_safe = safe_counts_agents > 0

        uncond_mean_speed = v_norm.mean(dim=(1, 2))

        safe_sum = (v_norm * safe_mask.unsqueeze(1)).sum(dim=(1, 2))
        safe_denom = safe_counts_agents * num_steps
        safe_mean_speed = safe_sum / torch.clamp_min(safe_denom, 1)

        mean_speed = torch.where(has_safe, safe_mean_speed, uncond_mean_speed)

        self.average_speed[self.model_idx, :] = mean_speed
        self.collision_rate_with_agents[self.model_idx, :] = (
            is_collision_with_agents.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        self.collision_rate_with_lanelets[self.model_idx, :] = (
            is_collision_with_lanelets.squeeze(-1).any(dim=-1).sum(dim=-1) / num_steps
        )
        self.distance_ref_average[self.model_idx, :] = distance_ref.squeeze(-1).mean(
            dim=(-2, -1)
        )

        actions = out_td["agents", "action"]
        u_lon = actions[..., 0]
        u_steer = actions[..., 1]
        du_lon = u_lon[:, 1:, :] - u_lon[:, :-1, :]
        du_steer = u_steer[:, 1:, :] - u_steer[:, :-1, :]
        du_lon_norm = (du_lon.abs() / self.u_lon_max).clamp(0.0, 1.0)
        du_steer_norm = (du_steer.abs() / self.u_steer_max).clamp(0.0, 1.0)
        beta = self.smoothness_beta
        jerk = beta * du_lon_norm + (1.0 - beta) * du_steer_norm
        jerk_mean = jerk.mean(dim=(1, 2))
        jerk_lon_mean = du_lon_norm.mean(dim=(1, 2))
        jerk_steer_mean = du_steer_norm.mean(dim=(1, 2))
        smoothness_score = jerk_mean.clamp(0.0, 1.0) * 100.0
        smoothness_lon_score = jerk_lon_mean.clamp(0.0, 1.0) * 100.0
        smoothness_lat_score = jerk_steer_mean.clamp(0.0, 1.0) * 100.0
        self.smoothness[self.model_idx, :] = smoothness_score
        self.smoothness_lon[self.model_idx, :] = smoothness_lon_score
        self.smoothness_lat[self.model_idx, :] = smoothness_lat_score

    def _get_simulation_outputs(self):
        # Load the model with the highest reward
        self.parameters.episode_reward_mean_current = (
            find_the_highest_reward_among_all_models(self.model_i_path)
        )
        self.parameters.model_name = get_model_name(parameters=self.parameters)
        path_eval_out_td = (
            self.parameters.where_to_save
            + self.parameters.model_name
            + f"_out_td_{self.parameters.scenario_type}.pth"
        )
        # Load simulation outputs if exist; otherwise run simulation.
        # Measuring inference time requires a fresh rollout, because cached
        # TensorDicts do not execute the policy.
        should_load_cached_out_td = (
            os.path.exists(path_eval_out_td)
            and (not self.is_render)
            and (not self.is_measure_policy_inference_time)
        )
        if should_load_cached_out_td:
            cprint("[INFO] Simulation outputs exist and will be loaded.", "grey")
            out_td = torch.load(path_eval_out_td)
        else:
            env, policy, priority_module, _ = mappo_cavs(parameters=self.parameters)
            rollout_policy = (
                TimedPolicy(policy)
                if self.is_measure_policy_inference_time
                else policy
            )

            cprint("[INFO] Run simulation...", "grey")
            sim_begin = time.time()

            if self.parameters.is_save_simulation_video:
                with torch.no_grad():
                    out_td, frame_list = env.rollout(
                        max_steps=self.parameters.max_steps - 1,
                        policy=rollout_policy,
                        priority_module=priority_module,
                        callback=lambda env, _: env.render(
                            mode="rgb_array", visualize_when_rgb=False
                        ),  # mode \in {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=self.parameters.is_save_simulation_video,
                    )
                sim_end = time.time() - sim_begin

                save_video(
                    f"{self.model_i_path}{self.video_names[self.model_i]}_{self.scenario_type}",
                    frame_list,
                    fps=1 / self.parameters.dt,
                )
                print(
                    colored(f"[INFO] Video saved under ", "grey"),
                    colored(f"{self.model_i_path}.", "blue"),
                )
            else:
                with torch.no_grad():
                    out_td = env.rollout(
                        max_steps=self.parameters.max_steps - 1,
                        policy=rollout_policy,
                        priority_module=priority_module,
                        callback=(lambda env, _: env.render(mode="human"))
                        if self.parameters.num_vmas_envs == 1
                        else None,  # mode should be one of {"human", "rgb_array"}
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        is_save_simulation_video=False,
                    )
                sim_end = time.time() - sim_begin

            if self.is_measure_policy_inference_time:
                self._record_policy_inference_time(rollout_policy)

            # Save simulation outputs
            torch.save(out_td, path_eval_out_td)
            print(colored("[INFO] Simulation outputs saved.", "grey"))

            # print(
            #     colored(
            #         f"[INFO] Total execution time for {self.parameters.num_vmas_envs} simulations (each has {self.parameters.max_steps} steps): {sim_end:.3f} sec.",
            #         "blue",
            #     )
            # )
            # print(
            #     colored(
            #         f"[INFO] One-step execution time {(sim_end / self.parameters.num_vmas_envs / self.parameters.max_steps):.4f} sec.",
            #         "blue",
            #     )
            # )

        return out_td

    def _record_policy_inference_time(self, timed_policy):
        total_s = float(timed_policy.total_time_s)
        num_calls = int(timed_policy.num_calls)
        num_agent_calls = max(
            1, num_calls * self.parameters.num_vmas_envs * self.num_agents
        )
        latency_ms = np.asarray(timed_policy.latencies_s, dtype=float) * 1000.0
        if latency_ms.size > 0:
            ms_per_call = float(latency_ms.mean())
            ms_per_call_std = float(latency_ms.std(ddof=0))
            ms_per_call_p95 = float(np.percentile(latency_ms, 95))
        else:
            ms_per_call = float("nan")
            ms_per_call_std = float("nan")
            ms_per_call_p95 = float("nan")
        us_per_agent_call = total_s / num_agent_calls * 1_000_000.0

        self.policy_inference_total_s[self.model_idx] = total_s
        self.policy_inference_num_calls[self.model_idx] = float(num_calls)
        self.policy_inference_ms_per_call[self.model_idx] = ms_per_call
        self.policy_inference_ms_per_call_std[self.model_idx] = ms_per_call_std
        self.policy_inference_ms_per_call_p95[self.model_idx] = ms_per_call_p95
        self.policy_inference_us_per_agent_call[self.model_idx] = us_per_agent_call

        cprint(
            "[TIME] Policy inference only: "
            f"total={total_s:.6f}s, calls={num_calls}, "
            f"ms/call mean={ms_per_call:.4f}, "
            f"std={ms_per_call_std:.4f}, "
            f"p95={ms_per_call_p95:.4f}, "
            f"us/agent-call={us_per_agent_call:.4f}",
            "blue",
        )

    @staticmethod
    def smooth_data(data, window_size=5):
        """
        Smooths the data using a simple moving average.

        Args:
            data: The input data to smooth.
            window_size (int): The size of the smoothing window.

        Returns:
            numpy.ndarray: The smoothed data.
        """
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
        return np.concatenate((data[: window_size - 1], smoothed))

    def plot(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Check if the directory exists
        if not os.path.exists(self.where_to_save_eva_results):
            os.makedirs(self.where_to_save_eva_results)
            print(f"[INFO] Directory '{self.where_to_save_eva_results}' was created.")
        else:
            print(
                f"[INFO] Directory '{self.where_to_save_eva_results}' already exists."
            )
        ###############################
        ## Fig 1 - Episode reward
        ###############################
        data_np = self.episode_reward.numpy()
        plt.clf()
        plt.figure(figsize=self.fig_sizes["episode_reward"])

        for i in range(data_np.shape[0]):
            # Original data with transparency
            plt.plot(
                data_np[i, :], color=colors[i], alpha=0.2, linestyle="-", linewidth=0.2
            )

            # Smoothed data
            smoothed_reward = self.smooth_data(data_np[i, :])
            plt.plot(
                smoothed_reward,
                label=self.legends[i],
                color=colors[i],
                linestyle="-",
                linewidth=0.8,
            )

        plt.xlim([0, data_np.shape[1]])
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.legend(
            bbox_to_anchor=(0.5, 1.88),
            loc="upper center",
            fontsize=9,
            ncol=1,
            handleheight=0.7,
            handletextpad=0.5,
            labelspacing=0.3,
        )
        # plt.legend(loc='lower right', fontsize="small", ncol=4)
        # plt.legend(bbox_to_anchor=(1, 0.5), loc='center right', fontsize=fontsize)

        plt.ylim(
            self.y_limits["episode_reward"],
        )
        plt.tight_layout(
            rect=[0, 0, 1, 1]
        )  # Adjust the layout to make space for the legend on the right
        plt.subplots_adjust(top=0.4)  # Adjust top value to make space for the legend

        # Save figure
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_episode_reward.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_episode_reward.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

        plt.clf()

        ###############################
        ## Fig 2 - collision rate [%]
        ###############################
        if not self.is_show_different_collisions:
            self._boxplot(
                figsize=self.fig_sizes["collision_rate"],
                data=self.collision_rate_sum,
                y_limits=self.y_limits["collision_rate"],
                y_label=r"Collision rate [$\%$]",
                fig_name="collision_rate",
            )
        else:
            plt.clf()
            fig, ax = plt.subplots(figsize=self.fig_sizes["collision_rate"])
            data_np = self.collision_rate_sum.numpy()
            data_with_agents_np = self.collision_rate_with_agents.numpy()
            data_with_lanelets_np = self.collision_rate_with_lanelets.numpy()

            # Positions of the violin plots (adjust as needed to avoid overlap)
            positions = np.arange(1, self.num_models + 1)
            offset = 0.2  # Offset for positioning the violins side by side

            # Plot a horizontal line indicating the median of our model
            median_our = np.median(data_np[self.idx_our])
            max_our = np.max(data_np[self.idx_our])
            plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
            ax.text(
                self.idx_our + 1,
                max_our,
                f"Our: {median_our:.2f}$\%$",
                ha="center",
                va="bottom",
                fontsize="small",
                color="grey",
            )

            # Plot each dataset with different colors
            parts1 = ax.violinplot(
                dataset=data_np.T,
                positions=positions - offset,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )
            parts2 = ax.violinplot(
                dataset=data_with_agents_np.T,
                positions=positions,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )
            parts3 = ax.violinplot(
                dataset=data_with_lanelets_np.T,
                positions=positions + offset,
                showmeans=False,
                showmedians=True,
                widths=0.2,
            )

            # Set colors for each violin plot
            self.custom_violinplot_color(parts1, Color.red100, Color.black100, 0.5)
            self.custom_violinplot_color(parts2, Color.blue100, Color.black100, 0.15)
            self.custom_violinplot_color(parts3, Color.green100, Color.black100, 0.15)

            # Setting ticks and labels
            ax.set_xticks(np.arange(len(self.x_ticks)))
            ax.set_xticklabels(
                self.x_ticks,
                rotation=self.x_tick_label_rotation,
                ha="right",
                fontsize="small",
            )
            ax.set_ylim(self.y_limits["collision_rate"])
            ax.set_ylabel(r"Collision rate [$\%$]")
            # ax.set_ylim([0, 2.0]) # [%]
            ax.yaxis.set_major_formatter(
                ticker.FormatStrFormatter("%.2f")
            )  # Set y-axis tick labels to have two digits after the comma
            ax.xaxis.set_minor_locator(
                ticker.NullLocator()
            )  # Make minor x-ticks invisible
            ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

            # # Add text to shown the median of M1
            # median_M1 = np.median(data_np[0])
            # ax.text(positions[0], 0.2, f'Median\n(total):\n{median_M1:.2f}$\%$', ha='center', va='bottom', fontsize='small', color='grey')

            # Adding legend
            ax.legend(
                [parts1["bodies"][0], parts2["bodies"][0], parts3["bodies"][0]],
                ["Total", "Agent-agent", "Agent-boundary"],
                loc="upper right",
                fontsize=8,
            )

            plt.tight_layout()
            # Save figure
            if self.parameters.is_save_eval_results:
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_collision_rate.pdf"
                plt.savefig(
                    path_save_eval_fig,
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.01,
                    dpi=300,
                )
                path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_collision_rate.png"
                plt.savefig(
                    path_save_eval_fig,
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0.01,
                    dpi=600,
                )
                print(
                    colored(f"[INFO] A fig has been saved under", "black"),
                    colored(f"{path_save_eval_fig}", "blue"),
                )

        ###############################
        ## Fig 3 - relative centerline deviation [%]
        ###############################
        self._boxplot(
            figsize=self.fig_sizes["centerline_deviation"],
            data=self.distance_ref_average,
            y_limits=self.y_limits["centerline_deviation"],
            y_label=r"Relative centerline deviation [$\%$]",
            fig_name="deviation_average",
        )

        ###############################
        ## Fig 4 - relative average velocity [%]
        ###############################
        self._boxplot(
            figsize=self.fig_sizes["average_speed"],
            data=self.average_speed,
            y_limits=self.y_limits["average_speed"],
            y_label=r"Rel. avg. spd. [$\%$]",
            fig_name="average_speed",
        )

        self._boxplot(
            figsize=self.fig_sizes["smoothness"],
            data=self.smoothness,
            y_limits=self.y_limits["smoothness"],
            y_label=r"Smoothness [$\%$]",
            fig_name="smoothness",
        )

    def _boxplot(self, figsize, data, y_label, fig_name, y_limits: None):
        plt.clf()
        fig, ax = plt.subplots(figsize=figsize)

        data_np = data.numpy()

        # Plot a horizontal line indicating the median of our model
        median_our = np.median(data_np[self.idx_our])
        max_our = np.max(data_np[self.idx_our])
        plt.axhline(y=median_our, color="black", linestyle="--", alpha=0.3)
        ax.text(
            self.idx_our + 1,
            max_our,
            f"Our: {median_our:.2f}$\%$",
            ha="center",
            va="bottom",
            fontsize="small",
            color="grey",
        )

        ax.violinplot(dataset=data_np.T, showmeans=False, showmedians=True)
        ax.set_xticks(np.arange(len(self.x_ticks)))
        ax.set_xticklabels(
            self.x_ticks,
            rotation=self.x_tick_label_rotation,
            ha="right",
            fontsize="small",
        )
        ax.set_ylabel(y_label)

        if y_limits:
            ax.set_ylim(y_limits)

        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Make minor x-ticks invisible
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.1)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.1)

        # Save figure
        plt.tight_layout()
        if self.parameters.is_save_eval_results:
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_{fig_name}.pdf"
            plt.savefig(
                path_save_eval_fig,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=300,
            )
            path_save_eval_fig = f"{self.where_to_save_eva_results}/{self.timestamp}_{self.parameters.scenario_type}_{fig_name}.png"
            plt.savefig(
                path_save_eval_fig,
                format="png",
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=600,
            )
            print(
                colored(f"[INFO] A fig has been saved under", "black"),
                colored(f"{path_save_eval_fig}", "blue"),
            )

    def _init_eva_matrices(self):
        """
        Initialize evaluation matrices.
        """
        self.average_speed = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.collision_rate_with_agents = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.collision_rate_with_lanelets = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.distance_ref_average = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.smoothness = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.smoothness_lon = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.smoothness_lat = torch.zeros(
            (self.num_models, self.parameters.num_vmas_envs),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.episode_reward = torch.zeros(
            (self.num_models, self.parameters.n_iters),
            device=self.parameters.device,
            dtype=torch.float32,
        )
        self.policy_inference_total_s = torch.full(
            (self.num_models,), float("nan"), dtype=torch.float64
        )
        self.policy_inference_num_calls = torch.zeros(
            (self.num_models,), dtype=torch.float64
        )
        self.policy_inference_ms_per_call = torch.full(
            (self.num_models,), float("nan"), dtype=torch.float64
        )
        self.policy_inference_ms_per_call_std = torch.full(
            (self.num_models,), float("nan"), dtype=torch.float64
        )
        self.policy_inference_ms_per_call_p95 = torch.full(
            (self.num_models,), float("nan"), dtype=torch.float64
        )
        self.policy_inference_us_per_agent_call = torch.full(
            (self.num_models,), float("nan"), dtype=torch.float64
        )

    def run_evaluation(self):
        """
        Main method to run evaluation over all provided model paths.
        """
        self.model_idx = 0

        for model_i in range(self.num_models):
            print("------------------------------------------")
            print(
                colored("-- [INFO] Model ", "black"),
                colored(f"{model_i + 1}", "blue"),
                colored(f"({self.legends[model_i]})", color="grey"),
            )
            print("------------------------------------------")

            self.model_i = model_i
            self.model_i_path = self.model_paths[model_i]

            self._load_parameters()
            self._adjust_parameters()

            if self.model_idx == 0:
                self._init_eva_matrices()  # Only need to be done once

            self.episode_reward[self.model_idx, :] = torch.tensor(
                [self.saved_data.episode_reward_mean_list]
            )

            self._evaluate_model_i()

            self.model_idx += 1

        if not self.parameters.is_save_simulation_video:
            self.compute_performance_metrics()
            self.plot()

    def compute_performance_metrics(
        self, is_remove_max_min: bool = False, is_relative_values: bool = True
    ):
        """
        This function computes the three performance metrices related to safety, lane following, and traffic efficienty.
        """
        if is_relative_values:
            # Use relative values
            self.distance_ref_average = (
                self.distance_ref_average / self.agent_width * 100
            )
            self.average_speed = self.average_speed / self.max_speed * 100

        self.collision_rate_with_agents = self.collision_rate_with_agents * 100
        self.collision_rate_with_lanelets = self.collision_rate_with_lanelets * 100

        self.collision_rate_sum = (
            self.collision_rate_with_agents[:] + self.collision_rate_with_lanelets[:]
        )

        # Collision rate [%]
        self.CR_AA_avg = self.collision_rate_with_agents.mean(dim=-1)
        self.CR_AL_avg = self.collision_rate_with_lanelets.mean(dim=-1)
        self.CR_total_avg = self.collision_rate_sum.mean(dim=-1)

        # Relative centerline deviation [%]
        self.CD_avg = self.distance_ref_average.mean(dim=-1)

        # Relative average speed [%]
        self.AS_avg = self.average_speed.mean(dim=-1)

        self.SM_avg = self.smoothness.mean(dim=-1)
        self.SM_lon_avg = self.smoothness_lon.mean(dim=-1)
        self.SM_lat_avg = self.smoothness_lat.mean(dim=-1)

        penalty_eff = getattr(self, "collision_efficiency_penalty", 0.0)
        if penalty_eff != 0.0:
            penalty_tensor_eff = torch.zeros_like(self.average_speed)
            has_collision_env = self.collision_rate_sum > 0
            penalty_tensor_eff[has_collision_env] = (
                penalty_eff * self.collision_rate_sum[has_collision_env]
            )
            self.average_speed_penalized = self.average_speed - penalty_tensor_eff
            self.AS_penalized_avg = self.average_speed_penalized.mean(dim=-1)
        else:
            self.average_speed_penalized = self.average_speed
            self.AS_penalized_avg = self.AS_avg

        penalty_sm = getattr(self, "collision_smoothness_penalty", 0.0)
        beta = self.smoothness_beta
        if penalty_sm != 0.0:
            penalty_tensor_sm = torch.zeros_like(self.smoothness)
            has_collision_env = self.collision_rate_sum > 0
            penalty_tensor_sm[has_collision_env] = (
                penalty_sm * self.collision_rate_sum[has_collision_env]
            )
            self.smoothness_lon_penalized = self.smoothness_lon + penalty_tensor_sm
            self.smoothness_lat_penalized = self.smoothness_lat + penalty_tensor_sm
            self.SM_lon_penalized_avg = self.smoothness_lon_penalized.mean(dim=-1)
            self.SM_lat_penalized_avg = self.smoothness_lat_penalized.mean(dim=-1)
        else:
            self.smoothness_lon_penalized = self.smoothness_lon
            self.smoothness_lat_penalized = self.smoothness_lat
            self.SM_lon_penalized_avg = self.SM_lon_avg
            self.SM_lat_penalized_avg = self.SM_lat_avg

        self.smoothness_penalized = (
            beta * self.smoothness_lon_penalized
            + (1.0 - beta) * self.smoothness_lat_penalized
        )
        self.SM_penalized_avg = self.smoothness_penalized.mean(dim=-1)

        # Compute composite score
        w1 = 1 / self.CR_total_avg.mean()
        w2 = 1 / self.CD_avg.mean()
        w3 = 1 / self.AS_penalized_avg.mean()
        # composite score = - w1 * CR_total_avg - w2 * CD_avg + w3 * AS_avg
        self.composite_score = (
            -w1 * self.CR_total_avg - w2 * self.CD_avg + w3 * self.AS_penalized_avg
        )

        print(f"composite_score: {self.composite_score}")

        self._log()

    def _log(self):
        CR_AA_std = self.collision_rate_with_agents.std(dim=-1, unbiased=False)
        CR_AA_min = self.collision_rate_with_agents.min(dim=-1).values
        CR_AA_max = self.collision_rate_with_agents.max(dim=-1).values
        CR_AL_std = self.collision_rate_with_lanelets.std(dim=-1, unbiased=False)
        CR_AL_min = self.collision_rate_with_lanelets.min(dim=-1).values
        CR_AL_max = self.collision_rate_with_lanelets.max(dim=-1).values
        CR_total_std = self.collision_rate_sum.std(dim=-1, unbiased=False)
        CR_total_min = self.collision_rate_sum.min(dim=-1).values
        CR_total_max = self.collision_rate_sum.max(dim=-1).values

        CD_std = self.distance_ref_average.std(dim=-1, unbiased=False)
        CD_min = self.distance_ref_average.min(dim=-1).values
        CD_max = self.distance_ref_average.max(dim=-1).values

        AS_std = self.average_speed.std(dim=-1, unbiased=False)
        AS_min = self.average_speed.min(dim=-1).values
        AS_max = self.average_speed.max(dim=-1).values
        AS_penalized_std = self.average_speed_penalized.std(dim=-1, unbiased=False)

        SM_std = self.smoothness.std(dim=-1, unbiased=False)
        SM_min = self.smoothness.min(dim=-1).values
        SM_max = self.smoothness.max(dim=-1).values
        SM_penalized_std = self.smoothness_penalized.std(dim=-1, unbiased=False)
        SM_lon_std = self.smoothness_lon.std(dim=-1, unbiased=False)
        SM_lon_min = self.smoothness_lon.min(dim=-1).values
        SM_lon_max = self.smoothness_lon.max(dim=-1).values
        SM_lon_penalized_std = self.smoothness_lon_penalized.std(
            dim=-1, unbiased=False
        )
        SM_lat_std = self.smoothness_lat.std(dim=-1, unbiased=False)
        SM_lat_min = self.smoothness_lat.min(dim=-1).values
        SM_lat_max = self.smoothness_lat.max(dim=-1).values
        SM_lat_penalized_std = self.smoothness_lat_penalized.std(
            dim=-1, unbiased=False
        )

        log_CR_AA = (
            "[LOG] Agent-agent collision rate [%]: "
            f"mean={self.format_array(self.CR_AA_avg.numpy())}, "
            f"std={self.format_array(CR_AA_std.numpy())}, "
            f"min={self.format_array(CR_AA_min.numpy())}, "
            f"max={self.format_array(CR_AA_max.numpy())}"
        )
        log_CR_AL = (
            "[LOG] Agent-lanelet collision rate [%]: "
            f"mean={self.format_array(self.CR_AL_avg.numpy())}, "
            f"std={self.format_array(CR_AL_std.numpy())}, "
            f"min={self.format_array(CR_AL_min.numpy())}, "
            f"max={self.format_array(CR_AL_max.numpy())}"
        )
        log_CR_total = (
            "[LOG] Total collision rate [%]: "
            f"mean={self.format_array(self.CR_total_avg.numpy())}, "
            f"std={self.format_array(CR_total_std.numpy())}, "
            f"min={self.format_array(CR_total_min.numpy())}, "
            f"max={self.format_array(CR_total_max.numpy())}"
        )
        log_CD = (
            "[LOG] Relative centerline deviation [%]: "
            f"mean={self.format_array(self.CD_avg.numpy())}, "
            f"std={self.format_array(CD_std.numpy())}, "
            f"min={self.format_array(CD_min.numpy())}, "
            f"max={self.format_array(CD_max.numpy())}"
        )
        log_AS = (
            "[LOG] Relative average speed [%]: "
            f"mean={self.format_array(self.AS_avg.numpy())}, "
            f"std={self.format_array(AS_std.numpy())}, "
            f"min={self.format_array(AS_min.numpy())}, "
            f"max={self.format_array(AS_max.numpy())}"
        )
        log_AS_penalized = (
            "[LOG] Relative average speed with collision penalty [%]: "
            f"mean={self.format_array(self.AS_penalized_avg.numpy())}, "
            f"std={self.format_array(AS_penalized_std.numpy())}"
        )
        log_SM = (
            "[LOG] Smoothness [%]: "
            f"mean={self.format_array(self.SM_avg.numpy())}, "
            f"std={self.format_array(SM_std.numpy())}, "
            f"min={self.format_array(SM_min.numpy())}, "
            f"max={self.format_array(SM_max.numpy())}"
        )

        log_SM_penalized = (
            "[LOG] Smoothness with collision penalty [%]: "
            f"mean={self.format_array(self.SM_penalized_avg.numpy())}, "
            f"std={self.format_array(SM_penalized_std.numpy())}"
        )

        log_SM_lon = (
            "[LOG] Smoothness longitudinal [%]: "
            f"mean={self.format_array(self.SM_lon_avg.numpy())}, "
            f"std={self.format_array(SM_lon_std.numpy())}, "
            f"min={self.format_array(SM_lon_min.numpy())}, "
            f"max={self.format_array(SM_lon_max.numpy())}"
        )

        log_SM_lon_penalized = (
            "[LOG] Smoothness longitudinal with collision penalty [%]: "
            f"mean={self.format_array(self.SM_lon_penalized_avg.numpy())}, "
            f"std={self.format_array(SM_lon_penalized_std.numpy())}"
        )

        log_SM_lat = (
            "[LOG] Smoothness lateral [%]: "
            f"mean={self.format_array(self.SM_lat_avg.numpy())}, "
            f"std={self.format_array(SM_lat_std.numpy())}, "
            f"min={self.format_array(SM_lat_min.numpy())}, "
            f"max={self.format_array(SM_lat_max.numpy())}"
        )

        log_SM_lat_penalized = (
            "[LOG] Smoothness lateral with collision penalty [%]: "
            f"mean={self.format_array(self.SM_lat_penalized_avg.numpy())}, "
            f"std={self.format_array(SM_lat_penalized_std.numpy())}"
        )

        log_CS = (
            f"[LOG] Composite scores: {self.format_array(self.composite_score.numpy())}"
        )
        if hasattr(self, "policy_inference_total_s") and torch.isfinite(
            self.policy_inference_total_s
        ).any():
            log_policy_time = (
                "[LOG] Policy inference time only: "
                f"total_s={self.format_array_precision(self.policy_inference_total_s.numpy(), 6)}, "
                f"calls={self.format_array_precision(self.policy_inference_num_calls.numpy(), 0)}, "
                f"ms_per_call_mean={self.format_array_precision(self.policy_inference_ms_per_call.numpy(), 4)}, "
                f"ms_per_call_std={self.format_array_precision(self.policy_inference_ms_per_call_std.numpy(), 4)}, "
                f"ms_per_call_p95={self.format_array_precision(self.policy_inference_ms_per_call_p95.numpy(), 4)}, "
                "us_per_agent_call="
                f"{self.format_array_precision(self.policy_inference_us_per_agent_call.numpy(), 4)}"
            )
        else:
            log_policy_time = (
                "[LOG] Policy inference time only: not measured "
                "(simulation outputs were loaded from cache or timing was disabled)"
            )

        cprint(log_CR_total, "black")
        cprint(log_CR_AA, "black")
        cprint(log_CR_AL, "black")
        cprint(log_CD, "black")
        cprint(log_AS, "black")
        cprint(log_AS_penalized, "black")
        cprint(log_SM, "black")
        cprint(log_SM_penalized, "black")
        cprint(log_SM_lon, "black")
        cprint(log_SM_lon_penalized, "black")
        cprint(log_SM_lat, "black")
        cprint(log_SM_lat_penalized, "black")
        cprint(log_policy_time, "black")
        cprint(log_CS)

        # Save the evaluation results to a txt file
        with open(self.where_to_save_logging, "a") as file:
            file.write(f"Scenario: {self.parameters.scenario_type}\n")
            file.write(f"{log_CR_AA}\n")
            file.write(f"{log_CR_AL}\n")
            file.write(f"{log_CR_total}\n")
            file.write(f"{log_CD}\n")
            file.write(f"{log_AS}\n")
            file.write(f"{log_AS_penalized}\n")
            file.write(f"{log_SM}\n")
            file.write(f"{log_SM_penalized}\n")
            file.write(f"{log_SM_lon}\n")
            file.write(f"{log_SM_lon_penalized}\n")
            file.write(f"{log_SM_lat}\n")
            file.write(f"{log_SM_lat_penalized}\n")
            file.write(f"{log_policy_time}\n")
            file.write(f"{log_CS}\n")
            file.write("=========================================")
            file.write("\n")

    @staticmethod
    def remove_max_min(tensor):
        """
        Remove the maximum and minimum values from each row of the tensor.

        Args:
            tensor: A 2D tensor with shape [a, b]

        Returns:
            A 2D tensor with the max and min values removed from each row.
        """
        # Find the indices of the max and min in each row
        max_vals, max_indices = torch.max(tensor, dim=1, keepdim=True)
        min_vals, min_indices = torch.min(tensor, dim=1, keepdim=True)

        # Deal with the case in which one whole row has the same value
        is_row_same_value = max_indices == min_indices
        is_row_different_values = max_indices != min_indices

        row_same_value, _ = torch.where(is_row_same_value)
        row_different_values, _ = torch.where(is_row_different_values)

        # Replace max and min values with inf and -inf
        tensor[row_same_value.unsqueeze(-1), 0] = float("inf")
        tensor[row_same_value.unsqueeze(-1), 1] = float("-inf")

        tensor[
            row_different_values.unsqueeze(-1),
            max_indices[is_row_different_values].unsqueeze(-1),
        ] = float("inf")
        tensor[
            row_different_values.unsqueeze(-1),
            min_indices[is_row_different_values].unsqueeze(-1),
        ] = float("-inf")

        # Remove the inf and -inf values
        mask = (tensor != float("inf")) & (tensor != float("-inf"))
        filtered_tensor = tensor[mask].view(tensor.size(0), -1)

        return filtered_tensor

    # Function to add custom median markers
    @staticmethod
    def custom_violinplot_color(parts, color_face, color_lines, alpha):
        for pc in parts["bodies"]:
            pc.set_facecolor(color_face)
            pc.set_edgecolor(color_face)
            pc.set_alpha(alpha)

        parts["cmedians"].set_colors(color_lines)
        parts["cmedians"].set_alpha(alpha)
        parts["cmaxes"].set_colors(color_lines)
        parts["cmaxes"].set_alpha(alpha)
        parts["cmins"].set_colors(color_lines)
        parts["cmins"].set_alpha(alpha)
        parts["cbars"].set_colors(color_lines)
        parts["cbars"].set_alpha(alpha)

    @staticmethod
    def format_array(arr):
        return ", ".join([f"{x:.2f}" for x in arr])

    @staticmethod
    def format_array_precision(arr, precision):
        return ", ".join([f"{x:.{precision}f}" for x in arr])

    @staticmethod
    def min_max_normalize(tensor: torch.Tensor, **kwargs):
        """
        This function min-max normalize the given tensor.
        """
        min_val = kwargs.pop("min", tensor.min())
        max_val = kwargs.pop("max", tensor.max())
        return (tensor - min_val) / (max_val - min_val)
