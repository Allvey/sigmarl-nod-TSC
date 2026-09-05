# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import random
from termcolor import colored, cprint

# Add project root to system path if you want to run this file directly
script_dir = os.path.dirname(__file__)  # Directory of the current script
project_root = os.path.dirname(script_dir)  # Project root directory
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.scenario import BaseScenario

# from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle

from utilities.kinematic_bicycle import KinematicBicycle
from utilities.colors import Color, colors
from utilities.topology_labels import generate_e_labels_with_corridor
from utilities.nod_marl.interaction import build_directed_interactions

from utilities.helper_training import Parameters

from utilities.helper_scenario import (
    Distances,
    Normalizers,
    Observations,
    Penalties,
    ReferencePathsAgentRelated,
    ReferencePathsMapRelated,
    Rewards,
    Thresholds,
    Collisions,
    Timer,
    Constants,
    CircularBuffer,
    StateBuffer,
    InitialStateBuffer,
    exponential_decreasing_fcn,
    get_distances_between_agents,
    get_perpendicular_distances,
    get_rectangle_vertices,
    get_short_term_reference_path,
    interX,
    angle_eliminate_two_pi,
    transform_from_global_to_local_coordinate,
)

from utilities.map_manager import MapManager

from utilities.constants import SCENARIOS, AGENTS


class ScenarioRoadTraffic(BaseScenario):
    """
    This scenario aims to design an MARL framework with information-dense observation design to enable fast training and to empower agents the ability to generalize to unseen scenarios.

    We propose five observation-design strategies. They correspond to five parameters in this file, and their default
    values are True.
        - is_ego_view: Whether to use ego view (otherwise bird view)
        - is_observe_distance_to_agents: Whether to observe the distance to other agents
        - is_observe_distance_to_boundaries: Whether to observe the distance to labelet boundaries (otherwise the points on lanelet boundaries)
        - is_observe_distance_to_center_line: Whether to observe the distance to reference path (otherwise None)
        - is_observe_vertices: Whether to observe the vertices of other agents (otherwise center points)

    In addition, there are some commonly used parameters you may want to adjust to suit your case:
        - n_agents: Number of agents
        - dt: Sample time in seconds
        - scenario_type: One of {"CPM_entire", "CPM_mixed", "intersection_1", ...}. See SCENARIOS in utilities/constants.py for more scenarios.
                         "CPM_entire": the entire CPM map will be used
                         "CPM_mixed": a specific part of the CPM map (intersection, merge-in, or merge-out) will be used for each env when making or resetting it. You can control the probability of using each of them by the parameter `scenario_probabilities`. It is an array with three values. The first value corresponds to the probability of using intersection. The second and the third values correspond to merge-in and merge-out, respectively. If you only want to use one specific part of the map for all parallel envs, you can set the other two values to zero. For example, if you want to train a RL policy only for intersection, they can set `scenario_probabilities` to [1.0, 0.0, 0.0].
                         "intersection_1": the intersection scenario with ID 1
        - is_partial_observation: Whether to enable partial observation (to model partially observable MDP)
        - n_nearing_agents_observed: Number of nearing agents to be observed (consider limited sensor range)

        is_testing_mode: Testing mode is designed to test the learned policy.
                         In non-testing mode, once a collision occurs, all agents will be reset with random initial states.
                         To ensure these initial states are feasible, the initial positions are conservatively large (1.2*diagonalLengthOfAgent).
                         This ensures agents are initially safe and avoids putting agents in an immediate dangerous situation at the beginning of a new scenario.
                         During testing, only colliding agents will be reset, without changing the states of other agents, who are possibly interacting with other agents.
                         This may allow for more effective testing.

    For other parameters, see the class Parameter defined in this file.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self._init_params(batch_dim, device, **kwargs)
        world = self._init_world(batch_dim, device)
        self._init_agents(world)
        return world

    def _init_params(self, batch_dim, device, **kwargs):
        """
        Initialize parameters.
        """
        scenario_type = kwargs.pop(
            "scenario_type", "CPM_mixed"
        )  # Scenario type. See all scenario types in SCENARIOS in utilities/constants
        self.n_agents = SCENARIOS[scenario_type]["n_agents"]  # Number of agents
        self.agent_width = AGENTS["width"]  # The width of the agent in [m]
        self.agent_length = AGENTS["length"]  # The length of the agent in [m]
        # Monotone identity generations prevent a reset agent from inheriting
        # the edge history of the previous vehicle occupying the same slot.
        self.nod_agent_generation = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.long
        )
        lane_width = SCENARIOS[scenario_type][
            "lane_width"
        ]  # The (rough) width of each lane in [m]

        # Reward
        r_p_normalizer = (
            100  # This parameter normalizes rewards and penalties to [-1, 1].
        )
        # This is useful for RL algorithms with an actor-critic architecture where the critic's
        # output is limited to [-1, 1] (e.g., due to tanh activation function).

        reward_progress = (
            kwargs.pop("reward_progress", 10) / r_p_normalizer
        )  # Reward for moving along reference paths
        reward_vel = (
            kwargs.pop("reward_vel", 5) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_reach_goal = (
            kwargs.pop("reward_reach_goal", 0) / r_p_normalizer
        )  # Goal-reaching reward

        # Penalty
        penalty_deviate_from_ref_path = kwargs.pop(
            "penalty_deviate_from_ref_path", -2 / r_p_normalizer
        )  # Penalty for deviating from reference paths
        penalty_near_boundary = kwargs.pop(
            "penalty_near_boundary", -20 / r_p_normalizer
        )  # Penalty for being too close to lanelet boundaries
        penalty_near_other_agents = kwargs.pop(
            "penalty_near_other_agents", -20 / r_p_normalizer
        )  # Penalty for being too close to other agents
        penalty_collide_with_agents = kwargs.pop(
            "penalty_collide_with_agents", -100 / r_p_normalizer
        )  # Penalty for colliding with other agents
        penalty_collide_with_boundaries = kwargs.pop(
            "penalty_collide_with_boundaries", -100 / r_p_normalizer
        )  # Penalty for colliding with lanelet boundaries
        penalty_change_steering = kwargs.pop(
            "penalty_change_steering", -2 / r_p_normalizer
        )  # Penalty for changing steering too quick
        penalty_time = kwargs.pop(
            "penalty_time", 5 / r_p_normalizer
        )  # Penalty for losing time

        threshold_deviate_from_ref_path = kwargs.pop(
            "threshold_deviate_from_ref_path", (lane_width - self.agent_width) / 2
        )  # Use for penalizing of deviating from reference path

        threshold_reach_goal = kwargs.pop(
            "threshold_reach_goal", self.agent_width / 2
        )  # Threshold less than which agents are considered at their goal positions

        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]

        threshold_near_boundary_high = kwargs.pop(
            "threshold_near_boundary_high", (lane_width - self.agent_width) / 2 * 0.9
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to lanelet boundaries
        threshold_near_boundary_low = kwargs.pop(
            "threshold_near_boundary_low", 0
        )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

        threshold_near_other_agents_c2c_high = kwargs.pop(
            "threshold_near_other_agents_c2c_high", self.agent_length + self.agent_width
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to other agents (for center-to-center distance)
        threshold_near_other_agents_c2c_low = kwargs.pop(
            "threshold_near_other_agents_c2c_low",
            (self.agent_length + self.agent_width) / 2,
        )  # Threshold above which agents will be penalized (for center-to-center distance,
        # If a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

        threshold_no_reward_if_too_close_to_boundaries = kwargs.pop(
            "threshold_no_reward_if_too_close_to_boundaries", self.agent_width / 10
        )
        threshold_no_reward_if_too_close_to_other_agents = kwargs.pop(
            "threshold_no_reward_if_too_close_to_other_agents", self.agent_width / 6
        )

        threshold_near_other_agents_MTV_low = kwargs.pop(
            "threshold_near_other_agents_MTV_low", 0
        )

        threshold_near_other_agents_MTV_high = kwargs.pop(
            "threshold_near_other_agents_MTV_high", self.agent_length
        )

        # Visualization
        self.resolution_factor = kwargs.pop("resolution_factor", 200)  # Default 200

        # Reference path
        sample_interval_ref_path = kwargs.pop(
            "sample_interval_ref_path", 2
        )  # Integer, sample interval from the long-term reference path for the short-term reference paths
        max_ref_path_points = kwargs.pop(
            "max_ref_path_points", 200
        )  # The estimated maximum points on the reference path

        noise_level = kwargs.pop(
            "noise_level", 0.2 * self.agent_width
        )  # Noise will be generated by the standary normal distribution. This parameter controls the noise level

        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 1
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

        # World dimensions
        self.world_x_dim = SCENARIOS[scenario_type]["world_x_dim"]
        self.world_y_dim = SCENARIOS[scenario_type]["world_y_dim"]

        self.render_origin = [
            self.world_x_dim / 2,
            self.world_y_dim / 2,
        ]

        self.viewer_size = (
            int(self.world_x_dim * self.resolution_factor),
            int(self.world_y_dim * self.resolution_factor),
        )
        self.viewer_zoom = SCENARIOS[scenario_type]["viewer_zoom"]

        self.max_steering_angle = kwargs.pop(
            "max_steering_angle",
            torch.deg2rad(
                torch.tensor(AGENTS["max_steering"], device=device, dtype=torch.float32)
            ),
        )  # Maximum allowed steering angle in degree
        self.max_speed = kwargs.pop(
            "max_speed", AGENTS["max_speed"]
        )  # Maximum allowed speed in [m/s]

        n_points_nearing_boundary = kwargs.pop(
            "n_points_nearing_boundary", 5
        )  # The number of points on nearing boundaries to be observed

        probability_record = kwargs.pop("probability_record", 1.0)
        probability_use_recording = kwargs.pop("probability_use_recording", 0.2)
        buffer_size = kwargs.pop("buffer_size", 100)

        if not hasattr(self, "parameters"):
            self.parameters = Parameters(
                n_agents=self.n_agents,
                scenario_type=scenario_type,
                is_partial_observation=kwargs.pop("is_partial_observation", True),
                is_testing_mode=kwargs.pop("is_testing_mode", False),
                is_visualize_short_term_path=kwargs.pop(
                    "is_visualize_short_term_path", True
                ),
                n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2),
                is_real_time_rendering=kwargs.pop("is_real_time_rendering", False),
                n_steps_stored=kwargs.pop("n_steps_stored", 10),
                n_points_short_term=kwargs.pop("n_points_short_term", 3),
                dt=kwargs.pop("dt", 0.05),
                is_ego_view=kwargs.pop("is_ego_view", True),
                is_observe_vertices=kwargs.pop("is_observe_vertices", True),
                is_observe_distance_to_agents=kwargs.pop(
                    "is_observe_distance_to_agents", True
                ),
                is_observe_distance_to_boundaries=kwargs.pop(
                    "is_observe_distance_to_boundaries", True
                ),
                is_observe_distance_to_center_line=kwargs.pop(
                    "is_observe_distance_to_center_line", True
                ),
                is_apply_mask=kwargs.pop("is_apply_mask", False),
                is_challenging_initial_state_buffer=kwargs.pop(
                    "is_challenging_initial_state_buffer", True
                ),
                cpm_scenario_probabilities=kwargs.pop(
                    "cpm_scenario_probabilities", [1.0, 0.0, 0.0]
                ),  # Probabilities of training agents in intersection, merge-in, and merge-out scenario
                is_add_noise=kwargs.pop("is_add_noise", True),
                is_observe_ref_path_other_agents=kwargs.pop(
                    "is_observe_ref_path_other_agents", False
                ),
                is_visualize_extra_info=kwargs.pop("is_visualize_extra_info", False),
                is_visualize_agent_id=kwargs.pop("is_visualize_agent_id", True),
                render_title=kwargs.pop(
                    "render_title",
                    "Multi-Agent Reinforcement Learning for Road Traffic (CPM Lab Scenario)",
                ),
                is_visualize_agent_trajectory=kwargs.pop(
                    "is_visualize_agent_trajectory", True
                ),
                agent_trajectory_len=kwargs.pop("agent_trajectory_len", 25),
                agent_trajectory_thickness_m=kwargs.pop(
                    "agent_trajectory_thickness_m", 0.06
                ),
                agent_trajectory_thickness_m_beautify=kwargs.pop(
                    "agent_trajectory_thickness_m_beautify", 4.0
                ),
                agent_trajectory_interp_points_per_segment=kwargs.pop(
                    "agent_trajectory_interp_points_per_segment", 4
                ),
                agent_trajectory_interp_use_catmull_rom=kwargs.pop(
                    "agent_trajectory_interp_use_catmull_rom", True
                ),
                is_using_opponent_modeling=kwargs.pop(
                    "is_using_opponent_modeling", False
                ),
                is_using_prioritized_marl=kwargs.pop(
                    "is_using_prioritized_marl", False
                ),
                is_visualize_observed_neighbors=kwargs.pop(
                    "is_visualize_observed_neighbors", False
                ),
            )

        # Logs
        if self.parameters.is_testing_mode:
            print(colored(f"[INFO] Testing mode", "red"))
        print(colored(f"[INFO] Scenario type: {self.parameters.scenario_type}", "red"))
        if self.parameters.is_prb:
            print(colored("[INFO] Enable prioritized replay buffer", "red"))
        if self.parameters.is_challenging_initial_state_buffer:
            print(colored("[INFO] Enable challenging initial state buffer", "red"))
        if self.parameters.is_using_opponent_modeling:
            print(colored("[INFO] Using opponent modeling", "red"))
        if self.parameters.is_using_prioritized_marl:
            if self.parameters.prioritization_method == "marl":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with MARL-generated priorities",
                        "red",
                    )
                )
            elif self.parameters.prioritization_method == "random":
                print(
                    colored(
                        "[INFO] Using prioritized MARL with random priorities", "red"
                    )
                )
            else:
                raise ValueError(
                    f"The given prioritization method is not supported. Obtained: {self.parameters.prioritization_method}. Expected: 'marl' or 'random'."
                )

        self.parameters.n_nearing_agents_observed = min(
            self.parameters.n_nearing_agents_observed, self.parameters.n_agents - 1
        )
        self.n_agents = self.parameters.n_agents

        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(
                batch_dim, device=device, dtype=torch.int32
            ),  # Each environment has its own time step
            step_duration=torch.zeros(
                self.parameters.max_steps, device=device, dtype=torch.float32
            ),
            step_begin=time.time(),
            render_begin=0,
        )

        # Get map data
        self.map = MapManager(
            scenario_type=self.parameters.scenario_type,
            device=device,
        )

        cprint("[INFO] Map parsed.", "blue")
        # Determine the maximum number of points on the reference path
        if "CPM_mixed" in self.parameters.scenario_type:
            # Mixed scenarios including intersection, merge in, and merge out
            max_ref_path_points = (
                max(
                    [
                        ref_p["center_line"].shape[0]
                        for ref_p in self.map.parser.reference_paths_intersection
                        + self.map.parser.reference_paths_merge_in
                        + self.map.parser.reference_paths_merge_out
                    ]
                )
                + self.parameters.n_points_short_term * sample_interval_ref_path
                + 2
            )  # Append a smaller buffer
        else:
            # Single scenario
            max_ref_path_points = (
                max(
                    [
                        ref_p["center_line"].shape[0]
                        for ref_p in self.map.parser.reference_paths
                    ]
                )
                + self.parameters.n_points_short_term * sample_interval_ref_path
                + 2
            )  # Append a smaller buffer

        # Get all reference paths
        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=self.map.parser.reference_paths,
            long_term_intersection=self.map.parser.reference_paths_intersection,
            long_term_merge_in=self.map.parser.reference_paths_merge_in,
            long_term_merge_out=self.map.parser.reference_paths_merge_out,
            point_extended_all=torch.zeros(
                (
                    len(self.map.parser.reference_paths),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros(
                (
                    len(self.map.parser.reference_paths_intersection),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            point_extended_merge_in=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_in),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            point_extended_merge_out=torch.zeros(
                (
                    len(self.map.parser.reference_paths_merge_out),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            sample_interval=torch.tensor(
                sample_interval_ref_path, device=device, dtype=torch.int32
            ),
        )

        # Extended the reference path by several points along the last vector of the center line
        idx_broadcasting_entend = torch.arange(
            1,
            self.parameters.n_points_short_term * sample_interval_ref_path + 1,
            device=device,
            dtype=torch.int32,
        ).unsqueeze(1)
        for idx, i_path in enumerate(self.map.parser.reference_paths):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(self.map.parser.reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )

        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),  # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            left_boundary=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            right_boundary=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            entry=torch.zeros(
                (batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32
            ),
            exit=torch.zeros(
                (batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32
            ),
            is_loop=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            n_points_long_term=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            n_points_left_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            n_points_right_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            short_term=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_points_short_term, 2),
                device=device,
                dtype=torch.float32,
            ),  # Short-term reference path
            short_term_indices=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_points_short_term),
                device=device,
                dtype=torch.int32,
            ),
            n_points_nearing_boundary=torch.tensor(
                n_points_nearing_boundary, device=device, dtype=torch.int32
            ),
            nearing_points_left_boundary=torch.zeros(
                (batch_dim, self.n_agents, n_points_nearing_boundary, 2),
                device=device,
                dtype=torch.float32,
            ),  # Nearing left boundary
            nearing_points_right_boundary=torch.zeros(
                (batch_dim, self.n_agents, n_points_nearing_boundary, 2),
                device=device,
                dtype=torch.float32,
            ),  # Nearing right boundary
            scenario_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which scenarios agents are (1 for intersection, 2 for merge-in, 3 for merge-out)
            path_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which paths agents are
            point_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which points agents are
        )

        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32
        )

        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.parameters.n_points_short_term,
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,  # Progress in the weighted directions (directions indicating by closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(
                reward_reach_goal, device=device, dtype=torch.float32
            ),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(
                penalty_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
            near_boundary=torch.tensor(
                penalty_near_boundary, device=device, dtype=torch.float32
            ),
            near_other_agents=torch.tensor(
                penalty_near_other_agents, device=device, dtype=torch.float32
            ),
            collide_with_agents=torch.tensor(
                penalty_collide_with_agents, device=device, dtype=torch.float32
            ),
            collide_with_boundaries=torch.tensor(
                penalty_collide_with_boundaries, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(
                penalty_change_steering, device=device, dtype=torch.float32
            ),
            time=torch.tensor(penalty_time, device=device, dtype=torch.float32),
        )

        self.observations = Observations(
            n_nearing_agents=torch.tensor(
                self.parameters.n_nearing_agents_observed,
                device=device,
                dtype=torch.int32,
            ),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(
                n_stored_steps, device=device, dtype=torch.int32
            ),
            n_observed_steps=torch.tensor(
                n_observed_steps, device=device, dtype=torch.int32
            ),
            nearing_agents_indices=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed),
                device=device,
                dtype=torch.int32,
            ),
        )
        assert (
            self.observations.n_stored_steps >= 1
        ), "The number of stored steps should be at least 1."
        assert (
            self.observations.n_observed_steps >= 1
        ), "The number of observed steps should be at least 1."
        assert (
            self.observations.n_stored_steps >= self.observations.n_observed_steps
        ), "The number of stored steps should be greater or equal than the number of observed steps."

        if self.parameters.is_ego_view:
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.parameters.n_points_short_term,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.ref_paths_agent_related.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.ref_paths_agent_related.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 4, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.parameters.n_points_short_term,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.ref_paths_agent_related.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.ref_paths_agent_related.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )

        self.observations.past_action_vel = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_action_steering = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_ref_path = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_boundaries = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_left_boundary = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_right_boundary = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_agents = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_lengths = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_widths = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )

        self.normalizers = Normalizers(
            pos=torch.tensor(
                [self.agent_length * 10, self.agent_length * 10],
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=self.max_steering_angle,
            action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 10, device=device, dtype=torch.float32
            ),
        )

        # Distances to boundaries and reference path, and also the closest point on the reference paths of agents
        if self.parameters.is_use_mtv_distance:
            distance_type = "MTV"  # One of {"c2c", "MTV"}
        else:
            distance_type = "c2c"  # One of {"c2c", "MTV"}
        # print(colored("[INFO] Distance type: ", "black"), colored(distance_type, "blue"))

        self.distances = Distances(
            type=distance_type,  # Type of distances between agents
            agents=torch.zeros(
                batch_dim, self.n_agents, self.n_agents, dtype=torch.float32
            ),
            left_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),  # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),
            boundaries=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            ref_paths=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            closest_point_on_ref_path=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_left_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_right_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
        )

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(
                threshold_reach_goal, device=device, dtype=torch.float32
            ),
            deviate_from_ref_path=torch.tensor(
                threshold_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
            near_boundary_low=torch.tensor(
                threshold_near_boundary_low, device=device, dtype=torch.float32
            ),
            near_boundary_high=torch.tensor(
                threshold_near_boundary_high, device=device, dtype=torch.float32
            ),
            near_other_agents_low=torch.tensor(
                (
                    threshold_near_other_agents_c2c_low
                    if self.distances.type == "c2c"
                    else threshold_near_other_agents_MTV_low
                ),
                device=device,
                dtype=torch.float32,
            ),
            near_other_agents_high=torch.tensor(
                (
                    threshold_near_other_agents_c2c_high
                    if self.distances.type == "c2c"
                    else threshold_near_other_agents_MTV_high
                ),
                device=device,
                dtype=torch.float32,
            ),
            change_steering=torch.tensor(
                threshold_change_steering, device=device, dtype=torch.float32
            ).deg2rad(),
            no_reward_if_too_close_to_boundaries=torch.tensor(
                threshold_no_reward_if_too_close_to_boundaries,
                device=device,
                dtype=torch.float32,
            ),
            no_reward_if_too_close_to_other_agents=torch.tensor(
                threshold_no_reward_if_too_close_to_other_agents,
                device=device,
                dtype=torch.float32,
            ),
            distance_mask_agents=self.agent_length * 5,
        )

        self.constants = Constants(
            env_idx_broadcasting=torch.arange(
                batch_dim, device=device, dtype=torch.int32
            ).unsqueeze(-1),
            empty_action_vel=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                AGENTS["length"] ** 2 + self.agent_width**2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
            * 1.2,
        )

        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros(
                (batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.bool,
            ),
            with_lanelets=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            with_entry_segments=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            with_exit_segments=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
        )

        if self.parameters.is_challenging_initial_state_buffer:
            self.initial_state_buffer = InitialStateBuffer(
                # Used only when self.parameters.is_challenging_initial_state_buffer is True
                probability_record=torch.tensor(
                    probability_record, device=device, dtype=torch.float32
                ),
                probability_use_recording=torch.tensor(
                    probability_use_recording, device=device, dtype=torch.float32
                ),
                buffer=torch.zeros(
                    (buffer_size, self.n_agents, 8), device=device, dtype=torch.float32
                ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
            )

        # Store the states of agents at previous several time steps
        self.state_buffer = StateBuffer(
            buffer=torch.zeros(
                (self.parameters.n_steps_stored, batch_dim, self.n_agents, 8),
                device=device,
                dtype=torch.float32,
            ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )

        traj_len = int(getattr(self.parameters, "agent_trajectory_len", 25))
        traj_len = max(2, traj_len)
        self.traj_pos_buffer = CircularBuffer(
            buffer=torch.zeros(
                (traj_len, batch_dim, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            )
        )

        # Store the computed observations of each agent for reuse in `info()`
        # to generate `base_obs` and `prio_obs`, which are used for prioritized MARL.
        self.stored_observations = [None] * self.n_agents

    def _init_world(self, batch_dim: int, device: torch.device):
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=torch.tensor(
                self.world_x_dim, device=device, dtype=torch.float32
            ),
            y_semidim=torch.tensor(
                self.world_y_dim, device=device, dtype=torch.float32
            ),
            dt=self.parameters.dt,
        )
        return world

    def _init_agents(self, world):
        # Create agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=AGENTS["length"], width=AGENTS["width"]),
                color=tuple(colors[i % len(colors)]),
                alpha=1.0,
                collide=False,
                render_action=False,
                u_range=[
                    self.max_speed,
                    self.max_steering_angle,
                ],  # Control command serves as velocity command
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                    world,
                    width=AGENTS["width"],
                    l_f=AGENTS["l_f"],
                    l_r=AGENTS["l_r"],
                    max_steering_angle=self.max_steering_angle,
                    integration="rk4",  # one of {"euler", "rk4"}
                ),
            )
            world.add_agent(agent)

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        agents = self.world.agents

        is_reset_single_agent = agent_index is not None

        if is_reset_single_agent:
            assert env_index is not None

        for env_i in (
            [env_index] if env_index is not None else range(self.world.batch_dim)
        ):
            if is_reset_single_agent:
                reset_agent_index = int(
                    agent_index.item()
                    if isinstance(agent_index, torch.Tensor)
                    else agent_index
                )
                self.nod_agent_generation[env_i, reset_agent_index] += 1
            else:
                self.nod_agent_generation[env_i] += 1

            # Begining of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.step_duration[:] = 0
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0

            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            (
                ref_paths_scenario,
                extended_points,
            ) = self._reset_scenario_related_ref_paths(
                env_i, is_reset_single_agent, agent_index
            )

            if (
                self.parameters.is_challenging_initial_state_buffer
                and (
                    torch.rand(1) < self.initial_state_buffer.probability_use_recording
                )
                and (self.initial_state_buffer.valid_size >= 1)
            ):
                # Use initial state buffer
                is_use_state_buffer = True
                initial_state = self.initial_state_buffer.get_random()
                self.ref_paths_agent_related.scenario_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_scenario
                ]  # Update
                self.ref_paths_agent_related.path_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_path
                ]  # Update
                self.ref_paths_agent_related.point_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_point
                ]  # Update
                # print(colored(f"[LOG] Reset with path ids: {initial_state[:, -2]}", "red"))
            else:
                is_use_state_buffer = False
                initial_state = None

            for i_agent in (
                range(self.n_agents)
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):
                ref_path, path_id = self._reset_init_state(
                    env_i,
                    i_agent,
                    is_reset_single_agent,
                    is_use_state_buffer,
                    initial_state,
                    ref_paths_scenario,
                    agents,
                )

                self._reset_agent_related_ref_path(
                    env_i, i_agent, ref_path, path_id, extended_points
                )

            # The operations below can be done for all envs in parallel
            if env_index is None:
                if env_i == (self.world.batch_dim - 1):
                    env_j = slice(None)  # `slice(None)` is equivalent to `:`
                else:
                    continue
            else:
                env_j = env_i

            for i_agent in (
                range(self.n_agents)
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):
                self._reset_init_distances_and_short_term_ref_path(
                    env_j, i_agent, agents
                )

            # Compute mutual distances between agents
            # TODO Enable the possibility of computing the mutual distances of agents in a single env
            mutual_distances = get_distances_between_agents(
                self=self, distance_type=self.distances.type, is_set_diagonal=True
            )
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_entry_segments[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False

        if not is_reset_single_agent:
            self.state_buffer.reset()
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in agents], dim=1),
                    torch.stack([a.state.rot for a in agents], dim=1),
                    torch.stack([a.state.vel for a in agents], dim=1),
                    self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
                ),
                dim=-1,
            )
            self.state_buffer.add(state_add)

        if hasattr(self, "traj_pos_buffer"):
            if not is_reset_single_agent:
                pos_add = torch.stack([a.state.pos for a in agents], dim=1)
                self.traj_pos_buffer.buffer[:] = pos_add.unsqueeze(0).expand(
                    self.traj_pos_buffer.buffer_size, -1, -1, -1
                )
                self.traj_pos_buffer.pointer = 0
                self.traj_pos_buffer.valid_size = self.traj_pos_buffer.buffer_size
            else:
                env_i = env_index
                agent_i = agent_index
                pos_i = agents[int(agent_i)].state.pos[env_i]
                self.traj_pos_buffer.buffer[
                    :, env_i, int(agent_i), :
                ] = pos_i.unsqueeze(0).expand(self.traj_pos_buffer.buffer_size, -1)

    def _reset_scenario_related_ref_paths(
        self, env_i, is_reset_single_agent, agent_index
    ):
        # Get the center line and boundaries of the long-term reference path for each agent
        if self.parameters.scenario_type == "CPM_mixed":
            if is_reset_single_agent:
                scenario_id = self.ref_paths_agent_related.scenario_id[
                    env_i, agent_index
                ]  # Keep the same scenario
            else:
                scenario_id = (
                    torch.multinomial(
                        torch.tensor(
                            self.parameters.cpm_scenario_probabilities,
                            device=self.world.device,
                            dtype=torch.float32,
                        ),
                        1,
                        replacement=True,
                    ).item()
                    + 1
                )  # A random interger {1, 2, 3}
                self.ref_paths_agent_related.scenario_id[env_i, :] = scenario_id
            if scenario_id == 1:
                # Intersection scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
                extended_points = self.ref_paths_map_related.point_extended_intersection
            elif scenario_id == 2:
                # Merge-in scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
                extended_points = self.ref_paths_map_related.point_extended_merge_in
            elif scenario_id == 3:
                # Merge-out scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
                extended_points = self.ref_paths_map_related.point_extended_merge_out
        else:
            ref_paths_scenario = self.ref_paths_map_related.long_term_all
            extended_points = self.ref_paths_map_related.point_extended_all
            self.ref_paths_agent_related.scenario_id[
                env_i, :
            ] = 0  # 0 for others, 1 for intersection, 2 for merge-in, 3 for merge-out scenario
        return ref_paths_scenario, extended_points

    def _reset_init_state(
        self,
        env_i,
        i_agent,
        is_reset_single_agent,
        is_use_state_buffer,
        initial_state,
        ref_paths_scenario,
        agents,
    ):
        """
        This function resets the initial position, rotation, and velocity for an agent based on the provided
        initial state buffer if it is used. Otherwise, it randomly generates initial states ensuring they
        are feasible and do not collide with other agents.
        """
        if is_use_state_buffer:
            path_id = initial_state[i_agent, self.initial_state_buffer.idx_path].int()
            ref_path = ref_paths_scenario[path_id]

            agents[i_agent].set_pos(initial_state[i_agent, 0:2], batch_index=env_i)
            agents[i_agent].set_rot(initial_state[i_agent, 2], batch_index=env_i)
            agents[i_agent].set_vel(initial_state[i_agent, 3:5], batch_index=env_i)

        else:
            is_feasible_initial_position_found = False
            random_count = 0

            # Ramdomly generate initial states for each agent
            while not is_feasible_initial_position_found:
                if random_count >= 20:
                    cprint(
                        f"Reset agent(s): random_count = {random_count}.",
                        "grey",
                    )
                random_count += 1
                path_id = torch.randint(
                    0, len(ref_paths_scenario), (1,)
                ).item()  # Select randomly a path
                self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id  # Update
                ref_path = ref_paths_scenario[path_id]

                num_points = ref_path["center_line"].shape[0]

                if self.parameters.scenario_type == "CPM_mixed":
                    # In the mixed scenarios of the CPM case, we aovid using the beginning part of a path, making agents encounter each other more frequently. Additionally, We avoid initializing agents to be at a very end of a path.
                    start_point_idx = 6
                    end_point_idx = int(num_points / 2)
                else:
                    start_point_idx = 3  # Do not set to an overly small value to make sure agents are fully inside its lane
                    end_point_idx = num_points - 3

                random_point_id = torch.randint(
                    start_point_idx, end_point_idx, (1,)
                ).item()

                self.ref_paths_agent_related.point_id[
                    env_i, i_agent
                ] = random_point_id  # Update
                position_start = ref_path["center_line"][random_point_id]
                agents[i_agent].set_pos(position_start, batch_index=env_i)

                # Check if the initial position is feasible
                if not is_reset_single_agent:
                    if i_agent == 0:
                        # The initial position of the first agent is always feasible
                        is_feasible_initial_position_found = True
                        continue
                    else:
                        positions = torch.stack(
                            [
                                self.world.agents[i].state.pos[env_i]
                                for i in range(i_agent + 1)
                            ]
                        )
                else:
                    # Check if the initial position of the agent to be reset is collision-free with other agents
                    positions = torch.stack(
                        [
                            self.world.agents[i].state.pos[env_i]
                            for i in range(self.n_agents)
                        ]
                    )

                diff_sq = (
                    positions[i_agent, :] - positions
                ) ** 2  # Calculate pairwise squared differences in positions
                initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                initial_mutual_distances_sq[i_agent] = (
                    torch.max(initial_mutual_distances_sq) + 1
                )  # Set self-to-self distance to a sufficiently high value
                min_distance_sq = torch.min(initial_mutual_distances_sq)

                is_feasible_initial_position_found = min_distance_sq >= (
                    self.constants.reset_agent_min_distance**2
                )

            rot_start = ref_path["center_line_yaw"][random_point_id]
            vel_start_abs = (
                torch.rand(1, dtype=torch.float32, device=self.world.device)
                * agents[i_agent].max_speed
            )  # Random initial velocity
            vel_start = torch.hstack(
                [
                    vel_start_abs * torch.cos(rot_start),
                    vel_start_abs * torch.sin(rot_start),
                ]
            )

            agents[i_agent].set_rot(rot_start, batch_index=env_i)
            agents[i_agent].set_vel(vel_start, batch_index=env_i)

        return ref_path, path_id

    def _reset_agent_related_ref_path(
        self, env_i, i_agent, ref_path, path_id, extended_points
    ):
        """
        This function resets the agent-related reference paths and updates various related attributes
        for a specified agent in an environment.
        """
        # Long-term reference paths for agents
        n_points_long_term = ref_path["center_line"].shape[0]

        self.ref_paths_agent_related.long_term[
            env_i, i_agent, 0:n_points_long_term, :
        ] = ref_path["center_line"]
        self.ref_paths_agent_related.long_term[
            env_i,
            i_agent,
            n_points_long_term : (
                n_points_long_term
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ),
            :,
        ] = extended_points[path_id, :, :]
        self.ref_paths_agent_related.long_term[
            env_i,
            i_agent,
            (
                n_points_long_term
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ) :,
            :,
        ] = extended_points[path_id, -1, :]
        self.ref_paths_agent_related.n_points_long_term[
            env_i, i_agent
        ] = n_points_long_term

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i, i_agent, 0 : n_points_long_term - 1, :
        ] = ref_path["center_line_vec_normalized"]
        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i,
            i_agent,
            (n_points_long_term - 1) : (
                n_points_long_term
                - 1
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ),
            :,
        ] = ref_path["center_line_vec_normalized"][-1, :]

        n_points_left_b = ref_path["left_boundary_shared"].shape[0]
        self.ref_paths_agent_related.left_boundary[
            env_i, i_agent, 0:n_points_left_b, :
        ] = ref_path["left_boundary_shared"]
        self.ref_paths_agent_related.left_boundary[
            env_i, i_agent, n_points_left_b:, :
        ] = ref_path["left_boundary_shared"][-1, :]
        self.ref_paths_agent_related.n_points_left_b[env_i, i_agent] = n_points_left_b

        n_points_right_b = ref_path["right_boundary_shared"].shape[0]
        self.ref_paths_agent_related.right_boundary[
            env_i, i_agent, 0:n_points_right_b, :
        ] = ref_path["right_boundary_shared"]
        self.ref_paths_agent_related.right_boundary[
            env_i, i_agent, n_points_right_b:, :
        ] = ref_path["right_boundary_shared"][-1, :]
        self.ref_paths_agent_related.n_points_right_b[env_i, i_agent] = n_points_right_b

        self.ref_paths_agent_related.entry[env_i, i_agent, 0, :] = ref_path[
            "left_boundary_shared"
        ][0, :]
        self.ref_paths_agent_related.entry[env_i, i_agent, 1, :] = ref_path[
            "right_boundary_shared"
        ][0, :]

        self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = ref_path[
            "left_boundary_shared"
        ][-1, :]
        self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = ref_path[
            "right_boundary_shared"
        ][-1, :]

        self.ref_paths_agent_related.is_loop[env_i, i_agent] = ref_path["is_loop"]

    def _reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[env_j, i_agent],
            self.distances.closest_point_on_ref_path[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, i_agent
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                env_j, i_agent
            ],
        )
        self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - (
            agents[i_agent].shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                env_j, i_agent
            ],
        )
        self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - (
            agents[i_agent].shape.width / 2
        )
        # Calculate the positions of the four vertices of the agents
        self.vertices[env_j, i_agent] = get_rectangle_vertices(
            center=agents[i_agent].state.pos[env_j, :],
            yaw=agents[i_agent].state.rot[env_j, :],
            width=agents[i_agent].shape.width,
            length=agents[i_agent].shape.length,
            is_close_shape=True,
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    env_j, i_agent
                ],
            )
            (
                self.distances.right_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    env_j, i_agent
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[env_j, i_agent], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_j, i_agent],
                    self.distances.right_boundaries[env_j, i_agent],
                )
            ),
            dim=-1,
        )

        # Get the short-term reference paths
        (
            self.ref_paths_agent_related.short_term[env_j, i_agent],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            index_closest_point=self.distances.closest_point_on_ref_path[
                env_j, i_agent
            ],
            n_points_to_return=self.parameters.n_points_short_term,
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, i_agent
            ],
            sample_interval=self.ref_paths_map_related.sample_interval,
            n_points_shift=1,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_left_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, i_agent
                ],
                sample_interval=1,
                n_points_shift=1,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_right_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, i_agent
                ],
                sample_interval=1,
                n_points_shift=1,
            )

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to lane boundaries
                Too close to other agents
                Deviating from reference paths
                Changing steering too quick
                Colliding with other agents
                Colliding with lane boundaries

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        # Initialize
        self.rew[:] = 0

        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        self._update_state_before_rewarding(agent, agent_index)

        ##################################################
        ## [reward] forward movement
        ##################################################
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
            1
        )  # Vector of the current movement

        ref_points_vecs = self.ref_paths_agent_related.short_term[
            :, agent_index
        ] - latest_state[:, agent_index, 0:2].unsqueeze(
            1
        )  # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected, self.rewards.weighting_ref_directions
        )  # Put more weights on nearing reference points

        reward_movement = (
            move_projected_weighted
            / (agent.max_speed * self.world.dt)
            * self.rewards.progress
        )
        self.rew += reward_movement  # Relative to the maximum possible movement

        ##################################################
        ## [reward] high velocity
        ##################################################
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
            -1
        )
        factor_moving_direction = torch.where(
            v_proj > 0, 1, 2
        )  # Get penalty if move in negative direction

        reward_vel = (
            factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        )
        self.rew += reward_vel

        ##################################################
        ## [reward] reach goal
        ##################################################
        reward_goal = (
            self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        )
        self.rew += reward_goal

        ##################################################
        ## [penalty] close to lanelet boundaries
        ##################################################
        penalty_close_to_lanelets = (
            exponential_decreasing_fcn(
                x=self.distances.boundaries[:, agent_index],
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
            )
            * self.penalties.near_boundary
        )
        self.rew += penalty_close_to_lanelets

        ##################################################
        ## [penalty] close to other agents
        ##################################################
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_close_to_agents = (
            torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        )
        self.rew += penalty_close_to_agents

        ##################################################
        ## [penalty] deviating from reference path
        ##################################################
        self.rew += (
            self.distances.ref_paths[:, agent_index]
            / self.thresholds.deviate_from_ref_path
            * self.penalties.deviate_from_ref_path
        )

        ##################################################
        ## [penalty] changing steering too quick
        ##################################################
        steering_current = self.observations.past_action_steering.get_latest(n=1)[
            :, agent_index
        ]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[
            :, agent_index
        ]

        steering_change = torch.clamp(
            (steering_current - steering_past).abs() * self.normalizers.action_steering
            - self.thresholds.change_steering,  # Not forget to denormalize
            min=0,
        )
        steering_change_reward_factor = steering_change / (
            2 * agent.u_range[1] - 2 * self.thresholds.change_steering
        )
        penalty_change_steering = (
            steering_change_reward_factor * self.penalties.change_steering
        )
        self.rew += penalty_change_steering

        # ##################################################
        # ## [penalty] colliding with other agents
        # ##################################################
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_other_agents = (
            is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        self.rew += penalty_collide_other_agents

        ##################################################
        ## [penalty] colliding with lanelet boundaries
        ##################################################
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        self.rew += penalty_collide_lanelet

        ##################################################
        ## [penalty/reward] time
        ##################################################
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = (
            torch.where(v_proj > 0, 1, -1)
            * agent.state.vel.norm(dim=-1)
            / agent.max_speed
            * self.penalties.time
        )
        self.rew += time_reward

        # [update] previous positions and short-term reference paths
        self._update_state_after_rewarding(agent_index)

        # [update] previous positions and short-term reference paths
        if agent_index == (self.n_agents - 1):  # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                    self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
                ),
                dim=-1,
            )
            self.state_buffer.add(state_add)
            if getattr(
                self.parameters, "is_visualize_agent_trajectory", False
            ) and hasattr(self, "traj_pos_buffer"):
                traj_add = torch.stack([a.state.pos for a in self.world.agents], dim=1)
                self.traj_pos_buffer.add(traj_add)

        (
            self.ref_paths_agent_related.short_term[:, agent_index],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                :, agent_index
            ],
            n_points_to_return=self.parameters.n_points_short_term,
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
            sample_interval=self.ref_paths_map_related.sample_interval,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    :, agent_index
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[
                    :, agent_index
                ],
                n_points_to_return=self.ref_paths_agent_related.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )

        assert not self.rew.isnan().any(), "Rewards contain nan."
        assert not self.rew.isinf().any(), "Rewards contain inf."

        # Clamed the reward to avoid abs(reward) being too large
        rew_clamed = torch.clamp(self.rew, min=-1, max=1)

        return rew_clamed

    def _update_state_before_rewarding(self, agent, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """
        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        if agent_index == 0:  # Avoid repeated computations
            # Timer
            self.timer.step_duration[self.timer.step] = (
                time.time() - self.timer.step_begin
            )
            self.timer.step_begin = (
                time.time()
            )  # Set to the current time as the begin of the current time step
            self.timer.step += 1  # Increment step by 1
            # print(self.timer.step)

            # Update distances between agents
            self.distances.agents = get_distances_between_agents(
                self=self, distance_type=self.distances.type, is_set_diagonal=True
            )
            self.collisions.with_agents[:] = False  # Reset
            self.collisions.with_lanelets[:] = False  # Reset
            self.collisions.with_entry_segments[:] = False  # Reset
            self.collisions.with_exit_segments[:] = False  # Reset

            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                if self.distances.type == "c2c":
                    for a_j in range(a_i + 1, self.n_agents):
                        # Check for collisions between agents using the interX function
                        collision_batch_index = interX(
                            self.vertices[:, a_i], self.vertices[:, a_j], False
                        )
                        self.collisions.with_agents[
                            torch.nonzero(collision_batch_index), a_i, a_j
                        ] = True
                        self.collisions.with_agents[
                            torch.nonzero(collision_batch_index), a_j, a_i
                        ] = True
                elif self.distances.type == "MTV":
                    # Two agents collide if their MTV-based distance is zero
                    self.collisions.with_agents[:] = self.distances.agents == 0

                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                self.collisions.with_lanelets[
                    (collision_with_left_boundary | collision_with_right_boundary), a_i
                ] = True

                # Check for collisions with entry or exit segments (only need if agents' reference paths are not a loop)
                if not self.ref_paths_agent_related.is_loop[:, a_i].any():
                    self.collisions.with_entry_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.entry[:, a_i],
                        is_return_points=False,
                    )
                    self.collisions.with_exit_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )

        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[:, agent_index],
            self.distances.closest_point_on_ref_path[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos,
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                :, agent_index
            ],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (
            agent.shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                :, agent_index
            ],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (
            agent.shape.width / 2
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    :, agent_index
                ],
            )
            (
                self.distances.right_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    :, agent_index
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index],
                )
            ),
            dim=-1,
        )

    def _update_state_after_rewarding(self, agent_index):
        """
        Update some states (such as previous positions and short-term reference paths) after rewarding agents.
        """
        if agent_index == (self.n_agents - 1):  # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                    self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
                ),
                dim=-1,
            )
            self.state_buffer.add(state_add)

        (
            self.ref_paths_agent_related.short_term[:, agent_index],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                :, agent_index
            ],
            n_points_to_return=self.parameters.n_points_short_term,
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
            sample_interval=self.ref_paths_map_related.sample_interval,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    :, agent_index
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[
                    :, agent_index
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )

    def observation(self, agent: Agent):
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs, which consists of the observation of this agent itself and possibly the observation of its surrounding agents.
                The observation of this agent itself includes
                    position (in case of using bird view),
                    rotation (in case of using bird view),
                    velocity,
                    short-term reference path,
                    distance to its reference path (optional), and
                    lane boundaries (or distances to them).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)

        if agent_index == 0:  # Avoid repeated computations
            self._update_observation_and_normalize(agent, agent_index)

        # Observation of other agents
        obs_other_agents = self._observe_other_agents(agent_index)

        obs_self = self._observe_self(agent_index)

        obs_self.append(obs_other_agents)  # Append the observations of other agents

        obs_all = [o for o in obs_self if o is not None]  # Filter out None values

        obs = torch.hstack(obs_all)  # Convert from list to tensor

        if self.parameters.is_using_opponent_modeling:
            # Zero-padding as a placeholder for actions of surrounding agents
            obs = F.pad(
                obs,
                (0, self.parameters.n_nearing_agents_observed * AGENTS["n_actions"]),
            )

        if self.parameters.is_add_noise:
            # Add sensor noise if required
            obs = obs + (
                self.observations.noise_level
                * torch.rand_like(obs, device=self.world.device, dtype=torch.float32)
            )

        # Store observation for reuse in `info()`, only relevant for prioritized MARL
        self.stored_observations[agent_index] = obs

        return obs

    def _update_observation_and_normalize(self, agent, agent_index):
        """Update observation and normalize them."""
        positions_global = torch.stack(
            [a.state.pos for a in self.world.agents], dim=0
        ).transpose(0, 1)
        rotations_global = (
            torch.stack([a.state.rot for a in self.world.agents], dim=0)
            .transpose(0, 1)
            .squeeze(-1)
        )

        lengths_global = torch.tensor(
            [a.shape.length for a in self.world.agents],
            device=self.world.device,
            dtype=torch.float32,
        ).repeat(self.world.batch_dim, 1)

        widths_global = torch.tensor(
            [a.shape.width for a in self.world.agents],
            device=self.world.device,
            dtype=torch.float32,
        ).repeat(self.world.batch_dim, 1)

        # Add new observation & normalize
        self.observations.past_distance_to_agents.add(
            self.distances.agents / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_ref_path.add(
            self.distances.ref_paths / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_left_boundary.add(
            torch.min(self.distances.left_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_right_boundary.add(
            torch.min(self.distances.right_boundaries, dim=-1)[0]
            / self.normalizers.distance_lanelet
        )
        self.observations.past_distance_to_boundaries.add(
            self.distances.boundaries / self.normalizers.distance_lanelet
        )
        self.observations.past_lengths.add(
            lengths_global / self.normalizers.distance_agent
        )  # Use distance to agents as the normalizer
        self.observations.past_widths.add(
            widths_global / self.normalizers.distance_agent
        )

        if self.parameters.is_ego_view:
            pos_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.world.device,
                dtype=torch.float32,
            )  # Positions of other agents relative to agent i
            rot_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents),
                device=self.world.device,
                dtype=torch.float32,
            )  # Rotations of other agents relative to agent i
            vel_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.world.device,
                dtype=torch.float32,
            )  # Velocities of other agents relative to agent i
            ref_i_others = torch.zeros_like(
                (self.observations.past_short_term_ref_points.get_latest())
            )  # Reference paths of other agents relative to agent i
            l_b_i_others = torch.zeros_like(
                (self.observations.past_left_boundary.get_latest())
            )  # Left boundaries of other agents relative to agent i
            r_b_i_others = torch.zeros_like(
                (self.observations.past_right_boundary.get_latest())
            )  # Right boundaries of other agents relative to agent i
            ver_i_others = torch.zeros_like(
                (self.observations.past_vertices.get_latest())
            )  # Vertices of other agents relative to agent i

            for a_i in range(self.n_agents):
                pos_i = self.world.agents[a_i].state.pos
                rot_i = self.world.agents[a_i].state.rot

                # Store new observation - position
                pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                    pos_i=pos_i,
                    pos_j=positions_global,
                    rot_i=rot_i,
                )

                # Store new observation - rotation
                rot_i_others[:, a_i] = rotations_global - rot_i

                for a_j in range(self.n_agents):
                    # Store new observation - velocities
                    rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                    vel_abs = torch.norm(
                        self.world.agents[a_j].state.vel, dim=1
                    ).unsqueeze(
                        1
                    )  # TODO Check if relative velocities here are better
                    vel_i_others[:, a_i, a_j] = torch.hstack(
                        (vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel))
                    )

                    # Store new observation - reference paths
                    ref_i_others[
                        :, a_i, a_j
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.ref_paths_agent_related.short_term[:, a_j],
                        rot_i=rot_i,
                    )

                    # Store new observation - left boundary
                    if not self.parameters.is_observe_distance_to_boundaries:
                        l_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                        # Store new observation - right boundary
                        r_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                    # Store new observation - vertices
                    ver_i_others[
                        :, a_i, a_j
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.vertices[:, a_j, 0:4, :],
                        rot_i=rot_i,
                    )
            # Add new observations & normalize
            self.observations.past_pos.add(
                pos_i_others
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
            self.observations.past_vel.add(vel_i_others / self.normalizers.v)
            self.observations.past_short_term_ref_points.add(
                ref_i_others
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_left_boundary.add(
                l_b_i_others
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_right_boundary.add(
                r_b_i_others
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_vertices.add(
                ver_i_others
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )

        else:  # Global coordinate system
            # Store new observations
            self.observations.past_pos.add(
                positions_global
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_vel.add(
                torch.stack([a.state.vel for a in self.world.agents], dim=1)
                / self.normalizers.v
            )
            self.observations.past_rot.add(rotations_global[:] / self.normalizers.rot)
            self.observations.past_vertices.add(
                self.vertices[:, :, 0:4, :]
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_short_term_ref_points.add(
                self.ref_paths_agent_related.short_term[:]
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_left_boundary.add(
                self.ref_paths_agent_related.nearing_points_left_boundary
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )
            self.observations.past_right_boundary.add(
                self.ref_paths_agent_related.nearing_points_right_boundary
                / (
                    self.normalizers.pos
                    if self.parameters.is_ego_view
                    else self.normalizers.pos_world
                )
            )

        # Add new observation - actions & normalize
        if agent.action.u is None:
            self.observations.past_action_vel.add(self.constants.empty_action_vel)
            self.observations.past_action_steering.add(
                self.constants.empty_action_steering
            )
        else:
            self.observations.past_action_vel.add(
                torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1)
                / self.normalizers.action_vel
            )
            self.observations.past_action_steering.add(
                torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1)
                / self.normalizers.action_steering
            )

            if self.parameters.is_apply_mask:
                # Determine the current lanelet IDs of all agents of all envs for later use
                self.map.determine_current_lanelet(positions_global)

    def _observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        ##################################################
        ## Observation of other agents
        ##################################################
        # Optional topology-driven neighbor selection
        use_topo_select = bool(
            getattr(self.parameters, "use_topology_neighbor_selection", False)
        )
        topo_model = getattr(self, "topology_learner", None)
        K_policy = int(self.observations.n_nearing_agents)
        if use_topo_select and (topo_model is not None):
            # Preserve heuristic indices for downstream modules and logging
            # (We do not use them to build the policy observation when topology selection is enabled)
            nearing_agents_distances_h, heur_idx = torch.topk(
                self.distances.agents[:, agent_index],
                k=self.observations.n_nearing_agents,
                largest=False,
            )
            self.observations.nearing_agents_indices[:, agent_index] = heur_idx

            # Build candidate set for topology scoring (exclude ego, no additional filtering)
            K_topo = int(
                getattr(self.parameters, "n_topology_nearing_agents_observed", K_policy)
                or K_policy
            )
            K_topo = max(1, min(K_topo, int(self.n_agents) - 1))
            dist_row = self.distances.agents[:, agent_index]  # [B, N]
            dist_row_excl = dist_row.clone()
            dist_row_excl[:, agent_index] = torch.tensor(
                float("inf"), device=self.world.device, dtype=dist_row.dtype
            )
            _, cand_idx = torch.topk(
                dist_row_excl, k=K_topo, largest=False
            )  # [B, K_topo]

            # Gather per-neighbor features for candidates (no masks applied in topo path)
            B = self.world.batch_dim
            indexing_tuple_topo = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.parameters.is_ego_view else ())
                + (cand_idx,)
            )
            obs_pos_cand = self.observations.past_pos.get_latest()[indexing_tuple_topo]
            obs_rot_cand = self.observations.past_rot.get_latest()[indexing_tuple_topo]
            obs_vel_cand = self.observations.past_vel.get_latest()[indexing_tuple_topo]
            obs_ref_cand = self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_topo
            ]
            obs_vertices_cand = self.observations.past_vertices.get_latest()[
                indexing_tuple_topo
            ]
            obs_dist_cand = self.observations.past_distance_to_agents.get_latest()[
                self.constants.env_idx_broadcasting, agent_index, cand_idx
            ]
            obs_len_cand = self.observations.past_lengths.get_latest()[
                self.constants.env_idx_broadcasting, cand_idx
            ]
            obs_wid_cand = self.observations.past_widths.get_latest()[
                self.constants.env_idx_broadcasting, cand_idx
            ]

            # Compose ego observation (reuse self composition)
            ego_obs_list = [o for o in self._observe_self(agent_index) if o is not None]
            ego_b = torch.hstack(ego_obs_list)  # [B, D_ego]

            # Compose neighbors_observation for topology model: [B, K_topo, D_nei]
            pos_flat = obs_pos_cand.reshape(B, K_topo, -1)
            rot_flat = obs_rot_cand.reshape(B, K_topo, -1)
            vel_flat = obs_vel_cand.reshape(B, K_topo, -1)
            ref_flat = (
                obs_ref_cand.reshape(B, K_topo, -1)
                if self.parameters.is_observe_ref_path_other_agents
                else None
            )
            vert_flat = obs_vertices_cand.reshape(B, K_topo, -1)
            dist_flat = (
                obs_dist_cand.reshape(B, K_topo, -1)
                if self.parameters.is_observe_distance_to_agents
                else None
            )
            len_flat = obs_len_cand.reshape(B, K_topo, -1)
            wid_flat = obs_wid_cand.reshape(B, K_topo, -1)
            nei_list = [
                (
                    vert_flat
                    if self.parameters.is_observe_vertices
                    else torch.cat([pos_flat, rot_flat, len_flat, wid_flat], dim=-1)
                ),
                vel_flat,
                dist_flat,
                ref_flat,
            ]
            nei_list = [x for x in nei_list if x is not None]
            nei_b = torch.cat(nei_list, dim=-1)

            # Relative features in ego local frame: [B, K_topo, D_rel]
            past_pos = self.observations.past_pos.get_latest()
            past_rot = self.observations.past_rot.get_latest()
            past_vel = self.observations.past_vel.get_latest()
            pos_i_allj = past_pos[:, agent_index]
            rot_i_allj = past_rot[:, agent_index]
            vel_i_allj = past_vel[:, agent_index]
            idx_for_2 = cand_idx.unsqueeze(-1).expand(-1, -1, pos_i_allj.size(-1))
            idx_for_1 = cand_idx
            rel_pos = torch.gather(pos_i_allj, dim=1, index=idx_for_2)
            rel_yaw = torch.gather(rot_i_allj, dim=1, index=idx_for_1)
            nei_v = torch.gather(vel_i_allj, dim=1, index=idx_for_2)
            nei_v_forward = nei_v[..., 0]
            ego_v_forward = past_vel[:, agent_index, agent_index, 0]
            d_speed = nei_v_forward - ego_v_forward.unsqueeze(-1)
            rel_b = torch.cat(
                [rel_pos, rel_yaw.unsqueeze(-1), d_speed.unsqueeze(-1)], dim=-1
            )

            # Score candidates with topology model
            edge_logits = topo_model(ego_b, nei_b, rel_b).squeeze(-1)  # [B, K_topo]
            edge_probs = torch.sigmoid(edge_logits)

            # Apply threshold and select top-K_policy neighbors by probability
            thr = float(getattr(self.parameters, "topology_selection_threshold", 0.5))
            sorted_probs, sorted_idx = torch.sort(
                edge_probs, dim=1, descending=True
            )  # [B, K_topo]
            # Map sorted indices back to agent ids
            cand_idx_sorted = torch.gather(cand_idx, 1, sorted_idx)
            # Distances of candidates (for stable reordering later)
            dist_cand = torch.gather(dist_row, dim=1, index=cand_idx)  # [B, K_topo]

            # Build selection tensors
            # 注意：特征收集需使用候选列表中的“位置索引”，而非全局 agent 索引
            sel_pos = torch.full(
                (B, K_policy), -1, device=self.world.device, dtype=torch.long
            )
            sel_ids = torch.full(
                (B, K_policy), -1, device=self.world.device, dtype=torch.long
            )
            sel_probs = torch.zeros(
                (B, K_policy), device=self.world.device, dtype=sorted_probs.dtype
            )
            for b in range(B):
                valid = (sorted_probs[b] >= thr).nonzero(as_tuple=False).squeeze(-1)
                n_take = min(K_policy, valid.numel())
                if n_take > 0:
                    # 位置索引（在 K_topo 列表内）
                    sel_pos[b, :n_take] = sorted_idx[b, valid[:n_take]]
                    # 对应的全局 agent 索引与概率，仅用于缓存与日志
                    sel_ids[b, :n_take] = cand_idx_sorted[b, valid[:n_take]]
                    sel_probs[b, :n_take] = sorted_probs[b, valid[:n_take]]

                    # --- Stable reordering by distance (tie-break by agent id) ---
                    # 仅对有效槽位进行稳定重排，确保与“距离升序”一致，从而稳定策略输入分布
                    pos_take = sel_pos[b, :n_take]
                    # 对应候选的距离
                    dist_take = dist_cand[b].index_select(0, pos_take)
                    ids_take = sel_ids[b, :n_take].to(torch.float32)
                    # 将 agent_id 作为微小扰动用于平局打破，避免频繁抖动
                    order = torch.argsort(dist_take + 1e-6 * ids_take, dim=0)
                    sel_pos[b, :n_take] = pos_take.index_select(0, order)
                    sel_ids[b, :n_take] = sel_ids[b, :n_take].index_select(0, order)
                    sel_probs[b, :n_take] = sel_probs[b, :n_take].index_select(0, order)

            # Cache selected indices/probabilities per agent for info() and training logs
            if not hasattr(self.observations, "topology_selected_indices"):
                self.observations.topology_selected_indices = torch.full(
                    (self.world.batch_dim, int(self.n_agents), K_policy),
                    -1,
                    device=self.world.device,
                    dtype=torch.long,
                )
            if not hasattr(self.observations, "topology_selected_probs"):
                self.observations.topology_selected_probs = torch.zeros(
                    (self.world.batch_dim, int(self.n_agents), K_policy),
                    device=self.world.device,
                    dtype=sel_probs.dtype,
                )
            self.observations.topology_selected_indices[:, agent_index] = sel_ids
            self.observations.topology_selected_probs[:, agent_index] = sel_probs

            # Gather selected neighbor features (pad missing with masks)
            def gather_or_mask(tensor, expand_last):
                # 通用 gather：按 dim=1 收集，index 需与输入张量维度一致
                idx_for = sel_pos.clamp(min=0)
                # 为所有后续维度添加轴，并扩展到与输入相同的形状（除dim=1外）
                add_dims = tensor.dim() - 2
                for _ in range(add_dims):
                    idx_for = idx_for.unsqueeze(-1)
                if add_dims > 0:
                    idx_for = idx_for.expand(-1, -1, *tensor.shape[2:])
                gathered = torch.gather(tensor, dim=1, index=idx_for)
                pad_mask = sel_pos < 0
                return gathered, pad_mask

            pos_sel, pad_mask = gather_or_mask(
                obs_pos_cand, expand_last=obs_pos_cand.size(-1)
            )
            rot_sel, _ = gather_or_mask(obs_rot_cand, expand_last=None)
            vel_sel, _ = gather_or_mask(obs_vel_cand, expand_last=obs_vel_cand.size(-1))
            ref_sel, _ = (
                gather_or_mask(
                    obs_ref_cand,
                    expand_last=(
                        obs_ref_cand.size(-1)
                        if self.parameters.is_observe_ref_path_other_agents
                        else None
                    ),
                )
                if self.parameters.is_observe_ref_path_other_agents
                else (None, pad_mask)
            )
            vert_sel, _ = gather_or_mask(
                obs_vertices_cand, expand_last=obs_vertices_cand.size(-1)
            )
            dist_sel, _ = (
                gather_or_mask(obs_dist_cand, expand_last=None)
                if self.parameters.is_observe_distance_to_agents
                else (None, pad_mask)
            )
            len_sel, _ = gather_or_mask(obs_len_cand, expand_last=None)
            wid_sel, _ = gather_or_mask(obs_wid_cand, expand_last=None)

            # Apply mask values to padded slots
            pos_sel[pad_mask] = self.constants.mask_one
            rot_sel[pad_mask] = self.constants.mask_zero
            vel_sel[pad_mask] = self.constants.mask_zero
            vert_sel[pad_mask] = self.constants.mask_one
            len_sel[pad_mask] = 0.0
            wid_sel[pad_mask] = 0.0
            if dist_sel is not None:
                dist_sel[pad_mask] = self.constants.mask_one
            if ref_sel is not None:
                ref_sel[pad_mask] = self.constants.mask_one

            # Flatten per neighbor and compose final policy observation for others
            pos_flat = pos_sel.reshape(B, K_policy, -1)
            rot_flat = rot_sel.reshape(B, K_policy, -1)
            vel_flat = vel_sel.reshape(B, K_policy, -1)
            vert_flat = vert_sel.reshape(B, K_policy, -1)
            len_flat = len_sel.reshape(B, K_policy, -1)
            wid_flat = wid_sel.reshape(B, K_policy, -1)
            dist_flat = (
                dist_sel.reshape(B, K_policy, -1) if (dist_sel is not None) else None
            )
            ref_flat = (
                ref_sel.reshape(B, K_policy, -1) if (ref_sel is not None) else None
            )

            obs_others_list = [
                (
                    vert_flat
                    if self.parameters.is_observe_vertices
                    else torch.cat([pos_flat, rot_flat, len_flat, wid_flat], dim=-1)
                ),
                vel_flat,
                dist_flat,
                ref_flat,
            ]
            obs_others_list = [o for o in obs_others_list if o is not None]
            obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(B, -1)

            # Lightweight logging (env 0) to compare selection vs heuristic
            if getattr(self.parameters, "is_visualize_short_term_path", False):
                env0 = 0
                probs_e0 = edge_probs[env0].detach().cpu().tolist()
                cand_e0 = cand_idx[env0].detach().cpu().tolist()
                sel_ids_e0 = sel_ids[env0].detach().cpu().tolist()
                sel_probs_e0 = sel_probs[env0].detach().cpu().tolist()
                heur_e0 = heur_idx[env0].detach().cpu().tolist()
                # print(
                #     colored(
                #         f"[TOPO-SELECT][ON] ego={agent_index} raw_probs="
                #         + str([f"{i}:{p:.2f}" for i, p in zip(cand_e0, probs_e0)])
                #         + f" | selected="
                #         + str([f"{i}:{p:.2f}" for i, p in zip(sel_ids_e0, sel_probs_e0)])
                #         + f" | heuristic={heur_e0}",
                #         "blue",
                #     )
                # )

            return obs_other_agents

        if self.parameters.is_partial_observation:
            # Each agent observes only a fixed number of nearest agents
            (
                nearing_agents_distances,
                self.observations.nearing_agents_indices[:, agent_index],
            ) = torch.topk(
                self.distances.agents[:, agent_index],
                k=self.observations.n_nearing_agents,
                largest=False,
            )

            if self.parameters.is_apply_mask:
                # Two kinds of agents will be masked by ego agents:
                # 1. By distance: agents that are distant to the ego agents
                # 2. By lanelet relation: agents whose lanelets are not the neighboring lanelets or the same lanelets of the ego agents
                masked_agents_by_distance = (
                    nearing_agents_distances >= self.thresholds.distance_mask_agents
                )
                # print(f"masked_agents_by_distance = {masked_agents_by_distance}")
                if len(self.map.parser.neighboring_lanelets_idx) != 0:
                    # Mask agents by lanelets
                    masked_agents_by_lanelets = (
                        self.map.determine_masked_agents_by_lanelets(
                            agent_index,
                            self.observations.nearing_agents_indices[:, agent_index],
                        )
                    )
                else:
                    masked_agents_by_lanelets = torch.zeros(
                        (
                            self.world.batch_dim,
                            self.parameters.n_nearing_agents_observed,
                        ),
                        device=self.world.device,
                        dtype=torch.bool,
                    )

                masked_agents = masked_agents_by_distance | masked_agents_by_lanelets

            else:
                # Otherwise no agents will be masked
                masked_agents = torch.zeros(
                    (self.world.batch_dim, self.parameters.n_nearing_agents_observed),
                    device=self.world.device,
                    dtype=torch.bool,
                )

            indexing_tuple_1 = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.parameters.is_ego_view else ())
                + (self.observations.nearing_agents_indices[:, agent_index],)
            )

            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_rot_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Rotation mask

            # Lengths and widths of nearing agents
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                self.constants.env_idx_broadcasting,
                self.observations.nearing_agents_indices[:, agent_index],
            ]

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_vel_other_agents[
                masked_agents
            ] = self.constants.mask_zero  # Velocity mask

            # Reference paths of nearing agents
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_1
                ]
            )  # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Reference-path mask

            # Distances to nearing agents
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[
                    self.constants.env_idx_broadcasting,
                    agent_index,
                    self.observations.nearing_agents_indices[:, agent_index],
                ]
            )  # [batch_size, n_nearing_agents]
            obs_distance_other_agents[
                masked_agents
            ] = self.constants.mask_one  # Distance mask

        else:
            indexing_tuple_2 = (self.constants.env_idx_broadcasting.squeeze(-1),) + (
                (agent_index,) if self.parameters.is_ego_view else ()
            )

            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_2
                ]
            )  # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_2
            ]  # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[indexing_tuple_2]
            )  # [batch_size, n_agents]
            obs_distance_other_agents[
                indexing_tuple_2
            ] = 0  # Reset self-self distance to zero
            obs_lengths_other_agents = self.observations.past_lengths.get_latest()[
                indexing_tuple_2
            ]
            obs_widths_other_agents = self.observations.past_widths.get_latest()[
                indexing_tuple_2
            ]

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_lengths_other_agents_flat = obs_lengths_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_widths_other_agents_flat = obs_widths_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )

        # Observation of other agents
        obs_others_list = [
            (
                obs_vertices_other_agents_flat
                if self.parameters.is_observe_vertices
                else torch.cat(  # [other] vertices
                    [
                        obs_pos_other_agents_flat,  # [others] positions
                        obs_rot_other_agents_flat,  # [others] rotations
                        obs_lengths_other_agents_flat,  # [others] lengths
                        obs_widths_other_agents_flat,  # [others] widths
                    ],
                    dim=-1,
                )
            ),
            obs_vel_other_agents_flat,  # [others] velocities
            (
                obs_distance_other_agents_flat
                if self.parameters.is_observe_distance_to_agents
                else None
            ),  # [others] mutual distances
            (
                obs_ref_path_other_agents_flat
                if self.parameters.is_observe_ref_path_other_agents
                else None
            ),  # [others] reference paths
        ]
        obs_others_list = [
            o for o in obs_others_list if o is not None
        ]  # Filter out None values
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            self.world.batch_dim, -1
        )  # [batch_size, -1]

        return obs_other_agents

    def _observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index,) if self.parameters.is_ego_view else ())
        )
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0) if self.parameters.is_ego_view else ())
        )  # In local coordinate system, only the first component is interesting, as the second is always 0
        # All observations
        obs_self = [
            (
                None
                if self.parameters.is_ego_view
                else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(
                    self.world.batch_dim, -1
                )
            ),  # [own] position,
            (
                None
                if self.parameters.is_ego_view
                else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(
                    self.world.batch_dim, -1
                )
            ),  # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(
                self.world.batch_dim, -1
            ),  # [own] velocity
            self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] short-term reference path
            (
                self.observations.past_distance_to_ref_path.get_latest()[
                    :, agent_index
                ].reshape(self.world.batch_dim, -1)
                if self.parameters.is_observe_distance_to_center_line
                else None
            ),  # [own] distances to reference paths
            (
                self.observations.past_distance_to_left_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.world.batch_dim, -1)
                if self.parameters.is_observe_distance_to_boundaries
                else self.observations.past_left_boundary.get_latest()[
                    indexing_tuple_3
                ].reshape(self.world.batch_dim, -1)
            ),  # [own] left boundaries
            (
                self.observations.past_distance_to_right_boundary.get_latest()[
                    :, agent_index
                ].reshape(self.world.batch_dim, -1)
                if self.parameters.is_observe_distance_to_boundaries
                else self.observations.past_right_boundary.get_latest()[
                    indexing_tuple_3
                ].reshape(self.world.batch_dim, -1)
            ),  # [own] right boundaries
        ]

        return obs_self

    def done(self):
        # print("[DEBUG] done()")
        is_collision_with_agents = self.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        is_leaving_entry_segment = self.collisions.with_entry_segments.any(dim=-1) & (
            self.timer.step >= 20
        )
        is_any_agents_leaving_exit_segment = self.collisions.with_exit_segments.any(
            dim=-1
        )
        is_max_steps_reached = self.timer.step == (self.parameters.max_steps - 1)

        if (
            self.parameters.is_challenging_initial_state_buffer
        ):  # Record challenging initial states
            if torch.rand(1) > (
                1 - self.initial_state_buffer.probability_record
            ):  # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(
                        self.state_buffer.get_latest(n=self.parameters.n_steps_stored)[
                            env_collide
                        ]
                    )
                    # print(colored(f"[LOG] Record states with path ids: {self.ref_paths_agent_related.path_id[env_collide]}.", "blue"))

        if self.parameters.is_testing_mode:
            is_done = is_max_steps_reached  # In test mode, we only reset the whole env if the maximum time steps are reached

            # Reset single agent
            agents_reset = (
                self.collisions.with_agents.any(dim=-1)
                | self.collisions.with_lanelets
                | self.collisions.with_entry_segments
                | self.collisions.with_exit_segments
            )
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(
                agents_reset_indices[0], agents_reset_indices[1]
            ):
                if not is_done[env_idx]:
                    self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            is_done = (
                is_max_steps_reached
                | is_collision_with_agents
                | is_collision_with_lanelets
            )
            if (
                self.parameters.scenario_type != "CPM_entire"
            ):  # This part only applies to the map that have loop-shaped paths
                # Reset the whole system only when collisions occur. Reset a single agents if it leaves an entry or an exit

                # Reset single agnet
                agents_reset = (
                    self.collisions.with_entry_segments
                    | self.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(
                    agents_reset_indices[0], agents_reset_indices[1]
                ):
                    if not is_done[env_idx]:
                        # Skip envs with done flag since later they will be reset anyway
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
                        # print(f"Reset agent {agent_idx} in env {env_idx}")
            else:
                # Reset the whole system once collisions occur. There is no entry or exit in this scenario.
                assert not is_leaving_entry_segment.any()
                assert not is_any_agents_leaving_exit_segment.any()

            assert not (is_collision_with_agents & (self.timer.step == 0)).any()
            assert not (is_collision_with_lanelets & (self.timer.step == 0)).any()
            assert not (is_leaving_entry_segment & (self.timer.step == 0)).any()
            assert not (is_max_steps_reached & (self.timer.step == 0)).any()
            assert not (
                is_any_agents_leaving_exit_segment & (self.timer.step == 0)
            ).any()

        # Logs
        # if is_collision_with_agents.any():
        #     print("Collide with other agents.")
        # if is_collision_with_lanelets.any():
        #     print("Collide with lanelet.")
        # if is_leaving_entry_segment.any():
        #     print("At least one agent is leaving its entry segment.")
        # if is_max_steps_reached.any():
        #     print("The number of the maximum steps is reached.")
        # if is_any_agents_leaving_exit_segment.any():
        #     print("At least one agent is leaving its exit segment.")

        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent)  # Index of the current agent

        is_action_empty = agent.action.u is None

        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)

        # Zero-padding as a placeholder for actions of surrounding agents
        base_obs = F.pad(
            self.stored_observations[agent_index].clone(),
            (0, self.parameters.n_nearing_agents_observed * AGENTS["n_actions"]),
        )

        prio_obs = self.stored_observations[agent_index].clone()

        # -------------------------------------------------------------
        # Structured observation for topology learning and later Top-K selection
        # We precompute and store:
        # - ego_observation: per-agent semantic vector (no neighbors)
        # - neighbors_observation: per-agent, per-neighbor semantic vectors
        # - neighbors_observation_flat: flattened neighbors_observation (optional, for quick inspection)
        # - relative_features: lightweight relative geometry (dx, dy, d_yaw, d_speed)
        # These keys are written under ("agents","info",*) to avoid changing the policy inputs.
        # -------------------------------------------------------------

        # Build ego observation by reusing _observe_self() composition
        ego_obs_list = [o for o in self._observe_self(agent_index) if o is not None]
        ego_observation = torch.hstack(ego_obs_list)  # [batch_dim, D_ego]

        # Flat neighbors observation using the original helper (preserves scenario masking & composition)
        neighbors_observation_flat = self._observe_other_agents(agent_index)

        # Structured neighbors_observation for TopoDecoder: [B, K, D_nei]
        if neighbors_observation_flat is not None:
            B = neighbors_observation_flat.shape[0]
            K = self.parameters.n_nearing_agents_observed
            D_nei = neighbors_observation_flat.shape[1] // K
            neighbors_observation = neighbors_observation_flat.reshape(B, K, D_nei)
        else:
            neighbors_observation = None

        # Compute relative features (dx, dy, d_yaw, d_speed) in ego local frame
        # Only implemented for ego-view, which is the default in this project
        if self.parameters.is_ego_view:
            # Neighbor indices for current ego (B, K)
            neighbor_idx = self.observations.nearing_agents_indices[
                :, agent_index
            ].long()

            # Convenience tensors
            past_pos = self.observations.past_pos.get_latest()  # [B, N, N, 2]
            past_rot = self.observations.past_rot.get_latest()  # [B, N, N]
            past_vel = self.observations.past_vel.get_latest()  # [B, N, N, 2]

            # Slice the j-dimension for current ego i, then gather neighbors along j
            pos_i_allj = past_pos[:, agent_index]  # [B, N, 2]
            rot_i_allj = past_rot[:, agent_index]  # [B, N]
            vel_i_allj = past_vel[:, agent_index]  # [B, N, 2]

            # Build gather indices
            idx_for_2 = neighbor_idx.unsqueeze(-1).expand(
                -1, -1, pos_i_allj.size(-1)
            )  # [B, K, 2]
            idx_for_1 = neighbor_idx  # [B, K]

            # Relative position and yaw in ego local frame
            rel_pos = torch.gather(pos_i_allj, dim=1, index=idx_for_2)  # [B, K, 2]
            rel_yaw = torch.gather(rot_i_allj, dim=1, index=idx_for_1)  # [B, K]

            # Neighbor forward speed (component 0)
            nei_v = torch.gather(vel_i_allj, dim=1, index=idx_for_2)  # [B, K, 2]
            nei_v_forward = nei_v[..., 0]  # [B, K]

            # Ego forward speed (first component): [B]
            ego_v_forward = past_vel[:, agent_index, agent_index, 0]  # [B]

            # Speed difference: [B, K]
            d_speed = nei_v_forward - ego_v_forward.unsqueeze(-1)

            # Apply masking consistent with scenario design
            if self.parameters.is_partial_observation:
                # Distances to all agents for ego i: [B, N] -> gather to [B, K]
                dist_i_allj = self.distances.agents[:, agent_index]  # [B, N]
                nearing_agents_distances = torch.gather(
                    dist_i_allj, dim=1, index=neighbor_idx
                )  # [B, K]
                if self.parameters.is_apply_mask:
                    masked_agents_by_distance = (
                        nearing_agents_distances >= self.thresholds.distance_mask_agents
                    )  # [B, K]
                    if len(self.map.parser.neighboring_lanelets_idx) != 0:
                        try:
                            masked_agents_by_lanelets = (
                                self.map.determine_masked_agents_by_lanelets(
                                    agent_index, neighbor_idx
                                )
                            )
                        except Exception:
                            masked_agents_by_lanelets = torch.zeros(
                                (
                                    self.world.batch_dim,
                                    self.parameters.n_nearing_agents_observed,
                                ),
                                device=self.world.device,
                                dtype=torch.bool,
                            )
                    else:
                        masked_agents_by_lanelets = torch.zeros(
                            (
                                self.world.batch_dim,
                                self.parameters.n_nearing_agents_observed,
                            ),
                            device=self.world.device,
                            dtype=torch.bool,
                        )
                    masked_agents = (
                        masked_agents_by_distance | masked_agents_by_lanelets
                    )
                else:
                    masked_agents = torch.zeros(
                        (
                            self.world.batch_dim,
                            self.parameters.n_nearing_agents_observed,
                        ),
                        device=self.world.device,
                        dtype=torch.bool,
                    )

                rel_pos[masked_agents] = self.constants.mask_one
                rel_yaw[masked_agents] = self.constants.mask_zero
                d_speed[masked_agents] = self.constants.mask_zero

            # Concatenate to [B, K, 4]
            relative_features = torch.cat(
                [rel_pos, rel_yaw.unsqueeze(-1), d_speed.unsqueeze(-1)], dim=-1
            )
        else:
            # Bird-view fallback: compute differences in world frame (approximate)
            # pos_i: [B, 2], yaw_i: [B], v_i_forward: [B] (approximate with x-component)
            pos_all = self.observations.past_pos.get_latest()[:, :, :, :]
            yaw_all = self.observations.past_rot.get_latest()[:, :, :]
            v_all = self.observations.past_vel.get_latest()[:, :, :, 0]
            pos_i = pos_all[:, :, agent_index, :]
            yaw_i = yaw_all[:, :, agent_index]
            v_i = v_all[:, :, agent_index]
            neighbor_idx = self.observations.nearing_agents_indices[:, agent_index]
            pos_j = pos_all[:, (slice(None),), neighbor_idx, :].squeeze(1)
            yaw_j = yaw_all[:, (slice(None),), neighbor_idx].squeeze(1)
            v_j = v_all[:, (slice(None),), neighbor_idx].squeeze(1)
            rel_pos = pos_j - pos_i.unsqueeze(1)
            rel_yaw = angle_eliminate_two_pi(yaw_j - yaw_i.unsqueeze(1))
            d_speed = v_j - v_i.unsqueeze(1)
            relative_features = torch.cat(
                [rel_pos, rel_yaw.unsqueeze(-1), d_speed.unsqueeze(-1)], dim=-1
            )

        # --- Topology labels相关：提供短期参考路径（自车与K邻居）与距离掩码 ---
        B = self.world.batch_dim
        K = self.parameters.n_nearing_agents_observed
        neighbor_idx_local = neighbor_idx.long()
        # 所有代理的短期参考路径 [B, N, T, 2]
        short_term_all = self.ref_paths_agent_related.short_term  # [B, N, T, 2]
        short_term_all_norm = short_term_all / self.normalizers.pos_world
        # 可选：在短期参考前追加“当前位置”，统一尺度归一化
        if getattr(
            self.parameters, "is_append_current_pos_to_short_refs_for_topology", False
        ):
            current_pos_all = torch.stack(
                [ag.state.pos for ag in self.world.agents], dim=1
            )  # [B, N, 2]
            current_pos_all_norm = current_pos_all / self.normalizers.pos_world
            short_term_use = torch.cat(
                [current_pos_all_norm.unsqueeze(2), short_term_all_norm], dim=2
            )  # [B, N, T+1, 2]
        else:
            short_term_use = short_term_all_norm
        # 扩展索引用于按代理维 gather -> [B, K, T, 2]
        T_points = short_term_use.size(2)
        idx_agents = (
            neighbor_idx_local.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_points, 2)
        )
        ref_neighbors_local = torch.gather(
            short_term_use, dim=1, index=idx_agents
        )  # [B, K, T, 2]
        ref_neighbors_local_flat = ref_neighbors_local.reshape(B, K * T_points * 2)

        # 自车短期参考（展平）与现有键保持一致
        ref_local_flat = short_term_use[:, agent_index].reshape(B, -1)

        # 邻居距离与掩码（统一距离屏蔽，与相对特征的屏蔽保持一致）
        dist_i_allj = self.distances.agents[:, agent_index]  # [B, N]
        nearing_agents_distances = torch.gather(
            dist_i_allj, dim=1, index=neighbor_idx_local
        )  # [B, K]
        try:
            if self.parameters.is_partial_observation:
                if self.parameters.is_apply_mask:
                    masked_agents_by_distance = (
                        nearing_agents_distances >= self.thresholds.distance_mask_agents
                    )  # [B, K]
                    if len(self.map.parser.neighboring_lanelets_idx) != 0:
                        masked_agents_by_lanelets = (
                            self.map.determine_masked_agents_by_lanelets(
                                agent_index, neighbor_idx_local
                            )
                        )
                    else:
                        masked_agents_by_lanelets = torch.zeros(
                            (B, K), device=self.world.device, dtype=torch.bool
                        )
                    neighbors_mask_distance = (
                        masked_agents_by_distance | masked_agents_by_lanelets
                    )
                else:
                    neighbors_mask_distance = torch.zeros(
                        (B, K), device=self.world.device, dtype=torch.bool
                    )
            else:
                neighbors_mask_distance = torch.zeros(
                    (B, K), device=self.world.device, dtype=torch.bool
                )
        except Exception:
            neighbors_mask_distance = torch.zeros(
                (B, K), device=self.world.device, dtype=torch.bool
            )

        # --- Topology-specific neighbor observations (K_topo can be larger than K) ---
        try:
            K_topo = int(self.parameters.n_topology_nearing_agents_observed)
        except Exception:
            K_topo = self.parameters.n_nearing_agents_observed

        topology_neighbors_observation_flat = None
        topology_relative_features = None
        topology_ref_neighbors_local_flat = None
        topology_neighbors_distance = None
        topology_neighbors_mask_distance = None
        topo_neighbor_idx = None

        if K_topo is not None and K_topo > 0 and K_topo != K:
            # 约束 K_topo 不超过可用邻居数量（排除自车）
            try:
                K_topo = int(K_topo)
            except Exception:
                pass
            K_topo = max(1, min(int(K_topo), int(self.n_agents) - 1))

            # 选择 K_topo 个最近邻居，并在选择前排除自车（自车距离为 0）
            # 这样后续调试打印无需再过滤自车，数量与 K_topo 一致
            dist_row = self.distances.agents[:, agent_index]  # [B, n_agents]
            # 排除自车条目：将自车对应列置为 +inf，使其不会被 topk 选中
            dist_row_excl = dist_row.clone()
            dist_row_excl[:, agent_index] = torch.tensor(
                float("inf"), device=self.world.device, dtype=dist_row_excl.dtype
            )

            topology_neighbors_distance, topo_neighbor_idx = torch.topk(
                dist_row_excl, k=K_topo, largest=False
            )  # [B, K_topo]

            # Build distance & lanelet masks, consistent with scenario logic
            if self.parameters.is_partial_observation:
                if self.parameters.is_apply_mask:
                    masked_topo_by_distance = (
                        topology_neighbors_distance
                        >= self.thresholds.distance_mask_agents
                    )
                    if len(self.map.parser.neighboring_lanelets_idx) != 0:
                        try:
                            masked_topo_by_lanelets = (
                                self.map.determine_masked_agents_by_lanelets(
                                    agent_index, topo_neighbor_idx
                                )
                            )
                        except Exception:
                            masked_topo_by_lanelets = torch.zeros(
                                (B, K_topo), device=self.world.device, dtype=torch.bool
                            )
                    else:
                        masked_topo_by_lanelets = torch.zeros(
                            (B, K_topo), device=self.world.device, dtype=torch.bool
                        )
                    topology_neighbors_mask_distance = (
                        masked_topo_by_distance | masked_topo_by_lanelets
                    )
                else:
                    topology_neighbors_mask_distance = torch.zeros(
                        (B, K_topo), device=self.world.device, dtype=torch.bool
                    )
            else:
                topology_neighbors_mask_distance = torch.zeros(
                    (B, K_topo), device=self.world.device, dtype=torch.bool
                )

            # Indexing tuple for ego-view gathers
            indexing_tuple_topo = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.parameters.is_ego_view else ())
                + (topo_neighbor_idx,)
            )

            # Gather per-neighbor features for topology
            obs_pos_topo = self.observations.past_pos.get_latest()[indexing_tuple_topo]
            obs_rot_topo = self.observations.past_rot.get_latest()[indexing_tuple_topo]
            obs_vel_topo = self.observations.past_vel.get_latest()[indexing_tuple_topo]
            obs_ref_path_topo = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_topo
                ]
            )
            obs_vertices_topo = self.observations.past_vertices.get_latest()[
                indexing_tuple_topo
            ]
            obs_lengths_topo = self.observations.past_lengths.get_latest()[
                self.constants.env_idx_broadcasting, topo_neighbor_idx
            ]
            obs_widths_topo = self.observations.past_widths.get_latest()[
                self.constants.env_idx_broadcasting, topo_neighbor_idx
            ]
            obs_distance_topo = self.observations.past_distance_to_agents.get_latest()[
                self.constants.env_idx_broadcasting, agent_index, topo_neighbor_idx
            ]

            # Apply masks
            obs_pos_topo[topology_neighbors_mask_distance] = self.constants.mask_one
            obs_rot_topo[topology_neighbors_mask_distance] = self.constants.mask_zero
            obs_vel_topo[topology_neighbors_mask_distance] = self.constants.mask_zero
            obs_ref_path_topo[
                topology_neighbors_mask_distance
            ] = self.constants.mask_one
            obs_vertices_topo[
                topology_neighbors_mask_distance
            ] = self.constants.mask_one
            obs_distance_topo[
                topology_neighbors_mask_distance
            ] = self.constants.mask_one

            # Flatten per neighbor
            obs_pos_topo_flat = obs_pos_topo.reshape(B, K_topo, -1)
            obs_rot_topo_flat = obs_rot_topo.reshape(B, K_topo, -1)
            obs_vel_topo_flat = obs_vel_topo.reshape(B, K_topo, -1)
            obs_ref_path_topo_flat = obs_ref_path_topo.reshape(B, K_topo, -1)
            obs_vertices_topo_flat = obs_vertices_topo.reshape(B, K_topo, -1)
            obs_distance_topo_flat = obs_distance_topo.reshape(B, K_topo, -1)
            obs_lengths_topo_flat = obs_lengths_topo.reshape(B, K_topo, -1)
            obs_widths_topo_flat = obs_widths_topo.reshape(B, K_topo, -1)

            obs_topo_list = [
                (
                    obs_vertices_topo_flat
                    if self.parameters.is_observe_vertices
                    else torch.cat(
                        [
                            obs_pos_topo_flat,
                            obs_rot_topo_flat,
                            obs_lengths_topo_flat,
                            obs_widths_topo_flat,
                        ],
                        dim=-1,
                    )
                ),
                obs_vel_topo_flat,
                (
                    obs_distance_topo_flat
                    if self.parameters.is_observe_distance_to_agents
                    else None
                ),
                (
                    obs_ref_path_topo_flat
                    if self.parameters.is_observe_ref_path_other_agents
                    else None
                ),
            ]
            obs_topo_list = [o for o in obs_topo_list if o is not None]
            topology_neighbors_observation_flat = torch.cat(
                obs_topo_list, dim=-1
            ).reshape(B, -1)

            # Relative features for topology neighbors
            if self.parameters.is_ego_view:
                past_pos = self.observations.past_pos.get_latest()
                past_rot = self.observations.past_rot.get_latest()
                past_vel = self.observations.past_vel.get_latest()

                pos_i_allj = past_pos[:, agent_index]
                rot_i_allj = past_rot[:, agent_index]
                vel_i_allj = past_vel[:, agent_index]

                idx_for_2 = topo_neighbor_idx.unsqueeze(-1).expand(
                    -1, -1, pos_i_allj.size(-1)
                )
                idx_for_1 = topo_neighbor_idx

                rel_pos_topo = torch.gather(pos_i_allj, dim=1, index=idx_for_2)
                rel_yaw_topo = torch.gather(rot_i_allj, dim=1, index=idx_for_1)
                nei_v_topo = torch.gather(vel_i_allj, dim=1, index=idx_for_2)
                nei_v_forward_topo = nei_v_topo[..., 0]
                ego_v_forward = past_vel[:, agent_index, agent_index, 0]
                d_speed_topo = nei_v_forward_topo - ego_v_forward.unsqueeze(-1)

                rel_pos_topo[topology_neighbors_mask_distance] = self.constants.mask_one
                rel_yaw_topo[
                    topology_neighbors_mask_distance
                ] = self.constants.mask_zero
                d_speed_topo[
                    topology_neighbors_mask_distance
                ] = self.constants.mask_zero

                topology_relative_features = torch.cat(
                    [
                        rel_pos_topo,
                        rel_yaw_topo.unsqueeze(-1),
                        d_speed_topo.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                # Bird-view fallback
                pos_all = self.observations.past_pos.get_latest()[:, :, :, :]
                yaw_all = self.observations.past_rot.get_latest()[:, :, :]
                v_all = self.observations.past_vel.get_latest()[:, :, :, 0]
                pos_i = pos_all[:, :, agent_index, :]
                yaw_i = yaw_all[:, :, agent_index]
                v_i = v_all[:, :, agent_index]
                pos_j = pos_all[:, (slice(None),), topo_neighbor_idx, :].squeeze(1)
                yaw_j = yaw_all[:, (slice(None),), topo_neighbor_idx].squeeze(1)
                v_j = v_all[:, (slice(None),), topo_neighbor_idx].squeeze(1)
                rel_pos_topo = pos_j - pos_i.unsqueeze(1)
                rel_yaw_topo = angle_eliminate_two_pi(yaw_j - yaw_i.unsqueeze(1))
                d_speed_topo = v_j - v_i.unsqueeze(1)
                topology_relative_features = torch.cat(
                    [
                        rel_pos_topo,
                        rel_yaw_topo.unsqueeze(-1),
                        d_speed_topo.unsqueeze(-1),
                    ],
                    dim=-1,
                )

            # Neighbor short-term references for topology
            short_term_all_norm = (
                self.ref_paths_agent_related.short_term / self.normalizers.pos_world
            )
            if getattr(
                self.parameters,
                "is_append_current_pos_to_short_refs_for_topology",
                False,
            ):
                current_pos_all = torch.stack(
                    [ag.state.pos for ag in self.world.agents], dim=1
                )  # [B, N, 2]
                current_pos_all_norm = current_pos_all / self.normalizers.pos_world
                short_term_use = torch.cat(
                    [current_pos_all_norm.unsqueeze(2), short_term_all_norm], dim=2
                )  # [B, N, T+1, 2]
            else:
                short_term_use = short_term_all_norm
            T_points = short_term_use.size(2)
            idx_agents_topo = (
                topo_neighbor_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, T_points, 2)
            )
            ref_neighbors_local_topo = torch.gather(
                short_term_use, dim=1, index=idx_agents_topo
            )
            topology_ref_neighbors_local_flat = ref_neighbors_local_topo.reshape(
                B, K_topo * T_points * 2
            )

        # --- NOD auxiliary branch: stable directed candidates and physical data ---
        # These fields are diagnostics/training targets only.  They are not
        # concatenated to the actor observation in phases 1-3.
        with torch.no_grad():
            nod_positions = torch.stack(
                [world_agent.state.pos for world_agent in self.world.agents], dim=1
            )
            nod_velocities = torch.stack(
                [world_agent.state.vel for world_agent in self.world.agents], dim=1
            )
            nod_yaws = torch.stack(
                [world_agent.state.rot for world_agent in self.world.agents], dim=1
            )
            nod_interaction = build_directed_interactions(
                positions=nod_positions,
                velocities=nod_velocities,
                yaws=nod_yaws,
                short_term_paths=self.ref_paths_agent_related.short_term,
                ego_index=agent_index,
                sensing_range=float(
                    getattr(
                        self.parameters,
                        "nod_sensing_range",
                        self.agent_length * 5,
                    )
                ),
                interaction_distance=float(
                    getattr(
                        self.parameters,
                        "nod_interaction_distance",
                        self.agent_length * 3,
                    )
                ),
                ttc_limit=float(getattr(self.parameters, "nod_ttc_limit", 2.0)),
                conflict_radius=float(
                    getattr(
                        self.parameters,
                        "nod_conflict_radius",
                        self.agent_width,
                    )
                ),
                max_speed=float(self.max_speed),
            )
            nod_neighbor_indices = nod_interaction["neighbor_indices"]
            nod_neighbor_generation = torch.gather(
                self.nod_agent_generation, dim=1, index=nod_neighbor_indices
            )

        info = {
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (
                (agent.action.u[:, 0] / self.normalizers.action_vel)
                if not is_action_empty
                else self.constants.empty_action_vel[:, agent_index]
            ),
            "act_steer": (
                (agent.action.u[:, 1] / self.normalizers.action_steering)
                if not is_action_empty
                else self.constants.empty_action_steering[:, agent_index]
            ),
            "ref": (
                self.ref_paths_agent_related.short_term[:, agent_index]
                / self.normalizers.pos_world
            ).reshape(self.world.batch_dim, -1),
            # 拓扑需要的别名与邻居参考
            "ref_local": ref_local_flat,
            "ref_neighbors_local": ref_neighbors_local_flat,
            "neighbors_distance": nearing_agents_distances,
            "neighbors_mask_distance": neighbors_mask_distance,
            "distance_ref": self.distances.ref_paths[:, agent_index]
            / self.normalizers.distance_ref,
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(
                dim=-1
            )[0]
            / self.normalizers.distance_lanelet,
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(
                dim=-1
            )[0]
            / self.normalizers.distance_lanelet,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
            **(
                {"base_observation": base_obs}
                if self.parameters.is_using_prioritized_marl
                else {}
            ),
            **(
                {"priority_observation": prio_obs}
                if self.parameters.is_using_prioritized_marl
                and self.parameters.prioritization_method.lower() == "marl"
                else {}
            ),
            # New structured keys for topology learning (kept under agents.info)
            "ego_observation": ego_observation,
            # "neighbors_observation": neighbors_observation,  # can be enabled if structured [B,K,D_nei] is required
            "neighbors_observation_flat": neighbors_observation_flat,
            "relative_features": relative_features,
            # Topology-specific keys
            **(
                {
                    "topology_neighbors_observation_flat": topology_neighbors_observation_flat,
                    "topology_relative_features": topology_relative_features,
                    "topology_ref_neighbors_local": topology_ref_neighbors_local_flat,
                    "topology_neighbors_distance": topology_neighbors_distance,
                    "topology_neighbors_mask_distance": topology_neighbors_mask_distance,
                    "topology_neighbors_indices": topo_neighbor_idx,
                    # Selected neighbors used by policy when topology selection is ON
                    **(
                        {
                            "topology_selected_indices": self.observations.topology_selected_indices[
                                :, agent_index
                            ],
                            "topology_selected_probs": self.observations.topology_selected_probs[
                                :, agent_index
                            ],
                        }
                        if (
                            hasattr(self.observations, "topology_selected_indices")
                            and hasattr(self.observations, "topology_selected_probs")
                        )
                        else {}
                    ),
                }
                if topology_neighbors_observation_flat is not None
                else {}
            ),
            # 策略分支使用的 K 个近邻编号（用于对比）
            "neighbors_indices": neighbor_idx_local,
            # NOD fields remain separate from the policy observation.
            "nod_pair_features": nod_interaction["features"],
            "nod_edge_mask": nod_interaction["edge_mask"],
            "nod_neighbor_indices": nod_neighbor_indices,
            "nod_ego_generation": self.nod_agent_generation[:, agent_index],
            "nod_neighbor_generation": nod_neighbor_generation,
            "nod_conflict_valid": nod_interaction["conflict_valid"],
            "nod_ttc": nod_interaction["ttc"],
            "nod_eta_gap": nod_interaction["eta_gap"],
            "nod_overlap_risk": nod_interaction["overlap_risk"],
            "nod_world_pos": agent.state.pos,
            "nod_world_vel": agent.state.vel,
        }

        return info

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0  # At step 0, skip pause
            else:
                # Allow slowing down visual rendering without affecting physics
                pause_scale = float(getattr(self.parameters, "render_pause_scale", 1.0))
                pause_duration = self.world.dt * pause_scale - (
                    time.time() - self.timer.render_begin
                )
            if pause_duration > 0:
                time.sleep(pause_duration)
            # print(f"Paused for {pause_duration} sec.")

            self.timer.render_begin = time.time()  # Update
        geoms = []

        is_beautify_map = bool(getattr(self.parameters, "is_beautify_map", False))

        def _polyline_key(v):
            v_np = v.detach().cpu()
            v_rounded = torch.round(v_np * 1000.0) / 1000.0
            a = tuple(map(tuple, v_rounded.tolist()))
            b = tuple(reversed(a))
            return a if a < b else b

        def _append_polyline(v, color, linewidth, close=False):
            geom = rendering.PolyLine(v=v, close=close)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*color)
            geom.set_linewidth(linewidth)
            geoms.append(geom)

        def _append_dashed(v, color, linewidth, stride=3):
            n = int(v.shape[0])
            if n < 2:
                return
            stride = max(2, int(stride))
            for k in range(0, n - 1, stride):
                _append_polyline(v[k : k + 2], color, linewidth, close=False)

        def _densify_trajectory(points: torch.Tensor) -> torch.Tensor:
            interp_points_per_seg = int(
                getattr(
                    self.parameters, "agent_trajectory_interp_points_per_segment", 1
                )
            )
            if interp_points_per_seg <= 1 or int(points.shape[0]) < 2:
                return points

            use_catmull_rom = bool(
                getattr(
                    self.parameters, "agent_trajectory_interp_use_catmull_rom", True
                )
            )
            ts_full = torch.linspace(
                0.0,
                1.0,
                steps=interp_points_per_seg,
                device=points.device,
                dtype=points.dtype,
            )

            n_pts = int(points.shape[0])
            out = []
            for i in range(n_pts - 1):
                ts = ts_full if i == 0 else ts_full[1:]
                p1 = points[i]
                p2 = points[i + 1]
                if not use_catmull_rom or n_pts < 4:
                    t = ts.unsqueeze(-1)
                    out.append((1.0 - t) * p1 + t * p2)
                    continue

                i0 = 0 if i - 1 < 0 else i - 1
                i3 = n_pts - 1 if i + 2 >= n_pts else i + 2
                p0 = points[i0]
                p3 = points[i3]

                t = ts.unsqueeze(-1)
                t2 = t * t
                t3 = t2 * t
                out.append(
                    0.5
                    * (
                        (2.0 * p1)
                        + (-p0 + p2) * t
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                    )
                )
            return torch.cat(out, dim=0) if out else points

        if is_beautify_map:
            bg = rendering.make_polygon(
                v=torch.tensor(
                    [
                        [-self.world.x_semidim, -self.world.y_semidim],
                        [-self.world.x_semidim, self.world.y_semidim],
                        [self.world.x_semidim, self.world.y_semidim],
                        [self.world.x_semidim, -self.world.y_semidim],
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                filled=True,
                draw_border=False,
            )
            xform = rendering.Transform()
            bg.add_attr(xform)
            bg.set_color(*Color.green10)
            geoms.append(bg)

        drawn_boundaries = set()
        for lanelet in self.map.parser.lanelets_all:
            left = lanelet["left_boundary"]
            right = lanelet["right_boundary"]
            center = lanelet["center_line"]

            if is_beautify_map:
                road_poly = torch.vstack([left, torch.flip(right, dims=[0])])
                road = rendering.make_polygon(
                    v=road_poly,
                    filled=True,
                    draw_border=False,
                )
                xform = rendering.Transform()
                road.add_attr(xform)
                road.set_color(*Color.black25)
                geoms.append(road)

                left_marking = lanelet.get("left_line_marking", None)
                right_marking = lanelet.get("right_line_marking", None)
                center_marking = lanelet.get("center_line_marking", "dashed")

                lk = _polyline_key(left)
                if lk not in drawn_boundaries:
                    if left_marking == "dashed":
                        _append_dashed(left, Color.black50, 1.6, stride=3)
                    else:
                        _append_polyline(left, Color.black50, 1.8, close=False)
                    drawn_boundaries.add(lk)

                rk = _polyline_key(right)
                if rk not in drawn_boundaries:
                    if right_marking == "dashed":
                        _append_dashed(right, Color.black50, 1.6, stride=3)
                    else:
                        _append_polyline(right, Color.black50, 1.8, close=False)
                    drawn_boundaries.add(rk)

                if center is not None and int(center.shape[0]) >= 2:
                    if center_marking == "solid":
                        _append_polyline(center, Color.yellow50, 1.0, close=False)
                    else:
                        _append_dashed(center, Color.yellow50, 1.0, stride=4)
            else:
                _append_polyline(left, Color.black50, 1.6, close=False)
                _append_polyline(right, Color.black50, 1.6, close=False)
                _append_polyline(center, Color.black10, 1.2, close=False)

        if self.parameters.is_visualize_extra_info:
            hight_a = -0.10
            hight_b = -0.20
            hight_c = -0.30

            # Title
            geom = rendering.TextLine(
                text=self.parameters.render_title,
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_a) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Time and time step
            geom = rendering.TextLine(
                text=f"t: {self.timer.step[0]*self.parameters.dt:.2f} sec",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_b) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            geom = rendering.TextLine(
                text=f"n: {self.timer.step[0]}",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_c) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Mean velocity
            # mean_vel = torch.vstack([a.state.vel for a in self.world.agents]).norm(dim=-1).mean()
            # geom = rendering.TextLine(
            #     text=f"Mean velocity: {mean_vel:.2f} m/s",
            #     x=1.68 * self.resolution_factor,
            #     y=(self.world.y_semidim + hight_b) * self.resolution_factor,
            #     font_size=14,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)
            # geoms.append(geom)

            # Mean deviation from lane center line
            # mean_deviation_from_center_line = self.distances.ref_paths[0].mean()
            # geom = rendering.TextLine(
            #     text=f"Mean deviation: {mean_deviation_from_center_line:.2f} m",
            #     x=3.15 * self.resolution_factor,
            #     y=(self.world.y_semidim + hight_b) * self.resolution_factor,
            #     font_size=14,
            # )
            # xform = rendering.Transform()
            # geom.add_attr(xform)
            # geoms.append(geom)

        trajectory_items = []
        for agent_i in range(self.n_agents):
            # # Visualize goal
            # if not self.ref_paths_agent_related.is_loop[env_index, agent_i]:
            #     circle = rendering.make_circle(radius=self.thresholds.reach_goal, filled=True)
            #     xform = rendering.Transform()
            #     circle.add_attr(xform)
            #     xform.set_translation(
            #         self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 0],
            #         self.ref_paths_agent_related.long_term[env_index, agent_i, -1, 1]
            #     )
            #     circle.set_color(*colors[agent_i])
            #     geoms.append(circle)

            # Visualize short-term reference paths of agents
            # if self.parameters.is_visualize_short_term_path & (agent_i == 0):
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*colors[agent_i])
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)

            # Visualize only first three future points per agent (independent of full path flag)
            if getattr(self.parameters, "is_visualize_future_three_points", False):
                path = self.ref_paths_agent_related.short_term[env_index, agent_i]
                n_pts = path.shape[0]
                n_show = min(3, n_pts)
                for i in range(n_show):
                    i_p = path[i]
                    circle = rendering.make_circle(radius=0.018, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)

            if getattr(
                self.parameters, "is_visualize_agent_trajectory", False
            ) and hasattr(self, "traj_pos_buffer"):
                buf = self.traj_pos_buffer.buffer
                valid = int(self.traj_pos_buffer.valid_size)
                if valid >= 2:
                    ptr = int(self.traj_pos_buffer.pointer)
                    if valid < int(buf.shape[0]):
                        traj = buf[:valid, env_index, agent_i]
                    else:
                        traj = torch.cat(
                            (
                                buf[ptr:, env_index, agent_i],
                                buf[:ptr, env_index, agent_i],
                            ),
                            dim=0,
                        )

                    n = int(traj.shape[0])
                    if n >= 2:
                        traj = _densify_trajectory(traj)
                        n = int(traj.shape[0])
                        base = colors[agent_i]
                        if is_beautify_map:
                            thickness_m = float(
                                getattr(
                                    self.parameters,
                                    "agent_trajectory_thickness_m_beautify",
                                    4.0,
                                )
                            )
                        else:
                            thickness_m = float(
                                getattr(
                                    self.parameters,
                                    "agent_trajectory_thickness_m",
                                    0.06,
                                )
                            )
                        fade_min = 0.0
                        fade_max = 0.85
                        denom = max(1, n - 1)
                        for seg_i in range(n - 1):
                            age = seg_i / denom
                            fade = fade_max + (fade_min - fade_max) * age
                            col = (
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(base[0]) + (1.0 - float(base[0])) * fade,
                                    ),
                                ),
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(base[1]) + (1.0 - float(base[1])) * fade,
                                    ),
                                ),
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        float(base[2]) + (1.0 - float(base[2])) * fade,
                                    ),
                                ),
                            )
                            p0 = traj[seg_i]
                            p1 = traj[seg_i + 1]
                            d = p1 - p0
                            seg_len = torch.norm(d).item()
                            if seg_len <= 1e-8:
                                continue

                            ux = float(d[0].item()) / seg_len
                            uy = float(d[1].item()) / seg_len
                            nx = -uy
                            ny = ux
                            half = thickness_m * 0.5

                            overlap = half
                            p0x = float(p0[0].item()) - ux * overlap
                            p0y = float(p0[1].item()) - uy * overlap
                            p1x = float(p1[0].item()) + ux * overlap
                            p1y = float(p1[1].item()) + uy * overlap

                            quad = torch.tensor(
                                [
                                    [p0x + nx * half, p0y + ny * half],
                                    [p0x - nx * half, p0y - ny * half],
                                    [p1x - nx * half, p1y - ny * half],
                                    [p1x + nx * half, p1y + ny * half],
                                ],
                                device=self.world.device,
                                dtype=torch.float32,
                            )
                            geom = rendering.make_polygon(
                                v=quad, filled=True, draw_border=False
                            )
                            xform = rendering.Transform()
                            geom.add_attr(xform)
                            geom.set_color(*col)
                            trajectory_items.append((float(age), geom))

                            if seg_i == 0:
                                cap0 = rendering.make_circle(radius=half, filled=True)
                                xform0 = rendering.Transform()
                                cap0.add_attr(xform0)
                                xform0.set_translation(
                                    float(p0[0].item()), float(p0[1].item())
                                )
                                cap0.set_color(*col)
                                trajectory_items.append((0.0, cap0))

                            cap1 = rendering.make_circle(radius=half, filled=True)
                            xform1 = rendering.Transform()
                            cap1.add_attr(xform1)
                            xform1.set_translation(
                                float(p1[0].item()), float(p1[1].item())
                            )
                            cap1.set_color(*col)
                            age_end = (seg_i + 1) / denom
                            trajectory_items.append((float(age_end), cap1))

            # Visualize nearing points on boundaries
            if not self.parameters.is_observe_distance_to_boundaries:
                # Left boundary
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_left_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*colors[agent_i])
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)

                # Right boundary
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_right_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*colors[agent_i])
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*colors[agent_i])
                    geoms.append(circle)

            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    # Left boundary
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.left_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Right boundary
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.right_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Entry
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.entry[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)
                    # Exit
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.exit[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*colors[agent_i])
                    geoms.append(geom)

        if trajectory_items:
            trajectory_items.sort(key=lambda item: item[0])
            geoms.extend([geom for _, geom in trajectory_items])

        # Visualize observed neighbors of a fixed agent by connecting line segments
        if (
            getattr(self.parameters, "is_visualize_observed_neighbors", False)
            and self.parameters.is_partial_observation
        ):
            # Use a fixed agent index if provided; default to 0 to avoid jitter
            ego_i_param = getattr(
                self.parameters, "visualize_observed_neighbors_agent_index", None
            )
            if ego_i_param is None:
                ego_i = 0
            else:
                ego_i = int(ego_i_param)
                if ego_i < 0 or ego_i >= self.n_agents:
                    ego_i = max(0, min(self.n_agents - 1, ego_i))

            pos_ego = self.world.agents[ego_i].state.pos[env_index]
            neighbor_idx = self.observations.nearing_agents_indices[:, ego_i]  # [B, K]
            B = self.world.batch_dim
            K = self.parameters.n_nearing_agents_observed

            # Distances for current env only: [K]
            nearing_agents_distances = self.distances.agents[
                env_index, ego_i, neighbor_idx[env_index]
            ]
            masked_by_distance = (
                nearing_agents_distances >= self.thresholds.distance_mask_agents
            )
            # Lanelet-based mask only when apply_mask is enabled and neighbor relations exist
            if (
                self.parameters.is_apply_mask
                and len(self.map.parser.neighboring_lanelets_idx) != 0
            ):
                masked_all_envs = self.map.determine_masked_agents_by_lanelets(
                    ego_i, neighbor_idx
                )  # [B, K]
                masked_by_lanelets = masked_all_envs[env_index]
            else:
                masked_by_lanelets = torch.zeros_like(
                    masked_by_distance, dtype=torch.bool
                )
            masked_agents = masked_by_distance | masked_by_lanelets  # [K]

            # --- Draw distance/lanelet-filtered neighbors (red), slightly offset ---
            valid_idx = neighbor_idx[env_index][~masked_agents]
            for j in valid_idx.tolist():
                pos_j = self.world.agents[int(j)].state.pos[env_index]
                # Perpendicular offset to avoid overlapping with topology lines
                v = pos_j - pos_ego
                v_norm = torch.norm(v)
                if float(v_norm.item()) > 1e-6:
                    n = torch.tensor(
                        [-v[1].item(), v[0].item()],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    n = n / torch.norm(n)
                else:
                    n = torch.tensor(
                        [0.0, 0.0], device=self.world.device, dtype=torch.float32
                    )
                offset = -0.04 * n  # meters

                line = rendering.Line(
                    tuple((pos_ego + offset).tolist()),
                    tuple((pos_j + offset).tolist()),
                    width=2,
                )
                xform = rendering.Transform()
                line.add_attr(xform)
                line.set_color(*Color.red50)
                geoms.append(line)

            # --- Compute topology-based positive labels e_ij and draw (blue), offset the other side ---
            try:
                # Normalize short-term refs and gather ego/neighbors
                short_term_all_norm = (
                    self.ref_paths_agent_related.short_term / self.normalizers.pos_world
                )  # [B,N,T,2]
                if getattr(
                    self.parameters,
                    "is_append_current_pos_to_short_refs_for_topology",
                    False,
                ):
                    current_pos_all = torch.stack(
                        [ag.state.pos for ag in self.world.agents], dim=1
                    )  # [B,N,2]
                    current_pos_all_norm = current_pos_all / self.normalizers.pos_world
                    short_term_use = torch.cat(
                        [current_pos_all_norm.unsqueeze(2), short_term_all_norm], dim=2
                    )  # [B,N,T+1,2]
                else:
                    short_term_use = short_term_all_norm
                T_points = short_term_use.size(2)
                idx_agents = (
                    neighbor_idx.long()
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, T_points, 2)
                )
                ref_neighbors_local = torch.gather(
                    short_term_use, dim=1, index=idx_agents
                )  # [B,K,T,2]
                ref_neighbors_flat = ref_neighbors_local.reshape(B, K * T_points * 2)
                ref_local_flat = short_term_use[:, ego_i].reshape(B, -1)

                # Distances and mask over all envs
                dist_row = self.distances.agents[:, ego_i]  # [B,N]
                neighbors_distance = torch.gather(
                    dist_row, dim=1, index=neighbor_idx.long()
                )  # [B,K]

                if (
                    self.parameters.is_apply_mask
                    and self.parameters.is_partial_observation
                ):
                    masked_by_distance_all = (
                        neighbors_distance >= self.thresholds.distance_mask_agents
                    )
                    if len(self.map.parser.neighboring_lanelets_idx) != 0:
                        masked_by_lanelets_all = (
                            self.map.determine_masked_agents_by_lanelets(
                                ego_i, neighbor_idx.long()
                            )
                        )  # [B,K]
                    else:
                        masked_by_lanelets_all = torch.zeros(
                            (B, K), device=self.world.device, dtype=torch.bool
                        )
                    neighbors_mask_distance = (
                        masked_by_distance_all | masked_by_lanelets_all
                    )  # [B,K]
                else:
                    neighbors_mask_distance = torch.zeros(
                        (B, K), device=self.world.device, dtype=torch.bool
                    )

                # 打印：用于生成拓扑标签的参考轨迹世界坐标（两位小数）
                try:
                    ego_pts_env = short_term_use[env_index, ego_i]  # [T,2] (normalized)
                    nei_ids_env = neighbor_idx.long()[env_index]  # [K]
                    nei_pts_env = ref_neighbors_local[env_index]  # [K,T,2] (normalized)

                    # 反归一化为世界坐标
                    ego_pts_env_world = ego_pts_env * self.normalizers.pos_world
                    nei_pts_env_world = nei_pts_env * self.normalizers.pos_world

                    def fmt_pts_row(t_row):
                        return (
                            "["
                            + ",".join(
                                [
                                    f"({float(t_row[k,0].item()):.2f},{float(t_row[k,1].item()):.2f})"
                                    for k in range(t_row.size(0))
                                ]
                            )
                            + "]"
                        )

                    print(
                        f"[TopoRefs] n={int(self.timer.step[0])}, ego={ego_i}, T={int(T_points)} (world)"
                    )
                    print(f"[TopoRefs] ego refs: {fmt_pts_row(ego_pts_env_world)}")
                    for j in range(int(K)):
                        nid = int(nei_ids_env[j].item())
                        print(
                            f"[TopoRefs] neighbor id={nid}: {fmt_pts_row(nei_pts_env_world[j])}"
                        )
                except Exception:
                    pass

                # Corridor width uses agent width + buffer
                corridor_buffer = float(
                    getattr(self.parameters, "topology_corridor_buffer_m", 0.4)
                )
                e_labels = generate_e_labels_with_corridor(
                    ref_local_flat,
                    ref_neighbors_flat,
                    neighbors_distance,
                    neighbors_mask_distance,
                    float(self.thresholds.distance_mask_agents),
                    int(K),
                    int(T_points),
                    self.normalizers.pos_world,
                    float(self.agent_width),
                    corridor_buffer,
                    use_intersection=True,
                    use_corridor=True,
                    max_time_lag_steps=2,
                )  # [B,K]

                # Positive edges for current env
                pos_edge_mask = e_labels[env_index] > 0.5  # [K]
                topo_idx = neighbor_idx[env_index][pos_edge_mask]

                for j in topo_idx.tolist():
                    pos_j = self.world.agents[int(j)].state.pos[env_index]
                    # Perpendicular offset opposite side
                    v = pos_j - pos_ego
                    v_norm = torch.norm(v)
                    if float(v_norm.item()) > 1e-6:
                        n = torch.tensor(
                            [-v[1].item(), v[0].item()],
                            device=self.world.device,
                            dtype=torch.float32,
                        )
                        n = n / torch.norm(n)
                    else:
                        n = torch.tensor(
                            [0.0, 0.0], device=self.world.device, dtype=torch.float32
                        )
                    offset = 0.04 * n  # meters

                    line = rendering.Line(
                        tuple((pos_ego + offset).tolist()),
                        tuple((pos_j + offset).tolist()),
                        width=3,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*Color.blue75)
                    geoms.append(line)

                # --- Output text overlays and console logs for the two neighbor sets ---
                if getattr(self.parameters, "is_visualize_extra_info", False):
                    # Position overlays below the default info HUD
                    hight_d = -0.40
                    hight_e = -0.50

                    # Dist/Lane neighbors set
                    geom = rendering.TextLine(
                        text=f"Dist/Lane: {','.join(map(str, valid_idx.tolist()))}",
                        x=0.05 * self.resolution_factor,
                        y=(self.world.y_semidim + hight_d) * self.resolution_factor,
                        font_size=14,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*Color.red75)
                    geoms.append(geom)

                    # Topology-positive neighbors set
                    geom = rendering.TextLine(
                        text=f"Topo=1: {','.join(map(str, topo_idx.tolist()))}",
                        x=0.05 * self.resolution_factor,
                        y=(self.world.y_semidim + hight_e) * self.resolution_factor,
                        font_size=14,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*Color.blue75)
                    geoms.append(geom)

                # Console log per time step n
                try:
                    print(
                        f"[Neighbors] n={int(self.timer.step[0])}, t={self.timer.step[0]*self.parameters.dt:.2f}s, "
                        f"ego={ego_i}, dist/lane={valid_idx.tolist()}, topo_pos={topo_idx.tolist()}"
                    )
                except Exception:
                    pass
            except Exception:
                # Fail-safe: skip topology overlay if any error
                pass

        if getattr(self.parameters, "is_visualize_agent_id", True):
            try:
                width = float(self.viewer_size[0])
                height = float(self.viewer_size[1])
                aspect_ratio = width / height if height > 0 else 1.0
                zoom = float(self.viewer_zoom)

                if aspect_ratio < 1:
                    cam_range = torch.tensor(
                        [zoom, zoom / aspect_ratio],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                else:
                    cam_range = torch.tensor(
                        [zoom * aspect_ratio, zoom],
                        device=self.world.device,
                        dtype=torch.float32,
                    )

                all_poses = torch.stack(
                    [agent.state.pos[env_index] for agent in self.world.agents], dim=0
                )
                max_agent_radius = max(
                    [agent.shape.circumscribed_radius() for agent in self.world.agents]
                )
                viewer_size_fit = (
                    torch.stack(
                        [
                            torch.max(
                                torch.abs(all_poses[:, 0] - self.render_origin[0])
                            ),
                            torch.max(
                                torch.abs(all_poses[:, 1] - self.render_origin[1])
                            ),
                        ]
                    )
                    + 2 * max_agent_radius
                )
                viewer_size = torch.maximum(
                    viewer_size_fit / cam_range,
                    torch.tensor(zoom, device=self.world.device, dtype=torch.float32),
                )
                cam_range = cam_range * torch.max(viewer_size)

                left = -cam_range[0] + self.render_origin[0]
                right = cam_range[0] + self.render_origin[0]
                bottom = -cam_range[1] + self.render_origin[1]
                top = cam_range[1] + self.render_origin[1]

                scalex = width / float((right - left).item())
                scaley = height / float((top - bottom).item())
                transx = -float(left.item()) * scalex
                transy = -float(bottom.item()) * scaley

                for agent_i, agent in enumerate(self.world.agents):
                    pos = agent.state.pos[env_index]
                    label_pos = pos + torch.tensor(
                        [0.0, float(self.agent_length) * 0.55],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    px = float(label_pos[0].item()) * scalex + transx
                    py = float(label_pos[1].item()) * scaley + transy
                    geom = rendering.TextLine(
                        text=str(agent_i + 1),
                        x=px,
                        y=py,
                        font_size=14,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*Color.black100)
                    geoms.append(geom)
            except Exception:
                pass

        return geoms


if __name__ == "__main__":
    scenario = ScenarioRoadTraffic()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
        shared_reward=False,
    )
