#!/usr/bin/env python3
"""Host-side frozen Actor/NOD inference, driven only by externally measured states."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
from utilities.helper_training import SaveData
from utilities.helper_scenario import get_distances_between_agents
from utilities.mappo_cavs import mappo_cavs
from utilities.sota_evaluation import select_best_checkpoint
from utilities.constants import AGENTS
from policy_contract import PROTOCOL, atomic_json, previous_action, receive_packet, send_packet
from local_junction import select_route_indices


class ExternalPolicy:
    def __init__(self, model_path, metadata, geometry, space_scale=3.0, speed_limit=0.2, maneuver="configured"):
        torch.set_num_threads(1)
        self.metadata = metadata
        self.names = [car["name"] for car in metadata["cars"]]
        self.n_agents = len(self.names)
        if metadata.get("scenario") != "CPM_mixed_local_2lane" or self.n_agents not in (3, 4):
            raise ValueError("This adapter requires a three- or four-car local two-lane junction")
        _, data_path, self.prefix = select_best_checkpoint(Path(model_path))
        parameters = SaveData.from_dict(json.loads(data_path.read_text())).parameters
        if not (parameters.is_using_nod_actor and parameters.is_using_nod_opinion and
                parameters.nod_observation_mode == "local_kinematics" and parameters.is_ego_view and
                parameters.is_observe_distance_to_boundaries and not parameters.is_apply_mask):
            raise ValueError("Unsupported checkpoint observation contract")
        self.space_scale = float(space_scale)
        if not math.isclose(metadata["lane_width"] / self.space_scale, 0.15, abs_tol=1e-6):
            raise ValueError("Space scale must map the local lane width to the trained 0.15 m lane")
        self.time_scale = self.space_scale * AGENTS["max_speed"] / speed_limit
        parameters.device = "cpu"
        parameters.scenario_type = "CPM_mixed"
        parameters.n_agents, parameters.num_vmas_envs = self.n_agents, 1
        parameters.max_steps = 128
        parameters.seed = 101
        parameters.cpm_scenario_probabilities = [1.0, 0.0, 0.0]
        parameters.where_to_save = str(Path(model_path).resolve()) + os.sep
        parameters.is_load_model = parameters.is_testing_mode = True
        parameters.is_continue_train = parameters.is_load_final_model = parameters.is_add_noise = False
        parameters.is_using_deadlock_critic = False
        self.env, self.policy, _, self.parameters = mappo_cavs(parameters=parameters)
        if self.parameters.model_name != self.prefix:
            raise ValueError("Factory selected a different checkpoint than its metadata")
        self.policy.eval()
        self.scenario = self.env.scenario
        self.session = None
        self.last_stamp = None
        self.last_seq = None
        self.frames = 0
        routes = geometry["routes"]["intersection"]
        self.route_indices = select_route_indices(metadata["cars"], routes, maneuver)
        self.vehicle_routes = {car["name"]: {"route_index": index, "approach": routes[index]["approach"],
                                           "maneuver": routes[index]["maneuver"],
                                           "spawn_distance_m": car["spawn_distance"],
                                           "spawn_xy_m": [car["x"], car["y"]]}
                               for car, index in zip(metadata["cars"], self.route_indices)}
        # Keep the original normalization and feature ordering, but use actual
        # JetAuto footprints and local route geometry in training spatial units.
        for agent in self.scenario.world.agents:
            agent.shape._length = metadata["vehicle"]["length"] / self.space_scale
            agent.shape._width = metadata["vehicle"]["width"] / self.space_scale
        paths = []
        # Match the native map's typical reference-point spacing, not its indices.
        native = self.scenario.map.parser.reference_paths_intersection[0]["center_line"]
        spacing = float(torch.linalg.vector_norm(torch.diff(native, dim=0), dim=-1).median())
        for index in self.route_indices:
            points = np.array(routes[index]["center_line"], dtype=np.float64) / self.space_scale
            arc = np.r_[0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
            distances = np.linspace(0, arc[-1], max(3, int(math.ceil(arc[-1] / spacing)) + 1))
            line = np.column_stack([np.interp(distances, arc, points[:, axis]) for axis in (0, 1)])
            tangent = np.gradient(line, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) * metadata["lane_width"] / self.space_scale / 2
            vectors = np.diff(line, axis=0)
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            paths.append({"center_line": torch.tensor(line, dtype=torch.float32),
                          "center_line_vec_normalized": torch.tensor(vectors, dtype=torch.float32),
                          "left_boundary_shared": torch.tensor(line + normal, dtype=torch.float32),
                          "right_boundary_shared": torch.tensor(line - normal, dtype=torch.float32), "is_loop": False})
        refs = self.scenario.ref_paths_agent_related
        extra = int(parameters.n_points_short_term * self.scenario.ref_paths_map_related.sample_interval)
        capacity = max(len(path["center_line"]) for path in paths) + extra + 2
        for key in ("long_term", "long_term_vec_normalized", "left_boundary", "right_boundary"):
            setattr(refs, key, torch.zeros(1, self.n_agents, capacity, 2))
        for i, path in enumerate(paths):
            direction = path["center_line_vec_normalized"][-1]
            extension = path["center_line"][-1] + torch.arange(1, extra + 1)[:, None] * spacing * direction
            self.scenario._reset_agent_related_ref_path(0, i, path, 0, extension[None])
            refs.path_id[0, i] = self.route_indices[i]
            refs.scenario_id[0, i] = 1
        files = [data_path] + [Path(model_path) / (self.prefix + "_" + suffix + ".pth")
                               for suffix in ("policy", "nod", "safety_value")]
        self.manifest = {"protocol": PROTOCOL, "checkpoint": self.prefix,
                         "n_agents": self.n_agents,
                         "scene_sha256": hashlib.sha256((ROOT / "generated/ros2/arena_metadata.json").read_bytes()).hexdigest(),
                         "checkpoint_files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                         "state_source": "Gazebo ground_truth", "space_scale": self.space_scale,
                         "time_scale": self.time_scale, "model_dt": parameters.dt,
                         "control_interval_sim_s": parameters.dt * self.time_scale,
                         "route_indices": dict(zip(self.names, self.route_indices)),
                         "vehicle_routes": self.vehicle_routes,
                         "action_reference": "virtual bicycle CG mapped to JetAuto base_footprint",
                         "model_observation_dim": int(self.env.observation_spec[("agents", "observation")].shape[-1]),
                         "footprints_model_m": [metadata["vehicle"]["length"] / self.space_scale,
                                                metadata["vehicle"]["width"] / self.space_scale]}

    def infer(self, request):
        if request.get("protocol") != PROTOCOL or set(request["states"]) != set(self.names):
            raise ValueError("Incorrect state packet or vehicle identities")
        if not isinstance(request["session"], str) or not request["session"]:
            raise ValueError("A nonempty policy session identity is required")
        states = np.array([request["states"][name] for name in self.names], dtype=float)
        commands = np.array([request["previous_commands"][name] for name in self.names], dtype=float)
        stamp = float(request["stamp"])
        if states.shape != (self.n_agents, 6) or commands.shape != (self.n_agents, 3) or not np.isfinite(states).all() or not np.isfinite(commands).all() or not math.isfinite(stamp):
            raise ValueError("Invalid state values: expected x,y,yaw,vx_body,vy_body,wz")
        new_episode = request["session"] != self.session or (self.last_stamp is not None and stamp < self.last_stamp)
        if not isinstance(request["seq"], int) or isinstance(request["seq"], bool) or request["seq"] < 0:
            raise ValueError("Invalid policy request sequence")
        if not new_episode and (request["seq"] <= self.last_seq or stamp <= self.last_stamp):
            raise ValueError("Duplicate or out-of-order policy state")
        if new_episode:
            self.scenario.nod_manager.reset_online_state()
            self.scenario.nod_agent_generation += 1
            for key, value in vars(self.scenario.observations).items():
                if key.startswith("past_") and hasattr(value, "buffer"):
                    value.buffer.zero_()
            self.session = request["session"]
        self.last_stamp = stamp
        self.last_seq = request["seq"]
        with torch.inference_mode():
            for i, (agent, state) in enumerate(zip(self.scenario.world.agents, states)):
                x, y, heading, vx, vy, angular = state
                c, s = math.cos(heading), math.sin(heading)
                agent.set_pos(torch.tensor([[x, y]], dtype=torch.float32) / self.space_scale, batch_index=None)
                agent.set_rot(torch.tensor([[heading]], dtype=torch.float32), batch_index=None)
                agent.set_vel(torch.tensor([[c * vx - s * vy, s * vx + c * vy]], dtype=torch.float32) * self.time_scale / self.space_scale, batch_index=None)
                agent.state.ang_vel[:] = angular * self.time_scale
                agent.action.u = torch.tensor([previous_action(commands[i], self.space_scale, self.time_scale)], dtype=torch.float32)
            for i in range(self.n_agents):
                self.scenario._reset_init_distances_and_short_term_ref_path(0, i, self.scenario.world.agents)
            self.scenario.distances.agents = get_distances_between_agents(self.scenario, distance_type=self.scenario.distances.type, is_set_diagonal=True)
            observations = torch.stack([self.scenario.observation(a) for a in self.scenario.world.agents], dim=1)
            infos = [self.scenario.info(a) for a in self.scenario.world.agents]
            info = TensorDict({key: torch.stack([item[key] for item in infos], dim=1) for key in infos[0]}, batch_size=[1, self.n_agents])
            td = TensorDict({"agents": TensorDict({"observation": observations, "info": info}, batch_size=[1, self.n_agents])}, batch_size=[1])
            if not torch.isfinite(observations).all():
                raise ValueError("Nonfinite policy observation")
            with set_exploration_type(ExplorationType.MODE):
                result = self.policy(td)
            actions = result[("agents", "action")][0]
            if not torch.isfinite(actions).all():
                raise ValueError("Nonfinite policy actions")
            self.frames += 1
            return {"protocol": PROTOCOL, "session": request["session"], "seq": request["seq"], "stamp": stamp,
                    "actions": dict(zip(self.names, actions.tolist())), "checkpoint": self.prefix,
                    "opinion_active_edges": int(result[("agents", "info", "nod_opinion_active")].sum()),
                    "max_abs_opinion": float(result[("agents", "info", "nod_actor_edge_context")][..., -1].abs().max()),
                    "observations": dict(zip(self.names, observations[0].tolist())),
                    "reference_points_model": dict(zip(self.names, self.scenario.ref_paths_agent_related.short_term[0].tolist())),
                    "observation_shape": list(observations.shape), "frames": self.frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--space-scale", type=float, default=3)
    parser.add_argument("--speed-limit", type=float, default=0.2)
    parser.add_argument("--maneuver", choices=("configured", "straight", "left", "right"), default="configured",
                        help="Default keeps each car's configured route; other values override all cars")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())
    geometry = json.loads((ROOT / "generated/ros2/cpm_map.json").read_text())
    model = ExternalPolicy(args.model_path, metadata, geometry, args.space_scale, args.speed_limit, args.maneuver)
    if args.self_test:
        request = {"protocol": PROTOCOL, "session": "self-test", "seq": 0, "stamp": 0,
                   "states": {c["name"]: [c["x"], c["y"], c["yaw"], 0, 0, 0] for c in metadata["cars"]},
                   "previous_commands": {c["name"]: [0, 0, 0] for c in metadata["cars"]}}
        print(json.dumps({"manifest": model.manifest, "result": model.infer(request)}, indent=2))
        return
    path = ROOT / "runtime/ros2/policy.sock"
    path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(1)
        atomic_json(ROOT / "runtime/ros2/policy_manifest.json", model.manifest)
        print("Policy ready: " + str(path), flush=True)
        try:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                while True:
                    request = receive_packet(stream)
                    started = time.monotonic()
                    response = model.infer(request)
                    response["inference_ms"] = (time.monotonic() - started) * 1000
                    send_packet(stream, response)
        except ConnectionError:
            pass
        finally:
            path.unlink(missing_ok=True)
            model.env.close()


if __name__ == "__main__":
    main()
