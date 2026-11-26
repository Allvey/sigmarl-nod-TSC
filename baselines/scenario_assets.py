"""Evaluation-only native-map adapter for isolated SigmaRL/XP-MARL/TSC.

No upstream source or policy is replaced. Only the two new maps are registered;
their coordinate frame, entry spawning and physical road contacts match the
current project's assets. This file is loaded by path inside isolated workers.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_SCENARIOS = ("roundabout_realistic", "bottleneck_1")


def scene_asset_record(name):
    if name not in ASSET_SCENARIOS:
        return None
    metadata = ROOT/"assets/maps"/(name+".json")
    scene = json.loads(metadata.read_text())["scene"]
    paths = [metadata] + [ROOT/scene[key] for key in
                         ("map_path", "geometry_asset_path", "render_asset_path")]
    paths += [Path(__file__).resolve(), ROOT/"utilities/road_collision.py"]
    return dict(scenario=name, road_collision_mode=scene["road_collision_mode"],
                lane_collision_margin=scene.get("lane_collision_margin", 0.),
                spawning="asset_entry_slots_with_original_minimum_distance",
                source_root=str(ROOT),
                sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_from_entry_slots(self, env_i, i_agent, is_reset_single_agent,
                           is_use_state_buffer, initial_state, ref_paths_scenario, agents):
    """Match current all-Actor entry sampling, including bounded retries.

    Keep original random route/slot selection, yaw, speed and separation. The
    only new rule is that samples must come from the scene's entry metadata.
    State buffers are disabled by both rollout workers.
    """
    import torch
    if is_use_state_buffer:
        raise ValueError("Shared-asset evaluation requires state-buffer replay to be disabled")
    i_agent = int(i_agent)
    for _ in range(2000):
        path_id = torch.randint(0, len(ref_paths_scenario), (1,)).item()
        path = ref_paths_scenario[path_id]
        indices = path["spawn_point_indices"]
        point_id = indices[torch.randint(0, len(indices), (1,)).item()]
        self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id
        self.ref_paths_agent_related.point_id[env_i, i_agent] = point_id
        agents[i_agent].set_pos(path["center_line"][point_id], batch_index=env_i)
        if not is_reset_single_agent and i_agent == 0:
            break
        count = self.n_agents if is_reset_single_agent else i_agent+1
        positions = torch.stack([a.state.pos[env_i] for a in agents[:count]])
        distances_sq = (positions[i_agent]-positions).square().sum(-1)
        distances_sq[i_agent] = distances_sq.max()+1
        if distances_sq.min() >= self.constants.reset_agent_min_distance**2:
            break
    else:
        raise RuntimeError(f"No collision-free entry spawn for agent {i_agent+1} of {self.n_agents} "
                           f"in {self.parameters.scenario_type} after 2000 attempts; "
                           "expand entry spawn regions or reduce traffic density")
    yaw = path["center_line_yaw"][point_id]
    speed = torch.rand(1, dtype=torch.float32, device=self.world.device)*agents[i_agent].max_speed
    agents[i_agent].set_rot(yaw, batch_index=env_i)
    agents[i_agent].set_vel(torch.hstack((speed*torch.cos(yaw), speed*torch.sin(yaw))), batch_index=env_i)
    return path, path_id


def install_scene_assets(base_class, name):
    if name not in ASSET_SCENARIOS:
        return base_class
    import torch
    constants = importlib.import_module("utilities.constants")
    parser_module = importlib.import_module("utilities.parse_osm")
    manager_module = importlib.import_module("utilities.map_manager")
    scene = json.loads((ROOT/"assets/maps"/(name+".json")).read_text())["scene"]
    for key in ("map_path", "geometry_asset_path", "render_asset_path"):
        scene[key] = str(ROOT/scene[key])
    constants.SCENARIOS[name] = scene
    original_parser = parser_module.ParseOSM
    road_module = _load_file("shared_asset_road_collision", ROOT/"utilities/road_collision.py")

    class AssetParser(original_parser):
        def _parse_map_file(self):
            super()._parse_map_file()
            if self._scenario_type == name:
                # Upstream uses lane_width*1.2; the new assets explicitly
                # specify their origin, shared with collision/artwork JSON.
                shift = scene["map_margin"]-self._width*1.2
                self._nodes = {key: (x+shift, y+shift) for key, (x, y) in self._nodes.items()}

        def _get_reference_paths(self):
            super()._get_reference_paths()
            if self._scenario_type == name:
                if len(self.reference_paths) != len(scene["route_metadata"]):
                    raise ValueError("Shared map route metadata does not match parsed routes")
                for path, item in zip(self.reference_paths, scene["route_metadata"]):
                    start, end = item["spawn_start_index"], item["spawn_end_index"]
                    indices = item.get("spawn_point_indices", list(range(start, end)))
                    if (not 0 < start < end < len(path["center_line"])-1 or not indices
                            or len(set(indices)) != len(indices)
                            or any(type(i) is not int or not start <= i < end for i in indices)):
                        raise ValueError("Invalid shared-map entry spawn points")
                    path["spawn_point_indices"] = list(indices)

    parser_module.ParseOSM = manager_module.ParseOSM = AssetParser

    class SharedAssetScenario(base_class):
        _evaluation_scene_assets = scene_asset_record(name)

        def _init_params(self, batch_dim, device, **kwargs):
            super()._init_params(batch_dim, device, **kwargs)
            # Retain the runtime's observation/threshold defaults, as the
            # current trainer does. Align only explicit map extent fields.
            self.world_x_dim, self.world_y_dim = scene["world_x_dim"], scene["world_y_dim"]
            self.normalizers.pos_world = torch.tensor([self.world_x_dim, self.world_y_dim],
                                                      dtype=torch.float32, device=device)
            self.render_origin = [self.world_x_dim/2, self.world_y_dim/2]
            self.viewer_size, self.viewer_zoom = tuple(scene["viewer_size"]), scene["viewer_zoom"]
            self._asset_road_collision = road_module.RoadBoundaryCollision.from_asset(
                scene["geometry_asset_path"], device=device)

        _reset_init_state = reset_from_entry_slots

        def _update_state_before_rewarding(self, agent, agent_index):
            result = super()._update_state_before_rewarding(agent, agent_index)
            if agent_index == 0:
                # Apply before rewards, terminal snapshots and resets. Merely
                # recomputing a metric afterwards would leave different physics.
                self.collisions.with_lanelets.copy_(self._asset_road_collision(self.vertices))
            return result

    return SharedAssetScenario


def inspect_scene(name):
    """Inspect the installed parser/curbs in a child without an environment."""
    if name not in ASSET_SCENARIOS:
        raise ValueError("Not a shared asset scenario: " + str(name))
    import torch
    parser = importlib.import_module("utilities.map_manager").ParseOSM(name, "cpu")
    constants = importlib.import_module("utilities.constants")
    collision = _load_file("inspect_asset_road_collision", ROOT/"utilities/road_collision.py").RoadBoundaryCollision
    road = collision.from_asset(constants.SCENARIOS[name]["geometry_asset_path"])
    bodies = []
    for route in parser.reference_paths:
        points = route["center_line"]
        yaw = route["center_line_yaw"]
        yaw = torch.cat((yaw, yaw[-1:]))
        tangent = torch.stack((yaw.cos(), yaw.sin()), -1)
        normal = torch.stack((-yaw.sin(), yaw.cos()), -1)
        bodies.append(torch.stack([points+s*.08*tangent+r*.04*normal
                                  for s, r in ((-1,-1),(1,-1),(1,1),(-1,1))], 1))
    return dict(scene_assets=scene_asset_record(name), lanelet_count=len(parser.lanelets_all),
                routes=[dict(center_line=p["center_line"].tolist(),
                             spawn_point_indices=p["spawn_point_indices"])
                        for p in parser.reference_paths],
                nominal_body_road_contacts=int(road(torch.cat(bodies)).sum()))
