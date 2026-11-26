"""Passive evaluation fields, optionally over shared new-map environment assets."""


def instrument_scenario(base_class, scenario_name=None):
    import torch

    if scenario_name in ("roundabout_realistic", "bottleneck_1"):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/"scenario_assets.py"
        spec = importlib.util.spec_from_file_location("evaluation_scene_assets", path)
        assets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assets)
        base_class = assets.install_scene_assets(base_class, scenario_name)

    class MeasuredScenario(base_class):
        def reset_world_at(self, env_index=None, agent_index=None):
            if not hasattr(self, "_evaluation_generation"):
                self._evaluation_generation = torch.zeros(
                    self.world.batch_dim, len(self.world.agents),
                    dtype=torch.long, device=self.world.device,
                )
                self._evaluation_snapshot = None
            envs = slice(None) if env_index is None else env_index
            agents = slice(None) if agent_index is None else agent_index
            self._evaluation_generation[envs, agents] += 1
            if agent_index is None:
                self._evaluation_snapshot = None
            return super().reset_world_at(env_index=env_index, agent_index=agent_index)

        def _evaluation_fields(self):
            return {
                "evaluation_generation": self._evaluation_generation.clone(),
                "evaluation_world_pos": torch.stack([a.state.pos for a in self.world.agents], dim=1).clone(),
                "evaluation_world_vel": torch.stack([a.state.vel for a in self.world.agents], dim=1).clone(),
                "evaluation_world_rot": torch.stack([a.state.rot for a in self.world.agents], dim=1).clone(),
                "testing_route_is_loop": self.ref_paths_agent_related.is_loop.clone(),
                "testing_exit_reached": self.collisions.with_exit_segments.clone(),
                "testing_entry_contact": self.collisions.with_entry_segments.clone(),
                "testing_road_contact": self.collisions.with_lanelets.clone(),
                "evaluation_vehicle_contact": self.collisions.with_agents.any(-1).clone(),
            }

        def done(self):
            # Upstream done() immediately respawns individual cars. Read terminal
            # events/velocities before that happens; do not change its decision.
            self._evaluation_snapshot = self._evaluation_fields()
            return super().done()

        def reward(self, agent):
            value = super().reward(agent)
            if agent is self.world.agents[-1]:
                # VMAS 1.4.1 collects rewards, observations, infos, then done.
                # Refresh before info as well as before done; otherwise the
                # previous step's terminal snapshot would lag by one frame.
                self._evaluation_snapshot = self._evaluation_fields()
            return value

        def info(self, agent):
            info = super().info(agent)
            index = self.world.agents.index(agent)
            snapshot = self._evaluation_snapshot
            if snapshot is None:
                snapshot = self._evaluation_fields()
            for key, value in snapshot.items():
                info[key] = value[:, index].reshape(self.world.batch_dim, -1).clone()
            return info

    return MeasuredScenario
