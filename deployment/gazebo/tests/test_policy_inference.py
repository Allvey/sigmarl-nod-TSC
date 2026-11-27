"""Checkpoint integration: externally measured states, frozen weights, recurrent reset."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchrl"), "Run in the sigmarl environment for checkpoint integration")
class ExternalInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from policy_worker import ExternalPolicy
        cls.model_path = ROOT.parents[1] / "checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control"
        if not cls.model_path.is_dir() or not (ROOT / "generated/ros2/cpm_map.json").is_file():
            raise unittest.SkipTest("Checkpoint and generated local map required")
        cls.meta = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())
        cls.model = ExternalPolicy(cls.model_path, cls.meta, json.loads((ROOT / "generated/ros2/cpm_map.json").read_text()))

    @classmethod
    def tearDownClass(cls):
        cls.model.env.close()

    def request(self, session="test"):
        return {"protocol": 1, "session": session, "seq": 1, "stamp": 1.0,
                "states": {c["name"]: [c["x"], c["y"], c["yaw"], 0, 0, 0] for c in self.meta["cars"]},
                "previous_commands": {c["name"]: [0, 0, 0] for c in self.meta["cars"]}}

    def test_frozen_inference_never_steps_a_predicted_world(self):
        import torch
        model = self.model
        weights = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        original = model.env.step
        model.env.step = lambda *_: (_ for _ in ()).throw(AssertionError("Inference must not advance VMAS physics"))
        try:
            request = self.request("first")
            first = model.infer(request)
            self.assertEqual(first["observation_shape"], [1, 3, 32])
            self.assertEqual(model.manifest["n_agents"], 3)
            self.assertEqual(set(first["observations"]), {"car_0", "car_1", "car_2"})
            self.assertTrue(all(len(values) == 32 for values in first["observations"].values()))
            request["session"] = "reset"
            second = model.infer(request)
            self.assertEqual(first["actions"], second["actions"])
            for i, agent in enumerate(model.scenario.world.agents):
                self.assertAlmostEqual(float(agent.state.pos[0, 0]), self.meta["cars"][i]["x"] / 3, places=6)
            self.assertTrue(all(torch.equal(tensor, model.policy.state_dict()[name]) for name, tensor in weights.items()))
        finally:
            model.env.step = original

    def test_identity_and_nonfinite_states_are_rejected(self):
        packet = self.request()
        packet["states"]["car_0"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.model.infer(packet)
        packet = self.request()
        packet["states"]["unknown"] = packet["states"].pop("car_0")
        with self.assertRaises(ValueError):
            self.model.infer(packet)
        packet = self.request()
        packet["states"]["car_3"] = [0, 0, 0, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.model.infer(packet)

    def test_duplicate_state_does_not_advance_recurrent_history(self):
        packet = self.request("duplicate")
        self.model.infer(packet)
        before = self.model.frames
        with self.assertRaises(ValueError):
            self.model.infer(packet)
        self.assertEqual(self.model.frames, before)

    def test_default_policy_retains_each_cars_maneuver_and_reference(self):
        import torch
        self.assertEqual(self.model.route_indices, [car["route_index"] for car in self.meta["cars"]])
        self.assertEqual([self.model.vehicle_routes[name]["maneuver"] for name in self.model.names],
                         ["left", "right", "straight"])
        self.assertEqual(len({self.model.vehicle_routes[name]["spawn_distance_m"] for name in self.model.names}), 3)
        refs = self.model.scenario.ref_paths_agent_related
        for i, car in enumerate(self.meta["cars"]):
            self.assertEqual(int(refs.path_id[0, i]), car["route_index"])
            expected_end = [[-0.225, -2.5], [-2.5, 0.225], [2.5, -0.225], [0.225, 2.5]][i]
            last = int(refs.n_points_long_term[0, i]) - 1
            torch.testing.assert_close(refs.long_term[0, i, last] * self.model.space_scale,
                                       torch.tensor(expected_end), atol=1e-6, rtol=0)

    def test_nearby_measured_motion_produces_online_nod_context(self):
        packet = self.request("nearby")
        for car, state in packet["states"].items():
            state[0] *= 0.4
            state[1] *= 0.4
            state[3] = 0.1
        self.model.infer(packet)
        packet["seq"], packet["stamp"] = 2, 1.75
        for state in packet["states"].values():
            state[0] += 0.075 * math.cos(state[2])
            state[1] += 0.075 * math.sin(state[2])
        result = self.model.infer(packet)
        self.assertGreater(result["opinion_active_edges"], 0)
        self.assertTrue(math.isfinite(result["max_abs_opinion"]))


if __name__ == "__main__":
    unittest.main()
