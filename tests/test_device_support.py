import unittest

import torch

from utilities.helper_training import Parameters, resolve_device
from utilities.topology_labels import generate_soft_labels_full_graph


def _soft_label_reference(paths: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Small scalar reference used to protect the vectorized implementation."""
    batch, agents, _, _ = paths.shape
    result = torch.zeros(batch, agents, agents, device=paths.device)
    vectors = paths[:, :, 1] - paths[:, :, 0]
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    for b in range(batch):
        for ego in range(agents):
            c_ego, s_ego = torch.cos(theta[b, ego]), torch.sin(theta[b, ego])
            origin_ego = paths[b, ego, 0]
            ego_delta = paths[b, ego] - origin_ego
            ego_y = -s_ego * ego_delta[:, 0] + c_ego * ego_delta[:, 1]
            for neighbor in range(agents):
                if ego == neighbor:
                    continue
                neighbor_delta = paths[b, neighbor] - origin_ego
                neighbor_y = (
                    -s_ego * neighbor_delta[:, 0]
                    + c_ego * neighbor_delta[:, 1]
                )
                d_ij = (ego_y - neighbor_y).abs().amin()

                c_nei, s_nei = (
                    torch.cos(theta[b, neighbor]),
                    torch.sin(theta[b, neighbor]),
                )
                origin_nei = paths[b, neighbor, 0]
                nei_delta = paths[b, neighbor] - origin_nei
                nei_y = -s_nei * nei_delta[:, 0] + c_nei * nei_delta[:, 1]
                ego_in_nei = paths[b, ego] - origin_nei
                ego_y_in_nei = (
                    -s_nei * ego_in_nei[:, 0] + c_nei * ego_in_nei[:, 1]
                )
                d_ji = (nei_y - ego_y_in_nei).abs().amin()
                result[b, ego, neighbor] = torch.sigmoid((d_ji - d_ij) / sigma)
    return result


class DeviceSupportTests(unittest.TestCase):
    def test_resolve_device_cpu(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_device_matches_runtime(self):
        expected = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.assertEqual(Parameters(device="auto").device, expected)

    def test_vectorized_soft_labels_match_reference(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        generator = torch.Generator().manual_seed(7)
        paths_cpu = torch.randn(3, 4, 4, 2, generator=generator)
        for device in devices:
            with self.subTest(device=device):
                paths = paths_cpu.to(device)
                actual = generate_soft_labels_full_graph(paths, sigma=0.8)
                expected = _soft_label_reference(paths, sigma=0.8)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_vmas_cuda_reset_and_step(self):
        from torchrl.envs.libs.vmas import VmasEnv

        from scenarios.road_traffic import ScenarioRoadTraffic

        parameters = Parameters(
            device="cuda:0",
            frames_per_batch=8,
            n_iters=1,
            max_steps=4,
            scenario_type="CPM_mixed",
            is_apply_mask=True,
            is_add_noise=False,
            is_using_opponent_modeling=True,
            n_nearing_agents_observed=2,
            n_topology_nearing_agents_observed=3,
        )
        scenario = ScenarioRoadTraffic()
        scenario.parameters = parameters
        env = VmasEnv(
            scenario=scenario,
            num_envs=2,
            continuous_actions=True,
            max_steps=parameters.max_steps,
            device=parameters.device,
            n_agents=parameters.n_agents,
        )
        tensordict = env.reset()
        self.assertEqual(tensordict.device.type, "cuda")
        for _ in range(2):
            tensordict = env.rand_step(tensordict)
            self.assertEqual(tensordict.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
