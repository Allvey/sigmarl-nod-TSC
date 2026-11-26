"""Collect TSC trajectories in its original project, with passive measurements."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_runtime(project):
    project = Path(project).resolve()
    os.chdir(project)
    sys.path.insert(0, str(project))
    trainer = importlib.import_module("utilities.mappo_cavs")
    helper = importlib.import_module("utilities.helper_training")
    scenario = importlib.import_module("scenarios.road_traffic")
    for module, relative in ((trainer, "utilities/mappo_cavs.py"),
                             (helper, "utilities/helper_training.py"),
                             (scenario, "scenarios/road_traffic.py")):
        if Path(module.__file__).resolve() != project / relative:
            raise RuntimeError("TSC imported a module from another project: " + module.__file__)
    tracked = ["utilities/mappo_cavs.py", "utilities/helper_training.py",
               "utilities/topology_module.py", "utilities/topology_labels.py",
               "utilities/kinematic_bicycle.py", "scenarios/road_traffic.py"]
    tracked += [str(p.relative_to(project)) for p in sorted((project / "assets/maps").glob("*"))
                if p.suffix in {".xml", ".osm"}]
    record = dict(runtime_backend="tsc_external", project=str(project),
                  trainer=trainer.__file__, helper=helper.__file__, scenario=scenario.__file__,
                  source_sha256={p: hashlib.sha256((project / p).read_bytes()).hexdigest()
                                 for p in tracked})
    return trainer, helper, record


def load_measurement_hooks():
    # Reuse the exact passive hooks used by SigmaRL/XP-MARL without importing
    # this project's utilities into the TSC interpreter.
    path = Path(__file__).resolve().parents[1] / "original_sigmarl/instrumentation.py"
    spec = importlib.util.spec_from_file_location("evaluation_measurements", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.instrument_scenario


def validate_checkpoint(metadata, raw):
    if not raw.get("use_topology_neighbor_selection"):
        raise ValueError("TSC evaluation requires its topology-based neighbor selection")
    if not raw.get("is_using_opponent_modeling"):
        raise ValueError("Expected the complete TSC model with its action predictor")
    prefix = str(Path(metadata).resolve())[:-len("_data.json")]
    for suffix in ("policy", "topology", "action_predictor"):
        if not Path(prefix + "_" + suffix + ".pth").is_file():
            raise FileNotFoundError("Missing TSC checkpoint: " + prefix + "_" + suffix + ".pth")
    return prefix


def verify_loaded_modules(scenario, parameters, prefix):
    import torch
    if not getattr(parameters, "is_topology_action_predictor_loaded", False):
        raise RuntimeError("TSC failed to load its action predictor")
    # The original loader catches topology errors. Never silently evaluate a
    # random/partially loaded auxiliary network as the full TSC method.
    for attribute, suffix in (("topology_learner", "topology"),
                              ("topology_action_predictor", "action_predictor")):
        module = getattr(scenario, attribute, None)
        if module is None:
            raise RuntimeError("TSC did not attach " + attribute)
        expected = torch.load(prefix + "_" + suffix + ".pth", map_location="cpu")
        actual = module.state_dict()
        if expected.keys() != actual.keys() or any(
            not torch.equal(actual[k].detach().cpu(), value.cpu())
            for k, value in expected.items()
        ):
            raise RuntimeError("TSC loaded different weights for " + attribute)
        module.eval()


def rollout(args):
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    raw = json.loads(Path(args.metadata).read_text())["parameters"]
    prefix = validate_checkpoint(args.metadata, raw)
    trainer, helper, record = load_runtime(args.project)
    trainer.ScenarioRoadTraffic = load_measurement_hooks()(trainer.ScenarioRoadTraffic, args.scenario)
    p = helper.Parameters.from_dict(raw)
    p.where_to_save = str(Path(args.metadata).resolve().parent) + os.sep
    p.is_load_model, p.is_continue_train = True, False
    p.is_load_final_model, p.is_testing_mode = False, True
    p.is_load_out_td, p.is_save_eval_results = False, False
    p.is_prb, p.is_challenging_initial_state_buffer = False, False
    p.scenario_type, p.n_agents = args.scenario, args.num_agents
    p.max_steps, p.num_vmas_envs = args.steps, args.num_envs
    p.frames_per_batch = args.steps * args.num_envs
    p.is_visualize_short_term_path = p.is_visualize_lane_boundary = False
    p.is_visualize_extra_info = p.is_real_time_rendering = False
    p.is_save_simulation_video = bool(args.video_output)
    if args.observation_noise != "checkpoint":
        p.is_add_noise = args.observation_noise == "on"
    if args.seed is not None:
        p.seed = args.seed
    env, policy, priority, _ = trainer.mappo_cavs(parameters=p)
    try:
        if p.model_name != Path(prefix).name:
            raise RuntimeError("TSC selected another checkpoint than the evaluation metadata")
        verify_loaded_modules(env.base_env.scenario, p, prefix)
        if args.seed is not None:
            env.set_seed(args.seed)
        # Only forward-call latency, consistent with other methods. Does not
        # represent total topology inference + environment stepping latency.
        latencies = []
        import time

        class TimedPolicy:
            def __call__(self, td):
                start = time.perf_counter()
                result = policy(td)
                latencies.append(time.perf_counter() - start)
                return result

            def __getattr__(self, name):
                return getattr(policy, name)

        with torch.no_grad(), set_exploration_type(
            ExplorationType.MODE if args.deterministic else ExplorationType.RANDOM
        ):
            result = env.rollout(
                max_steps=args.steps - 1,
                policy=TimedPolicy() if args.measure_inference_time else policy,
                priority_module=priority, auto_cast_to_device=True,
                break_when_any_done=False,
                is_save_simulation_video=bool(args.video_output),
                callback=(lambda env, _: env.render(
                    mode="rgb_array" if args.video_output else "human",
                    visualize_when_rgb=False,
                )) if args.render or args.video_output else None,
            )
        if args.video_output:
            from vmas.simulator.utils import save_video
            result, frames = result
            video = Path(args.video_output)
            video.parent.mkdir(parents=True, exist_ok=True)
            save_video(str(video.with_suffix("")), frames, fps=1 / p.dt)
        torch.save(result.cpu(), args.output)
        record.update(
            checkpoint=Path(prefix).name,
            checkpoint_sha256={s: hashlib.sha256(Path(prefix + "_" + s + ".pth").read_bytes()).hexdigest()
                               for s in ("policy", "topology", "action_predictor")},
            instrumentation="read-only generation, terminal events and world kinematics",
            seed=args.seed, observation_noise=p.is_add_noise,
            deterministic_actions=args.deterministic, num_agents=p.n_agents,
            dt=p.dt, nominal_steps=args.steps, recorded_steps=result.shape[1],
            seed_application="after_model_construction_before_rollout",
            policy_neighbors=p.n_nearing_agents_observed,
            topology_neighbors=p.n_topology_nearing_agents_observed,
            scene_assets=getattr(trainer.ScenarioRoadTraffic, "_evaluation_scene_assets", None),
        )
        if args.measure_inference_time:
            record["policy_latencies_s"] = latencies
        Path(args.provenance_output).write_text(json.dumps(record, indent=2) + "\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--inspect", action="store_true", help="Inspect imports without creating an environment")
    parser.add_argument("--metadata")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance-output")
    parser.add_argument("--scenario")
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--observation-noise", choices=("checkpoint", "on", "off"), default="off")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--measure-inference-time", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-output")
    args = parser.parse_args()
    if args.inspect:
        _, _, record = load_runtime(args.project)
        if args.scenario:
            path = Path(__file__).resolve().parents[1]/"scenario_assets.py"
            spec = importlib.util.spec_from_file_location("inspection_scene_assets", path)
            assets = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(assets)
            scenario = importlib.import_module("scenarios.road_traffic")
            assets.install_scene_assets(scenario.ScenarioRoadTraffic, args.scenario)
            record.update(assets.inspect_scene(args.scenario))
        Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    else:
        for key in ("metadata", "provenance_output", "scenario", "num_agents", "num_envs", "steps"):
            if getattr(args, key) is None:
                parser.error("Missing --" + key.replace("_", "-"))
        rollout(args)


if __name__ == "__main__":
    main()
