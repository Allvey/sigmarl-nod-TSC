"""Isolated process entry point; imports only the vendored baseline utilities."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__:
    from .runtime import PROJECT_ROOT, UPSTREAM, validate_method, verify_upstream
else:
    from runtime import PROJECT_ROOT, UPSTREAM, validate_method, verify_upstream


def load_runtime():
    provenance = verify_upstream()
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    trainer = importlib.import_module("utilities.mappo_cavs")
    helper = importlib.import_module("utilities.helper_training")
    for module, relative in ((trainer, "utilities/mappo_cavs.py"),
                             (helper, "utilities/helper_training.py")):
        if Path(module.__file__).resolve() != (UPSTREAM / relative).resolve():
            raise RuntimeError("Original baseline imported a non-upstream module")
    return trainer, helper, provenance


def configure_training(raw, method):
    validate_method(raw, method)
    if raw.get("is_load_model") or raw.get("is_continue_train") or raw.get("is_testing_mode"):
        raise ValueError("Original baseline training profiles must start from scratch")
    output = Path(raw["where_to_save"])
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    configured = dict(raw, where_to_save=str(output.resolve()) + os.sep)
    return configured, output.resolve()


def train(args):
    raw = json.loads(Path(args.config).read_text())
    raw, output = configure_training(raw, args.method)
    trainer, helper, provenance = load_runtime()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*_policy.pth")):
        raise FileExistsError("Fresh training directory already contains policies: " + str(output))
    provenance.update(method=args.method,
                      runtime_backend="xpmarl_1_2_0" if args.method == "xp_marl" else "sigmarl_1_2_0",
                      config=str(Path(args.config).resolve()), parameters=raw,
                      training_seed=0, instrumentation=False)
    (output / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    print("[ORIGINAL]", args.method, provenance["commit"], "output:", output, flush=True)
    trainer.mappo_cavs(parameters=helper.Parameters.from_dict(raw))


class TimedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.latencies_s = []

    def __call__(self, td):
        start = time.perf_counter()
        result = self.policy(td)
        self.latencies_s.append(time.perf_counter() - start)
        return result

    def __getattr__(self, name):
        return getattr(self.policy, name)


def rollout(args):
    data = json.loads(Path(args.metadata).read_text())
    raw = data["parameters"]
    validate_method(raw, args.method)
    trainer, helper, provenance = load_runtime()
    import numpy as np
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    if __package__:
        from .instrumentation import instrument_scenario
    else:
        from instrumentation import instrument_scenario

    # Passive measurements, plus shared environmental assets for the two new
    # maps only. Training above uses byte-identical upstream sources.
    trainer.ScenarioRoadTraffic = instrument_scenario(trainer.ScenarioRoadTraffic, args.scenario)
    p = helper.Parameters.from_dict(raw)
    p.where_to_save = str(Path(args.metadata).resolve().parent) + os.sep
    p.is_load_model = True
    p.is_continue_train = False
    p.is_load_final_model = False
    p.is_testing_mode = True
    p.is_load_out_td = False
    p.is_save_eval_results = False
    p.is_prb = False
    p.is_challenging_initial_state_buffer = False
    p.scenario_type = args.scenario
    p.n_agents = args.num_agents
    p.max_steps = args.steps
    p.num_vmas_envs = args.num_envs
    p.frames_per_batch = args.steps * args.num_envs
    p.is_visualize_short_term_path = False
    p.is_visualize_lane_boundary = False
    p.is_visualize_extra_info = False
    p.is_real_time_rendering = False
    p.is_save_simulation_video = bool(args.video_output)
    if args.observation_noise != "checkpoint":
        p.is_add_noise = args.observation_noise == "on"
    if args.seed is not None:
        # Explicit matched evaluation protocol, applied after upstream import's
        # torch.manual_seed(0). No changes to model/observation/action code.
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
    env, policy, priority, _ = trainer.mappo_cavs(parameters=p)
    if args.seed is not None:
        # check_env_specs inside the untouched trainer sets seed=0. Restore the
        # requested evaluation seed after construction, just before rollout.
        env.set_seed(args.seed)
    if args.method == "xp_marl" and priority is None:
        raise RuntimeError("XP-MARL must load its original learned priority policy")
    wrapped = TimedPolicy(policy) if args.measure_inference_time else policy
    try:
        with torch.no_grad(), set_exploration_type(
            ExplorationType.MODE if args.deterministic else ExplorationType.RANDOM
        ):
            result = env.rollout(
                max_steps=args.steps - 1, policy=wrapped, priority_module=priority,
                auto_cast_to_device=True, break_when_any_done=False,
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
        provenance.update(
            method=args.method,
            runtime_backend=("xpmarl_1_2_0" if args.method == "xp_marl" else "sigmarl_1_2_0"),
            trainer=str(Path(trainer.__file__).resolve()),
            policy_module=str(Path(helper.__file__).resolve()),
            instrumentation="read-only generation, terminal events and world kinematics",
            seed=args.seed, observation_noise=p.is_add_noise,
            deterministic_actions=args.deterministic, num_agents=p.n_agents,
            seed_application="after_model_construction_before_rollout",
            nominal_steps=args.steps, recorded_steps=result.shape[1],
            scene_assets=getattr(trainer.ScenarioRoadTraffic, "_evaluation_scene_assets", None),
        )
        if args.measure_inference_time:
            provenance["policy_latencies_s"] = wrapped.latencies_s
        Path(args.provenance_output).write_text(json.dumps(provenance, indent=2) + "\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--method", choices=("sigmarl", "xp_marl"), required=True)
    training.add_argument("--config", required=True)
    inspection = sub.add_parser("inspect", help="Check actual module origins without creating an environment")
    inspection.add_argument("--output", required=True)
    inspection.add_argument("--scenario", help="Inspect new map assets without creating an environment")
    evaluation = sub.add_parser("rollout")
    evaluation.add_argument("--method", choices=("sigmarl", "xp_marl"), required=True)
    evaluation.add_argument("--metadata", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--provenance-output", required=True)
    evaluation.add_argument("--scenario", required=True)
    evaluation.add_argument("--num-agents", type=int, required=True)
    evaluation.add_argument("--num-envs", type=int, required=True)
    evaluation.add_argument("--steps", type=int, required=True)
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--observation-noise", choices=("checkpoint", "on", "off"), default="off")
    evaluation.add_argument("--deterministic", action="store_true")
    evaluation.add_argument("--measure-inference-time", action="store_true")
    evaluation.add_argument("--render", action="store_true")
    evaluation.add_argument("--video-output")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "inspect":
        trainer, helper, provenance = load_runtime()
        scenario = importlib.import_module("scenarios.road_traffic")
        provenance.update(trainer=trainer.__file__, helper=helper.__file__, scenario=scenario.__file__)
        if args.scenario:
            path = Path(__file__).resolve().parents[1]/"scenario_assets.py"
            spec = importlib.util.spec_from_file_location("inspection_scene_assets", path)
            assets = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(assets)
            assets.install_scene_assets(trainer.ScenarioRoadTraffic, args.scenario)
            provenance.update(assets.inspect_scene(args.scenario))
        Path(args.output).write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        rollout(args)


if __name__ == "__main__":
    main()
