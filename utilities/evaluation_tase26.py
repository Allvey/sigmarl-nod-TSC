import argparse
import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation


parser = argparse.ArgumentParser(description="Evaluate pinned baselines and controlled DGPPO experiments.")
parser.add_argument("--comparison", choices=["best574", "value-lr", "lateral-reset"], default="value-lr")
args = parser.parse_args()

if args.comparison == "best574":
    model_paths = ["outputs/dgppo_minimal_v2/", "outputs/dgppo_v2_respawn_training/"]
    expected_checkpoint_names = ["reward6.38", "reward5.74"]
    load_final_models = [False, False]
    refresh_respawn_observations = [False, True]
    legends = ["v2", "respawn fix (reward5.74)"]
    video_names = ["v2", "respawn_reward574"]
    comparison_output = "outputs/dgppo_best574_comparison"
elif args.comparison == "lateral-reset":
    model_paths = ["outputs/dgppo_minimal_v2/", "outputs/dgppo_v2_respawn_training/",
                   "outputs/dgppo_lateral_reset_control/", "outputs/dgppo_lateral_reset/"]
    # Same 50-batch endpoint; do not select checkpoints on these test maps.
    expected_checkpoint_names = ["reward6.38", "reward5.74", "final", "final"]
    load_final_models = [False, False, True, True]
    refresh_respawn_observations = [False, True, True, True]
    legends = ["v2", "initial reward5.74", "centerline reset (final)", "20% lateral reset (final)"]
    video_names = ["v2", "initial", "lateral_reset_control_final", "lateral_reset_final"]
    comparison_output = "outputs/dgppo_lateral_reset_comparison"
else:
    model_paths = ["outputs/dgppo_minimal_v2/", "outputs/dgppo_v2_respawn_training/", "outputs/dgppo_value_lr_control/",
                   "outputs/dgppo_value_lr_low/"]
    # Compare both trials at the same 50-batch endpoint, after safety warmup.
    expected_checkpoint_names = ["reward6.38", "reward5.74", "final", "final"]
    load_final_models = [False, False, True, True]
    refresh_respawn_observations = [False, True, True, True]
    legends = ["v2", "initial reward5.74", "Value LR 1e-3 (final)", "Value LR 3e-4 (final)"]
    video_names = ["v2", "initial", "value_lr_control_final", "value_lr_low_final"]
    comparison_output = "outputs/dgppo_value_lr_final_comparison"

num_models = len(model_paths)
x_ticks = [f"$M_{{{idx}}}$" for idx in range(num_models)]
render_titles = legends

fig_sizes = {
    "episode_reward": (3.8, 4.2),
    "collision_rate": (3.5, 2.0),
    "centerline_deviation": (3.5, 2.0),
    "average_speed": (3.5, 2.0),
    "smoothness": (3.5, 2.0),
}

y_limits = {
    "episode_reward": [-1, 8],
    "collision_rate": [0, 10],
    "centerline_deviation": [0, 100],
    "average_speed": [70, 100],
    "smoothness": [0, 100],
}

is_show_different_collisions = True

scenario_types = [
    "CPM_entire",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
]


for i_scenario in scenario_types:
    print("*****************************************")
    print("*****************************************")
    print(f"[INFO] Scenario: {i_scenario}")
    print("*****************************************")
    print("*****************************************")

    n_agents = SCENARIOS[i_scenario]["n_agents"]

    evaluator = Evaluation(
        scenario_type=i_scenario,
        model_paths=model_paths,
        expected_checkpoint_names=expected_checkpoint_names,
        load_final_models=load_final_models,
        refresh_respawn_observations=refresh_respawn_observations,
        fitst_model_index=0,
        num_agents=n_agents,
        fig_sizes=fig_sizes,
        y_limits=y_limits,
        simulation_steps=1200,
        is_show_different_collisions=is_show_different_collisions,
        x_ticks=x_ticks,
        where_to_save_eva_results=f"{comparison_output}/eva_{i_scenario}",
        where_to_save_logging=f"{comparison_output}/log.txt",
        legends=legends,
        render_titles=render_titles,
        num_simulations_per_model=8,
        is_render=False,
        is_save_simulation_video=False,
        is_measure_policy_inference_time=True,
        video_names=video_names,
    )

    evaluator.run_evaluation()
