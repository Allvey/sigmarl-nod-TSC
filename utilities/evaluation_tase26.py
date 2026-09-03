import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation


model_paths = [
    "outputs/TSC/seed8/",
]

num_models = len(model_paths)
x_ticks = [f"$M_{{{idx}}}$" for idx in range(0, num_models)]

fig_sizes = {
    "episode_reward": (3.8, 4.2),
    "collision_rate": (3.5, 2.0),
    "centerline_deviation": (3.5, 2.0),
    "average_speed": (3.5, 2.0),
    "smoothness": (3.5, 2.0),
}

y_limits = {
    "episode_reward": [-1, 8],
    "collision_rate": [0, 3],
    "centerline_deviation": [0, 100],
    "average_speed": [70, 100],
    "smoothness": [0, 100],
}

legends = [
    "Our (tase26)",
]
is_show_different_collisions = True

render_titles = [path.rsplit("/", 2)[-2] for path in model_paths]

video_names = [path.rsplit("/", 2)[-2][0:2] for path in model_paths]

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
        fitst_model_index=0,
        num_agents=n_agents,
        fig_sizes=fig_sizes,
        y_limits=y_limits,
        simulation_steps=120,
        is_show_different_collisions=is_show_different_collisions,
        x_ticks=x_ticks,
        where_to_save_eva_results=f"outputs/TSC/seed8/eva_{i_scenario}",
        where_to_save_logging="outputs/TSC/seed8/log.txt",
        legends=legends,
        render_titles=render_titles,
        num_simulations_per_model=2,
        is_render=False,
        is_save_simulation_video=False,
        is_measure_policy_inference_time=True,
        video_names=video_names,
    )

    evaluator.run_evaluation()
