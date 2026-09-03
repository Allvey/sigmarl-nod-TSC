import os
import sys
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
input_dir = os.path.join(script_dir, "inputs")
output_dir = os.path.join(script_dir, "outputs")
model_dir = os.path.join(project_root, "checkpoints", "CPM_entire", "TSC-Net")
result_dir = os.path.join(output_dir, "TSC-Net")

from utilities.constants import SCENARIOS
from utilities.evaluation_base import Evaluation


model_paths = [
    model_dir + os.sep,
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

scenario_order = [
    "CPM_entire",
    "intersection_2",
    "on_ramp_1",
    "roundabout_1",
]


def _clear_cached_sim_outputs():
    for rel_path in model_paths:
        model_dir = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.join(project_root, rel_path)
        )
        pattern = os.path.join(model_dir, "*_out_td_*.pth")
        for p in glob.glob(pattern):
            try:
                os.remove(p)
            except OSError:
                pass


def run_evaluation_for_current_config(log_path: str):
    for name in scenario_order:
        print("*****************************************")
        print("*****************************************")
        print(f"[INFO] Scenario: {name}")
        print("*****************************************")
        print("*****************************************")

        n_agents = SCENARIOS[name]["n_agents"]

        evaluator = Evaluation(
            scenario_type=name,
            model_paths=model_paths,
            fitst_model_index=0,
            num_agents=n_agents,
            fig_sizes=fig_sizes,
            y_limits=y_limits,
            simulation_steps=1200,
            is_show_different_collisions=is_show_different_collisions,
            x_ticks=x_ticks,
            where_to_save_eva_results=os.path.join(
                result_dir, f"eva_{name}"
            ),
            where_to_save_logging=log_path,
            legends=legends,
            render_titles=render_titles,
            num_simulations_per_model=32,
            is_render=False,
            is_save_simulation_video=False,
            video_names=video_names,
        )

        evaluator.run_evaluation()


def collect_safety_metrics():
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "log.txt")
    safe_log_path = os.path.join(result_dir, "safe_log.txt")

    _clear_cached_sim_outputs()
    run_evaluation_for_current_config(log_path)

    scenarios = {}
    current_scenario = None
    buffer = []

    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("Scenario: "):
                if current_scenario is not None:
                    scenarios[current_scenario] = buffer
                current_scenario = line.strip().split("Scenario: ", 1)[1]
                buffer = []
            elif current_scenario is not None:
                if line.startswith("[LOG] Agent-agent collision rate"):
                    buffer.append(line.rstrip("\n"))
                elif line.startswith("[LOG] Agent-lanelet collision rate"):
                    buffer.append(line.rstrip("\n"))
                elif line.startswith("[LOG] Total collision rate"):
                    buffer.append(line.rstrip("\n"))
                elif line.startswith("========================================="):
                    scenarios[current_scenario] = buffer
                    current_scenario = None
                    buffer = []

        if current_scenario is not None:
            scenarios[current_scenario] = buffer

    mode = "a" if os.path.exists(safe_log_path) else "w"

    with open(safe_log_path, mode) as out:
        for name in scenario_order:
            lines = scenarios.get(name)
            if not lines:
                continue
            n_agents = SCENARIOS[name]["n_agents"]
            out.write(f"Scenario: {name}\n")
            out.write(f"Num agents: {n_agents}\n")
            for l in lines:
                out.write(l + "\n")
            out.write("=========================================\n")


if __name__ == "__main__":
    collect_safety_metrics()
