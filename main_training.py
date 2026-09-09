from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters
import argparse

config_file = "config.json"  # Stage 9; use "config_stage8a.json" for Stage 8A.
parser = argparse.ArgumentParser(description="Train with a selected method configuration.")
parser.add_argument("--config", default=config_file, help="Training configuration JSON file")
config_file = parser.parse_args().config
parameters = Parameters.from_json(config_file)
print(f"[INFO] Config: {config_file}; safety mode: {parameters.safety_control_mode}; output: {parameters.where_to_save}")
mappo_cavs(parameters=parameters)
