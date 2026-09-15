from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters
import argparse

config_file = "config.json"  # Alternatives: config_stage7.json, config_stage8a.json, config_stage8b.json.
parser = argparse.ArgumentParser(description="Train with a selected method configuration.")
parser.add_argument("--config", default=config_file, help="Training configuration JSON file")
config_file = parser.parse_args().config
parameters = Parameters.from_json(config_file)
print(f"[INFO] Navigation boundary: {parameters.use_navigation_boundary}; "
      f"fresh training Safety Value: {parameters.training_reset_safety_value}")
print(f"[INFO] Config: {config_file}; safety mode: {parameters.safety_control_mode}; output: {parameters.where_to_save}")
print(f"[INFO] PPO profile: {parameters.ppo_training_profile}; epochs={parameters.num_epochs}; "
      f"lr={parameters.lr}; GAE lambda={parameters.lmbda}; clip={parameters.clip_epsilon}")
print(f"[INFO] Safety training: {parameters.safety_training_mode}; "
      f"task mode={parameters.dgppo_task_mode}; warmup fits={parameters.safety_barrier_warmup_batches}; "
      f"KL limit={parameters.safety_finetune_target_kl}")
mappo_cavs(parameters=parameters)
