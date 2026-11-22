from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters
import argparse

config_file = "configs/current/config_dgppo_nod_gain1_control_finetune.json"
parser = argparse.ArgumentParser(description="Train with a selected method configuration.")
parser.add_argument("--config", default=config_file, help="Training configuration JSON file")
config_file = parser.parse_args().config
parameters = Parameters.from_json(config_file)
print(f"[INFO] Config: {config_file}; safety mode: {parameters.safety_control_mode}; output: {parameters.where_to_save}")
print(f"[INFO] PPO profile: {parameters.ppo_training_profile}; epochs={parameters.num_epochs}; "
      f"lr={parameters.lr}; GAE lambda={parameters.lmbda}; clip={parameters.clip_epsilon}")
print(f"[INFO] Safety training: {parameters.safety_training_mode}; "
      f"task mode={parameters.dgppo_task_mode}; warmup fits={parameters.safety_barrier_warmup_batches}; "
      f"KL limit={parameters.safety_finetune_target_kl}")
if parameters.safety_control_mode == "dgppo":
    print(f"[INFO] DGPPO alpha: base={parameters.dgppo_alpha}; "
          f"road safe/recovery={parameters.dgppo_road_alpha_safe}/"
          f"{parameters.dgppo_road_alpha_recovery}; opinion gain={parameters.dgppo_alpha_gain}; "
          f"deadzone={parameters.dgppo_opinion_deadzone}")
print(f"[INFO] Actor opinion input: {parameters.nod_actor_opinion_mode}")
if parameters.is_using_nod_opinion:
    print(f"[INFO] NOD evidence: {parameters.nod_evidence_mode}, "
          f"target={parameters.nod_evidence_target}; variance={parameters.nod_variance_mode}, "
          f"fixed value={parameters.nod_fixed_variance_value}; "
          f"supervision={parameters.nod_variance_supervision}")
    print(f"[INFO] Control weight: {parameters.nod_evidence_gate_mode}; "
          f"target={parameters.nod_evidence_gate_target}; "
          f"gate lr={parameters.nod_evidence_gate_lr}; "
          f"max logit shift={parameters.nod_evidence_gate_max_logit_shift}")
    if parameters.nod_evidence_gate_mode == "ppo":
        print(f"[INFO] Gate optimization: ordered replay length={parameters.nod_evidence_gate_sequence_length}; "
              f"gate-only batches after warmup={parameters.nod_evidence_gate_only_iters}; "
              f"joint Actor lr={parameters.lr}; message lr={parameters.nod_message_lr}")
    if parameters.training_checkpoint_preupdate:
        print(f"[INFO] Checkpoints use the scored pre-update policy; "
              f"periodic control-batch interval={parameters.training_checkpoint_interval}")
mappo_cavs(parameters=parameters)
