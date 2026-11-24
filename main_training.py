import argparse
from pathlib import Path
import sys

config_file = "configs/current/config_dgppo_nod_gain1_control_finetune.json"
parser = argparse.ArgumentParser(description="Train with a selected method configuration.")
parser.add_argument("--config", help="Training configuration JSON file")
parser.add_argument("--method", choices=("nod", "sigmarl", "xp_marl"), default="nod",
                    help="SigmaRL/XP-MARL run the isolated official 1.2.0 implementation")
args = parser.parse_args()
original_profiles = {
    Path("configs/baselines/sigmarl_original.json").resolve(): "sigmarl",
    Path("configs/baselines/xp_marl_original.json").resolve(): "xp_marl",
}
profile_method = original_profiles.get(Path(args.config).resolve()) if args.config else None
if profile_method and args.method not in ("nod", profile_method):
    parser.error("--method disagrees with the selected original baseline profile")
method = profile_method or args.method
if method != "nod":
    from baselines.original_sigmarl.runtime import PROJECT_ROOT, launch_worker
    config = Path(args.config or ("configs/baselines/" + method + "_original.json"))
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    launch_worker(["train", "--method", method, "--config", str(config.resolve())])
    sys.exit(0)

from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters

config_file = args.config or config_file
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
