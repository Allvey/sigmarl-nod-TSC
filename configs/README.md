# Configuration layout

The root directory no longer stores experiment JSON files.

## Current

- `current/config_dgppo_nod_gain1_control_finetune.json`: current full
  NOD-DGPPO model used for testing and the upcoming benchmark.
- `current/config_dgppo_nod_candidate_only.json`: frozen candidate NOD stage
  used to initialize the current control branches.
- `current/config_dgppo_nod_future_candidate_only.json` and
  `current/config_dgppo_nod_future_quantile_candidate.json`: staged future
  mitigation predictor and calibrated lower-quantile candidate.
- `current/config_dgppo_nod_future_fixed_alpha_finetune.json` and
  `current/config_dgppo_nod_future_lower_alpha_finetune.json`: matched control
  branches that share the frozen future NOD and differ only in pair alpha.
- `current/config_dgppo_nod_future_lower_gain0p4_finetune.json`: lower-gain
  follow-up branch, initialized from the same checkpoint as those controls.

## Ablations

- `ablations/config_dgppo_nod_future_evidence_ppo_attention_joint_control.json`:
  shared-A optimization follow-up from the fixed controller `reward7.08`.
  After 20 Safety warmup iterations, jointly train Actor/messages/A for 50
  iterations without an A-only phase. Initial learning rates are respectively
  `1e-5`, `3e-5`, `5e-6`; use 30 PPO epochs and KL limit `0.005`.
  Uses profile `current` to honor explicit settings; profile `original` would
  override the 30-epoch budget to 60. GAE lambda/clip remain explicitly `0.9/0.2`.
  Scored checkpoints store pre-update weights paired with their rollout reward;
  every 5 control batches retain an independently loadable iteration folder.
  Frozen future evidence, shared-A dynamics and message scale are unchanged.
  Outputs: `dgppo_nod_future_evidence_ppo_attention_joint_control/`.
  Training is complete: selected aligned checkpoint `reward7.26` (rollout 37).
  Evaluation is complete: late reward deterioration eased and A stayed near A0,
  but the tested control means do not improve on fixed A. See `EXPERIMENTS.md`.
- `ablations/config_dgppo_nod_future_evidence_ppo_attention_sequence_control.json`:
  original-equation follow-up with supervised future evidence frozen and PPO
  learning one shared A in both self-amplification and evidence fusion. Starts
  from the fixed-weight controller `reward7.08`; reuses 20 Safety warmup + 10
  A-only + joint training, ordered 8-step replay and the same learning rates.
  `nod_evidence_gate_target="attention"` selects this role; existing gate configs
  default to `"evidence"` and keep their independent beta. Unified-A policy
  checkpoints cannot be exchanged with beta checkpoints. Outputs use the new
  `dgppo_nod_future_evidence_ppo_attention_sequence_control/` directory;
  training and one independent three-model assessment are complete. The selected
  `reward7.34` increases vehicle collision time rate versus fixed A in the tested
  protocol; see `EXPERIMENTS.md` for the comparison and limits.
- `ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json`:
  selected paper branch for supervised prediction and PPO evidence assimilation,
  initialized from the trained fixed-weight controller `reward7.08`.
  After 20 Safety Value warmup iterations, train the PPO evidence weight alone
  for 10 iterations, then jointly fine-tune Actor/message layers at `1e-5`.
  Ordered 8-step opinion replay gives the weight short-horizon recurrent credit;
  the GRU, forecast and risk attention stay frozen. The policy checkpoint also
  stores the gate-phase counter. This is a new optimization trial, not a matched
  single-factor ablation of the old gate experiment.
- `ablations/config_dgppo_nod_future_evidence_ppo_gate_control.json`: freezes
  the trained future-evidence NOD and trains a separate bounded evidence-use
  weight with PPO through the actor's differentiable one-step opinion update.
  It starts at beta=A and shares the source candidate, seed and control budget
  with the fixed-weight future-evidence control. Gate weights live in the policy
  checkpoint, while the version-8 NOD sidecar remains the frozen predictor.
- `ablations/config_dgppo_nod_future_evidence_candidate.json` and
  `ablations/config_dgppo_nod_future_evidence_control.json`: future-evidence
  counterpart to the learned error-scale branch. The head predicts future
  mitigation from causal history plus current geometry, supplies evidence to
  the original NOD solve, and uses fixed variance 1. Candidate training uses
  future-evidence BCE plus opinion BCE, with no old NLL or geometric teacher
  loss; the control stage freezes this version-8 NOD. This is separate from
  the older post-filter future-head/lower-bound experiment.
- `ablations/config_dgppo_nod_evidence_error_variance_candidate.json`: independent
  error-scale supervision for the geometric responsibility evidence. The new
  candidate starts its scale at 1, predicts squared error against future labels,
  and blocks opinion-calibration gradients into the scale head. Its wider range
  is not applied to the frozen behavior NOD collecting the trajectories.
- `ablations/config_dgppo_nod_evidence_error_variance_control.json`: frozen
  error-scale NOD control fine-tuning. Loads all four final checkpoint modules
  from that candidate, trains Actor/message aggregator, task Critic and Safety
  Value, and retains gain 1 and opinion deadzone 0.1. The candidate's learned
  variance range and supervision contract are preserved during loading.
- `ablations/config_dgppo_nod_fixed_variance_candidate.json` and
  `ablations/config_dgppo_nod_learned_evidence_candidate.json`: matched
  responsibility-NOD candidates. Both keep the frozen behavior policy and
  replace the learned variance with the same first-rollout median; the second
  candidate additionally learns evidence from the causal geometric teacher.
- `ablations/config_dgppo_nod_fixed_variance_scratch_candidate.json` and
  `ablations/config_dgppo_nod_learned_evidence_scratch_candidate.json`: the
  same objectives with randomly initialized learner NODs. The behavior policy
  stays frozen, and both runs pin the variance measured in the warm-start runs.
- `ablations/config_dgppo_nod_fixed_variance_control.json` and
  `ablations/config_dgppo_nod_learned_evidence_control.json`: corresponding
  frozen-NOD Actor/Safety fine-tunes, each initialized from its own candidate.
- `ablations/config_dgppo_nod_ablation_fixed_alpha.json`: Actor receives the
  online opinion, while the safety barrier uses fixed alpha.
- `ablations/config_dgppo_nod_ablation_neutral_z.json`: the Actor opinion input
  is neutral and the safety barrier uses fixed alpha.

## Benchmarks

- `benchmarks/sota_all_actor.json`: paired-seed all-Actor comparison among the
  current NOD-DGPPO checkpoint, official SigmaRL, and official XP-MARL.
- `benchmarks/future_lower_control.json`: paired comparison of both future-NOD
  control branches, the current NOD-DGPPO model, SigmaRL, and XP-MARL.
- `benchmarks/README.md`: checkpoint preparation, commands, and output layout.

## Archive

`archive/` contains earlier DGPPO experiments, NOD development stages, and the
legacy staged configurations. They are retained for reproducibility and are not
part of the current benchmark.

All paths inside the JSON files were updated to the organized output layout.
