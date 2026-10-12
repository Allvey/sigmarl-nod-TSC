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
