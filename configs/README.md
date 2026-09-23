# Configuration layout

The root directory no longer stores experiment JSON files.

## Current

- `current/config_dgppo_nod_gain1_control_finetune.json`: current full
  NOD-DGPPO model used for testing and the upcoming benchmark.
- `current/config_dgppo_nod_candidate_only.json`: frozen candidate NOD stage
  used to initialize the current control branches.

## Ablations

- `ablations/config_dgppo_nod_ablation_fixed_alpha.json`: Actor receives the
  online opinion, while the safety barrier uses fixed alpha.
- `ablations/config_dgppo_nod_ablation_neutral_z.json`: the Actor opinion input
  is neutral and the safety barrier uses fixed alpha.

## Archive

`archive/` contains earlier DGPPO experiments, NOD development stages, and the
legacy staged configurations. They are retained for reproducibility and are not
part of the current benchmark.

All paths inside the JSON files were updated to the organized output layout.
