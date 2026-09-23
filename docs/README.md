# Documentation index

The project documentation is grouped by whether it describes the selected
method, reusable repository details, or an earlier experiment.

## Current method

- [`DGPPO_NOD_RESPONSIBILITY.md`](current/DGPPO_NOD_RESPONSIBILITY.md): current
  responsibility-opinion definition, Candidate NOD workflow, gain-1 Safety
  integration, diagnostics, and matched ablations.
- [`RULE_VEHICLES_TESTING.md`](current/RULE_VEHICLES_TESTING.md): mixed Actor and
  rule-vehicle visualization setup and output files.
- [`RULE_VEHICLE_COORDINATION.md`](current/RULE_VEHICLE_COORDINATION.md): current
  centralized coordination used only by rule vehicles during testing.

The selected checkpoint, output locations, and common commands are maintained
in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

## Repository reference

- [`MODEL_STRUCTURE.md`](reference/MODEL_STRUCTURE.md): stage9-to-current network
  evolution; its stage9 sections are retained as a historical structure
  snapshot.
- [`NETWORK_STRUCTURE_DETAILS.md`](reference/NETWORK_STRUCTURE_DETAILS.md):
  detailed legacy network dimensions.
- [`OBSERVATION_SPACE_DETAILS.md`](reference/OBSERVATION_SPACE_DETAILS.md):
  observation-space reference and historical fields.
- [`RL_ENVIRONMENT_DESIGN.md`](reference/RL_ENVIRONMENT_DESIGN.md): reward,
  dynamics, maps, and the historical environment design.

## Historical experiments

`archive/` contains the earlier DGPPO minimal branch, staged NOD plans, fixed and
opinion-alpha steps, PPO profile experiments, safety fine-tuning, road-safety
diagnostics, and single-car comparisons. These files explain old checkpoints
but do not define the current benchmark configuration.

## Papers and data

- `papers/`: locally stored reference PDFs.
- `data/`: extracted comparison and leader-set spreadsheets.
