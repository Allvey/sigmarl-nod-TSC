# Documentation index

The project documentation is grouped by whether it describes the selected
method, reusable repository details, or an earlier experiment.

## Current method

- [LEARNING_IN_NOD_TECHNICAL_PLAN.md](current/LEARNING_IN_NOD_TECHNICAL_PLAN.md):
  Paper-method specification: supervised future risk-mitigation evidence,
  risk attention A learned during candidate supervision and frozen during
  control learning, and an independent PPO evidence-assimilation weight beta.
  Defines the extended NOD equation, Actor/message and opinion-adjusted safety
  learning, architecture, losses, staged sequence replay, gradient boundaries,
  theoretical conditions and reproduction configs. Alternatives and measured
  tradeoffs are kept in appendices; later experiments do not automatically
  change the main method.
- [`RESPONSIBILITY_OPINION_TECHNICAL_REPORT.md`](current/RESPONSIBILITY_OPINION_TECHNICAL_REPORT.md):
  Chinese technical report covering the problem definition, counterfactual
  responsibility, network architecture, loss functions, training sequence,
  safety learning, and theoretical limits. Its paper-writing section separates
  measured results from explicitly marked E1/E2 evidence placeholders.
- [`DGPPO_NOD_RESPONSIBILITY.md`](current/DGPPO_NOD_RESPONSIBILITY.md): current
  responsibility-opinion definition, Candidate NOD workflow, gain-1 Safety
  integration, diagnostics, and matched ablations.
- [`RULE_VEHICLES_TESTING.md`](current/RULE_VEHICLES_TESTING.md): mixed Actor and
  rule-vehicle visualization setup and output files.
- [`RULE_VEHICLE_COORDINATION.md`](current/RULE_VEHICLE_COORDINATION.md): current
  centralized coordination used only by rule vehicles during testing.
- [`ORIGINAL_BASELINES.md`](current/ORIGINAL_BASELINES.md): in-project official
  SigmaRL/XP-MARL 1.2.0 implementation, official root-config training with only
  priority toggles differing, provenance and matched five-metric evaluation.
- [`NOD_FROM_SCRATCH.md`](current/NOD_FROM_SCRATCH.md): two-stage joint NOD/control
  training from random initialization, frozen-predictor PPO beta, budget and evaluation commands.
- [`ERROR_SCALE_RETRAIN.md`](current/ERROR_SCALE_RETRAIN.md): repeat the historical
  reward7.38 error-scale branch with its exact training seed and reward7.21 initialization.
- [`ERROR_SCALE_FROM_SCRATCH.md`](current/ERROR_SCALE_FROM_SCRATCH.md): 250 joint
  iterations learning geometric-evidence error scale, then 70 frozen-NOD control
  iterations, with no pretrained dependency or separate candidate stage.
- [`SOTA_BENCHMARK.md`](current/SOTA_BENCHMARK.md): matched all-Actor protocol,
  official SigmaRL and XP-MARL checkpoints, commands, metrics, and outputs.
- [`TSC_BASELINE.md`](current/TSC_BASELINE.md): four-method evaluation with the
  sibling TSC project's seed5 weights, six default scenarios, shared new-map
  assets in isolated runtimes and unified five-metric results.
- [`SCENARIO_VISUALIZATION.md`](current/SCENARIO_VISUALIZATION.md): fixed NOD
  model visualization on additional maps, all-Actor defaults and per-scenario video paths.
- [`ROUNDABOUT_REALISTIC.md`](current/ROUNDABOUT_REALISTIC.md): aerial-inspired
  roundabout vectorized from the approved traffic diagram, editable SVG,
  OSM/boundary/render assets, entry spawning and the `main_testing.py` command.
- [`BOTTLENECK_SCENARIO.md`](current/BOTTLENECK_SCENARIO.md): straight divided
  2-to-1-to-2 bottleneck, analytic OSM/SVG/render/collision assets, 20-vehicle
  visualization, rebuild commands and output paths.
- [`TRAFFIC_SCENARIO_CONCEPTS.md`](current/TRAFFIC_SCENARIO_CONCEPTS.md): proposed
  one-way crossroad, right-turn side-road merge and highway merge sketches,
  designed to reduce close oncoming interactions; existing four evaluation
  scenarios remain unchanged.
- [`FUTURE_RISK_MITIGATION_OPINION_TECHNICAL_REPORT.md`](current/FUTURE_RISK_MITIGATION_OPINION_TECHNICAL_REPORT.md):
  implemented future-prediction extension and its current limitations; this is
  an alternative to the frozen-NOD future-evidence mainline, with a different
  future-head placement and calibrated-lower-bound mechanism.

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

- [LEARNING_IN_NOD_EXPLORATION_RECORD.md](archive/LEARNING_IN_NOD_EXPLORATION_RECORD.md):
  Complete pre-finalization record of evidence/scale, beta and PPO shared-A
  experiments, including joint adaptation and measured limitations. Historical
  recommendations in this archive do not define the current paper method.

`archive/` contains the earlier DGPPO minimal branch, staged NOD plans, fixed and
opinion-alpha steps, PPO profile experiments, safety fine-tuning, road-safety
diagnostics, and single-car comparisons. These files explain old checkpoints
but do not define the current benchmark configuration.

## Papers and data

- `papers/`: locally stored reference PDFs.
- `data/`: extracted comparison and leader-set spreadsheets.
