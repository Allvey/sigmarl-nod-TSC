# Experiment index

## Current benchmark candidate

The model currently selected for comparison with SigmaRL and XP-MARL is:

```text
outputs/current/dgppo_nod_gain1_control_finetune/reward7.18
```

Its configuration is:

```text
configs/current/config_dgppo_nod_gain1_control_finetune.json
```

`main_testing.py`, `utilities/evaluation_tase26.py`, the multi-seed evaluator,
and the fixed-state opinion scan default to this model.

The current method and evaluation notes are indexed in
[`docs/current/`](docs/current/); historical implementation notes are kept in
[`docs/archive/`](docs/archive/).

## Current output layout

```text
outputs/
  current/
    dgppo_nod_gain1_control_finetune/       full model
    ablations/
      dgppo_nod_ablation_fixed_alpha/       online z, fixed alpha
      dgppo_nod_ablation_neutral_z/         neutral Actor z, fixed alpha
  dependencies/
    dgppo_nod_candidate_only/               initialization for current branches
    dgppo_nod_responsibility_finetune/      initialization for candidate NOD
  archive/                                  all earlier experiments and diagnostics
```

No checkpoints or evaluation results were deleted during the reorganization.
Historical directories were moved as complete directories, including videos,
CSVs, JSON summaries, and model weights.

## Common commands

Visualize the current model:

```bash
python main_testing.py
```

Run its five-seed mixed-controller evaluation:

```bash
python -m scripts.run_rule_multiseed_evaluation \
  --model-path outputs/current/dgppo_nod_gain1_control_finetune/ \
  --seeds 101 202 303 404 505 --no-video
```

Run the fixed-state opinion scan:

```bash
python -m scripts.validate_actor_opinion_sensitivity \
  --model-path outputs/current/dgppo_nod_gain1_control_finetune/ \
  --seed 123 --ego 1
```

## SigmaRL and XP-MARL benchmark

The first cross-method benchmark is an all-Actor comparison. This preserves
XP-MARL's learned priority action propagation and avoids coupling it to the
mixed rule-vehicle controller. Every method uses the same scenario, paired
environment seeds, deterministic actions, and no observation noise.

Download and validate the official checkpoints:

```bash
python -m scripts.prepare_sota_checkpoints
```

Run the paper-metric evaluator smoke test first:

```bash
python utilities/evaluation_tase26.py \
  --scenarios intersection_2 --num-simulations 2 --steps 200
```

Run the formal comparison using the same metrics as the ITSC24 and ICRA25
evaluation scripts:

```bash
python utilities/evaluation_tase26.py
```

Run a one-seed supplementary event-statistics check:

```bash
python -m scripts.run_sota_benchmark \
  --seeds 101 --max-steps 200
```

Run the supplementary five-seed event-statistics evaluation:

```bash
python -m scripts.run_sota_benchmark
```

Formal paper metrics are written below `outputs/benchmarks/paper_metrics/`.
The supplementary protocol is stored in
`configs/benchmarks/sota_all_actor.json`, with results below
`outputs/benchmarks/sota_all_actor/`.

Archived configurations and results should only be used when reproducing an
earlier development stage. New benchmark outputs should be placed under a new
`outputs/benchmarks/` directory rather than at the top level of `outputs/`.
