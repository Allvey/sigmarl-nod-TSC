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

## Future lower-bound control experiment

The frozen stage-2 future NOD and its calibration are shared by two new
fine-tuning branches. Both start from the same `reward7.18` policy and Safety
Value checkpoint. The only experimental difference is whether the calibrated
future lower bound changes the safe-side pair alpha. The NOD is not trained in
either branch.

```bash
python main_training.py --config configs/current/config_dgppo_nod_future_fixed_alpha_finetune.json
python main_training.py --config configs/current/config_dgppo_nod_future_lower_alpha_finetune.json
```

The lower-bound branch logs `future_lower_applied_count` and the existing
`opinion_advantage_changed_rate`. A nonzero alpha count alone does not show
that the Actor's update changed; check both metrics. After training, use the
same deterministic seed and scenario protocol for both branches:

```bash
python -m scripts.run_sota_benchmark \
  --config configs/benchmarks/future_lower_control.json \
  --output-root outputs/benchmarks/future_lower_control \
  --seeds 101 202 303 --max-steps 200
```

For the final comparison, omit `--seeds` and `--max-steps` to use the five-seed,
1200-step manifest. The benchmark writes collision, speed, stop, and return
summaries below `outputs/benchmarks/future_lower_control/`. This is an
empirical policy comparison, not an execution-time safety certificate.

The manifest also includes `future_lower_final` and `future_fixed_final`, which
evaluate the policies after all fine-tuning iterations. The `future_lower` and
`future_fixed` entries evaluate the reward-selected checkpoints. To add the
final checkpoints to an existing three-seed, 1200-step benchmark without
rerunning its completed methods, use the same output root and protocol:

```bash
python -m scripts.run_sota_benchmark \
  --config configs/benchmarks/future_lower_control.json \
  --output-root outputs/benchmarks/future_lower_control \
  --seeds 101 202 303 --max-steps 1200 --skip-existing
```

Compare `future_lower_final` with `future_fixed_final` separately from the
reward-selected pair; checkpoint selection can otherwise obscure the effect
of later safety updates.

The first final-checkpoint comparison showed fewer vehicle collisions with
full lower-bound alpha (1 versus 2 over three seeds), but its mean speed was
0.652 versus 0.678 m/s for fixed alpha. The follow-up isolates a smaller
alpha response: `dgppo_alpha_gain=0.4` scales the normalized positive lower
bound before the `dgppo_alpha_span=5` multiplier, so the maximum pair alpha
is 12 rather than 15. The training initialization, frozen NOD, calibration,
and other control settings are unchanged. Train this one additional branch:

```bash
python main_training.py --config configs/current/config_dgppo_nod_future_lower_gain0p4_finetune.json
```

Then add its final checkpoint to the existing three-seed evaluation. Completed
results for the other methods are reused:

```bash
python -m scripts.run_sota_benchmark \
  --config configs/benchmarks/future_lower_control.json \
  --output-root outputs/benchmarks/future_lower_control \
  --seeds 101 202 303 --max-steps 1200 --skip-existing
```

The primary paired comparison is `future_lower_gain0p4_final` against
`future_fixed_final`. Judge vehicle collisions and throughput together; do
not select a branch solely from training reward. Use this three-seed run as a
single decision gate: continue to a final five-seed comparison only if the
gain-0.4 branch has no more vehicle collisions than fixed alpha and its mean
speed is at least 98% of fixed alpha. Otherwise retain the fixed-alpha or
established NOD-DGPPO checkpoint rather than repeatedly tuning the gain on
these same three seeds. The gate is an experiment-selection rule, not a
statistical or safety guarantee.

The gain-0.4 run on seeds 101, 202, and 303 produced 1 vehicle collision in
total, versus 2 for fixed alpha, but mean speed was 0.6614 versus 0.6784 m/s
(97.49% of fixed alpha). Its mean return was 61.88 versus 56.14. Thus it
passed the collision condition but missed the predeclared speed condition.
Stop gain tuning on these seeds. Keep the gain-0.4 checkpoint as a documented
safety-throughput tradeoff; use the established NOD-DGPPO checkpoint as the
current primary model until a separately justified method change is evaluated.

## Existing SigmaRL and XP-MARL protocol

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
