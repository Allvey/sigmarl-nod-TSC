# Experiment index

## Current evaluation selection

`python utilities/evaluation_tase26.py` now compares the fixed NOD error-scale
control model, original SigmaRL, ICRA25 XP-MARL M1 and external TSC seed5.
Exact directories, five metrics and commands are in
[TSC baseline guide](docs/current/TSC_BASELINE.md). TSC remains in sibling
`sigmarl-traffic`; four-method results stay in this project under
`outputs/benchmarks/paper_metrics_with_tsc/seed5/mixed/four_methods/six_scenarios/`.
The default maps are CPM_entire, intersection_2, on_ramp_1, roundabout_1,
roundabout_realistic and bottleneck_1, with shared native assets for the two new maps.
`--without-tsc` retains the fixed three-method evaluation. Model descriptions
below document earlier experiment selections and training branches.

## Current benchmark candidate

### Progressive safety-control candidate (2026-11-26)

Run `python main_training.py --method nod --config configs/current/config_dgppo_nod_progressive_control.json`.
This separate 70-iteration branch initializes Actor, task Critic and Safety Value
from `checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/reward7.38`,
freezes NOD and uses the same seed, `5080868027432654403`.
Results go to `checkpoints/nod_progressive_control/seed5080868027432654403/`.
The fixed benchmark models and evaluation defaults are unchanged.

The opt-in `dgppo_task_mode="progressive"` uses
`k=clip(1-max_valid(relu(C)/(|alpha*V|+max(eps,1e-8)))/scale,0,1)` and
`advantage=k*relu(task)+min(task,0)-weight*penalty`, with
`dgppo_progressive_scale=1`. Larger scale retains more positive task advantage.
Current or successor physical violation (`g>0`) or nonnegative Safety Value
forces `k=0`. Existing identity/validity masks, risk penalty and fallback rules
remain active; this is a training heuristic, not a collision-free guarantee.

`dgppo_opinion_deadzone_mode="continuous"` maps opinions through
`sign(z)*max(|z|-d,0)/(1-d)` before gain and clipping; here `d=0.1`, gain=1,
base alpha=10 and span=5. Training and the visualization overlay share this rule.
The checkpoint barrier contract records both changes and restarts Value warmup.
Legacy configurations default to `gated` and `hard`, preserving old behavior.
No longitudinal smoothness cost or vehicle dynamics change is included in this
first step; those belong to the next comparison after assessing throughput,
collision breakdown, action changes and jerk with the existing evaluator.

The following candidate descriptions record earlier experiments.

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

## Fixed paper method and exploration records

The paper method is specified in
[Learning in NOD technical plan](docs/current/LEARNING_IN_NOD_TECHNICAL_PLAN.md):
supervised future risk-mitigation evidence, A learned during candidate supervision
and frozen during control learning, and an independent PPO beta that assimilates
evidence in the extended NOD equation. Actor/message updates use opinion-adjusted
safety advantages.

Its candidate and preparatory fixed-weight control configurations are
[future evidence candidate](configs/ablations/config_dgppo_nod_future_evidence_candidate.json)
and [frozen-NOD control](configs/ablations/config_dgppo_nod_future_evidence_control.json).
The preparatory control checkpoint prefix is
outputs/current/ablations/dgppo_nod_future_evidence_control/reward7.08.
The selected paper branch uses
[staged sequence PPO beta](configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json),
with the assessed checkpoint prefix
outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/reward7.16.

This method specification is distinct from the repository's default reward7.18
benchmark candidate listed above. No default checkpoint has been switched.
The staged sequence beta section below now corresponds to the paper mainline;
the fixed-weight and PPO shared-A branches remain comparisons. Their full
technical evolution is preserved in
[the exploration archive](docs/archive/LEARNING_IN_NOD_EXPLORATION_RECORD.md).
New experiments update this index and the plan's appendices unless an explicit
method redesign is chosen.

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

## Fixed-variance / learned-evidence ablation

These are separate responsibility-opinion variants; the selected model above
remains unchanged. In both candidates, the frozen behavior NOD and Actor
collect the same seeded trajectories. On the first rollout containing a valid
interaction, the learner computes the median of the behavior NOD's predicted
variance over valid interactions, freezes it, and saves it in the candidate
NOD checkpoint. No test trajectories are used to choose this value. The
learned-evidence candidate additionally trains a bounded evidence head against
the original causal geometric evidence, while the existing NLL and opinion
calibration targets remain unchanged.

The two candidate runs have completed:

```bash
python main_training.py --config configs/ablations/config_dgppo_nod_fixed_variance_candidate.json
python main_training.py --config configs/ablations/config_dgppo_nod_learned_evidence_candidate.json
```

Candidate outputs are under `outputs/current/ablations/dgppo_nod_{fixed_variance,learned_evidence}_candidate/`;
their prepared but untrained controls would use the corresponding `_control/`
directories. Both candidates froze variance at `0.00673811`, essentially the
old model's minimum `exp(-5)`. The fixed-variance candidate's final Brier score
and future-gap correlation (`0.19207`, `0.50788`) match the original candidate
to within approximately `2e-5`. The learned-evidence candidate finished with
evidence MAE `0.518`, sign accuracy `0.784`, NLL `31.54`, and future-gap
correlation `0.31144`. These are candidate-training diagnostics on the frozen
behavior distribution, not policy performance results. They show that replacing
the causal geometric evidence with the current history-only predictor loses
substantial information. Do not use the learned-evidence checkpoint for control
fine-tuning without changing that evidence model and rerunning candidate
training. The fixed-variance control branch is also low priority because its
candidate is almost identical to the established NOD candidate.

### Scratch-initialized candidate check

To isolate whether warm-starting the learner NOD caused the evidence-head
result, rerun only the two candidates with `nod_candidate_init_mode=scratch`:

```bash
python main_training.py --config configs/ablations/config_dgppo_nod_fixed_variance_scratch_candidate.json
python main_training.py --config configs/ablations/config_dgppo_nod_learned_evidence_scratch_candidate.json
```

The Actor, Safety Value, and behavior NOD still load from the original
`reward7.21` checkpoint and remain frozen during collection. The learner NOD
alone starts with random weights. Both runs use the same explicit fixed
variance, `0.006738110911101103`, copied from the previous candidates so that
initialization is the primary changed factor. The outputs are saved under
`outputs/current/ablations/dgppo_nod_{fixed_variance,learned_evidence}_scratch_candidate/`.
Compare final evidence MAE/sign accuracy and future-gap correlation with the
warm-start results above before considering any Actor/Safety fine-tuning. This
small variance still heavily penalizes evidence prediction error; scratch
initialization by itself does not remove that limitation.

Both scratch candidate runs completed for 30 iterations. The fixed-variance
geometric-evidence candidate ended with Brier `0.19167` and future-gap
correlation `0.50831`, essentially matching its warm-start counterpart
(`0.19207`, `0.50788`). The scratch learned-evidence candidate improved its
evidence MAE from `0.736` initially to `0.516`, but ended with sign accuracy
`0.779`, NLL `31.87`, Brier `0.21224`, and future-gap correlation `0.30384`.
Its warm-start counterpart ended at MAE `0.518`, sign accuracy `0.784`, Brier
`0.20098`, and correlation `0.31144`. Scratch initialization did not recover
the lost evidence fidelity or opinion correlation. These are training-rollout
diagnostics under the same frozen behavior policy; the identical reward
(`6.4805`) is not a comparison of new control policies. Do not proceed to
Actor/Safety fine-tuning with either learned-evidence candidate on this basis.

## Independent evidence-error scale candidate

The new branch retains the causal geometric evidence and changes only how its
learned scale is supervised. With the existing future target `e*=2q*-1`, the
scale loss is `0.5*((e_geom-e*)^2/v + log(v) + log(2*pi))`. The two error terms
are detached supervision, not the opinion after assimilating the same evidence.
The scale is detached in the opinion solve so that opinion BCE trains risk
attention while the independent error loss trains the scale/history encoder.
Future-window tails and generation changes are excluded from both supervised
losses. This estimates conditional squared proxy error, including bias; it is
not a calibrated observation-noise variance or safety lower bound.

```bash
python -m pytest tests/test_nod_evidence_error_variance.py tests/test_nod_opinion.py tests/test_nod_behavior_isolation.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_evidence_error_variance_candidate.json
```

The run reuses the original frozen behavior policy/NOD and warm-starts the
learner history/attention, but resets its variance output to 1. The learner
range is `[exp(-5), 4]`, covering squared differences between bounded evidence
values. The behavior filter retains the original bounds from its saved data
JSON. Outputs go to
`outputs/current/ablations/dgppo_nod_evidence_error_variance_candidate/` and
carry a version-7 NOD checkpoint with the supervision/range contract.

Inspect `variance_lower_bound_ratio`, `variance_mean`, `variance_p10/p90`,
`variance_error_correlation`, and the predicted/actual squared errors in
`variance_low_bin_*` and `variance_high_bin_*`. A useful ranking should assign
larger scales to groups with larger actual proxy errors; merely leaving the
lower bound is insufficient. These logged metrics are training diagnostics,
not independent calibration or policy performance. No Actor/Safety fine-tuning
is part of this candidate run, and its frozen behavior reward is not evidence
of a new policy benefit.

The independent error-scale candidate completed 30 iterations. Its final
Brier was `0.15300`, opinion BCE `0.57710`, and future-gap correlation
`0.54456`. To avoid relying on one rollout, the last-five-iteration averages
are compared below with the earlier fixed-small-variance candidate:

| Training diagnostic (last five iterations) | Fixed small variance | Independent error scale |
| --- | ---: | ---: |
| Brier (lower is better) | 0.20470 | 0.15858 |
| Opinion BCE (lower is better) | 0.86587 | 0.59814 |
| Opinion/future-gap correlation | 0.46092 | 0.50827 |

Across those five iterations, neither scale bound was saturated and the
opinion solver failure ratio remained zero. Mean scale/error correlation was
`0.33196`; actual squared error averaged `0.34695` in the predicted low-scale
group and `1.58293` in the high-scale group (approximately `4.56` times larger).
The final scale mean was `0.95674`, with p10/p90 `0.06134`/`1.82963`. This
supports learning a nonconstant error ranking on the training distribution.
It does not establish calibration: in the final low-scale group the predicted
scale was `0.17975` versus actual squared error `0.33269`, while the high-scale
group predicted `1.87359` versus `1.53558` actual error.

These are post-update training-rollout metrics. The comparison changes both
supervision and allowed scale range, so it cannot isolate adaptive scaling
from simply using a larger fixed scale. The next control experiment should
freeze `final_nod.pth` from this candidate and fine-tune Actor/Safety before
assessing collisions and throughput. A matched constant-scale ablation is
needed if claiming that state-dependent scaling itself produces the benefit.

### Frozen error-scale NOD control fine-tuning

The next branch is configured in
`configs/ablations/config_dgppo_nod_evidence_error_variance_control.json`:

```bash
python main_training.py --config configs/ablations/config_dgppo_nod_evidence_error_variance_control.json
```

It loads the candidate's `final_policy`, `final_critic`, `final_safety_value`
and version-7 `final_nod` from the same pinned prefix. The first three modules
were frozen throughout candidate training; the NOD is the newly trained
learner. All NOD weights stay frozen during the 70-iteration control run,
while its history and opinions evolve online. Actor/message aggregator, task
Critic and Safety Value are updated. The existing 20-fit Safety warmup,
gain `1`, deadzone `0.1`, gated advantages and fine-tuning LR `5e-5` remain
unchanged. The new supervision/range contract is loaded directly, without
resetting its scale output or converting it back to the old NOD contract.
The training log should report `NOD weights frozen`; the saved NOD metrics
should show `training_frozen=1`, `optimizer_updates=0`, and `candidate_only=0`.

Outputs are saved separately in
`outputs/current/ablations/dgppo_nod_evidence_error_variance_control/`.
After training, visualize the highest-training-reward intermediate checkpoint:

```bash
python main_testing.py \
  --model-path outputs/current/ablations/dgppo_nod_evidence_error_variance_control/ \
  --env-seed 101 --no-speed-print --no-realtime
```

The printed `[Test output]` directory contains the mixed-controller video,
speed CSV and summary. This video is a qualitative check; the performance
comparison uses all learned-policy vehicles and the established paper metrics:

```bash
python utilities/evaluation_tase26.py \
  --model-paths \
    outputs/current/ablations/dgppo_nod_evidence_error_variance_control/ \
    outputs/current/dgppo_nod_gain1_control_finetune/ \
    'checkpoints/itsc24/M0 (our)/' \
    'checkpoints/icra25/M1 (XP-MARL)/' \
  --legends 'Error-scale NOD-DGPPO' 'Original NOD-DGPPO' 'SigmaRL' 'XP-MARL' \
  --output-root outputs/benchmarks/evidence_error_variance_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

Each method uses its highest-training-reward intermediate checkpoint, matched
environment seed, deterministic actions and disabled observation noise.
Results are under
`outputs/benchmarks/evidence_error_variance_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/`.
Compare vehicle collision rate, average speed and return with the original
NOD-DGPPO before claiming control improvement. This run does not isolate
state-dependent scaling from the changed overall scale; the matched
constant-scale ablation remains a separate follow-up if this control is useful.

The control run completed all 70 iterations and produced `reward7.38_*` and
`final_*` checkpoints. Every saved NOD metric entry reports
`training_frozen=1`, `optimizer_updates=0`, and `candidate_only=0`.
The final batch has active DGPPO barriers and Actor updates, with opinion
alpha spanning `5.91941` to `14.69404`. The last-20-iteration averages are:

| Training diagnostic | Original NOD control | Error-scale NOD control |
| --- | ---: | ---: |
| Mean episode return | 6.06695 | 6.60777 |
| Vehicle collision flag fraction | 0.00006104 | 0.00007935 |
| Road collision flag fraction | 0.00123291 | 0.00087891 |
| Opinion-changed advantage fraction | 0.03057 | 0.01710 |
| Barrier violation fraction | 0.25018 | 0.23063 |

Collision fractions here are means of per-agent, per-step Boolean flags on
stochastic training rollouts, not collision-event counts or the paper
evaluation metric. Returns improved by approximately `8.9%` over these last
20 iterations, but vehicle collision flags did not improve. Opinion gating
changed fewer advantages than in the old branch, consistent with a less
forceful opinion response; this alone does not prove better uncertainty use.
The old branch's maximum logged return includes frozen-Actor warmup, so it
must not be compared with the new `reward7.38` filename as if they were
equivalent selected policies. The independent four-method evaluation below
has now completed; no default model selection has changed.

### Error-scale control: first paper-metric comparison

The four-method run completed on `intersection_2` with 32 parallel
simulations, 1200 requested steps, seed 101, deterministic actions and no
observation noise or rollout-cache reuse. Results are in
[`metrics_summary.json`](outputs/benchmarks/evidence_error_variance_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/metrics_summary.json)
and [`log.txt`](outputs/benchmarks/evidence_error_variance_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/log.txt).

| Metric (mean, percent) | Error-scale NOD | Original NOD | SigmaRL | XP-MARL |
| --- | ---: | ---: | ---: | ---: |
| Vehicle collision rate (lower is better) | 0.22936 | 0.26585 | 0.35186 | 0.67244 |
| Road collision rate (lower is better) | 1.56380 | 1.58726 | 2.04598 | 1.11551 |
| Sum of collision rates (lower is better) | 1.79316 | 1.85311 | 2.39783 | 1.78795 |
| Normalized average speed (higher is better) | 83.27902 | 82.94552 | 84.68028 | 81.54013 |
| Collision-penalized speed (higher is better) | 79.69270 | 79.23930 | 79.88462 | 77.96423 |
| Centerline deviation (lower is better) | 56.07506 | 54.06508 | 57.20815 | 25.63027 |
| Action change (lower is smoother) | 7.24631 | 7.39857 | 6.34687 | 5.86492 |

Selected checkpoints were `reward7.38`, `reward7.18`, `reward6.03`, and
`reward5.71`, respectively. The new branch's mean vehicle collision rate is
13.7% lower than the old NOD branch, 34.8% lower than SigmaRL and 65.9% lower
than XP-MARL. Normalized speed improves by 0.33350 percentage points over old
NOD and by 1.73889 points over XP-MARL, but is 1.40126 points below SigmaRL.
The gain over old NOD is modest; this single evaluation seed with 32
simulations does not establish reproducibility across seeds or statistical
significance. Vehicle collision standard deviations across simulations were
0.15879, 0.14808, 0.14254 and 0.23399 percentage points.

The inherited evaluator counts the fraction of steps with at least one
colliding vehicle in each environment, not unique collision events or
collision-free-episode probability. Its normalized speed averages agents
that remain collision-free throughout the rollout, falling back to all
agents if none remain collision-free; it is not unconditional fleet speed
or throughput. Centerline deviation is normalized by vehicle width, not a
probability. No independent test-return metric is exported by this run.

Keep the new branch as a viable control candidate: vehicle-collision and
speed means improve over old NOD without a large speed loss to the baselines.
It is not uniformly superior: XP-MARL has much smaller centerline deviation
and smoother actions, and slightly lower summed collision rate; SigmaRL is
faster and smoother. This establishes the implemented mechanism and an
initial policy comparison, not the causal contribution of adaptive scaling.
The next priority is the future-evidence counterpart below, returning to the
original learned-evidence versus learned-variance comparison. A constant-scale
geometric-evidence control remains an optional follow-up for isolating adaptive
scaling itself. Merely changing variance at deployment on the trained new Actor
would test distribution sensitivity, rather than provide a matched training
ablation.

## Supervised future evidence with PPO evidence weight

This opt-in branch separates the frozen forecast from its task-dependent use.
It loads the existing future-evidence candidate's full `final` bundle and freezes
its GRU, evidence head, risk attention A and fixed variance 1. A new PPO-owned
MLP (`73 -> 32 -> 1`, 2401 parameters) produces beta in (0,1), modifying only
the evidence term of the KL opinion update. The bounded logit residual starts
at zero, so beta initially matches A. The risk self-amplification term retains
the frozen A. Beta is a control weight, not calibrated prediction confidence.

The policy caches causal inputs and previous z during collection. PPO re-solves
the current opinion step from those detached inputs so its actor objective can
train beta; it never advances the online history during shuffled replay. This
uses one-step truncated recurrent gradients, not a full history reconstruction.
Future labels do not enter the control update. Safety advantages use the
collected z and remain detached; beta learns through the actor's action
probability and entropy, with the existing safety-adjusted advantage. The
predictor's supervised weights remain frozen.

Run the offline checks, then the single new control branch:

```bash
python -m pytest tests/test_nod_evidence_gate.py tests/test_nod_future_evidence.py tests/test_nod_actor.py tests/test_nod_opinion.py tests/test_dgppo_nod.py tests/test_nod_visualization.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_control.json
```

No new candidate training is needed. Outputs are saved to
`outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_control/`.
Gate weights are stored in the policy checkpoint; the NOD version-8 sidecar is
still the frozen forecast/filter parameters. Load the saved configuration and
the complete bundle when evaluating or resuming. Introducing a fresh gate is
allowed only from the explicit initialization prefix; inference loading is
strict, and cannot silently discard a learned gate or initialize it randomly.

After warmup, `nod_metrics_list` records `evidence_gate_ppo_updates`,
`evidence_gate_grad_norm_mean` and `evidence_gate_parameter_delta_l2` to confirm
optimization, plus the collected `evidence_weight_mean/min/max` and
`evidence_weight_risk_delta_abs` for the active interactions. Existing
`training_frozen=1` and `optimizer_updates=0` refer to the predictor's supervised
optimizer, not to this PPO-owned gate. There may be no gate gradients in a batch
with no active interactions. Warmup freezes the actor and gate together.

Optional mixed-controller visualization:

```bash
python main_testing.py --model-path outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_control/ --env-seed 101 --no-speed-print --no-realtime
```

The overlay appends predicted e, learned beta and frozen A for each active pair.
It reads the collected decision snapshot and does not advance the opinion state.
Video and mixed-controller statistics stay under this branch's
`rule_vehicle_visualization/<run-name>/` directory.

For the first performance comparison, reuse the completed fixed-weight future
control with the same candidate, seed and 70-iteration budget:

```bash
python utilities/evaluation_tase26.py \
  --model-paths outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_control/ outputs/current/ablations/dgppo_nod_future_evidence_control/ \
  --legends 'PPO evidence weight' 'Fixed evidence weight' \
  --output-root outputs/benchmarks/future_evidence_ppo_gate_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

The PPO-gate branch has completed 70 iterations and saved `reward6.95` plus all
four `final` checkpoint modules. Every logged predictor entry has
`training_frozen=1` and `optimizer_updates=0`. The gate begins updating at
iteration 21, after 20 warmup iterations, with nonzero gradient and parameter
change in all 50 subsequent iterations (1348 PPO minibatch updates in total).
All recorded scalar NOD and Safety Value diagnostics are finite.

Last-20-iteration training means compare as follows:

| Training diagnostic | PPO evidence weight | Fixed evidence weight |
| --- | ---: | ---: |
| Episode return | 6.26325 | 6.42452 |
| Agent collision flag fraction | 0.00003662 | 0.00006104 |
| Road collision flag fraction | 0.00135498 | 0.00098877 |
| Barrier violation fraction | 0.34016 | 0.24361 |
| Opinion-changed advantage fraction | 0.01513 | 0.01647 |
| PPO minibatch updates per iteration | 56.35 | 63.35 |

The learned gate's mean gradient norm is `2.27e-5`, and mean per-iteration
parameter displacement is `0.001746` over the last 20 iterations. Its collected
mean beta is `0.41417`, but mean `abs(beta-A)` is only `0.001268`. Thus the PPO
gradient path works, while the learned modulation currently remains small.
These values do not establish that beta learned a useful state-dependent
evidence-use strategy. Agent-collision flags improve during training, but return
falls and road-collision flags rise; training diagnostics do not settle the
independent performance ranking.

The independent comparison is now complete at
`outputs/benchmarks/future_evidence_ppo_gate_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/`.
It uses deterministic actions, no observation noise, 32 parallel simulations,
1200 requested steps and evaluation seed 101. The selected checkpoints are
`reward6.95` for the PPO gate and `reward7.08` for the fixed weight, each selected
by training reward within its branch. The fixed branch's complete metric summary
exactly reproduces the preceding future-evidence-versus-variance evaluation.

| Independent metric | PPO evidence weight | Fixed evidence weight |
| --- | ---: | ---: |
| Agent-agent collision rate (%) | 0.28148 | 0.17462 |
| Road collision rate (%) | 1.19892 | 1.66024 |
| Summed collision rate (%) | 1.48040 | 1.83486 |
| Normalized average speed (%) | 83.62932 | 82.71497 |
| Penalized normalized speed (%) | 80.66853 | 79.04525 |
| Centerline deviation (%) | 49.22356 | 54.34002 |
| Action change (%) | 7.16872 | 7.48459 |

The PPO branch improves road collision rate by 27.8% relative, normalized speed
by 0.91435 percentage points, centerline deviation by 9.4% relative and action
change by 4.2% relative. However, agent-agent collision rate increases by 61.2%
relative (0.10686 percentage points). Its lower summed collision rate is driven
by fewer road contacts and does not establish improved vehicle-vehicle safety.
Collision rates count collision time steps, not independent accident events;
the sum may count overlapping vehicle and road contacts twice. The evaluator
preferentially computes speed from vehicles that were collision-free throughout
the rollout, so it is not unconditional fleet throughput. Thirty-two parallel
simulations with one evaluation seed are not thirty-two independent seed
replications; no statistical significance or cross-scenario claim is made.

The mechanism is implemented and its PPO gradient path works, but this first
comparison does not support making learned evidence use the main contribution
for improving vehicle-vehicle avoidance. With that objective prioritized, keep
fixed-weight future evidence as the preferred candidate and retain the PPO gate
as a tradeoff/ablation branch. The small beta-A displacement and jointly trained
Actor do not establish that the gate itself caused the change in behavior or
learned to deliberately accept more risk. Do not automatically increase the gate
gain from this result. This comparison adds the learned gate and the corresponding
one-step replay gradient path; the default project model has not been switched.

## Staged sequence PPO evidence weight

This follow-up addresses two optimization limitations of the first PPO gate:
Actor/message updates compete with the new gate for the same KL budget, and
flat one-step replay cuts credit from later actions to earlier evidence weights.
It initializes all four modules from the trained fixed-weight controller
`outputs/current/ablations/dgppo_nod_future_evidence_control/reward7.08`,
introducing only a fresh gate at beta=A. The old gate branch remains reproducible.

Configuration:
`configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json`.
The 70-iteration schedule is:

| Iterations | Policy optimization | Other training |
| --- | --- | --- |
| 1–20 | Actor, messages and beta frozen | Safety Value warmup |
| 21–30 | beta only; Actor/message parameter gradients cleared | Task Critic and Safety Value continue training |
| 31–70 | beta plus Actor/messages jointly | Task Critic and Safety Value continue training |

The phase counter advances only on rollout batches with an actual PPO update.
The fixed forecast, GRU and risk attention A never receive control gradients or
supervised updates. Actor/message initial LR is `1e-5`, gate LR is `5e-5`, and
task Critic initial LR is `1e-5`; the existing LR schedule still applies.
Target KL remains `0.01`, logit shift bound remains 2, variance remains 1,
alpha gain remains 1 and opinion deadzone remains 0.1.

Replay packs each environment's 128 collected steps into nonoverlapping ordered
8-step windows. Each minibatch has 64 windows (512 frames). Shuffle windows,
never time inside them. Starting opinions and the forecast/history caches are
detached; recomputed opinions and beta inputs form a gradient chain within a
window. Identity/generation changes, inactive interactions, cached state resets
and available terminal flags cut that chain. Replay never advances online NOD
state. This is truncated backpropagation through the opinion state, not through
the frozen GRU or the simulator, and not through detached Safety advantages.

Implementation includes checks for Actor/message parameter isolation during
the gate-only phase. The counter is stored in the policy checkpoint together
with its weights, so selecting an intermediate checkpoint retains its phase
progress. Evaluation/resume requires that checkpoint's matching configuration.

Run the offline checks and training locally:

```bash
python -m pytest tests/test_nod_evidence_gate_sequence.py tests/test_nod_evidence_gate.py tests/test_nod_future_evidence.py tests/test_nod_actor.py tests/test_nod_opinion.py tests/test_dgppo_nod.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_sequence_control.json
```

Outputs go to
`outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/`.
Inspect `nod_metrics_list`: `evidence_gate_only_phase` should be 1 for the first
10 active control batches and 0 afterward; `evidence_gate_body_parameter_delta_l2`
must be zero in that first phase. `evidence_gate_training_batches`,
`evidence_gate_ppo_updates`, `evidence_gate_parameter_delta_l2`, and
`evidence_weight_risk_delta_abs` distinguish actual optimization from a
gate remaining near initialization. `evidence_gate_preupdate_kl` checks numerical
replay consistency before each batch's updates; it should be near zero.

After training, use the existing visualization or one direct performance comparison:

```bash
python main_testing.py --model-path outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/ --env-seed 101 --no-speed-print --no-realtime
python utilities/evaluation_tase26.py \
  --model-paths outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/ outputs/current/ablations/dgppo_nod_future_evidence_control/ \
  --legends 'Staged sequence PPO weight' 'Fixed evidence weight' \
  --output-root outputs/benchmarks/future_evidence_ppo_gate_sequence_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

Status: the user completed all 70 training iterations and the independent
evaluation. The staged phase matches iterations 21–30 with exactly zero
Actor/message parameter displacement; forecast supervision remains frozen.
PPO made 18771 minibatch updates, compared with 1348 in the one-step branch.
The best active-rollout checkpoint is `reward7.16` (iteration 43); later training
returns decline, so the evaluation uses this intermediate checkpoint, not `final`.

Under the command above, vehicle collision time rate is 0.09904% versus
0.17462% for the fixed weight (43.3% relative reduction), normalized speed is
81.70083% versus 82.71497%, road collision rate is 1.63678% versus 1.66024%,
centerline deviation is 56.85823% versus 54.34002%, and action change is
7.84812% versus 7.48459%. The fixed baseline exactly reproduces the previous
evaluation. This supports a vehicle-safety tradeoff in the tested protocol,
with lower speed and worse smoothness, not uniform performance dominance.
The coding agent only read the artifacts and did not execute training or tests.
For the method synthesis and evidence limits, see
[Learning in NOD technical plan](docs/current/LEARNING_IN_NOD_TECHNICAL_PLAN.md).
The new initialization, extra fine-tuning budget, learning rates and replay all differ
from the original trial. This evaluates a practical optimization follow-up,
not a single-factor ablation or a proof that beta alone causes any improvement.
The default project model has not been switched.

## Original-equation PPO attention with frozen future evidence

The new branch retains the original NOD equation: one PPO-owned A multiplies
both `tanh(a*z)` and `(w/v)*(e-z)`. The frozen risk attention A0 supplies the
initialization prior; it is not a separate self-amplification coefficient.
Future evidence, GRU and variance 1 remain frozen. Actor messages receive the
same A as the opinion update, including during PPO replay. A can affect the
Actor both through z and through its existing attention input coordinate.

Configuration:
`configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_sequence_control.json`.
It differs from the staged beta configuration only in
`nod_evidence_gate_target="attention"` and its new output directory. Missing
target fields default to `"evidence"` for the historical split-beta behavior.
The complete `reward7.08` fixed-weight controller is the source. Reuse 20 Safety
warmup iterations, 10 active A-only PPO batches, subsequent joint training and
ordered 8-step replay. A-only updates preserve Actor/message parameters.
No candidate NOD retraining is needed. Unified-A policies carry a semantic
marker that prevents exchanging them with same-shaped beta policies.

The user executes tests and training:

```bash
python -m pytest tests/test_nod_unified_attention.py tests/test_nod_evidence_gate.py tests/test_nod_evidence_gate_sequence.py tests/test_nod_opinion.py tests/test_nod_visualization.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_sequence_control.json
```

Outputs:
`outputs/current/ablations/dgppo_nod_future_evidence_ppo_attention_sequence_control/`.
Logs add `attention_rl_mean`, `attention_prior_mean`,
`attention_prior_delta_abs`, and `control_weight_target_attention`; phase,
gradient and parameter-displacement diagnostics keep the `evidence_gate_*`
names. The visualization shows `e`, `A_RL`, and `A0`.

After training, use one direct comparison to confirm control performance:

```bash
python utilities/evaluation_tase26.py \
  --model-paths outputs/current/ablations/dgppo_nod_future_evidence_ppo_attention_sequence_control/ outputs/current/ablations/dgppo_nod_future_evidence_control/ outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/ \
  --legends 'PPO shared A' 'Fixed A' 'PPO beta' \
  --output-root outputs/benchmarks/future_evidence_ppo_attention_sequence_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

Optional visualization:

```bash
python main_testing.py --model-path outputs/current/ablations/dgppo_nod_future_evidence_ppo_attention_sequence_control/ --env-seed 101 --no-speed-print --no-realtime
```

Status: the user completed 70 training iterations and the three-model evaluation.
PPO made 17290 minibatch updates; predictor supervision remained frozen and
Actor/message displacement was zero in iterations 21–30. The last-20 mean A
was 0.22844 versus prior A0=0.37191. Highest active-rollout reward was 7.33555
at iteration 26, producing `reward7.34` during the A-only phase. Later joint
training declined: last-20 reward averaged 4.73516. A-only fixed-state action
mode changes averaged only 1.38e-6 per iteration, so parameter adaptation did
not establish strong action sensitivity. Checkpoint filenames use pre-update
rollout reward, while saving occurs after optimization.

Under the command above, shared A / fixed A / split beta respectively achieved:

| Metric | Shared A | Fixed A | Split beta |
| --- | ---: | ---: | ---: |
| Vehicle collision time rate % | 0.23978 | 0.17462 | 0.09904 |
| Road collision time rate % | 1.61854 | 1.66024 | 1.63678 |
| Normalized speed % | 82.95641 | 82.71497 | 81.70083 |
| Centerline deviation % | 54.40619 | 54.34002 | 56.85823 |
| Action change % | 7.50337 | 7.48459 | 7.84812 |

Fixed A and beta exactly reproduced their previous exported metrics. Shared A
increased vehicle collision time rate by 37.3% relative to fixed A, while speed
rose by only 0.24144 percentage points. This run confirms implementation and
learning, but does not support a vehicle-safety advantage for shared A. Split
beta remains the safer tested candidate at a speed/smoothness cost; this single
evaluation seed does not establish universal or statistically significant ranking.
Results are in
`outputs/benchmarks/future_evidence_ppo_attention_sequence_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/metrics_summary.json`.
The coding agent only analyzed saved artifacts and did not run training or simulation.

The measured 43.3% reduction above belongs to the separate beta branch and is
not evidence for this new A branch. Shared A also reinforces prior opinion;
unlike beta alone, increasing it need not pull z toward the predicted e.
Theory and gradient details for this exploratory branch are in section 13 of
[the archived exploration record](docs/archive/LEARNING_IN_NOD_EXPLORATION_RECORD.md).

## Shared-A joint adaptation with aligned checkpoints

This independent follow-up retains the original shared-A equation and starts
again from the fixed-weight controller `reward7.08`, rather than the degraded
final shared-A policy. Forecast, GRU, risk prior and variance 1 remain frozen.
After 20 Safety warmup iterations, use 50 joint Actor/message/A iterations;
remove the A-only phase. Initial Actor/Critic LR is `1e-5`, message LR `3e-5`,
A LR `5e-6`; use 30 PPO epochs, KL limit `0.005`, and the existing 8-step replay.
Set `ppo_training_profile="current"` so these explicit settings take effect;
the legacy `original` profile would force 60 epochs during parameter loading.
GAE lambda and clip remain explicitly `0.9` and `0.2`.
This is a practical optimization follow-up with several simultaneous changes,
not a single-factor ablation or a demonstrated stability guarantee.

Configuration:
`configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_joint_control.json`.

```bash
python -m pytest tests/test_nod_joint_control.py tests/test_nod_unified_attention.py tests/test_nod_evidence_gate_sequence.py tests/test_nod_visualization.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_ppo_attention_joint_control.json
```

Outputs:
`outputs/current/ablations/dgppo_nod_future_evidence_ppo_attention_joint_control/`.
`actor_message_grad_norm_mean` and `actor_message_parameter_delta_l2` supplement
the existing A gradient, displacement and action-response diagnostics.

`training_checkpoint_preupdate=true` snapshots the complete scored policy,
critic, frozen NOD and Safety Value before the current rollout's optimizer
updates. Best-score bundles keep the `rewardX.XX` convention; selection uses
the raw reward and JSON `checkpoint_selection` retains the selected rollout
iteration, pre-update policy-batch counter and timing even as history grows.
Final weights remain post-update and are not assigned a rollout score.

`training_checkpoint_interval=5` also stores scored bundles under
`iterations/iteration0025/`, `iterations/iteration0030/`, etc. Each folder has
the usual reward-named files and can be passed directly to existing evaluation.
These folders survive best-checkpoint cleanup. Old configurations default to
legacy checkpoint behavior; the aligned branch requires a fresh run and does
not claim complete optimizer/collector resume support.

After training, run one direct performance comparison:

```bash
python utilities/evaluation_tase26.py \
  --model-paths outputs/current/ablations/dgppo_nod_future_evidence_ppo_attention_joint_control/ outputs/current/ablations/dgppo_nod_future_evidence_control/ outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/ \
  --legends 'Joint shared A' 'Fixed A' 'PPO beta' \
  --output-root outputs/benchmarks/future_evidence_ppo_attention_joint_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

Status: the user completed all 70 training iterations. Saved settings confirm
30 epochs, profile=current, A LR `5e-6`, message LR `3e-5`, and KL `0.005`.
There were 4409 PPO minibatch updates and no predictor supervision updates;
46 of 50 joint batches reached the KL stop. Last-20 mean rollout reward was
6.26800 versus 4.73516 in the first shared-A run; final rollout reward was
6.87110. This eases the prior late training decline, but is not an independent
performance comparison.

The last-20 mean absolute A-prior deviation was only 0.00088186 (mean A=0.39773,
A0=0.39853), versus 0.15266 before. Message parameters update, but A outputs
remain nearly fixed; improved training behavior cannot be attributed to A.
The selected aligned bundle is `reward7.26`: rollout 37, raw reward 7.26266,
16 prior policy-update batches. The subsequent iteration-37 optimization is
logged as batch 17 and is not included in those saved weights. All 10 periodic
bundles and the root best bundle are complete, with aligned JSON provenance.

The user completed independent performance evaluation on 2026-10-14 under
the command above. Fixed A and beta reproduce all previously exported metrics.

| Metric | Joint shared A | Fixed A | Split beta |
| --- | ---: | ---: | ---: |
| Vehicle collision time rate % | 0.17723 | 0.17462 | 0.09904 |
| Road collision time rate % | 1.73322 | 1.66024 | 1.63678 |
| Normalized speed % | 82.48520 | 82.71497 | 81.70083 |
| Centerline deviation % | 58.36640 | 54.34002 | 56.85823 |
| Action change % | 7.62916 | 7.48459 | 7.84812 |

Joint adaptation reduces vehicle collision time rate by 26.1% versus the first
shared-A trial, but remains 1.49% above fixed A (absolute +0.002606 percentage
points). Speed is 0.22977 percentage points lower; centerline deviation and
action change are respectively 7.41% and 1.93% higher than fixed A. All five
tested means favor fixed A over this follow-up; this is not a significance,
equivalence or noninferiority test. Near-prior A outputs and improved training
stability have not established a shared-A control benefit.

Results are at
`outputs/benchmarks/future_evidence_ppo_attention_joint_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/metrics_summary.json`.
Implementation, training and direct performance confirmation are complete.
Fixed A remains the supported original-equation candidate; beta remains the
safer tested vehicle-avoidance candidate at an efficiency/smoothness cost.
No further parameter sweep or training change was made in this analysis.
The coding agent reads saved artifacts and performs static checks only;
the user executes runtime work.

## Future evidence versus learned error scale

This branch returns to the original question: should a network predict the
future evidence value, or estimate the squared future error of current
geometric evidence? It is separate from the older geometric-teacher learned
evidence experiment and from the post-filter future-head/lower-bound work.

The future-evidence head takes the GRU state and current causal geometric
evidence, predicts `q_hat = sigmoid(MLP([hidden, e_geom]))`, and supplies
`e_used = 2*q_hat-1` to the existing KL opinion update. Its MLP has widths
`65 -> 64 -> 1` with Tanh internally. The final layer starts at zero, producing
neutral evidence. Variance is explicitly fixed at 1, close to the error-scale
candidate's final training mean of 0.95674 and its initial value of 1; it is
not copied from the old collapsed scale. The same learned attention, fixed
`b=0,s=1`, opinion masks and reset rules remain in place.

The losses are future-evidence BCE and opinion BCE, each with weight 1, against
the same future label `q*`. The old observed-evidence NLL and geometric-teacher
Huber loss are excluded. Both losses require complete future windows, active
interaction and consistent vehicle identities. No future label enters online
inference. New diagnostics distinguish forecast quality
(`future_evidence_bce/brier/mae/bias`) from filtered opinion quality
(`calibration_bce`, `brier`, `z_counterfactual_gap_correlation`), with
`geometric_future_brier` evaluated against the same targets as a reference.
These candidate metrics remain training diagnostics.

The candidate reuses the exact frozen `reward7.21` behavior checkpoint,
training seed, 30 iterations and optimizer/minibatch budget of the learned
error-scale candidate. It warm-starts history and attention from that source,
while its prediction head starts neutral. The behavior NOD remains the old
geometric filter. This compares two implementations using the same causal
information at the system level; it is not a single-parameter ablation:
the learned targets, gradient paths and input wiring differ by design.

Run the targeted tests, then candidate training:

```bash
python -m pytest tests/test_nod_future_evidence.py tests/test_nod_evidence_error_variance.py tests/test_nod_behavior_isolation.py tests/test_nod_opinion.py -q
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_candidate.json
```

Outputs go to `outputs/current/ablations/dgppo_nod_future_evidence_candidate/`.
Version-8 NOD checkpoints record `evidence_target=future`, so they cannot be
silently loaded as the old geometric-teacher or learned-variance model.
Future-evidence candidate continuation from a loaded candidate is rejected
to avoid replacing the separately pinned behavior NOD with the learner.

The future-evidence candidate completed 30 iterations. Its logged behavior
reward and agent-collision-flag sequences exactly match the learned error-scale
candidate; every NOD metric entry confirms candidate-only training with frozen
behavior NOD. This supports the intended collection isolation, without claiming
that aggregate logs establish tensor-level equality of every trajectory.
Last-five-iteration opinion diagnostics compare as follows:

| Training diagnostic | Learned future evidence | Learned error scale |
| --- | ---: | ---: |
| Filtered-opinion Brier | 0.15899 | 0.15858 |
| Opinion BCE | 0.60556 | 0.59814 |
| Opinion/future-gap correlation | 0.53749 | 0.50827 |
| Opinion ECE | 0.18550 | 0.14970 |
| Opinion standard deviation | 0.18647 | 0.30352 |

The forecast head's own Brier is `0.11000`, versus `0.22906` when the current
geometric evidence is directly scored against the same future labels. Its
future-q MAE is `0.24769`, bias `+0.01367`, and evidence sign accuracy `0.78916`
(excluding future evidence within 0.1 of zero). Neither opinion solving nor
boundary saturation failed in these last five iterations; variance remained
fixed at 1 and the old NLL remained zero. Forecast and filtered-opinion Brier
must not be conflated: filtering increases Brier from 0.11000 to 0.15899 on
this training distribution. This may reflect temporal lag or shrinkage, but
these logs alone do not identify the cause.

The result removes the earlier impression that learning evidence inherently
performs poorly: the new future target gives comparable opinion Brier and
somewhat higher correlation, though opinion BCE/ECE favor the error-scale
branch. These are post-update candidate-training diagnostics, not independent
calibration or control results. Proceed with the already configured control
fine-tune without changing variance or opinion dynamics before comparing the
two trained control branches.

After reviewing candidate diagnostics, run the corresponding 70-iteration
control fine-tune (already configured):

```bash
python main_training.py --config configs/ablations/config_dgppo_nod_future_evidence_control.json
```

It loads the candidate's full `final` bundle. Actor/task Critic/Safety weights
in that bundle stayed frozen during candidate collection. NOD weights are
then frozen during control fine-tuning, with the same warmup, gain 1 and
deadzone 0.1 as the learned error-scale control. Outputs go to
`outputs/current/ablations/dgppo_nod_future_evidence_control/`. Candidate
training and all 70 control fine-tuning iterations have completed;
the independent two-branch performance evaluation has also completed.

The future-evidence control produced `reward7.08_*` and `final_*` checkpoints.
Every saved NOD metric entry confirms frozen NOD weights with zero optimizer
updates. Its last-batch alpha spans `7.97640` to `12.82799`. The last 20
training iterations compare as follows:

| Training diagnostic | Learned future evidence | Learned error scale |
| --- | ---: | ---: |
| Mean episode return | 6.42452 | 6.60777 |
| Vehicle collision flag fraction | 0.00006104 | 0.00007935 |
| Road collision flag fraction | 0.00098877 | 0.00087891 |
| Opinion-changed advantage fraction | 0.01647 | 0.01710 |
| Barrier violation fraction | 0.24361 | 0.23063 |

The future-evidence branch has lower vehicle collision flag frequency but
lower return in these stochastic training rollouts. Both branches actively
use opinion-modulated advantages. These mixed results do not rank independent
control performance; proceed with the focused evaluation below, keeping both
configurations unchanged.

The focused comparison uses the existing paper-metric evaluator:

```bash
python utilities/evaluation_tase26.py \
  --model-paths \
    outputs/current/ablations/dgppo_nod_future_evidence_control/ \
    outputs/current/ablations/dgppo_nod_evidence_error_variance_control/ \
  --legends 'Learned future evidence' 'Learned error scale' \
  --output-root outputs/benchmarks/future_evidence_vs_variance_paper_metrics \
  --scenarios intersection_2 --num-simulations 32 --steps 1200 --seed 101
```

Optionally visualize the learned-evidence control with `python main_testing.py
--model-path outputs/current/ablations/dgppo_nod_future_evidence_control/
--env-seed 101 --no-speed-print --no-realtime`. Both evaluators load the
highest-training-reward intermediate checkpoint. The two-method comparison
reuses the established protocol without retraining the variance branch or
launching new lower-bound/calibration experiments.

The focused evaluation completed using `reward7.08` for future evidence and
`reward7.38` for error scale, with the protocol above. The error-scale metrics
exactly reproduce that branch's previous four-method evaluation. Results are
in [`metrics_summary.json`](outputs/benchmarks/future_evidence_vs_variance_paper_metrics/intersection_2/steps_1200_envs_32_seed_101/metrics_summary.json).

| Evaluation mean (percent) | Learned future evidence | Learned error scale |
| --- | ---: | ---: |
| Vehicle collision rate (lower is better) | 0.17462 | 0.22936 |
| Road collision rate (lower is better) | 1.66024 | 1.56380 |
| Sum of collision rates (lower is better) | 1.83486 | 1.79316 |
| Normalized average speed (higher is better) | 82.71497 | 83.27902 |
| Collision-penalized speed (higher is better) | 79.04525 | 79.69270 |
| Centerline deviation (lower is better) | 54.34002 | 56.07506 |
| Action change (lower is smoother) | 7.48459 | 7.24631 |

Future evidence reduces the mean vehicle collision rate by 23.9%, an absolute
0.05473 percentage-point difference, while normalized speed falls by 0.56406
points (0.68% relative). Error scale has lower road and summed collision rates
and smoother actions; future evidence has smaller centerline deviation. The
vehicle-collision standard deviations are 0.12744 and 0.15879 percentage points.
The metric definitions and speed-cohort limitation are the same as in the
earlier paper-metric comparison; test return is not exported.

The original implementation comparison is now complete at the candidate,
control-training and first independent evaluation levels. If vehicle-vehicle
avoidance is the primary criterion and road contacts are temporarily secondary,
this run favors the future-evidence implementation at a small normalized-speed
cost. It does not show uniform superiority, statistical significance or a
universal preference for learning evidence over variance. Future evidence
prediction and geometric-error-scale estimation are different learned targets,
and the observation wiring/gradients differ; the conclusion is limited to
these two implementations on this one-scenario, one-seed protocol. Retain both
branches, with no further mechanism additions or default-model switch implied
by this result.

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
