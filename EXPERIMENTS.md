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
