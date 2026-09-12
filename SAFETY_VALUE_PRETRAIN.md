# Safety Value pretraining from the task-only baseline

This stage implements phases 1 and 2 of the gated Safety Critic recovery plan.
It loads the verified task-only checkpoint, keeps the Actor and task Critic
frozen for the whole run, and trains only the DGPPO Safety Value. It does not
apply a safety advantage to the Actor.

## Run

```bash
python main_training.py --config config_ppo_original_safety_value_pretrain.json
```

The source prefix is:

```text
outputs/ppo_original_task_only/reward7.36
```

Results are written to:

```text
outputs/ppo_original_safety_value_pretrain/
```

Use `final_safety_value.pth` for the next stage. Intermediate `reward*.pth`
names reflect random task-rollout returns from an unchanged Actor, so they do
not rank Safety Value quality. `final_policy.pth` and `final_critic.pth` are
saved for a self-contained next-stage prefix and must remain identical to the
source task-only weights.

## What is trained

- The original DGPPO lambda/max target is unchanged.
- Loss is computed as an equal mean over pair, road, and unattributed-collision
  heads. Positive targets receive a capped weight of 4 and underestimated
  positive targets receive an additional weight of 2.
- The deterministic shadow sampler uses 32 environments for 128 steps per
  batch. Eight environments are assigned to challenging starts after the
  corresponding buffers contain valid pre-danger states.
- Eight evenly spaced environments form a held-out validation partition. Their
  transitions are excluded from optimizer loss and from danger-start mining.
- The run uses 70 batches and four Safety Value epochs per batch.
- `dgppo_weight=0` and `is_using_safety_constraint=false`; `value_pretrain`
  independently forces zero Actor/task-Critic optimizer steps.

The training distribution remains the existing `CPM_mixed` intersection
setting (`cpm_scenario_probabilities=[1,0,0]`). This stage establishes a clean
pair/road critic experiment on the baseline distribution; it does not claim
held-out safety on the four evaluation scenarios.

## Metrics to inspect

Use the `validation_` metrics in the saved `reward*_data.json`, especially:

- `validation_pair_early_warning_recall`
- `validation_road_early_warning_recall`
- `validation_pair_early_warning_underestimate_rate`
- `validation_road_early_warning_underestimate_rate`
- `validation_pair_observed_safe_positive_rate`
- `validation_road_observed_safe_positive_rate`

Also check `challenge_*` metrics, `challenging_start_frame_ratio`, buffer sizes,
and per-head positive class weights. Early batches may have no challenging
frames while the start buffers are being populated.

Do not activate gated Actor updates solely because training loss decreases.
The provisional next-stage gate is validation early-warning recall of at least
60% for the active head, with underestimation below 30%, without an excessive
safe-state positive rate. If these conditions are not met, keep the task-only
policy unchanged and improve the Value data or targets first.
