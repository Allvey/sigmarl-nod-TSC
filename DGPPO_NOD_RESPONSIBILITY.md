# NOD responsibility opinion

This experiment gives the directed opinion `z_ij` one fixed meaning:

`z_ij = 2 q_ij - 1`, where `q_ij` is the fraction of the required collision-risk
reduction carried by neighbor `j` for ego `i`.

- `z=-1`: the neighbor has carried none of the required avoidance.
- `z=0`: the neighbor has carried half of it.
- `z=+1`: the neighbor has carried all of it.

At the start of a predicted conflict, the current velocities of both vehicles
are stored as counterfactual anchors. The online evidence replaces only the
neighbor anchor with its observed velocity. The training label replaces only
the neighbor counterfactual trajectory with its observed future trajectory.
The ego anchor is unchanged in both cases, so ego braking cannot be credited to
the neighbor.

Only active predicted conflicts produce opinions and supervised samples.
Unrelated visible vehicles no longer contribute a large neutral-label class.
The likelihood mapping is fixed to `b=0, s=1`; the learned parts are the history
encoder, uncertainty `sigma`, and risk attention.

Use `config_dgppo_nod_responsibility_finetune.json` for this mode. It loads the
same-shaped NOD checkpoint but interprets evidence through the fixed mapping and
writes results to `outputs/dgppo_nod_responsibility_finetune/`.

After the responsibility model has been trained, use
`config_dgppo_nod_responsibility_control_finetune.json` for the control stage.
It initializes Policy, Critic, Safety Value and NOD from the responsibility
checkpoint, freezes NOD weights, and lets the control networks adapt to stable
online opinions. Online `z` still evolves; only NOD parameter updates are
disabled.

## Isolated acceptance workflow

`config_dgppo_nod_candidate_only.json` implements the controlled NOD stage. A
frozen behavior NOD remains bound to the Actor during collection, while a
separate learner NOD updates from completed rollouts. PPO, Safety Value, Safety
Critic and Deadlock Critic do not update. The candidate is stored as
`outputs/dgppo_nod_candidate_only/final_nod.pth`; the frozen companion networks
are also copied under the same `final` prefix for the next stage.

Run the deterministic two-vehicle acceptance cases with:

```bash
python -m scripts.validate_nod_responsibility \
  --checkpoint outputs/dgppo_nod_candidate_only/final_nod.pth
```

After those cases pass, `config_dgppo_nod_gain1_control_finetune.json` loads the
candidate, freezes NOD, and enables opinion-conditioned DGPPO with gain 1 and a
`|z| <= 0.1` fixed-alpha deadzone.

## Road-safe follow-up

`config_dgppo_nod_gain1_road_safe_finetune.json` starts from the selected gain-1
checkpoint and runs a shorter 30-iteration follow-up. Pair constraints retain
the accepted opinion rule. The road head uses `alpha=7.5` while its value is on
the safe side, making approach to the boundary stricter, and `alpha=15` after
the road value becomes positive, strengthening recovery. The changed barrier
contract preserves Safety Value weights but restarts the 10-batch barrier
warmup.

The visualization-only `non_yielding` controller remains aggressive during
ordinary conflicts. It now applies an emergency response only when its
full-speed trajectory predicts a collision within 0.45 seconds. Rule-to-rule
trajectory reservations remain unchanged.

## Repeated evaluation across environment seeds

Keep the rule-vehicle assignment seed fixed while changing only the rollout
seed. The rollout seed controls environment reset and stochastic policy
sampling, while the fixed assignment seed preserves which agents use each rule
profile. This tests whether the selected controller remains safe and efficient
under different environment realizations. Run the default five seeds with
videos using:

```bash
python -m scripts.run_rule_multiseed_evaluation \
  --seeds 101 202 303 404 505
```

For faster statistical evaluation without MP4 encoding, add `--no-video`.
Interrupted batches can be resumed with `--skip-existing`; a seed is reused only
when its `summary.json` already exists.

Each run is saved under `rule_vehicle_visualization/` in a directory containing
both `assignseed` and `envseed`. The aggregate files are written to:

```text
outputs/dgppo_nod_gain1_road_safe_finetune/rule_vehicle_multiseed/
  intersection_2_assignseed123_coordinated_rules_v7/
    envseeds_101_202_303_404_505/
      multiseed_summary.json
      multiseed_summary.csv
```

The JSON contains per-seed results and mean, population standard deviation,
minimum, maximum and sum for collision, speed, stopped-time and emergency-brake
metrics. The CSV contains one row per environment seed.

## Actor opinion-sensitivity diagnostic

The next acceptance test holds one physical interaction state fixed and sweeps
only the directed opinion supplied to Agent 1:

```bash
python -m scripts.validate_actor_opinion_sensitivity \
  --model-path outputs/dgppo_nod_gain1_control_finetune/ \
  --seed 123 \
  --ego 1
```

The script searches a deterministic rollout for active pairs whose physical
constraint and learned Safety Value are both on the safe side. It prioritizes a
state where alpha 5 versus alpha 15 changes the pairwise gate; if no such state
exists, it selects the transition closest to the barrier boundary. It reuses
that exact state for `z = -1, -0.5, 0, 0.5, 1`, then records the mapped alpha,
deterministic velocity/steering commands, policy location and NOD message norm.
It also holds one naturally observed successor fixed and reports the pairwise
DGPPO delta, overall gate and penalty under each alpha. This latter quantity
isolates the alpha mapping; it is not presented as the successor caused by each
counterfactual action. The environment is never advanced between Actor sweep
points. Results are saved under
`actor_opinion_sensitivity/seed123_agent1/` in `sweep.csv` and `summary.json`.

A positive `positive_minus_negative_velocity` means that the Actor commands
more speed when it believes the neighbor is carrying more avoidance
responsibility. `velocity_nondecreasing_with_z` reports the stricter monotonic
test. These are diagnostics rather than assertions: a failed direction test is
evidence that the Actor did not learn the intended response.

## Matched training ablations

All three branches initialize Policy, Critic, Safety Value and frozen NOD from
`outputs/dgppo_nod_candidate_only/final` and retain identical optimization and
random-seed settings:

```bash
# Full mechanism: real z reaches both Actor and opinion-conditioned alpha.
python main_training.py --config config_dgppo_nod_gain1_control_finetune.json

# Real z reaches Actor, but the DGPPO barrier always uses alpha=10.
python main_training.py --config config_dgppo_nod_ablation_fixed_alpha.json

# Physical NOD context remains, but z is zeroed only at the Actor input;
# the DGPPO barrier also uses alpha=10.
python main_training.py --config config_dgppo_nod_ablation_neutral_z.json
```

The neutral branch does not erase hidden history, risk features, attention, or
neighbor geometry. It removes only the final opinion coordinate from the
message encoder. The original online context remains available for diagnostics.

Evaluate all three trained branches using the same environment seeds. The
causal comparisons are:

- Full versus Fixed-alpha: contribution of opinion-conditioned Safety training.
- Fixed-alpha versus Neutral-z: contribution of direct opinion input to Actor.
- Full versus Neutral-z: total contribution of the responsibility opinion.

For example, after training finishes:

```bash
python -m scripts.run_rule_multiseed_evaluation \
  --model-path outputs/dgppo_nod_gain1_control_finetune/ \
  --seeds 101 202 303 404 505 --no-video
python -m scripts.run_rule_multiseed_evaluation \
  --model-path outputs/dgppo_nod_ablation_fixed_alpha/ \
  --seeds 101 202 303 404 505 --no-video
python -m scripts.run_rule_multiseed_evaluation \
  --model-path outputs/dgppo_nod_ablation_neutral_z/ \
  --seeds 101 202 303 404 505 --no-video
```

Because alpha changes the Actor during PPO training and is not an online action
filter, switching `dgppo_opinion_alpha` only at test time is not a valid
ablation. Each branch must be trained independently from the common candidate.
