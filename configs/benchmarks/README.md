# Benchmark configurations

## Four-method, five-metric evaluation

Run all four scenarios with 32 parallel simulations per method, 1200 nominal
steps, and environment seed 101:

```bash
python utilities/evaluation_tase26.py
```

The methods, in plot and export order, are:

| Method | Checkpoint directory |
| --- | --- |
| NOD-DGPPO (PPO beta) | `outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/` |
| SigmaRL | `checkpoints/itsc24/M0 (our)/` |
| MAPPO | `checkpoints/icra25/M2 (Baseline)/` |
| XP-MARL | `checkpoints/icra25/M1 (XP-MARL)/` |

Each directory uses its highest-reward saved policy. MAPPO here denotes the
ICRA25 non-prioritized baseline; retain that checkpoint provenance in the paper.
All vehicles use learned policies. Actions are deterministic and observation
noise is disabled for every method during evaluation.

The five metrics are collision rate, exit throughput, trajectory jerk RMS,
relative average speed, and the original action-change smoothness. Collision
plots retain total, vehicle-vehicle, and vehicle-boundary distributions. Collision
rates count pre-transition time steps with any vehicle in contact, divided by
recorded steps and multiplied by 100, separately for vehicle and road-boundary
contacts. The total is the sum of these percentages. Smoothness is an action
change measure, so lower values mean smoother actions. Centerline deviation
and low-TTC exposure are disabled.

Results, including PNG/PDF plots, `metrics_runs.csv`, `metrics_summary.json`,
and `log.txt`, go to:

```text
outputs/benchmarks/paper_metrics_four_methods/<scenario>/steps_1200_envs_32_seed_101/
```

Scenarios are `CPM_entire`, `intersection_2`, `on_ramp_1`, and `roundabout_1`.
Throughput is N/A for the full-network loop routes (`CPM_entire`). Existing
three-method results keep their original paths and recorded checkpoint identities.

For one scenario:

```bash
python utilities/evaluation_tase26.py --scenarios intersection_2
```

For a short check:

```bash
python utilities/evaluation_tase26.py \
  --scenarios intersection_2 --num-simulations 2 --steps 200
```

Use `--model-paths` and `--legends` with matching numbers of entries to compare
other checkpoints, and `--output-root` to choose a separate results directory.
The entry point reuses the `Evaluation` class from the ITSC24 and ICRA25 scripts.

`sota_all_actor.json` defines the supplementary explicit-seed comparison among
NOD-DGPPO, SigmaRL, MAPPO, and XP-MARL. It uses:

- the same scenario and agent count for every method;
- paired environment seeds;
- 1200 steps per seed;
- deterministic policy actions;
- observation noise disabled for every method;
- the highest-reward intermediate checkpoint in each model directory;
- learned policies for every vehicle (no rule vehicles).

`future_lower_control.json` uses the same evaluator and protocol for the two
matched future-NOD control branches, the previous NOD-DGPPO checkpoint, and
the three comparison baselines. Train both branches listed in `EXPERIMENTS.md`
before running it. Keep the comparison under a separate
`--output-root outputs/benchmarks/future_lower_control` so the established
SOTA results remain intact.

It also includes final-checkpoint evaluations for both control branches and
the follow-up gain-0.4 lower-bound branch. Existing run summaries can be
retained with `--skip-existing` when adding the follow-up branch.

The JSON benchmark configurations use the same three baseline identities as
`evaluation_tase26.py`, but retain their own NOD control branches and explicit
five-seed protocol. They do not replace the five-metric entry point above.
Historical results retain the checkpoint recorded in their CSV/JSON metadata;
changing a configuration does not relabel previous measurements.

These checkpoints are included in the upstream SigmaRL assets
([ITSC24 package](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/itsc24.zip),
[ICRA25 package](https://github.com/bassamlab/assets/blob/main/sigmarl/checkpoints/icra25.zip)).

Prepare them with:

```bash
python -m scripts.prepare_sota_checkpoints
```

Run a short one-seed smoke evaluation first:

```bash
python -m scripts.run_sota_benchmark \
  --seeds 101 --max-steps 200
```

Then run the configured five-seed comparison:

```bash
python -m scripts.run_sota_benchmark
```

Results are written under
`outputs/benchmarks/sota_all_actor/<scenario>/steps_<N>/seeds_<...>/`. The
directory contains every run's `summary.json`, a combined `runs.csv`,
`aggregate.csv`, `pairwise.csv`, and `summary.json`. In `pairwise.csv`, a
positive `advantage_mean` means NOD-DGPPO performed better than the named
competitor for that metric.

Do not reuse the archived training-curve JSON files as evaluation checkpoints;
they contain no policy weights and their names do not reliably identify the
enabled prioritization behavior.
