# SigmaRL and XP-MARL benchmark

## Scope

The first SOTA comparison evaluates three learned controllers:

| Method | Checkpoint directory | Action mechanism |
| --- | --- | --- |
| NOD-DGPPO | `outputs/current/dgppo_nod_gain1_control_finetune/` | decentralized Actor with NOD input |
| SigmaRL | `checkpoints/itsc24/M0 (our)/` | decentralized Actor |
| XP-MARL | `checkpoints/icra25/M1 (XP-MARL)/` | learned priorities and sequential action propagation |

All vehicles remain learned vehicles in this comparison. The mixed rule-vehicle
wrapper is excluded because it does not implement XP-MARL's sequential action
propagation for rule-controlled agents.

## Supplementary explicit-seed protocol

The supplementary protocol is defined in
`configs/benchmarks/sota_all_actor.json`:

- scenario: `intersection_2`;
- environment seeds: 101, 202, 303, 404, and 505;
- duration: 1200 simulation steps per seed;
- deterministic policy actions;
- observation noise disabled for all methods;
- highest-reward intermediate checkpoint from each model directory;
- identical scenario agent count for all methods.

The old official checkpoint JSON files predate NOD and the safety modules. The
benchmark loader explicitly disables these missing additions when loading an
old model, so its Actor input dimension remains compatible with the published
weights. XP-MARL additionally requires and loads its matching priority-policy
checkpoint. Its Actor observation contains four trailing action-propagation
slots (two observed neighbors with two action dimensions each); the priority
network continues to receive the unpadded physical observation.

## Two evaluation layers

The formal comparison uses `utilities/evaluation_tase26.py`, which shares the
same `Evaluation` implementation as `evaluation_ITSC24.py` and
`evaluation_icra25.py`. Its primary metrics are the existing paper metrics:

- agent-agent and agent-lanelet collision rate;
- relative centerline deviation;
- relative average speed;
- longitudinal and lateral action change;
- the existing collision-penalized efficiency and smoothness variants.

The newer `scripts.run_sota_benchmark` workflow remains a supplementary
diagnostic. It measures cumulative collision events, stopping, route error,
and returns for explicit individual seeds. These diagnostics should not replace
the established paper metric table.

## Commands

Download and validate the official checkpoints:

```bash
python -m scripts.prepare_sota_checkpoints
```

Run a short formal-evaluator smoke test on one scenario and two parallel runs:

```bash
python utilities/evaluation_tase26.py \
  --scenarios intersection_2 --num-simulations 2 --steps 200
```

Run the formal four-scenario evaluation with 32 parallel simulations per
method, matching the sample count used by the earlier evaluation scripts:

```bash
python utilities/evaluation_tase26.py
```

The formal entry point uses the same seed, deterministic policy actions, and
disabled observation noise for all three methods. Pass `--stochastic-actions`
or `--checkpoint-observation-noise` only for a historical-protocol sensitivity
run.

Run the supplementary explicit-seed diagnostic:

```bash
python -m scripts.run_sota_benchmark \
  --seeds 101 --max-steps 200
```

Run the full supplementary diagnostic:

```bash
python -m scripts.run_sota_benchmark
```

Use `--skip-existing` to reuse completed per-seed summaries. Add
`--save-video` only when visual inspection is needed; the standard statistical
run does not encode videos.

## Outputs and interpretation

Results are stored in:

```text
outputs/benchmarks/sota_all_actor/
  <scenario>/
    steps_<N>/
      seeds_<seed list>/
        <method>/seed_<seed>/summary.json
        runs.csv
        aggregate.csv
        pairwise.csv
        summary.json
```

Formal paper-metric outputs are stored separately under:

```text
outputs/benchmarks/paper_metrics/<scenario>/steps_<N>_envs_<M>_seed_<seed>/
```

Each directory contains the original PDF/PNG plots and log, plus
`metrics_runs.csv` and `metrics_summary.json`.

For the supplementary files, the primary safety measure is the number of
vehicle collision events. Road collision events are retained even when they
are not used for the current selection decision. Efficiency is described by
mean speed, stopped time fraction, longest stop, and mean per-agent return.
Route error and action-change statistics are secondary diagnostics. The formal
paper outputs use collision-frame rate and the other established `Evaluation`
metrics listed above.

In `pairwise.csv`, positive `advantage_mean` always favors NOD-DGPPO. For
collision, stopping, route-error, and action-change metrics, lower raw values
are better; for speed and return, higher raw values are better.
