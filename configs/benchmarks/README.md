# Benchmark configurations

The formal paper-metric comparison is run through
`utilities/evaluation_tase26.py`. It reuses the same `Evaluation` class as the
ITSC24 and ICRA25 scripts while adding matched seeds, official-checkpoint
compatibility, and CSV/JSON export.

Run a short check with:

```bash
python utilities/evaluation_tase26.py \
  --scenarios intersection_2 --num-simulations 2 --steps 200
```

Run all four scenarios with 32 simulations per method using:

```bash
python utilities/evaluation_tase26.py
```

`sota_all_actor.json` defines the supplementary explicit-seed comparison among
NOD-DGPPO, SigmaRL, and XP-MARL. It uses:

- the same scenario and agent count for every method;
- paired environment seeds;
- 1200 steps per seed;
- deterministic policy actions;
- observation noise disabled for every method;
- the highest-reward intermediate checkpoint in each model directory;
- learned policies for every vehicle (no rule vehicles).

The official comparison checkpoints are:

```text
checkpoints/itsc24/M0 (our)/          SigmaRL
checkpoints/icra25/M1 (XP-MARL)/      XP-MARL
```

They are downloaded from the upstream SigmaRL assets
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
