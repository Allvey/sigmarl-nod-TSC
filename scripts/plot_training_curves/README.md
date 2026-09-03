# plot_training_curves

Plots training reward and agent-agent collision curves across random seeds and
methods.

## Inputs

Default local inputs:

- `scripts/plot_training_curves/inputs/TSC` as `TSC`
- `scripts/plot_training_curves/inputs/XPMarl` as `XP-MARL`
- `scripts/plot_training_curves/inputs/SigmaRL` as `SigmaRL`
- `scripts/plot_training_curves/inputs/MFPO` as `MFPO`

Each method folder should contain seed subfolders such as
`seed0/reward*_data.json`.

You can also pass method roots under `outputs/` or `checkpoints/`; those files
are read in place and do not need to be copied.

## Outputs

Default outputs are written under:

```text
scripts/plot_training_curves/outputs/
```

Generated files:

- `reward_mean_curve.png`
- `reward_mean_curve.pdf`
- `collision_agents_mean_curve.png`
- `collision_agents_mean_curve.pdf`

## Usage

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig /usr/bin/python3 \
  scripts/plot_training_curves/plot_training_curves.py \
  --smooth-window 10 \
  --smooth-direction forward
```

To choose methods explicitly:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig /usr/bin/python3 \
  scripts/plot_training_curves/plot_training_curves.py \
  --methods outputs/TSC:TSC:#8172b2 outputs/xpmarl:XP-MARL:#d62728 outputs/sigmarl:SigmaRL:#2ca02c outputs/MFPO:MFPO:#1f77b4
```

## Utilities

Add `0.003` to every value in `collision_agents_rate_list` under the default
ablation demo folder:

```bash
python scripts/plot_training_curves/add_collision_offset.py
```

The script creates `.bak` backups by default. Use `--no-backup` to disable
backups, or pass `--input-dir` and `--offset` for another dataset.
