# plot_leader_set_distribution

Draws stacked bar charts for local leader-set size distributions with `K=2`.

## Inputs

Default input:

```text
scripts/plot_leader_set_distribution/inputs/leader_set_distribution_extracted.xlsx
```

## Outputs

Default outputs are written under:

```text
scripts/plot_leader_set_distribution/outputs/
```

Generated files:

- `leader_set_distribution_stacked.png`
- `leader_set_distribution_stacked.pdf`

## Usage

From the repository root:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig /usr/bin/python3 \
  scripts/plot_leader_set_distribution/plot_leader_set_distribution.py
```

Use `--input path/to/file.xlsx` or `--output path/to/name_without_extension` to
override the defaults.
