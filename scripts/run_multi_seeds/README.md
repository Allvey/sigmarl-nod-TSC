# run_multi_seeds

Launches multiple training runs with different random seeds using a shared
configuration file.

## Inputs

Default config:

```text
scripts/run_multi_seeds/inputs/config.json
```

You can override it with `--config path/to/config.json`.

## Outputs

Default training outputs are derived from `where_to_save` in:

```text
scripts/run_multi_seeds/inputs/config.json
```

If `where_to_save` ends with a seed folder such as `outputs/TSC/seed8/`, the
script replaces the seed suffix for each run, for example:

```text
outputs/TSC/seed0/
outputs/TSC/seed1/
outputs/TSC/seed2/
```

If `where_to_save` is a method root such as `outputs/TSC/`, the script appends
`seed_0/`, `seed_1/`, etc.

Use `--save-root outputs/TSC/` to override the config path explicitly.

## Usage

From the repository root:

```bash
python scripts/run_multi_seeds/run_multi_seeds.py \
  --seeds 0 1 2
```

If `--seeds` is omitted, use `--runs N` to run seeds `0..N-1`.
