# collect_safety_tscnet

Runs the safety/performance evaluation workflow for configured TSC-Net
checkpoints.

## Inputs

- Model checkpoints are read in place from `checkpoints/CPM_entire/TSC-Net/`.
- Checkpoint/model inputs under `checkpoints/` are not copied into this folder.

## Outputs

Generated logs and evaluation artifacts are written under:

```text
scripts/collect_safety_tscnet/outputs/TSC-Net/
```

## Usage

From the repository root:

```bash
python scripts/collect_safety_tscnet/collect_safety_tscnet.py
```

Edit `model_dir`, `scenario_order`, and plotting/evaluation options near the top
of `collect_safety_tscnet.py` before running.
