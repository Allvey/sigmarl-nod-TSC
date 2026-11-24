# Original baseline training profiles

These profiles run the untouched SigmaRL 1.2.0 implementation vendored in
`baselines/original_sigmarl/upstream/`, not the NOD trainer with features disabled.

| Profile | Source parameters | Output |
| --- | --- | --- |
| `sigmarl_original.json` | official 1.2.0 root `config.json`, observation noise off | `checkpoints/sigmarl_original/config_v1_2_0/` |
| `xp_marl_original.json` | same root config, with learned priorities enabled | `checkpoints/xp_marl_original/config_v1_2_0/` |

The source is `baselines/original_sigmarl/upstream/config.json`, not parameters
extracted from published weights. SigmaRL changes only `where_to_save`. XP-MARL
also sets `is_using_prioritized_marl=true` and `prioritization_method="marl"`.
All other settings match exactly, including `is_add_noise=false` for both.
The source's `prioritization_method="random"` has no effect while prioritization
is disabled in SigmaRL. Environment count and total frames use upstream's
derived defaults.

Run from the project root:

```bash
python main_training.py --method sigmarl
python main_training.py --method xp_marl
```

Both train from scratch with the original centralized critic and distribution,
without loading published weights. Earlier training outputs in the parent
directories retain their original metadata and are not overwritten or selected
by `--baseline-checkpoints retrained`. See
[the implementation and comparison guide](../../docs/current/ORIGINAL_BASELINES.md).
