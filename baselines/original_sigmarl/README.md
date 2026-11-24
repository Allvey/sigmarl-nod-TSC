# Official SigmaRL / XP-MARL runtime

`upstream/` is a byte-identical snapshot of 35 files at
<https://github.com/bassamlab/SigmaRL/tree/1.2.0>, commit
`5fe715bdfba4ff3e33d901d69dfa220f1222c060`. Its original MIT license is
`upstream/LICENSE.txt`; file hashes are recorded in `provenance.json`.

The snapshot was exported from the verified official Git commit, not copied
from the modified working files of this project or its sibling baseline repo.
All runtime assets and imports are local to this project.

`runtime.py` verifies source integrity and launches `worker.py` with the
upstream root as its working directory and import path. The separate process
keeps upstream `utilities` / `scenarios` independent from the NOD modules.
Training invokes upstream `mappo_cavs` directly. Evaluation adds passive info
fields through `instrumentation.py`; policy, priority propagation, observations,
reward and reset behavior remain upstream code.

Both training profiles derive from the untouched upstream root `config.json`,
with observation noise disabled. SigmaRL changes only the output directory;
XP-MARL additionally enables `is_using_prioritized_marl` and sets
`prioritization_method="marl"`. Saved model metadata does not define the training
profiles. New outputs use `checkpoints/{method}_original/config_v1_2_0/` to keep
earlier, differently configured runs separate.

See [the reproduction guide](../../docs/current/ORIGINAL_BASELINES.md) for
training, published vs retrained weights, five-metric comparison and provenance.
