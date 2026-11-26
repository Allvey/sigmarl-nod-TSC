# Archived four-stage from-scratch plan

These four configurations are the superseded 250 + 30 + 70 + 70 iteration
proposal. They retain their original output paths and stage dependencies as
historical records; weights have not been moved or changed.

The active two-stage plan is in `configs/current/from_scratch/` and documented
in [the training guide](../../../docs/current/NOD_FROM_SCRATCH.md).
It directly learns future evidence during joint training, then freezes the
predictor while PPO jointly trains beta and the Actor.
