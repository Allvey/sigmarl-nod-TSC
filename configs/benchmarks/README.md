# Benchmark configurations

This directory is reserved for the matched SigmaRL, XP-MARL, and NOD-DGPPO
benchmark configurations. Each benchmark config should use the same scenario,
agent count, rollout duration, evaluation seeds, and checkpoint-selection rule.

Do not reuse the archived training-curve JSON files as benchmark configs; their
agent counts and training budgets are not matched to the current model.
