"""Use original baseline Actors in the project's shared visualization environment."""

from tensordict.nn.distributions import NormalParamExtractor

from baselines.original_sigmarl.runtime import validate_method


def configure_sigmarl_visualization(parameters, raw_parameters):
    """Do not let modern Parameters defaults add NOD to an old checkpoint."""
    validate_method(raw_parameters, "sigmarl")
    for name in (
        "is_using_nod_opinion", "is_using_nod_actor", "is_using_safety_critic",
        "is_using_safety_value_shadow", "is_using_safety_constraint",
        "is_using_deadlock_critic", "nod_freeze_training",
    ):
        setattr(parameters, name, False)
    parameters.safety_control_mode = "legacy_q"


def restore_original_actor_distribution(policy):
    """Restore upstream's unbounded scale extractor without changing weights.

    The current builder caps Gaussian scales at 1.0. Original SigmaRL uses the
    ordinary NormalParamExtractor. The replacement has no trainable parameters
    and leaves checkpoint keys and deterministic actions unchanged.
    """
    from utilities.mappo_cavs import BoundedNormalParamExtractor

    matches = [(parent, name) for parent in policy.modules()
               for name, child in parent.named_children()
               if isinstance(child, BoundedNormalParamExtractor)]
    if len(matches) != 1:
        raise ValueError("Expected one baseline Actor scale extractor")
    parent, name = matches[0]
    setattr(parent, name, NormalParamExtractor())
