"""Candidate initialization must also accept final weights without final JSON."""
import json

import pytest
import torch

from utilities.helper_training import Parameters
from utilities.mappo_cavs import _load_behavior_variance_bounds
from utilities.nod_marl.trainer import NODOpinionManager


def test_final_weights_without_json_keep_behavior_and_candidate_contracts(tmp_path):
    root = "configs/current/error_scale_three_stage/"
    first = Parameters.from_json(root + "01_bootstrap.json")
    second = Parameters.from_json(root + "02_candidate.json")
    source = NODOpinionManager(first)
    checkpoint = source.checkpoint_state()
    prefix = str(tmp_path / "final")
    torch.save(checkpoint, prefix + "_nod.pth")
    assert not (tmp_path / "final_data.json").exists()
    bounds = _load_behavior_variance_bounds(prefix, "cpu")
    assert bounds == (first.nod_min_log_sigma, first.nod_max_log_sigma)
    # Reproduce the actual behavior-manager setup, then learner migration.
    behavior_parameters = Parameters.from_dict(second.to_dict())
    behavior_parameters.nod_variance_supervision = "posterior"
    behavior_parameters.nod_min_log_sigma, behavior_parameters.nod_max_log_sigma = bounds
    behavior = NODOpinionManager(behavior_parameters)
    assert behavior.load_checkpoint(checkpoint)
    learner = NODOpinionManager(second)
    learner.initialize_error_variance_from_behavior(behavior.checkpoint_state())
    assert learner.checkpoint_state()["variance_log_bounds"] == [
        second.nod_min_log_sigma, second.nod_max_log_sigma]
    for key, tensor in source.model.state_dict().items():
        torch.testing.assert_close(behavior.model.state_dict()[key], tensor, rtol=0, atol=0)


def test_embedded_bounds_take_precedence_over_metadata(tmp_path):
    prefix = str(tmp_path / "final")
    torch.save({"variance_log_bounds": [-2., .4]}, prefix + "_nod.pth")
    (tmp_path / "final_data.json").write_text(json.dumps({"parameters": {
        "nod_min_log_sigma": -2.5, "nod_max_log_sigma": 0.}}))
    assert _load_behavior_variance_bounds(prefix, "cpu") == (-2., .4)


def test_legacy_bounds_require_matching_metadata(tmp_path):
    prefix = str(tmp_path / "reward7.21")
    torch.save({"version": 3}, prefix + "_nod.pth")
    with pytest.raises(FileNotFoundError, match="matching metadata"):
        _load_behavior_variance_bounds(prefix, "cpu")
    (tmp_path / "reward7.21_data.json").write_text(json.dumps({"parameters": {
        "nod_min_log_sigma": -2.2, "nod_max_log_sigma": -.1}}))
    assert _load_behavior_variance_bounds(prefix, "cpu") == (-2.2, -.1)


@pytest.mark.parametrize("bounds", [[0., -1.], [float('nan'), 0.], [-2.], "invalid"])
def test_invalid_embedded_bounds_are_rejected(tmp_path, bounds):
    prefix = str(tmp_path / "final")
    torch.save({"variance_log_bounds": bounds}, prefix + "_nod.pth")
    with pytest.raises(ValueError, match="Invalid behavior"):
        _load_behavior_variance_bounds(prefix, "cpu")
