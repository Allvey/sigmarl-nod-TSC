import torch

from scripts.calibrate_nod_future_quantile import (
    calibration_correction,
    lower_bound,
    relaxation_diagnostics,
    summarize,
)


def test_calibration_correction_is_conservative_and_bound_does_not_exceed_mean():
    raw = torch.tensor([0.2, 0.4, 0.6, 0.8])
    target = torch.tensor([0.3, 0.3, 0.5, 0.1])
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5])

    correction = calibration_correction(raw, target, eta=0.25)
    bound = lower_bound(raw, mean, correction)

    assert abs(correction - 0.7) < 1e-6
    assert (bound >= 0).all()
    assert (bound <= mean).all()
    assert (target >= bound).all()


def test_calibration_never_raises_raw_lower_bound():
    raw = torch.tensor([0.1, 0.2, 0.3])
    target = torch.tensor([0.8, 0.8, 0.8])
    correction = calibration_correction(raw, target, eta=0.1)
    assert correction == 0.0
    torch.testing.assert_close(lower_bound(raw, torch.ones_like(raw), correction), raw)


def test_relaxation_diagnostics_count_only_positive_bound_decisions():
    bound = torch.tensor([0.4, 0.6, 0.7, 0.9])
    target = torch.tensor([0.0, 0.4, 0.8, 0.7])
    report = relaxation_diagnostics(bound, target)

    assert report["count"] == 3
    assert report["target_below_threshold_count"] == 1
    assert report["bound_violation_count"] == 2
    assert abs(report["bound_violation_rate"] - 2 / 3) < 1e-6
    assert abs(report["bound_excess_mean_on_violations"] - 0.2) < 1e-6
    assert relaxation_diagnostics(torch.zeros(2), torch.ones(2))["bound_violation_rate"] is None


def test_profile_summary_recomputes_mae_for_selected_neighbors():
    result = {
        "seed": 1,
        "_quantile_raw": torch.tensor([0.2, 0.6]),
        "_prediction": torch.tensor([0.9, 0.6]),
        "_target": torch.tensor([0.0, 0.6]),
        "_neighbor_profile": torch.tensor([0, 2]),
    }
    selected = summarize(result, correction=0.0, profile_code=2)
    assert selected["valid_pairs"] == 1
    assert selected["future_mae"] == 0.0
    assert selected["relaxation"]["count"] == 1
