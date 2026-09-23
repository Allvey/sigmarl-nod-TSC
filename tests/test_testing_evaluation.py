import csv
import json

import pytest

from utilities.testing_evaluation import (
    aggregate_rule_runs,
    rule_run_name,
    summarize_rule_run,
)


def _write_run(tmp_path, seed, vehicle_collisions, road_collisions, speeds):
    output = tmp_path / f"seed_{seed}"
    output.mkdir()
    (output / "rule_setup.json").write_text(json.dumps({
        "seed": seed,
        "assignment_seed": 123,
        "checkpoint": "reward7.16",
        "controller_version": "coordinated_rules_v7",
        "vehicles": {"1": "moderate"},
    }))
    with (output / "agent_speeds.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "t_sec", "agent_1_speed", "agent_2_speed"])
        for step, (actor_speed, rule_speed) in enumerate(speeds, 1):
            writer.writerow([step, step * .05, actor_speed, rule_speed])
    with (output / "rule_diagnostics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "agent", "emergency_brake", "road_events",
            "vehicle_contact_events", "rule_contact",
        ])
        writer.writeheader()
        writer.writerow(dict(agent=2, emergency_brake=1, road_events=1,
                             vehicle_contact_events=0, rule_contact=0))
        writer.writerow(dict(agent=2, emergency_brake=0, road_events=1,
                             vehicle_contact_events=1, rule_contact=1))
    return output, summarize_rule_run(
        output,
        vehicle_collisions=vehicle_collisions,
        road_collisions=road_collisions,
        dt=.05,
    )


def test_rule_run_name_distinguishes_environment_and_assignment_seeds():
    first = rule_run_name(
        "intersection_2", .5,
        {"yielding": .25, "moderate": .5, "non_yielding": .25},
        assignment_seed=123, environment_seed=101, actor_index=0,
        controller_version="coordinated_rules_v7",
    )
    second = rule_run_name(
        "intersection_2", .5,
        {"yielding": .25, "moderate": .5, "non_yielding": .25},
        assignment_seed=123, environment_seed=202, actor_index=0,
        controller_version="coordinated_rules_v7",
    )
    assert "assignseed123_envseed101" in first
    assert first != second


def test_summarize_and_aggregate_rule_runs(tmp_path):
    first_dir, first = _write_run(
        tmp_path, 101, vehicle_collisions=1, road_collisions=2,
        speeds=[(1., 0.), (.8, 0.), (.6, .4)],
    )
    second_dir, second = _write_run(
        tmp_path, 202, vehicle_collisions=0, road_collisions=1,
        speeds=[(.6, .4), (.6, .4), (.6, .4)],
    )

    assert first["total_collisions"] == 3
    assert first["actor"]["mean_speed"] == pytest.approx(.8)
    assert first["rule"]["longest_stop_seconds"] == pytest.approx(.1)
    assert first["emergency_seconds"] == pytest.approx(.05)
    assert first["rule_rule_contact_detected"] is True

    aggregate, json_path, csv_path = aggregate_rule_runs(
        [first_dir / "summary.json", second_dir / "summary.json"],
        tmp_path / "aggregate",
    )
    assert aggregate["seeds"] == [101, 202]
    assert aggregate["aggregate"]["total_collisions"]["mean"] == pytest.approx(2.)
    assert aggregate["collision_free_runs"] == 0
    assert aggregate["collision_free_rate"] == 0.
    assert json_path.endswith("multiseed_summary.json")
    assert csv_path.endswith("multiseed_summary.csv")


def test_summary_rejects_post_reset_collision_counter(tmp_path):
    with pytest.raises(ValueError, match="after an environment reset"):
        _write_run(
            tmp_path, 303, vehicle_collisions=0, road_collisions=0,
            speeds=[(.6, .4), (.6, .4)],
        )
