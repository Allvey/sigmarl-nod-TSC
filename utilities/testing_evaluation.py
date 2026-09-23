"""File-based summaries for reproducible mixed-controller evaluations."""
import csv
import json
import os
import statistics


def rule_run_name(scenario, fraction, profile_weights, *, assignment_seed,
                  environment_seed, actor_index, controller_version):
    mix = "_".join(f"{name}_{profile_weights[name]:g}" for name in profile_weights)
    actor = "all" if actor_index is None else str(actor_index)
    name = (f"{scenario}_fraction_{fraction:g}_{mix}"
            f"_assignseed{assignment_seed}_envseed{environment_seed}"
            f"_actor{actor}_{controller_version}")
    return name.replace(".", "p")


def _group_metrics(values, ids, dt, stop_threshold):
    selected = [value for agent in ids for value in values[agent]]
    if not selected:
        return dict(mean_speed=0., median_speed=0., stop_fraction=0.,
                    slow_fraction=0., longest_stop_seconds=0.)
    longest = 0
    for agent in ids:
        current = 0
        for value in values[agent]:
            current = current + 1 if value < stop_threshold else 0
            longest = max(longest, current)
    return dict(
        mean_speed=statistics.mean(selected),
        median_speed=statistics.median(selected),
        stop_fraction=sum(value < stop_threshold for value in selected) / len(selected),
        slow_fraction=sum(value < .2 for value in selected) / len(selected),
        longest_stop_seconds=longest * dt,
    )


def summarize_rule_run(output_dir, *, vehicle_collisions, road_collisions,
                       dt, stop_threshold=.05,
                       collision_count_source="rollout_cumulative_counter"):
    setup_path = os.path.join(output_dir, "rule_setup.json")
    speed_path = os.path.join(output_dir, "agent_speeds.csv")
    diagnostic_path = os.path.join(output_dir, "rule_diagnostics.csv")
    with open(setup_path, encoding="utf-8") as file:
        setup = json.load(file)
    with open(speed_path, newline="", encoding="utf-8") as file:
        speed_rows = list(csv.DictReader(file))
    if not speed_rows:
        raise ValueError(f"No speed samples were written to {speed_path}")
    agent_fields = [name for name in speed_rows[0] if name.startswith("agent_")]
    agent_ids = [int(name.split("_")[1]) for name in agent_fields]
    values = {
        agent: [float(row[f"agent_{agent}_speed"]) for row in speed_rows]
        for agent in agent_ids
    }
    rule_ids = sorted(int(agent) + 1 for agent in setup.get("vehicles", {}))
    actor_ids = sorted(set(agent_ids) - set(rule_ids))
    with open(diagnostic_path, newline="", encoding="utf-8") as file:
        diagnostics = list(csv.DictReader(file))
    emergency_frames = sum(int(row.get("emergency_brake", 0)) for row in diagnostics)
    rule_road_involvements = sum(
        max((int(row["road_events"]) for row in diagnostics
             if int(row["agent"]) == agent), default=0)
        for agent in rule_ids
    )
    rule_vehicle_involvements = sum(
        max((int(row["vehicle_contact_events"]) for row in diagnostics
             if int(row["agent"]) == agent), default=0)
        for agent in rule_ids
    )
    if rule_vehicle_involvements > 2 * vehicle_collisions:
        raise ValueError(
            "Vehicle collision counter is inconsistent with rule diagnostics; "
            "the counter may have been read after an environment reset"
        )
    if rule_road_involvements > road_collisions:
        raise ValueError(
            "Road collision counter is inconsistent with rule diagnostics; "
            "the counter may have been read after an environment reset"
        )
    summary = dict(
        environment_seed=int(setup["seed"]),
        assignment_seed=setup.get("assignment_seed"),
        checkpoint=setup.get("checkpoint"),
        controller_version=setup.get("controller_version"),
        steps=len(speed_rows),
        duration_seconds=len(speed_rows) * dt,
        vehicle_collisions=int(vehicle_collisions),
        road_collisions=int(road_collisions),
        total_collisions=int(vehicle_collisions + road_collisions),
        collision_count_source=collision_count_source,
        emergency_frames=emergency_frames,
        emergency_seconds=emergency_frames * dt,
        rule_vehicle_collision_involvements=rule_vehicle_involvements,
        rule_road_collision_involvements=rule_road_involvements,
        rule_rule_contact_detected=any(int(row.get("rule_contact", 0)) for row in diagnostics),
        actor=_group_metrics(values, actor_ids, dt, stop_threshold),
        rule=_group_metrics(values, rule_ids, dt, stop_threshold),
        all=_group_metrics(values, agent_ids, dt, stop_threshold),
        per_agent={str(agent): _group_metrics(values, [agent], dt, stop_threshold)
                   for agent in agent_ids},
    )
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def aggregate_rule_runs(summary_paths, output_dir):
    runs = []
    for path in summary_paths:
        with open(path, encoding="utf-8") as file:
            run = json.load(file)
        run["summary_path"] = os.path.abspath(path)
        runs.append(run)
    if not runs:
        raise ValueError("At least one completed run is required")
    fields = {
        "vehicle_collisions": lambda r: r["vehicle_collisions"],
        "road_collisions": lambda r: r["road_collisions"],
        "total_collisions": lambda r: r["total_collisions"],
        "rule_vehicle_collision_involvements":
            lambda r: r["rule_vehicle_collision_involvements"],
        "rule_road_collision_involvements":
            lambda r: r["rule_road_collision_involvements"],
        "actor_mean_speed": lambda r: r["actor"]["mean_speed"],
        "rule_mean_speed": lambda r: r["rule"]["mean_speed"],
        "all_mean_speed": lambda r: r["all"]["mean_speed"],
        "actor_stop_fraction": lambda r: r["actor"]["stop_fraction"],
        "rule_stop_fraction": lambda r: r["rule"]["stop_fraction"],
        "all_stop_fraction": lambda r: r["all"]["stop_fraction"],
        "actor_longest_stop_seconds": lambda r: r["actor"]["longest_stop_seconds"],
        "rule_longest_stop_seconds": lambda r: r["rule"]["longest_stop_seconds"],
        "longest_stop_seconds": lambda r: r["all"]["longest_stop_seconds"],
        "emergency_seconds": lambda r: r["emergency_seconds"],
    }
    aggregate = {}
    for name, getter in fields.items():
        values = [float(getter(run)) for run in runs]
        aggregate[name] = dict(
            mean=statistics.mean(values),
            std=statistics.pstdev(values),
            min=min(values),
            max=max(values),
            sum=sum(values),
        )
    result = dict(
        n_runs=len(runs),
        seeds=[run["environment_seed"] for run in runs],
        collision_free_runs=sum(run["total_collisions"] == 0 for run in runs),
        collision_free_rate=sum(run["total_collisions"] == 0 for run in runs) / len(runs),
        rule_rule_contact_runs=sum(bool(run["rule_rule_contact_detected"]) for run in runs),
        aggregate=aggregate,
        runs=runs,
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "multiseed_summary.json")
    csv_path = os.path.join(output_dir, "multiseed_summary.csv")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    flat_fields = ["environment_seed", *fields, "rule_rule_contact_detected", "summary_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=flat_fields)
        writer.writeheader()
        for run in runs:
            row = {"environment_seed": run["environment_seed"],
                   "rule_rule_contact_detected": int(run["rule_rule_contact_detected"]),
                   "summary_path": run["summary_path"]}
            row.update({name: getter(run) for name, getter in fields.items()})
            writer.writerow(row)
    return result, json_path, csv_path
