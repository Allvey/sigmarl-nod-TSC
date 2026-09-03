import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REWARD_KEY = "episode_reward_mean_list"
DEFAULT_COLLISION_KEY = "collision_agents_rate_list"


COLLISION_LABELS = {
    "collision_agents_rate_list": "Agent-Agent Collision Rate",
    "collision_lanelets_rate_list": "Agent-Lanelet Collision Rate",
    "collision_total_rate_list": "Total collision rate",
}


COLLISION_ALIASES = {
    "collision_agents_rate_list": ("agent_collision_rate_list", 0.01),
    "collision_lanelets_rate_list": ("lanelet_collision_rate_list", 0.01),
    "collision_total_rate_list": ("total_collision_rate_list", 0.01),
}


# DEFAULT_METHODS = [
#     f"{SCRIPT_DIR / 'inputs' / 'TSC'}:TSC:#8172b2",
#     f"{SCRIPT_DIR / 'inputs' / 'XPMarl'}:XP-MARL:#d62728",
#     f"{SCRIPT_DIR / 'inputs' / 'SigmaRL'}:SigmaRL:#2ca02c",
#     f"{SCRIPT_DIR / 'inputs' / 'MFPO'}:MFPO:#1f77b4",
# ]

DEFAULT_METHODS = [
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'TSC'}:TSC:#8172b2",
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'demo'}:TSC w/o Top-K filter:#1f77b4",
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'demo1'}:TSC w/ random priority:#2ca02c",
    # f"{SCRIPT_DIR / 'inputs/Ablation' / 'demo2'}:TSC w/o Lcons:#d62728",
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'demo2'}:TSC w/o $L_{{\\mathrm{{cons}}}}$:#d62728",
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'randomP'}:TSC w/o Stackelberg:#ff9300",
    f"{SCRIPT_DIR / 'inputs/Ablation' / 'NearestTopK'}:TSC w/ Euclidean Top-K:#CC79A7",
    # f"{SCRIPT_DIR / 'inputs/Ablation' / 'woLcons'}:woLcons:#1f77b4",
]


# METHOD_LINESTYLES = {
#     "TSC": "-",
#     "XP-MARL": "--",
#     "SigmaRL": "--",
#     "MFPO": "--",
# }

METHOD_LINESTYLES = {
    "TSC": "-",
    "TSC w/o Top-K filter": "--",
    "TSC w/ random priority": "--",
    r"TSC w/o $L_{\mathrm{cons}}$": "--",
    "TSC w/o Stackelberg": "--",
    "TSC w/ Euclidean Top-K": "--",
}


def _get_pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import matplotlib. Your current Python environment may have "
            "an incompatible NumPy/matplotlib build. Try running this script with "
            "`/usr/bin/python3` on this machine, or reinstall matplotlib for the "
            "active environment."
        ) from exc
    return plt


def _apply_paper_style(plt):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_seed_args(seed_args):
    seeds = []
    for token in seed_args:
        for part in str(token).split(","):
            part = part.strip()
            if part:
                seeds.append(part)
    return seeds


def _reward_from_filename(path: Path):
    match = re.search(r"reward(-?[0-9]*\.?[0-9]+)_data\.json$", path.name)
    return float(match.group(1)) if match else float("-inf")


def _find_run_json(input_root: Path, seed: str):
    candidates = []
    for seed_dir_name in (f"seed{seed}", f"seed_{seed}"):
        seed_dir = input_root / seed_dir_name
        if seed_dir.exists():
            candidates.extend(seed_dir.glob("reward*_data.json"))
    if not candidates:
        raise FileNotFoundError(f"No reward*_data.json found for seed {seed}")
    return max(candidates, key=_reward_from_filename)


def _find_all_seed_jsons(input_root: Path):
    candidates = []
    for seed_dir in sorted(input_root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        reward_files = list(seed_dir.glob("reward*_data.json"))
        if reward_files:
            candidates.append(max(reward_files, key=_reward_from_filename))
    if not candidates:
        raise FileNotFoundError(f"No seed*/reward*_data.json found in {input_root}")
    return candidates


def _parse_method_specs(method_args):
    methods = []
    for spec in method_args:
        parts = spec.split(":")
        if len(parts) == 1:
            root = parts[0]
            label = Path(root).name
            color = None
        elif len(parts) == 2:
            root, label = parts
            color = None
        else:
            root, label, color = parts[0], parts[1], ":".join(parts[2:])
        methods.append({"root": Path(root), "label": label, "color": color})
    return methods


def _get_collision_series(data, collision_key: str):
    if collision_key in data:
        return data[collision_key]
    alias = COLLISION_ALIASES.get(collision_key)
    if alias is not None:
        alias_key, alias_scale = alias
        if alias_key in data:
            return [float(value) * alias_scale for value in data[alias_key]]
    expected = [collision_key]
    if alias is not None:
        expected.append(alias[0])
    raise KeyError(f"Missing collision series. Expected one of: {expected}")


def _load_series(json_paths, collision_key: str):
    runs = []
    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        reward = data.get(REWARD_KEY)
        if reward is None:
            raise KeyError(f"{path} must contain {REWARD_KEY}")
        collision = _get_collision_series(data, collision_key)
        runs.append(
            {
                "name": path.parent.name,
                "path": path,
                "reward": np.asarray(reward, dtype=float),
                "collision": np.asarray(collision, dtype=float),
                "frames_per_batch": data.get("parameters", {}).get(
                    "frames_per_batch", 1
                ),
            }
        )
    return runs


def _trim_to_common_length(arrays):
    min_len = min(len(arr) for arr in arrays)
    if min_len <= 0:
        raise ValueError("At least one run has an empty series")
    return np.vstack([arr[:min_len] for arr in arrays])


def _get_step_axis(runs, num_points: int):
    frames_per_batch_values = [run["frames_per_batch"] for run in runs]
    frames_per_batch = frames_per_batch_values[0]
    if any(value != frames_per_batch for value in frames_per_batch_values):
        print(
            "[WARN] Different frames_per_batch values found across runs; "
            f"using {frames_per_batch} from {runs[0]['name']} for the x-axis."
        )
    return np.arange(1, num_points + 1) * int(frames_per_batch)


def _truncate_by_max_steps(values, environment_steps, max_environment_steps):
    if max_environment_steps is None:
        return values, environment_steps
    keep_mask = environment_steps <= max_environment_steps
    if not np.any(keep_mask):
        raise ValueError(
            f"No data points remain after applying --max-environment-steps "
            f"{max_environment_steps}"
        )
    keep_count = int(keep_mask.sum())
    return values[:, :keep_count], environment_steps[:keep_count]


def _moving_average(values, window: int, direction: str):
    if window <= 1:
        return values
    result = np.empty_like(values, dtype=float)
    num_steps = values.shape[1]
    for run_idx in range(values.shape[0]):
        for step_idx in range(num_steps):
            if direction == "forward":
                end = min(num_steps, step_idx + window)
                start = max(0, end - window)
            elif direction == "centered":
                left = window // 2
                right = window - left
                start = max(0, step_idx - left)
                end = min(num_steps, step_idx + right)
            else:
                start = max(0, step_idx - window + 1)
                end = step_idx + 1
            result[run_idx, step_idx] = values[run_idx, start:end].mean()
    return result


def _plot_mean_curve(
    method_series,
    ylabel,
    title,
    output_base: Path,
    smooth_window: int,
    smooth_direction: str,
    xlabel: str,
    max_environment_steps,
    show_title: bool,
    legend_loc: str,
    is_collision_plot: bool,
):
    plt = _get_pyplot()
    _apply_paper_style(plt)
    from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

    fig, ax = plt.subplots(figsize=(4.6, 3.25))

    for series in method_series:
        values = _moving_average(series["values"], smooth_window, smooth_direction)
        x = series["environment_steps"][: values.shape[1]]
        values, x = _truncate_by_max_steps(values, x, max_environment_steps)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        color = series["color"]
        linestyle = series.get("linestyle", "-")

        ax.plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=series["label"],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    if show_title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, frameon=False)
    if is_collision_plot:
        ax.set_ylim(0.0, 0.02)
        ax.set_yticks(np.arange(0.0, 0.0201, 0.004))
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.55)
    ax.grid(True, which="minor", color="#f2f2f2", linewidth=0.35, alpha=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_powerlimits((6, 6))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    if max_environment_steps is not None:
        ax.set_xlim(0, max_environment_steps)
        ax.set_xticks(np.linspace(0, max_environment_steps, 6))
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.6,
    )
    ax.tick_params(which="minor", length=1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    if max_environment_steps is not None:
        x_pad = 0.02 * max_environment_steps
        ax.set_xlim(-x_pad, max_environment_steps + x_pad)
    else:
        x_span = float(x[-1] - x[0]) if len(x) > 1 else float(x[-1])
        x_pad = 0.02 * x_span
        ax.set_xlim(float(x[0]) - x_pad, float(x[-1]) + x_pad)
    ax.margins(y=0.05)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean reward and total collision curves across random seeds."
    )
    parser.add_argument("--input-root", type=Path, default=SCRIPT_DIR / "inputs" / "TSC")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help=(
            "Methods to plot as root:label:color. Default plots "
            "the local inputs/TSC, inputs/SigmaRL, and inputs/XPMarl folders."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=[],
        help=(
            "Optional seed ids to use for every method, e.g. --seeds 0 1 2. "
            "If omitted, each method uses all available seed folders."
        ),
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Use all seed*/reward*_data.json files under input-root.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing moving-average window. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--smooth-direction",
        type=str,
        default="forward",
        choices=["trailing", "forward", "centered"],
        help=(
            "Smoothing direction. trailing uses past points, forward uses future "
            "points, centered uses points on both sides."
        ),
    )
    parser.add_argument(
        "--collision-key",
        type=str,
        default=DEFAULT_COLLISION_KEY,
        choices=[
            "collision_agents_rate_list",
            "collision_lanelets_rate_list",
            "collision_total_rate_list",
        ],
        help="Collision series to plot. Default matches outputs/TSC/TSC.xlsx.",
    )
    parser.add_argument(
        "--collision-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the selected collision series. Default 1 keeps the "
            "raw values used in TSC.xlsx. Use 100 if you want percentage points."
        ),
    )
    parser.add_argument(
        "--max-environment-steps",
        type=int,
        default=1_000_000,
        help=(
            "Display curves up to this many environment steps. Smoothing still uses "
            "available data beyond this limit. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--method-label",
        type=str,
        default="TSC",
        help="Deprecated single-method legend label. Use --methods instead.",
    )
    parser.add_argument(
        "--line-color",
        type=str,
        default="#9467bd",
        help="Matplotlib color for the mean curve and std shading.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Show plot titles. Default is off to match paper-style figures.",
    )
    args = parser.parse_args()

    methods = _parse_method_specs(args.methods)
    input_root = args.input_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seed_args(args.seeds)
    reward_method_series = []
    collision_method_series = []
    default_colors = ["#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    loaded_method_runs = []

    for method_idx, method in enumerate(methods):
        root = method["root"]
        if seeds and not args.all_seeds:
            json_paths = [_find_run_json(root, seed) for seed in seeds]
        else:
            json_paths = _find_all_seed_jsons(root)

        runs = _load_series(json_paths, args.collision_key)
        reward_values = _trim_to_common_length([run["reward"] for run in runs])
        collision_values = _trim_to_common_length(
            [run["collision"] * args.collision_scale for run in runs]
        )
        environment_steps = _get_step_axis(runs, reward_values.shape[1])
        color = method["color"] or default_colors[method_idx % len(default_colors)]
        linestyle = METHOD_LINESTYLES.get(method["label"], "-")

        reward_method_series.append(
            {
                "label": method["label"],
                "color": color,
                "linestyle": linestyle,
                "values": reward_values,
                "environment_steps": environment_steps,
            }
        )
        collision_method_series.append(
            {
                "label": method["label"],
                "color": color,
                "linestyle": linestyle,
                "values": collision_values,
                "environment_steps": environment_steps,
            }
        )
        loaded_method_runs.append((method, runs))

    max_environment_steps = (
        args.max_environment_steps if args.max_environment_steps > 0 else None
    )

    collision_label = COLLISION_LABELS.get(args.collision_key, args.collision_key)
    collision_output_stem = args.collision_key.replace("_rate_list", "_mean_curve")

    reward_outputs = _plot_mean_curve(
        method_series=reward_method_series,
        ylabel="Episode Mean Reward",
        title="Training Reward Across Seeds",
        output_base=output_dir / "reward_mean_curve",
        smooth_window=max(1, args.smooth_window),
        smooth_direction=args.smooth_direction,
        xlabel="Environmental Step",
        max_environment_steps=max_environment_steps,
        show_title=args.show_title,
        legend_loc="upper left",
        is_collision_plot=False,
    )
    collision_outputs = _plot_mean_curve(
        method_series=collision_method_series,
        ylabel=f"{collision_label} [%]",
        title=f"{collision_label} Across Seeds",
        output_base=output_dir / collision_output_stem,
        smooth_window=max(1, args.smooth_window),
        smooth_direction=args.smooth_direction,
        xlabel="Environmental Step",
        max_environment_steps=max_environment_steps,
        show_title=args.show_title,
        legend_loc="upper right",
        is_collision_plot=True,
    )

    print("Loaded runs:")
    for method, runs in loaded_method_runs:
        print(f"  {method['label']} ({method['root']}):")
        for run in runs:
            print(f"    {run['name']}: {run['path']}")
        run_length = min(len(run["reward"]) for run in runs)
        displayed_steps = _get_step_axis(runs, run_length)
        if max_environment_steps is not None:
            displayed_steps = displayed_steps[displayed_steps <= max_environment_steps]
        print(
            "    displayed x-axis environment steps: "
            f"{int(displayed_steps[0])} to {int(displayed_steps[-1])}"
        )
    print("Saved figures:")
    for path in reward_outputs + collision_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
