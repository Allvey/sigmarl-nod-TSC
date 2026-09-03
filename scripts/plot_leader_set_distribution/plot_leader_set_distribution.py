import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import openpyxl


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_TITLES = {
    "Training Stage": "(a) Training Stage",
    "Training Stage (last 20%)": "(a) Training Stage",
    "Evaluation Stage": "(b) Evaluation Stage",
}


SERIES = [
    {
        "key": "p0",
        "label": r"$|\mathcal{L}_i^t|=0$",
        "color": "#c7ddf0",
        "column": 2,
    },
    {
        "key": "p1",
        "label": r"$|\mathcal{L}_i^t|=1$",
        "color": "#d4d4d4",
        "column": 3,
    },
    {
        "key": "p2",
        "label": r"$|\mathcal{L}_i^t|=2$",
        "color": "#6f8fb3",
        "column": 4,
    },
]


def _get_pyplot():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _apply_style(plt):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_delta(value, setting):
    if abs(float(value) - 0.0) < 1e-12:
        return "0"
    if setting and str(setting).lower() == "default":
        return f"{float(value):.2f}"
    return f"{float(value):.2f}"


def load_leader_set_distribution(xlsx_path: Path):
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=False)
    ws = wb.active
    rows_by_phase = {}

    for row in ws.iter_rows(min_row=3, values_only=True):
        phase = row[0]
        if phase is None:
            continue
        delta_p = row[1]
        if delta_p is None:
            continue
        item = {
            "delta_p": float(delta_p),
            "setting": row[6],
            "p0": float(row[2]),
            "p1": float(row[3]),
            "p2": float(row[4]),
        }
        rows_by_phase.setdefault(str(phase), []).append(item)

    for phase in rows_by_phase:
        rows_by_phase[phase].sort(key=lambda item: item["delta_p"])
    return rows_by_phase


def plot_distribution(rows_by_phase, output_base: Path):
    plt = _get_pyplot()
    _apply_style(plt)

    phases = ["Training Stage (last 20%)", "Evaluation Stage"]
    fig, axes = plt.subplots(1, 2, figsize=(5, 4.45), sharey=True)

    legend_handles = None
    legend_labels = None
    for ax, phase in zip(axes, phases):
        rows = rows_by_phase[phase]
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows), dtype=float)
        width = 0.62
        handles = []

        for series in SERIES:
            values = np.asarray([row[series["key"]] for row in rows], dtype=float)
            bars = ax.bar(
                x,
                values,
                width=width,
                bottom=bottoms,
                color=series["color"],
                edgecolor="white",
                linewidth=0.35,
                label=series["label"],
            )
            handles.append(bars[0])
            for idx, (bar, value, bottom) in enumerate(zip(bars, values, bottoms)):
                if value >= 2.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + value / 2,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_format_delta(row["delta_p"], row["setting"]) for row in rows]
        )
        ax.set_xlabel(r"$\Delta p$ (leader selection margin)")
        ax.text(
            0.5,
            -0.29,
            PHASE_TITLES.get(phase, phase),
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", linestyle="--", color="#bfbfbf", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

        legend_handles = handles[::-1]
        legend_labels = [series["label"] for series in SERIES[::-1]]

    axes[0].set_ylabel("Proportion (%)")
    axes[1].set_ylabel("Proportion (%)")
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=False,
            columnspacing=1.8,
            handlelength=2.4,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 0.90), w_pad=2.4)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot stacked bar charts for local leader-set size distribution."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "inputs" / "leader_set_distribution_extracted.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "leader_set_distribution_stacked",
    )
    args = parser.parse_args()

    rows_by_phase = load_leader_set_distribution(args.input)
    png_path, pdf_path = plot_distribution(rows_by_phase, args.output)
    print(f"Saved figures:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
