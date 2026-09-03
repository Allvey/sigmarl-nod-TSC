import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

raw_text = """
TSC:
=========================================
Scenario: CPM_entire
Num agents: 20
[LOG] Agent-agent collision rate [%]: mean=0.04, min=0.00, max=0.17
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=0.07, min=0.00, max=0.33
=========================================
Scenario: intersection_2
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.18, min=0.00, max=0.50
[LOG] Agent-lanelet collision rate [%]: mean=0.81, min=0.33, max=1.33
[LOG] Total collision rate [%]: mean=0.99, min=0.42, max=1.67
=========================================
Scenario: on_ramp_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.03, min=0.00, max=0.17
=========================================
Scenario: roundabout_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.08, min=0.00, max=0.25
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.10, min=0.00, max=0.33

=========================================
Scenario: CPM_entire
Num agents: 15
[LOG] Agent-agent collision rate [%]: mean=0.02, min=0.00, max=0.08
[LOG] Agent-lanelet collision rate [%]: mean=0.04, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=0.06, min=0.00, max=0.25
=========================================
Scenario: intersection_2
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.10, min=0.00, max=0.25
[LOG] Agent-lanelet collision rate [%]: mean=0.70, min=0.25, max=1.17
[LOG] Total collision rate [%]: mean=0.79, min=0.42, max=1.33
=========================================
Scenario: on_ramp_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.02, min=0.00, max=0.08
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.02, min=0.00, max=0.08
=========================================
Scenario: roundabout_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.01, min=0.00, max=0.08
[LOG] Agent-lanelet collision rate [%]: mean=0.24, min=0.00, max=0.67
[LOG] Total collision rate [%]: mean=0.26, min=0.00, max=0.67
=========================================

Scenario: CPM_entire
Num agents: 25
[LOG] Agent-agent collision rate [%]: mean=0.05, min=0.00, max=0.25
[LOG] Agent-lanelet collision rate [%]: mean=0.06, min=0.00, max=0.33
[LOG] Total collision rate [%]: mean=0.11, min=0.00, max=0.42
=========================================
Scenario: intersection_2
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=0.46, min=0.17, max=0.83
[LOG] Agent-lanelet collision rate [%]: mean=0.84, min=0.17, max=1.42
[LOG] Total collision rate [%]: mean=1.30, min=0.67, max=2.00
=========================================
Scenario: on_ramp_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=0.29, min=0.08, max=0.67
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.29, min=0.08, max=0.67
=========================================
Scenario: roundabout_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=0.44, min=0.17, max=0.83
[LOG] Agent-lanelet collision rate [%]: mean=0.02, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.45, min=0.17, max=0.92
=========================================

Scenario: CPM_entire
Num agents: 30
[LOG] Agent-agent collision rate [%]: mean=0.14, min=0.00, max=0.58
[LOG] Agent-lanelet collision rate [%]: mean=0.06, min=0.00, max=0.25
[LOG] Total collision rate [%]: mean=0.21, min=0.00, max=0.83
=========================================
Scenario: intersection_2
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=0.99, min=0.50, max=1.50
[LOG] Agent-lanelet collision rate [%]: mean=0.93, min=0.42, max=1.42
[LOG] Total collision rate [%]: mean=1.92, min=1.00, max=2.59
=========================================
Scenario: on_ramp_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=2.73, min=2.00, max=3.42
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=2.73, min=2.00, max=3.42
=========================================
Scenario: roundabout_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=1.23, min=0.67, max=1.75
[LOG] Agent-lanelet collision rate [%]: mean=0.02, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=1.25, min=0.67, max=1.75
=========================================

MFPO:
Scenario: CPM_entire
Num agents: 20
[LOG] Agent-agent collision rate [%]: mean=1.71, min=0.42, max=3.00
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.71, min=0.42, max=3.00
=========================================
Scenario: intersection_2
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=5.46, min=4.09, max=7.09
[LOG] Agent-lanelet collision rate [%]: mean=2.29, min=1.50, max=3.00
[LOG] Total collision rate [%]: mean=7.75, min=6.17, max=9.26
=========================================
Scenario: on_ramp_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=4.37, min=3.34, max=5.67
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=4.37, min=3.34, max=5.67
=========================================
Scenario: roundabout_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=4.80, min=3.84, max=5.59
[LOG] Agent-lanelet collision rate [%]: mean=0.78, min=0.33, max=1.33
[LOG] Total collision rate [%]: mean=5.57, min=4.59, max=6.26
=========================================
Scenario: CPM_entire
Num agents: 15
[LOG] Agent-agent collision rate [%]: mean=0.90, min=0.08, max=1.50
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.90, min=0.08, max=1.50
=========================================
Scenario: intersection_2
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=2.77, min=2.09, max=3.92
[LOG] Agent-lanelet collision rate [%]: mean=1.65, min=1.08, max=2.42
[LOG] Total collision rate [%]: mean=4.42, min=3.34, max=5.59
=========================================
Scenario: on_ramp_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=1.78, min=0.92, max=2.42
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.78, min=0.92, max=2.42
=========================================
Scenario: roundabout_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=2.71, min=2.17, max=3.42
[LOG] Agent-lanelet collision rate [%]: mean=0.42, min=0.08, max=0.92
[LOG] Total collision rate [%]: mean=3.13, min=2.25, max=3.75
=========================================
Scenario: CPM_entire
Num agents: 25
[LOG] Agent-agent collision rate [%]: mean=2.11, min=1.25, max=3.34
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=2.11, min=1.25, max=3.34
=========================================
Scenario: intersection_2
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=9.23, min=8.42, max=10.93
[LOG] Agent-lanelet collision rate [%]: mean=2.95, min=2.25, max=3.84
[LOG] Total collision rate [%]: mean=12.18, min=10.68, max=13.68
=========================================
Scenario: on_ramp_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=7.94, min=6.42, max=9.01
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=7.94, min=6.42, max=9.01
=========================================
Scenario: roundabout_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=6.93, min=5.84, max=7.92
[LOG] Agent-lanelet collision rate [%]: mean=0.77, min=0.50, max=1.17
[LOG] Total collision rate [%]: mean=7.71, min=7.01, max=8.76
=========================================
Scenario: CPM_entire
Num agents: 30
[LOG] Agent-agent collision rate [%]: mean=3.12, min=1.50, max=4.42
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=3.12, min=1.50, max=4.42
=========================================
Scenario: intersection_2
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=15.09, min=12.51, max=16.35
[LOG] Agent-lanelet collision rate [%]: mean=3.72, min=2.75, max=5.25
[LOG] Total collision rate [%]: mean=18.81, min=16.18, max=20.77
=========================================
Scenario: on_ramp_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=12.12, min=11.09, max=13.26
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=12.12, min=11.09, max=13.26
=========================================
Scenario: roundabout_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=9.29, min=8.34, max=10.34
[LOG] Agent-lanelet collision rate [%]: mean=0.88, min=0.50, max=1.58
[LOG] Total collision rate [%]: mean=10.18, min=9.17, max=11.18
=========================================

SigmaRL:
Scenario: CPM_entire
Num agents: 30
[LOG] Agent-agent collision rate [%]: mean=2.90, min=1.50, max=3.84
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=2.90, min=1.50, max=3.84
=========================================
Scenario: intersection_2
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=15.31, min=13.84, max=17.93
[LOG] Agent-lanelet collision rate [%]: mean=1.50, min=0.75, max=2.25
[LOG] Total collision rate [%]: mean=16.80, min=15.18, max=19.43
=========================================
Scenario: on_ramp_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=11.58, min=10.68, max=12.59
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=11.58, min=10.68, max=12.59
=========================================
Scenario: roundabout_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=9.77, min=8.92, max=10.43
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=9.79, min=8.92, max=10.43
=========================================


Scenario: CPM_entire
Num agents: 25
[LOG] Agent-agent collision rate [%]: mean=2.23, min=1.50, max=3.17
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=2.23, min=1.50, max=3.17
=========================================
Scenario: intersection_2
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=9.82, min=8.67, max=11.59
[LOG] Agent-lanelet collision rate [%]: mean=1.25, min=0.75, max=1.83
[LOG] Total collision rate [%]: mean=11.07, min=9.51, max=12.84
=========================================
Scenario: on_ramp_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=7.51, min=6.26, max=8.51
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=7.51, min=6.26, max=8.51
=========================================
Scenario: roundabout_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=7.32, min=6.17, max=8.34
[LOG] Agent-lanelet collision rate [%]: mean=0.01, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=7.34, min=6.17, max=8.34
=========================================

Scenario: CPM_entire
Num agents: 20
[LOG] Agent-agent collision rate [%]: mean=1.67, min=0.83, max=2.25
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.67, min=0.83, max=2.25
=========================================
Scenario: intersection_2
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=5.45, min=4.42, max=6.51
[LOG] Agent-lanelet collision rate [%]: mean=0.95, min=0.50, max=1.67
[LOG] Total collision rate [%]: mean=6.40, min=5.25, max=7.84
=========================================
Scenario: on_ramp_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=4.28, min=3.25, max=5.59
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=4.28, min=3.25, max=5.59
=========================================
Scenario: roundabout_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=5.02, min=4.17, max=5.92
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.25
[LOG] Total collision rate [%]: mean=5.06, min=4.25, max=5.92
=========================================

Scenario: CPM_entire
Num agents: 15
[LOG] Agent-agent collision rate [%]: mean=0.93, min=0.17, max=1.58
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.93, min=0.17, max=1.58
=========================================
Scenario: intersection_2
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=2.87, min=2.00, max=3.84
[LOG] Agent-lanelet collision rate [%]: mean=0.84, min=0.25, max=1.42
[LOG] Total collision rate [%]: mean=3.71, min=2.75, max=4.50
=========================================
Scenario: on_ramp_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=1.72, min=1.00, max=2.59
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.72, min=1.00, max=2.59
=========================================
Scenario: roundabout_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=3.01, min=2.50, max=3.92
[LOG] Agent-lanelet collision rate [%]: mean=0.08, min=0.00, max=0.33
[LOG] Total collision rate [%]: mean=3.09, min=2.59, max=4.00
=========================================

XP-MARL:
Scenario: CPM_entire
Num agents: 20
[LOG] Agent-agent collision rate [%]: mean=0.19, min=0.00, max=0.42
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.22, min=0.00, max=0.50
=========================================
Scenario: intersection_2
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.69, min=0.25, max=1.17
[LOG] Agent-lanelet collision rate [%]: mean=0.98, min=0.50, max=1.33
[LOG] Total collision rate [%]: mean=1.66, min=1.00, max=2.25
=========================================
Scenario: on_ramp_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.24, min=0.00, max=0.58
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.24, min=0.00, max=0.58
=========================================
Scenario: roundabout_1
Num agents: 8
[LOG] Agent-agent collision rate [%]: mean=0.56, min=0.08, max=1.25
[LOG] Agent-lanelet collision rate [%]: mean=0.07, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=0.62, min=0.08, max=1.25
=========================================
Scenario: CPM_entire
Num agents: 15
[LOG] Agent-agent collision rate [%]: mean=0.11, min=0.00, max=0.33
[LOG] Agent-lanelet collision rate [%]: mean=0.07, min=0.00, max=0.33
[LOG] Total collision rate [%]: mean=0.18, min=0.00, max=0.42
=========================================
Scenario: intersection_2
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.48, min=0.17, max=0.75
[LOG] Agent-lanelet collision rate [%]: mean=0.79, min=0.33, max=1.25
[LOG] Total collision rate [%]: mean=1.27, min=0.67, max=2.00
=========================================
Scenario: on_ramp_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.10, min=0.00, max=0.33
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.10, min=0.00, max=0.33
=========================================
Scenario: roundabout_1
Num agents: 6
[LOG] Agent-agent collision rate [%]: mean=0.24, min=0.00, max=0.67
[LOG] Agent-lanelet collision rate [%]: mean=0.27, min=0.00, max=0.50
[LOG] Total collision rate [%]: mean=0.51, min=0.25, max=0.92
=========================================
Scenario: CPM_entire
Num agents: 25
[LOG] Agent-agent collision rate [%]: mean=0.24, min=0.00, max=0.58
[LOG] Agent-lanelet collision rate [%]: mean=0.04, min=0.00, max=0.25
[LOG] Total collision rate [%]: mean=0.28, min=0.00, max=0.67
=========================================
Scenario: intersection_2
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=1.06, min=0.58, max=1.50
[LOG] Agent-lanelet collision rate [%]: mean=1.20, min=0.67, max=2.25
[LOG] Total collision rate [%]: mean=2.27, min=1.25, max=3.34
=========================================
Scenario: on_ramp_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=0.74, min=0.25, max=1.33
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.74, min=0.25, max=1.33
=========================================
Scenario: roundabout_1
Num agents: 10
[LOG] Agent-agent collision rate [%]: mean=1.36, min=0.92, max=2.00
[LOG] Agent-lanelet collision rate [%]: mean=0.01, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=1.37, min=0.92, max=2.00
=========================================
Scenario: CPM_entire
Num agents: 30
[LOG] Agent-agent collision rate [%]: mean=0.42, min=0.17, max=0.83
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.45, min=0.17, max=0.92
=========================================
Scenario: intersection_2
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=1.70, min=1.00, max=2.50
[LOG] Agent-lanelet collision rate [%]: mean=1.19, min=0.42, max=1.75
[LOG] Total collision rate [%]: mean=2.90, min=2.00, max=3.67
=========================================
Scenario: on_ramp_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=3.07, min=2.25, max=3.92
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=3.07, min=2.25, max=3.92
=========================================
Scenario: roundabout_1
Num agents: 12
[LOG] Agent-agent collision rate [%]: mean=2.75, min=1.83, max=3.42
[LOG] Agent-lanelet collision rate [%]: mean=0.01, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=2.76, min=1.83, max=3.42
=========================================
"""


def parse_collision_from_text(text):
    data = {}
    current_method = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_method = re.match(r"([\w\-]+):\s*$", line)
        if m_method:
            current_method = m_method.group(1)
            i += 1
            continue
        m_scenario = re.match(r"Scenario:\s+(\S+)", line)
        if m_scenario and current_method is not None:
            scenario = m_scenario.group(1)
            j = i + 1
            num_agents = None
            mean_val = None
            while (
                j < len(lines)
                and "=========================================" not in lines[j]
            ):
                line_j = lines[j]
                m_num = re.search(r"Num agents:\s*([0-9]+)", line_j)
                if m_num:
                    num_agents = int(m_num.group(1))
                m_total = re.search(
                    r"Total collision rate \[%\]: mean=([0-9.]+)", line_j
                )
                if m_total:
                    mean_val = float(m_total.group(1))
                j += 1
            if num_agents is not None and mean_val is not None:
                if scenario not in data:
                    data[scenario] = {}
                data[scenario].setdefault(current_method, []).append(
                    (num_agents, mean_val)
                )
            i = j
            continue
        i += 1
    return data


data = parse_collision_from_text(raw_text)

scenario_order = ["CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1"]
scenario_titles = {
    "CPM_entire": "Clover",
    "intersection_2": "Weave",
    "on_ramp_1": "Merge",
    "roundabout_1": "Bypass",
}

plt.style.use("seaborn-v0_8")

output_dir = os.path.join(SCRIPT_DIR, "outputs", "curve_plot")
os.makedirs(output_dir, exist_ok=True)

method_colors = {
    "SigmaRL": "#4C72B0",
    "TSC": "#55A868",
    "XP-MARL": "#C44E52",
    "MFPO": "#8172B2",
}
methods_order = ["MFPO", "SigmaRL", "XP-MARL", "TSC"]

for scenario in scenario_order:
    if scenario not in data:
        continue
    fig, ax = plt.subplots(figsize=(4, 4))
    available_methods = data[scenario].keys()
    methods = [m for m in methods_order if m in available_methods]
    for method in methods:
        pairs = data[scenario][method]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        xs = [p[0] for p in pairs_sorted]
        ys = [p[1] for p in pairs_sorted]
        ax.plot(
            xs,
            ys,
            marker="o",
            color=method_colors.get(method),
            label=method,
        )
    ax.set_title(scenario_titles.get(scenario, scenario), fontsize=14)
    ax.set_xlabel("Number of Agents", fontsize=14)
    ax.set_ylabel("Collision Rate [%]", fontsize=14)
    ax.tick_params(axis="x", labelsize=20)  # 设置 X 轴刻度字体大小
    ax.tick_params(axis="y", labelsize=12)  # 设置 Y 轴刻度字体大小
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"), fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True), fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()
    filename = f"curve_{scenario}.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)
