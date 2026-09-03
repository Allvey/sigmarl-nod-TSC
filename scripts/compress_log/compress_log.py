import re
from textwrap import dedent

raw_text = dedent(
    """
MFPO:
Scenario: CPM_entire
[LOG] Agent-agent collision rate [%]: mean=1.71, min=0.42, max=3.00
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.71, min=0.42, max=3.00
[LOG] Relative centerline deviation [%]: mean=39.60, min=38.23, max=41.83
[LOG] Relative average speed [%]: mean=90.01, min=89.35, max=90.75
[LOG] Relative average speed with collision penalty [%]: mean=86.59
[LOG] Smoothness [%]: mean=5.21, min=5.01, max=5.47
[LOG] Smoothness with collision penalty [%]: mean=6.92
[LOG] Smoothness longitudinal [%]: mean=0.92
[LOG] Smoothness longitudinal with collision penalty [%]: mean=2.63
[LOG] Smoothness lateral [%]: mean=9.49
[LOG] Smoothness lateral with collision penalty [%]: mean=11.20
[LOG] Composite scores: -1.00
=========================================
Scenario: intersection_2
[LOG] Agent-agent collision rate [%]: mean=5.46, min=4.09, max=7.09
[LOG] Agent-lanelet collision rate [%]: mean=2.29, min=1.50, max=3.00
[LOG] Total collision rate [%]: mean=7.75, min=6.17, max=9.26
[LOG] Relative centerline deviation [%]: mean=34.90, min=33.92, max=35.86
[LOG] Relative average speed [%]: mean=87.99, min=87.65, max=88.26
[LOG] Relative average speed with collision penalty [%]: mean=72.48
[LOG] Smoothness [%]: mean=5.21, min=5.00, max=5.36
[LOG] Smoothness with collision penalty [%]: mean=12.96
[LOG] Smoothness longitudinal [%]: mean=1.69
[LOG] Smoothness longitudinal with collision penalty [%]: mean=9.44
[LOG] Smoothness lateral [%]: mean=8.72
[LOG] Smoothness lateral with collision penalty [%]: mean=16.47
[LOG] Composite scores: -1.00
=========================================
Scenario: on_ramp_1
[LOG] Agent-agent collision rate [%]: mean=4.37, min=3.34, max=5.67
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=4.37, min=3.34, max=5.67
[LOG] Relative centerline deviation [%]: mean=34.96, min=34.52, max=35.65
[LOG] Relative average speed [%]: mean=87.37, min=87.24, max=87.52
[LOG] Relative average speed with collision penalty [%]: mean=78.62
[LOG] Smoothness [%]: mean=3.57, min=3.47, max=3.66
[LOG] Smoothness with collision penalty [%]: mean=7.95
[LOG] Smoothness longitudinal [%]: mean=1.47
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.85
[LOG] Smoothness lateral [%]: mean=5.67
[LOG] Smoothness lateral with collision penalty [%]: mean=10.04
[LOG] Composite scores: -1.00
=========================================
Scenario: roundabout_1
[LOG] Agent-agent collision rate [%]: mean=4.80, min=3.84, max=5.59
[LOG] Agent-lanelet collision rate [%]: mean=0.78, min=0.33, max=1.33
[LOG] Total collision rate [%]: mean=5.57, min=4.59, max=6.26
[LOG] Relative centerline deviation [%]: mean=37.74, min=36.65, max=38.70
[LOG] Relative average speed [%]: mean=87.55, min=87.27, max=87.79
[LOG] Relative average speed with collision penalty [%]: mean=76.41
[LOG] Smoothness [%]: mean=5.15, min=4.93, max=5.29
[LOG] Smoothness with collision penalty [%]: mean=10.73
[LOG] Smoothness longitudinal [%]: mean=1.36
[LOG] Smoothness longitudinal with collision penalty [%]: mean=6.93
[LOG] Smoothness lateral [%]: mean=8.95
[LOG] Smoothness lateral with collision penalty [%]: mean=14.52
[LOG] Composite scores: -1.00
=========================================

Sigma:
Scenario: CPM_entire
[LOG] Agent-agent collision rate [%]: mean=1.67, min=0.83, max=2.25
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=1.67, min=0.83, max=2.25
[LOG] Relative centerline deviation [%]: mean=38.41, min=37.00, max=40.22
[LOG] Relative average speed [%]: mean=90.39, min=89.18, max=92.26
[LOG] Relative average speed with collision penalty [%]: mean=87.06
[LOG] Smoothness [%]: mean=4.83, min=4.50, max=5.18
[LOG] Smoothness with collision penalty [%]: mean=6.50
[LOG] Smoothness longitudinal [%]: mean=0.91
[LOG] Smoothness longitudinal with collision penalty [%]: mean=2.58
[LOG] Smoothness lateral [%]: mean=8.75
[LOG] Smoothness lateral with collision penalty [%]: mean=10.42
[LOG] Composite scores: -1.00
=========================================
Scenario: intersection_2
[LOG] Agent-agent collision rate [%]: mean=5.45, min=4.42, max=6.51
[LOG] Agent-lanelet collision rate [%]: mean=0.95, min=0.50, max=1.67
[LOG] Total collision rate [%]: mean=6.40, min=5.25, max=7.84
[LOG] Relative centerline deviation [%]: mean=28.96, min=28.11, max=30.55
[LOG] Relative average speed [%]: mean=92.00, min=91.84, max=92.24
[LOG] Relative average speed with collision penalty [%]: mean=79.19
[LOG] Smoothness [%]: mean=4.85, min=4.68, max=5.12
[LOG] Smoothness with collision penalty [%]: mean=11.25
[LOG] Smoothness longitudinal [%]: mean=1.19
[LOG] Smoothness longitudinal with collision penalty [%]: mean=7.59
[LOG] Smoothness lateral [%]: mean=8.51
[LOG] Smoothness lateral with collision penalty [%]: mean=14.91
[LOG] Composite scores: -1.00
=========================================
Scenario: on_ramp_1
[LOG] Agent-agent collision rate [%]: mean=4.28, min=3.25, max=5.59
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=4.28, min=3.25, max=5.59
[LOG] Relative centerline deviation [%]: mean=25.59, min=25.12, max=26.19
[LOG] Relative average speed [%]: mean=91.68, min=91.59, max=91.81
[LOG] Relative average speed with collision penalty [%]: mean=83.11
[LOG] Smoothness [%]: mean=2.61, min=2.55, max=2.68
[LOG] Smoothness with collision penalty [%]: mean=6.90
[LOG] Smoothness longitudinal [%]: mean=0.95
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.23
[LOG] Smoothness lateral [%]: mean=4.28
[LOG] Smoothness lateral with collision penalty [%]: mean=8.56
[LOG] Composite scores: -1.00
=========================================
Scenario: roundabout_1
[LOG] Agent-agent collision rate [%]: mean=5.02, min=4.17, max=5.92
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.25
[LOG] Total collision rate [%]: mean=5.06, min=4.25, max=5.92
[LOG] Relative centerline deviation [%]: mean=31.64, min=30.87, max=32.43
[LOG] Relative average speed [%]: mean=91.54, min=91.36, max=91.81
[LOG] Relative average speed with collision penalty [%]: mean=81.43
[LOG] Smoothness [%]: mean=5.21, min=5.02, max=5.35
[LOG] Smoothness with collision penalty [%]: mean=10.27
[LOG] Smoothness longitudinal [%]: mean=1.12
[LOG] Smoothness longitudinal with collision penalty [%]: mean=6.18
[LOG] Smoothness lateral [%]: mean=9.31
[LOG] Smoothness lateral with collision penalty [%]: mean=14.36
[LOG] Composite scores: -1.00
=========================================

XP:
Scenario: CPM_entire
[LOG] Agent-agent collision rate [%]: mean=0.19, min=0.00, max=0.42
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.22, min=0.00, max=0.50
[LOG] Relative centerline deviation [%]: mean=50.71, min=43.19, max=59.64
[LOG] Relative average speed [%]: mean=85.25, min=82.75, max=88.45
[LOG] Relative average speed with collision penalty [%]: mean=84.80
[LOG] Smoothness [%]: mean=5.39, min=4.65, max=6.00
[LOG] Smoothness with collision penalty [%]: mean=5.61
[LOG] Smoothness longitudinal [%]: mean=2.05
[LOG] Smoothness longitudinal with collision penalty [%]: mean=2.27
[LOG] Smoothness lateral [%]: mean=8.73
[LOG] Smoothness lateral with collision penalty [%]: mean=8.95
[LOG] Composite scores: -1.00
=========================================
Scenario: intersection_2
[LOG] Agent-agent collision rate [%]: mean=0.69, min=0.25, max=1.17
[LOG] Agent-lanelet collision rate [%]: mean=0.98, min=0.50, max=1.33
[LOG] Total collision rate [%]: mean=1.66, min=1.00, max=2.25
[LOG] Relative centerline deviation [%]: mean=25.58, min=24.74, max=26.50
[LOG] Relative average speed [%]: mean=81.49, min=79.92, max=83.01
[LOG] Relative average speed with collision penalty [%]: mean=78.16
[LOG] Smoothness [%]: mean=5.88, min=5.55, max=6.26
[LOG] Smoothness with collision penalty [%]: mean=7.54
[LOG] Smoothness longitudinal [%]: mean=3.92
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.58
[LOG] Smoothness lateral [%]: mean=7.84
[LOG] Smoothness lateral with collision penalty [%]: mean=9.51
[LOG] Composite scores: -1.00
=========================================
Scenario: on_ramp_1
[LOG] Agent-agent collision rate [%]: mean=0.24, min=0.00, max=0.58
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.24, min=0.00, max=0.58
[LOG] Relative centerline deviation [%]: mean=28.47, min=27.91, max=29.11
[LOG] Relative average speed [%]: mean=81.90, min=79.28, max=84.46
[LOG] Relative average speed with collision penalty [%]: mean=81.41
[LOG] Smoothness [%]: mean=5.20, min=4.80, max=5.63
[LOG] Smoothness with collision penalty [%]: mean=5.45
[LOG] Smoothness longitudinal [%]: mean=5.33
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.57
[LOG] Smoothness lateral [%]: mean=5.08
[LOG] Smoothness lateral with collision penalty [%]: mean=5.32
[LOG] Composite scores: -1.00
=========================================
Scenario: roundabout_1
[LOG] Agent-agent collision rate [%]: mean=0.56, min=0.08, max=1.25
[LOG] Agent-lanelet collision rate [%]: mean=0.07, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=0.62, min=0.08, max=1.25
[LOG] Relative centerline deviation [%]: mean=30.91, min=30.23, max=31.86
[LOG] Relative average speed [%]: mean=79.24, min=77.46, max=81.93
[LOG] Relative average speed with collision penalty [%]: mean=77.99
[LOG] Smoothness [%]: mean=6.86, min=6.46, max=7.42
[LOG] Smoothness with collision penalty [%]: mean=7.48
[LOG] Smoothness longitudinal [%]: mean=5.32
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.94
[LOG] Smoothness lateral [%]: mean=8.39
[LOG] Smoothness lateral with collision penalty [%]: mean=9.02
[LOG] Composite scores: -1.00
=========================================

TSC:
Scenario: CPM_entire
[LOG] Agent-agent collision rate [%]: mean=0.04, min=0.00, max=0.17
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Total collision rate [%]: mean=0.07, min=0.00, max=0.33
[LOG] Relative centerline deviation [%]: mean=48.50, min=44.68, max=51.99
[LOG] Relative average speed [%]: mean=88.31, min=85.94, max=89.74
[LOG] Relative average speed with collision penalty [%]: mean=88.17
[LOG] Smoothness [%]: mean=5.34, min=4.74, max=6.36
[LOG] Smoothness with collision penalty [%]: mean=5.42
[LOG] Smoothness longitudinal [%]: mean=1.87
[LOG] Smoothness longitudinal with collision penalty [%]: mean=1.95
[LOG] Smoothness lateral [%]: mean=8.81
[LOG] Smoothness lateral with collision penalty [%]: mean=8.89
[LOG] Composite scores: -1.00
=========================================
Scenario: intersection_2
[LOG] Agent-agent collision rate [%]: mean=0.18, min=0.00, max=0.50
[LOG] Agent-lanelet collision rate [%]: mean=0.81, min=0.33, max=1.33
[LOG] Total collision rate [%]: mean=0.99, min=0.42, max=1.67
[LOG] Relative centerline deviation [%]: mean=36.56, min=35.53, max=37.72
[LOG] Relative average speed [%]: mean=83.81, min=81.10, max=85.08
[LOG] Relative average speed with collision penalty [%]: mean=81.83
[LOG] Smoothness [%]: mean=6.34, min=6.11, max=6.63
[LOG] Smoothness with collision penalty [%]: mean=7.33
[LOG] Smoothness longitudinal [%]: mean=4.30
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.29
[LOG] Smoothness lateral [%]: mean=8.39
[LOG] Smoothness lateral with collision penalty [%]: mean=9.38
[LOG] Composite scores: -1.00
=========================================
Scenario: on_ramp_1
[LOG] Agent-agent collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Agent-lanelet collision rate [%]: mean=0.00, min=0.00, max=0.00
[LOG] Total collision rate [%]: mean=0.03, min=0.00, max=0.17
[LOG] Relative centerline deviation [%]: mean=34.64, min=34.20, max=35.01
[LOG] Relative average speed [%]: mean=84.78, min=84.00, max=85.75
[LOG] Relative average speed with collision penalty [%]: mean=84.71
[LOG] Smoothness [%]: mean=5.80, min=5.43, max=6.28
[LOG] Smoothness with collision penalty [%]: mean=5.84
[LOG] Smoothness longitudinal [%]: mean=5.06
[LOG] Smoothness longitudinal with collision penalty [%]: mean=5.09
[LOG] Smoothness lateral [%]: mean=6.55
[LOG] Smoothness lateral with collision penalty [%]: mean=6.58
[LOG] Composite scores: -1.00
=========================================
Scenario: roundabout_1
[LOG] Agent-agent collision rate [%]: mean=0.08, min=0.00, max=0.25
[LOG] Agent-lanelet collision rate [%]: mean=0.03, min=0.00, max=0.08
[LOG] Total collision rate [%]: mean=0.10, min=0.00, max=0.33
[LOG] Relative centerline deviation [%]: mean=37.28, min=36.28, max=39.15
[LOG] Relative average speed [%]: mean=79.58, min=77.78, max=81.02
[LOG] Relative average speed with collision penalty [%]: mean=79.38
[LOG] Smoothness [%]: mean=8.20, min=7.46, max=9.11
[LOG] Smoothness with collision penalty [%]: mean=8.30
[LOG] Smoothness longitudinal [%]: mean=7.34
[LOG] Smoothness longitudinal with collision penalty [%]: mean=7.44
[LOG] Smoothness lateral [%]: mean=9.06
[LOG] Smoothness lateral with collision penalty [%]: mean=9.16
[LOG] Composite scores: -1.00
=========================================
"""
).strip("\n")


def compress_log(text: str) -> str:
    # 需要丢弃的指标关键词（出现这些字段的整行删掉）
    drop_keywords = [
        "Relative centerline deviation",
        "Relative average speed [%]:",  # 未惩罚速度
        "Smoothness [%]:",  # 未惩罚总 smooth
        "Smoothness longitudinal [%]:",  # 未惩罚纵向
        "Smoothness lateral [%]:",  # 未惩罚横向
        "Composite scores",
    ]

    def should_drop(line: str) -> bool:
        return any(k in line for k in drop_keywords)

    replace_map = {
        "Agent-agent collision rate [%]": "CR_AA",
        "Agent-lanelet collision rate [%]": "CR_AM",
        "Total collision rate [%]": "CR",
        "Relative average speed with collision penalty [%]": "AS",
        "Smoothness with collision penalty [%]": "SM",
        "Smoothness longitudinal with collision penalty [%]": "SM_LO",
        "Smoothness lateral with collision penalty [%]": "SM_LA",
    }

    scenario_map = {
        "CPM_entire": "Clover",
        "intersection_2": "Weave",
        "on_ramp_1": "Merge",
        "roundabout_1": "Bypass",
    }

    out_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("Scenario:"):
            _, name = line.split(":", 1)
            name = name.strip()
            new_name = scenario_map.get(name, name)
            line = f"Scenario: {new_name}"

        if "[LOG]" in line and should_drop(line):
            continue

        if line.startswith("[LOG]"):
            for old, new in replace_map.items():
                if old in line:
                    line = line.replace(f"[LOG] {old}:", f"[LOG] {new}:")
                    break

        out_lines.append(line)

    return "\n".join(out_lines)


if __name__ == "__main__":
    result = compress_log(raw_text)
    print(result)
