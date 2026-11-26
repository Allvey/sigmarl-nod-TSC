"""Launch the original TSC project with its own module search path."""
import os
from pathlib import Path
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parents[1]
DEFAULT_TSC_PROJECT = PROJECT_ROOT.parent / "sigmarl-traffic"


def launch_worker(arguments, project=DEFAULT_TSC_PROJECT):
    project = Path(project).resolve()
    for relative in ("utilities/mappo_cavs.py", "utilities/topology_module.py",
                     "scenarios/road_traffic.py"):
        if not (project / relative).is_file():
            raise FileNotFoundError("Missing TSC source: " + str(project / relative))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project)
    subprocess.run(
        [sys.executable, str(PACKAGE / "worker.py"), "--project", str(project)]
        + list(arguments), cwd=project, env=env, check=True,
    )
