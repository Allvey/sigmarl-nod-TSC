"""Standard-library-only launcher and integrity checks for official baselines."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parents[1]
UPSTREAM = PACKAGE / "upstream"
BACKENDS = ("current", "auto", "sigmarl_1_2_0", "xpmarl_1_2_0")


def verify_upstream():
    manifest = json.loads((PACKAGE / "provenance.json").read_text())
    for name, expected in manifest["files"].items():
        path = UPSTREAM / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Original SigmaRL source missing or modified: " + name)
    return {key: manifest[key] for key in ("upstream_url", "tag", "commit")}


def resolve_backend(backend, raw_parameters):
    if backend not in BACKENDS:
        raise ValueError("Unknown evaluation backend: " + str(backend))
    if backend == "auto":
        # Modified-project checkpoints carry this field even when NOD is off.
        if "is_using_nod_opinion" in raw_parameters:
            return "current"
        return ("xpmarl_1_2_0" if raw_parameters.get("is_using_prioritized_marl", False)
                else "sigmarl_1_2_0")
    return backend


def validate_method(parameters, method):
    if any(key.startswith(("nod_", "safety_", "dgppo_", "deadlock_"))
           or key.startswith("is_using_nod") or key.startswith("is_using_safety")
           or key == "is_using_deadlock_critic" for key in parameters):
        raise ValueError("Modified NOD-project metadata cannot be an original baseline")
    prioritized = bool(parameters.get("is_using_prioritized_marl", False))
    if prioritized != (method == "xp_marl"):
        raise ValueError("Checkpoint/config priority setting disagrees with " + method)
    if parameters.get("is_using_opponent_modeling", False):
        raise ValueError("This comparison uses the original main methods, without opponent modeling")
    if prioritized and parameters.get("prioritization_method", "marl") != "marl":
        raise ValueError("Original XP-MARL main method requires learned priorities")


def launch_worker(arguments):
    verify_upstream()
    env = os.environ.copy()
    # A separate process prevents utilities/scenarios imports from sharing the
    # NOD project's module cache. Prefer only the vendored upstream on PYTHONPATH.
    env["PYTHONPATH"] = str(UPSTREAM)
    subprocess.run(
        [sys.executable, str(PACKAGE / "worker.py")] + list(arguments),
        cwd=UPSTREAM, env=env, check=True,
    )
