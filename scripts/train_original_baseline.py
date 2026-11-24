"""Train official SigmaRL/XP-MARL inside this project, without the NOD runtime."""
import argparse
import json
from pathlib import Path

from baselines.original_sigmarl.runtime import PROJECT_ROOT, launch_worker, verify_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sigmarl", "xp_marl"), required=True)
    parser.add_argument("--config", help="Optional original-runtime configuration JSON")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_upstream(), indent=2))
        return
    config = Path(args.config or ("configs/baselines/" + args.method + "_original.json"))
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    launch_worker(["train", "--method", args.method, "--config", str(config.resolve())])


if __name__ == "__main__":
    main()
