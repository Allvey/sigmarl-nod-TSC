import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path when running as a script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
default_config_path = os.path.join(script_dir, "inputs", "config.json")

from utilities.helper_training import Parameters
from utilities.mappo_cavs import mappo_cavs


def _parse_seeds(arg, runs: int):
    if arg:
        if isinstance(arg, list):
            parts = []
            for token in arg:
                for p in token.split(","):
                    p = p.strip()
                    if p != "":
                        parts.append(p)
            return [int(p) for p in parts]
        else:
            parts = [s.strip() for s in str(arg).split(",") if s.strip() != ""]
            return [int(p) for p in parts]
    if runs and runs > 0:
        return list(range(runs))
    return [1, 2, 3]


def _ensure_trailing_slash(path: str):
    return path if path.endswith("/") else path + "/"


def _save_dir_from_config(config_save_path: str, seed: int):
    save_path = Path(config_save_path)
    if not save_path.is_absolute():
        save_path = Path(project_root) / save_path

    leaf = save_path.name
    if leaf == "":
        leaf = save_path.parent.name

    if leaf.startswith("seed") and leaf.replace("seed_", "seed").replace("seed", "").isdigit():
        separator = "_" if leaf.startswith("seed_") else ""
        return save_path.parent / f"seed{separator}{seed}"

    return save_path / f"seed_{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--seeds", nargs="*", default=[])
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--save-root", type=str, default=None)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds, args.runs)
    base_params = Parameters.from_json(args.config)

    for seed in seeds:
        if args.save_root:
            base_dir = _ensure_trailing_slash(args.save_root)
            save_dir = f"{base_dir}seed_{seed}/"
        else:
            save_dir = str(_save_dir_from_config(base_params.where_to_save, seed))
            save_dir = _ensure_trailing_slash(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        params = Parameters.from_json(args.config)
        params.where_to_save = save_dir
        params.is_load_model = False
        params.is_continue_train = False

        mappo_cavs(parameters=params)


if __name__ == "__main__":
    main()
