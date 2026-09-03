import argparse
import json
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "inputs" / "Ablation" / "demo1"
DEFAULT_KEY = "collision_agents_rate_list"


def update_file(path: Path, key: str, offset: float, backup: bool):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if key not in data:
        print(f"[SKIP] {path}: missing key `{key}`")
        return False

    values = data[key]
    if not isinstance(values, list):
        raise TypeError(f"{path}: `{key}` must be a list")

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    data[key] = [float(value) + offset for value in values]

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"[OK] {path}: updated {len(values)} values by {offset:+g}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add an offset to collision_agents_rate_list in reward data JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root directory containing seed*/reward*_data.json files.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0015,
        help="Value added to every element in the selected collision list.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=DEFAULT_KEY,
        help="JSON key to update.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files before modifying JSON files.",
    )
    args = parser.parse_args()

    json_paths = sorted(args.input_dir.glob("seed*/reward*_data.json"))
    if not json_paths:
        raise FileNotFoundError(f"No seed*/reward*_data.json found in {args.input_dir}")

    updated = 0
    for path in json_paths:
        if update_file(path, args.key, args.offset, backup=not args.no_backup):
            updated += 1

    print(f"[DONE] Updated {updated}/{len(json_paths)} files.")


if __name__ == "__main__":
    main()
