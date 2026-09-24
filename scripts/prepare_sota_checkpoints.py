"""Download and validate the official SigmaRL and XP-MARL checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "itsc24": "https://raw.githubusercontent.com/bassamlab/assets/main/sigmarl/checkpoints/itsc24.zip",
    "icra25": "https://raw.githubusercontent.com/bassamlab/assets/main/sigmarl/checkpoints/icra25.zip",
}
EXPECTED = {
    "itsc24": Path("itsc24") / "M0 (our)",
    "icra25": Path("icra25") / "M1 (XP-MARL)",
}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
        bundle.extractall(destination)


def _best_checkpoint(model_dir: Path) -> tuple[float, Path]:
    candidates = []
    for json_path in model_dir.glob("reward*_data.json"):
        try:
            reward = float(json_path.name[len("reward") : -len("_data.json")])
        except ValueError:
            continue
        if (model_dir / f"reward{reward:.2f}_policy.pth").is_file():
            candidates.append((reward, json_path))
    if not candidates:
        raise FileNotFoundError(f"No complete reward checkpoint in {model_dir}")
    return max(candidates, key=lambda item: item[0])


def _validate_model(name: str, model_dir: Path) -> None:
    reward, json_path = _best_checkpoint(model_dir)
    with json_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    params = raw.get("parameters", raw)
    prioritized = bool(params.get("is_using_prioritized_marl", False))
    if name == "icra25" and not prioritized:
        raise ValueError(f"{model_dir} is not an XP-MARL checkpoint")
    prefix = model_dir / f"reward{reward:.2f}"
    if prioritized and not Path(str(prefix) + "_priority_policy.pth").is_file():
        raise FileNotFoundError(f"Missing XP-MARL priority policy for {prefix}")
    print(
        f"[ready] {name}: {model_dir} | checkpoint=reward{reward:.2f} "
        f"| prioritized_marl={prioritized}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", default="checkpoints",
        help="Destination relative to the repository root (default: checkpoints).",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Download an archive again even when the expected model already exists.",
    )
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = ROOT / checkpoint_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    download_dir = checkpoint_root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    for name, url in ARCHIVES.items():
        model_dir = checkpoint_root / EXPECTED[name]
        if args.force_download or not model_dir.is_dir():
            archive = download_dir / f"{name}.zip"
            print(f"[download] {url}")
            with urllib.request.urlopen(url) as response, archive.open("wb") as file:
                shutil.copyfileobj(response, file)
            print(f"[extract] {archive} -> {checkpoint_root}")
            _safe_extract(archive, checkpoint_root)
        _validate_model(name, model_dir)


if __name__ == "__main__":
    main()
