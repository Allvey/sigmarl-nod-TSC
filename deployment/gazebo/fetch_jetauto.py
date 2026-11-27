#!/usr/bin/python3
"""Fetch a pinned official JetAuto description; verify each Git blob checksum."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "569b77d3ed5ae4a581cdd8e341f29eb8476284da"
PREFIX = "src/jetauto_simulations/jetauto_description/"
ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "generated/vendor/jetauto_description"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "sigmarl-jetauto-simulation"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def fetch():
    tree = json.loads(get(f"https://api.github.com/repos/Hiwonder/JetAuto/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned an incomplete model file list")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob"
               and entry["path"].startswith(PREFIX) and "/pro/" not in entry["path"]]

    def download(entry):
        relative = Path(entry["path"][len(PREFIX):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid upstream model path")
        destination = DESTINATION / relative

        def valid(content):
            header = f"blob {len(content)}\0".encode()
            return hashlib.sha1(header + content).hexdigest() == entry["sha"]

        if destination.exists() and valid(destination.read_bytes()):
            return
        content = get(f"https://raw.githubusercontent.com/Hiwonder/JetAuto/{COMMIT}/{entry['path']}")
        if not valid(content):
            raise RuntimeError(f"Upstream checksum mismatch: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.write_bytes(content)
        temporary.replace(destination)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(download, entries))
    provenance = {"repository": "https://github.com/Hiwonder/JetAuto", "commit": COMMIT,
                  "upstream_directory": PREFIX, "files": len(entries),
                  "note": "Official ROS1 description assets reused for ROS2 simulation; runtime plugins are adapted separately."}
    (DESTINATION / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {len(entries)} official JetAuto description files at {COMMIT}")


if __name__ == "__main__":
    fetch()
